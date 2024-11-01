// SPDX-License-Identifier: MPL-2.0

#![warn(unused)]

use alloc::{boxed::Box, sync::Arc};
use core::{
    fmt,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use ostd::{
    cpu::{all_cpus, CpuId, CpuSet, PinCurrentCpu},
    sync::SpinLock,
    task::{
        scheduler::{inject_scheduler, EnqueueFlags, LocalRunQueue, Scheduler, UpdateFlags},
        Task,
    },
    trap::disable_local,
};

mod time;

mod fair;
mod idle;
mod real_time;
mod stop;

use ostd::arch::read_tsc as sched_clock;

use super::priority::{Nice, Priority, RangedU8};
use crate::thread::Thread;

#[allow(unused)]
pub fn init() {
    inject_scheduler(Box::leak(Box::new(ClassScheduler::default())));
}

/// Represents the middle layer between scheduling classes and generic scheduler
/// traits. It consists of all the sets of run queues for CPU cores. Other global
/// information may also be stored here.
pub struct ClassScheduler {
    rqs: Box<[SpinLock<PerCpuClassRqSet>]>,
}

/// Represents the run queue for each CPU core. It stores a list of run queues for
/// scheduling classes in its corresponding CPU core. The current task of this CPU
/// core is also stored in this structure.
struct PerCpuClassRqSet {
    stop: Arc<stop::StopClassRq>,
    real_time: real_time::RealTimeClassRq,
    fair: fair::FairClassRq,
    idle: idle::IdleClassRq,
    current: Option<Arc<Task>>,
}

/// The run queue for scheduling classes (the main trait). Scheduling classes
/// should implement this trait to function as expected.
trait SchedClassRq: Send + fmt::Debug {
    /// Enqueues a task into the run queue.
    fn enqueue(&mut self, thread: Arc<Thread>);

    /// Checks if the run queue is empty.
    fn is_empty(&mut self) -> bool;

    /// Picks the next task for running.
    fn pick_next(&mut self) -> Option<Arc<Thread>>;

    /// Update the information of the current task.
    fn update_current(&mut self, now: u64, thread: &Thread, flags: UpdateFlags) -> bool;
}

pub use real_time::RealTimePolicy;

/// The User-chosen scheduling policy.
///
/// The scheduling policies are specified by the user, usually through its priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SchedPolicy {
    Stop,
    RealTime {
        rt_prio: real_time::RtPrio,
        rt_policy: RealTimePolicy,
    },
    Fair(Nice),
    Idle,
}

impl From<Priority> for SchedPolicy {
    fn from(priority: Priority) -> Self {
        match priority.range().get() {
            0 => SchedPolicy::Stop,
            rt @ 1..=99 => SchedPolicy::RealTime {
                rt_prio: RangedU8::new(rt),
                rt_policy: Default::default(),
            },
            100..=139 => SchedPolicy::Fair(priority.into()),
            _ => SchedPolicy::Idle,
        }
    }
}

/// The scheduling attribute for a thread.
///
/// This is used to store the scheduling policy and runtime parameters for each
/// scheduling class.
#[derive(Debug)]
pub struct SchedAttr {
    policy: SpinLock<SchedPolicy>,

    start: AtomicU64,
    period_start: AtomicU64,

    real_time: real_time::RealTimeAttr,
    fair: fair::FairAttr,
}

impl SchedAttr {
    /// Constructs a new `SchedAttr` with the given scheduling policy.
    pub fn new(policy: SchedPolicy) -> Self {
        Self {
            policy: SpinLock::new(policy),
            start: Default::default(),
            period_start: Default::default(),
            real_time: {
                let (prio, policy) = match policy {
                    SchedPolicy::RealTime { rt_prio, rt_policy } => (rt_prio.get(), rt_policy),
                    _ => (real_time::RtPrio::MAX, Default::default()),
                };
                real_time::RealTimeAttr::new(prio, policy)
            },
            fair: fair::FairAttr::new(match policy {
                SchedPolicy::Fair(nice) => nice,
                _ => Nice::default(),
            }),
        }
    }

    /// Retrieves the current scheduling policy of the thread.
    pub fn policy(&self) -> SchedPolicy {
        *self.policy.lock()
    }

    /// Updates the scheduling policy of the thread.
    ///
    /// Specifically for real-time policies, if the new policy doesn't
    /// specify a base slice factor for RR, the old one will be kept.
    pub fn set_policy(&self, mut policy: SchedPolicy) {
        let mut guard = self.policy.lock();
        match policy {
            SchedPolicy::RealTime { rt_prio, rt_policy } => {
                self.real_time.update(rt_prio.get(), rt_policy);
            }
            SchedPolicy::Fair(nice) => self.fair.update(nice),
            _ => {}
        }

        // Keep the old base slice factor if the new policy doesn't specify one.
        if let (
            SchedPolicy::RealTime {
                rt_policy:
                    RealTimePolicy::RoundRobin {
                        base_slice_factor: slot,
                    },
                ..
            },
            SchedPolicy::RealTime {
                rt_policy: RealTimePolicy::RoundRobin { base_slice_factor },
                ..
            },
        ) = (*guard, &mut policy)
        {
            *base_slice_factor = slot.or(*base_slice_factor);
        }

        *guard = policy;
    }
}

impl Scheduler for ClassScheduler {
    fn enqueue(&self, task: Arc<Task>, _flags: EnqueueFlags) -> Option<CpuId> {
        let thread = Thread::borrow_from_task(&task);

        let cpu_affinity = thread.atomic_cpu_affinity().load();
        let cpu = self.select_cpu(&cpu_affinity);
        task.schedule_info().cpu.set_if_is_none(cpu).ok()?;
        drop(cpu_affinity);

        let mut rq = self.rqs[cpu.as_usize()].disable_irq().lock();
        rq.enqueue_thread(thread);
        Some(cpu)
    }

    fn local_mut_rq_with(&self, f: &mut dyn FnMut(&mut dyn LocalRunQueue)) {
        let guard = disable_local();
        let mut lock = self.rqs[guard.current_cpu().as_usize()].lock();
        f(&mut *lock)
    }

    fn local_rq_with(&self, f: &mut dyn FnMut(&dyn LocalRunQueue)) {
        let guard = disable_local();
        f(&*self.rqs[guard.current_cpu().as_usize()].lock())
    }
}

impl ClassScheduler {
    // TODO: Implement a better algorithm and replace the current naive implementation.
    fn select_cpu(&self, affinity: &CpuSet) -> CpuId {
        let guard = disable_local();
        let cur = guard.current_cpu();
        if affinity.contains(cur) {
            cur
        } else {
            affinity.iter().next().expect("empty affinity")
        }
    }
}

impl PerCpuClassRqSet {
    fn pick_next_thread(&mut self) -> Option<Arc<Thread>> {
        (self.stop.pick_next())
            .or_else(|| self.real_time.pick_next())
            .or_else(|| self.fair.pick_next())
            .or_else(|| self.idle.pick_next())
    }

    fn enqueue_thread(&mut self, thread: &Arc<Thread>) {
        let attr = thread.sched_attr();

        attr.period_start.store(sched_clock(), Relaxed);
        attr.start.store(sched_clock(), Relaxed);

        let cloned = thread.clone();
        match *attr.policy.lock() {
            SchedPolicy::Stop => self.stop.enqueue(cloned),
            SchedPolicy::RealTime { .. } => self.real_time.enqueue(cloned),
            SchedPolicy::Fair(_) => self.fair.enqueue(cloned),
            SchedPolicy::Idle => self.idle.enqueue(cloned),
        };
    }
}

impl LocalRunQueue for PerCpuClassRqSet {
    fn current(&self) -> Option<&Arc<Task>> {
        self.current.as_ref()
    }

    fn pick_next_current(&mut self) -> Option<&Arc<Task>> {
        match self.pick_next_thread() {
            Some(next) => {
                let next_task = next.task();
                if let Some(old_task) = self.current.replace(next_task.clone()) {
                    if Arc::ptr_eq(&old_task, &next_task) {
                        return None;
                    }
                    let old = Thread::borrow_from_task(&old_task);
                    self.enqueue_thread(old);
                }
                self.current.as_ref()
            }
            None => None,
        }
    }

    fn update_current(&mut self, flags: UpdateFlags) -> bool {
        if let Some(cur_task) = &self.current {
            let cur = Thread::borrow_from_task(cur_task);
            let now = sched_clock();

            let (update, lookahead) = match &*cur.sched_attr().policy.lock() {
                SchedPolicy::Stop => (self.stop.update_current(now, cur, flags), 0),
                SchedPolicy::RealTime { .. } => (self.real_time.update_current(now, cur, flags), 1),
                SchedPolicy::Fair(_) => (self.fair.update_current(now, cur, flags), 2),
                SchedPolicy::Idle => (self.idle.update_current(now, cur, flags), 3),
            };

            cur.sched_attr().start.store(now, Relaxed);

            let lookahead = {
                let mut ret = false;
                if lookahead >= 1 {
                    ret |= !self.stop.is_empty();
                }
                if lookahead >= 2 {
                    ret |= !self.real_time.is_empty();
                }
                if lookahead >= 3 {
                    ret |= !self.fair.is_empty();
                }
                ret
            };
            update || lookahead
        } else {
            true
        }
    }

    fn dequeue_current(&mut self) -> Option<Arc<Task>> {
        self.current.take().inspect(|cur_task| {
            cur_task.schedule_info().cpu.set_to_none();
        })
    }
}

impl Default for ClassScheduler {
    fn default() -> Self {
        let stop = stop::StopClassRq::new();
        let class_rq = |cpu| {
            SpinLock::new(PerCpuClassRqSet {
                stop: stop.clone(),
                real_time: real_time::RealTimeClassRq::new(cpu),
                fair: fair::FairClassRq::new(cpu),
                idle: idle::IdleClassRq::new(),
                current: None,
            })
        };
        ClassScheduler {
            rqs: all_cpus().map(class_rq).collect(),
        }
    }
}
