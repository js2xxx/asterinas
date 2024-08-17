// SPDX-License-Identifier: MPL-2.0

#![warn(unused)]

use alloc::{boxed::Box, sync::Arc};
use core::fmt;

use ostd::{
    cpu::{all_cpus, CpuId, CpuSet, PinCurrentCpu},
    sync::{PreemptDisabled, SpinLock, SpinLockGuard},
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

use super::priority::Priority;
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
    type Attr;

    /// Enqueues a task into the run queue.
    fn enqueue(&mut self, thread: Arc<Thread>, attr: SpinLockGuard<'_, SchedAttr, PreemptDisabled>);

    fn is_empty(&mut self) -> bool;

    /// Picks the next task for running.
    fn pick_next(&mut self) -> Option<Arc<Thread>>;

    /// Update the information of the current task.
    fn update_current(&mut self, attr: &mut Self::Attr, flags: UpdateFlags) -> bool;
}

/// The scheduling attr. Users should not construct a scheduling attr
/// directly using its variant types.
#[derive(Debug)]
pub enum SchedAttr {
    Stop(stop::StopAttr),
    RealTime(real_time::RealTimeAttr),
    Fair(fair::FairAttr),
    Idle(idle::IdleAttr),
}

impl SchedAttr {
    /// Constructs a new scheduling attr object based from the given priority.
    pub fn new(priority: Priority) -> SchedAttr {
        match priority.range().get() {
            0 => SchedAttr::Stop(stop::StopAttr(())),
            1..100 => SchedAttr::RealTime(real_time::RealTimeAttr::new(priority)),
            100..=139 => SchedAttr::Fair(fair::FairAttr::new(priority.into())),
            _ => SchedAttr::Idle(idle::IdleAttr(())),
        }
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
        let cloned = thread.clone();
        let attr = thread.sched_attr().lock();
        match *attr {
            SchedAttr::Stop(_) => self.stop.enqueue(cloned, attr),
            SchedAttr::RealTime(_) => self.real_time.enqueue(cloned, attr),
            SchedAttr::Fair(_) => self.fair.enqueue(cloned, attr),
            SchedAttr::Idle(_) => self.idle.enqueue(cloned, attr),
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
            let (update, lookahead) = match &mut *cur.sched_attr().lock() {
                SchedAttr::Stop(stop_attr) => (self.stop.update_current(stop_attr, flags), 0),
                SchedAttr::RealTime(real_time_attr) => {
                    (self.real_time.update_current(real_time_attr, flags), 1)
                }
                SchedAttr::Fair(vruntime) => (self.fair.update_current(vruntime, flags), 2),
                SchedAttr::Idle(idle_attr) => (self.idle.update_current(idle_attr, flags), 3),
            };
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
