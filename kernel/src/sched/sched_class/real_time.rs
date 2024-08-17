// SPDX-License-Identifier: MPL-2.0

use alloc::collections::vec_deque::VecDeque;
use core::{array, num::NonZero};

use bitvec::{bitarr, BitArr};

use super::{time::base_slice_clocks, *};
use crate::sched::priority::Priority;

/// The scheduling attribute for the REAL-TIME scheduling class.
///
/// This structure provides not-only the priority of the thread,
/// but also the time slice for the thread, measured in [`sched_clock`]s.
///
/// - If the time slice is not set, the thread is considered to be a FIFO
///   thread, and will be executed to its end if there no thread with a
///   lower priority.
/// - If the time slice is set, the thread is considered to be an RR
///   (round-robin) thread, and will be executed for the time slice, and
///   then it will be put back to the inactive array.
#[derive(Debug, Clone, Copy)]
pub struct RealTimeAttr {
    priority: Priority,
    time_slice: Option<NonZero<u64>>, // SCHED_RR; SCHED_FIFO

    start: u64,
}

impl RealTimeAttr {
    pub fn new(priority: Priority) -> Self {
        let n = base_slice_clocks() * 20; // 0.75ms * 20 = 15ms
        RealTimeAttr {
            priority,
            // Defaults to be an RR thread.
            //
            // TODO: Add some scheduling strategy configuration to
            // be able to set the attr to be a FIFO thread.
            time_slice: NonZero::new(n),
            start: sched_clock(),
        }
    }
}

struct PrioArray {
    map: BitArr![for 100],
    queue: [VecDeque<Arc<Thread>>; 100],
}

impl core::fmt::Debug for PrioArray {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "map: ")?;
        f.debug_list().entries(self.map.iter_ones()).finish()?;
        for thread in self.queue.iter().flatten() {
            let attr = match *thread.sched_attr().lock() {
                SchedAttr::RealTime(real_time_attr) => real_time_attr,
                _ => unreachable!(),
            };
            writeln!(f, "    {attr:?}")?;
        }
        Ok(())
    }
}

impl PrioArray {
    fn enqueue(&mut self, thread: Arc<Thread>, prio: u8) {
        let queue = &mut self.queue[usize::from(prio)];
        let is_empty = queue.is_empty();
        queue.push_back(thread);
        if is_empty {
            self.map.set(usize::from(prio), true);
        }
    }

    fn pop(&mut self, target_cpu: CpuId) -> Option<Arc<Thread>> {
        let mut iter = self.map.iter_ones();
        let (thread, prio, queue) = 'find: loop {
            let prio = iter.next()? as u8;

            let queue = &mut self.queue[usize::from(prio)];
            for _ in 0..queue.len() {
                let thread = queue.pop_front()?;
                if thread.atomic_cpu_affinity().load().contains(target_cpu) {
                    break 'find (thread, prio, queue);
                }
                queue.push_back(thread);
            }
        };
        if queue.is_empty() {
            self.map.set(usize::from(prio), false);
        }
        Some(thread)
    }
}

/// The per-cpu run queue for the REAL-TIME scheduling class.
///
/// The REAL-TIME scheduling class is implemented as a classic O(1)
/// priority algorithm.
///
/// It uses a bit array to track which priority levels have runnable
/// threads, and a vector of queues to store the threads.
///
/// Threads are popped & dequeued from the active array (`array[index]`), and
/// are enqueued into the inactive array (`array[!index]`). When the active array
/// is empty, the 2 arrays are swapped by `index`.
#[derive(Debug)]
pub(super) struct RealTimeClassRq {
    cpu: CpuId,
    index: bool,
    array: [PrioArray; 2],
}

impl RealTimeClassRq {
    pub fn new(cpu: CpuId) -> RealTimeClassRq {
        RealTimeClassRq {
            cpu,
            index: false,
            array: array::from_fn(|_| PrioArray {
                map: bitarr![0; 100],
                queue: array::from_fn(|_| VecDeque::new()),
            }),
        }
    }

    fn active_array(&mut self) -> &mut PrioArray {
        &mut self.array[usize::from(self.index)]
    }

    fn inactive_array(&mut self) -> &mut PrioArray {
        &mut self.array[usize::from(!self.index)]
    }

    fn swap_arrays(&mut self) {
        self.index = !self.index;
    }
}

impl SchedClassRq for RealTimeClassRq {
    type Attr = RealTimeAttr;

    fn enqueue(
        &mut self,
        thread: Arc<Thread>,
        attr: SpinLockGuard<'_, SchedAttr, PreemptDisabled>,
    ) {
        let prio = match &*attr {
            SchedAttr::RealTime(attr) => attr.priority,
            _ => unreachable!(),
        };
        self.inactive_array().enqueue(thread, prio.range().get());
    }

    fn is_empty(&mut self) -> bool {
        self.active_array().map.is_empty() && self.inactive_array().map.is_empty()
    }

    fn pick_next(&mut self) -> Option<Arc<Thread>> {
        let target_cpu = self.cpu;
        self.active_array().pop(target_cpu).or_else(|| {
            self.swap_arrays();
            self.active_array().pop(target_cpu)
        })
    }

    fn update_current(&mut self, rt: &mut RealTimeAttr, flags: UpdateFlags) -> bool {
        let now = sched_clock();
        let should_preempt = match flags {
            UpdateFlags::Tick | UpdateFlags::Wait => match rt.time_slice {
                Some(ts) => ts.get() <= now - rt.start,
                None => (self.inactive_array().map.iter_ones().next())
                    .is_some_and(|prio| prio > usize::from(rt.priority.range().get())),
            },
            UpdateFlags::Yield => true,
        };
        if should_preempt {
            rt.start = now;
        }
        should_preempt
    }
}
