// SPDX-License-Identifier: MPL-2.0

use alloc::{collections::btree_set::BTreeSet, sync::Arc};
use core::{
    cmp,
    ops::Bound,
    sync::atomic::{AtomicU64, Ordering::*},
};

use ostd::{
    cpu::{num_cpus, CpuId},
    task::scheduler::UpdateFlags,
};

use super::{
    time::{base_slice_clocks, min_period_clocks},
    SchedClassRq,
};
use crate::{sched::priority::Nice, thread::Thread};

const WEIGHT_0: u64 = 1024;
pub const fn nice_to_weight(nice: Nice) -> u64 {
    // Calculated by the formula below:
    //
    //     weight = 1024 * 1.25^(-nice)
    //
    // We propose that every increment of the nice value results
    // in 12.5% change of the CPU load weight.
    const FACTOR_NUMERATOR: u64 = 5;
    const FACTOR_DENOMINATOR: u64 = 4;

    match nice.range().get() {
        0 => WEIGHT_0,
        nice @ 1.. => {
            let numerator = FACTOR_DENOMINATOR.pow(nice as u32);
            let denominator = FACTOR_NUMERATOR.pow(nice as u32);
            WEIGHT_0 * numerator / denominator
        }
        nice => {
            let numerator = FACTOR_NUMERATOR.pow((-nice) as u32);
            let denominator = FACTOR_DENOMINATOR.pow((-nice) as u32);
            WEIGHT_0 * numerator / denominator
        }
    }
}

/// The scheduling entity for the FAIR scheduling class.
///
/// The structure contains 2 significant indications:
/// `vruntime` & `period_start`.
///
/// # `vruntime`
///
/// The vruntime (virtual runtime) is calculated by the formula:
///
///     vruntime += (cur - start) * WEIGHT_0 / weight
///
/// and a thread with a lower vruntime gains a greater privilege to be
/// scheduled, making the whole run queue balanced on vruntime (thus FAIR).
///
/// # Scheduling periods
///
/// Scheduling periods is designed to calculate the time slice for each threads.
///
/// The time slice for each threads is calculated by the formula:
///
///     time_slice = period * weight / total_weight
///
/// where `total_weight` is the sum of all weights in the run queue including
/// the current thread and [`period`](FairClassRq::period) is calculated
/// regarding the number of running threads.
///
/// When a thread's time slice is exhausted, it will be put back to the
/// run queue.

#[derive(Debug)]
pub struct FairAttr {
    weight: AtomicU64,
    vruntime: AtomicU64,
}

impl FairAttr {
    pub fn new(nice: Nice) -> Self {
        FairAttr {
            weight: nice_to_weight(nice).into(),
            vruntime: Default::default(),
        }
    }

    pub fn update(&self, nice: Nice) {
        self.weight.store(nice_to_weight(nice), Relaxed);
    }

    fn tick(
        &self,
        now: u64,
        start: u64,
        period_start: u64,
        total_weight: u64,
        period: u64,
    ) -> bool {
        let weight = self.weight.load(Relaxed);

        // Update the vruntime.
        self.vruntime
            .fetch_add((now - start) * WEIGHT_0 / weight, Relaxed);

        debug_assert!(total_weight != 0);
        debug_assert!(period != 0);

        debug_assert!(now - period_start != 0);

        // Check if the time slice is exhausted.
        //
        // The expression is dedicated to avoid overflowing.
        let slice = period * weight / total_weight;
        now - period_start > slice
    }
}

/// The wrapper for threads in the FAIR run queue.
///
/// This structure is used to provide the capability for keying in the
/// run queue implemented by `BTreeSet` in the `FairClassRq`.
struct FairQueueItem(Arc<Thread>);

impl core::fmt::Debug for FairQueueItem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.key())
    }
}

impl FairQueueItem {
    fn key(&self) -> u64 {
        self.0.sched_attr().fair.vruntime.load(Relaxed)
    }
}

impl PartialEq for FairQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.key().eq(&other.key())
    }
}

impl Eq for FairQueueItem {}

impl PartialOrd for FairQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FairQueueItem {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

/// The per-cpu run queue for the FAIR scheduling class.
///
/// See [`FairAttr`] for the explanation of vruntimes and scheduling periods.
///
/// The structure contains a `BTreeSet` to store the threads in the run queue to
/// ensure the efficiency for finding next-to-run threads.
#[derive(Debug)]
pub(super) struct FairClassRq {
    #[allow(unused)]
    cpu: CpuId,
    /// The ready-to-run threads.
    threads: BTreeSet<FairQueueItem>,
    total_weight: u64,
}

impl FairClassRq {
    pub fn new(cpu: CpuId) -> Self {
        Self {
            cpu,
            threads: BTreeSet::new(),
            total_weight: 0,
        }
    }

    /// The scheduling period is calculated as the maximum of the following two values:
    ///
    /// 1. The minimum period value, defined by [`min_period_clocks`].
    /// 2. `period = min_granularity * n` where
    ///    `min_granularity = log2(1 + num_cpus) * base_slice_clocks`, and `n` is the number of
    ///    runnable threads (including the current running thread).
    ///
    /// The formula is chosen by 3 principles:
    ///
    /// 1. The scheduling period should reflect the running threads and CPUs;
    /// 2. The scheduling period should not be too low to limit the overhead of context switching;
    /// 3. The scheduling period should not be too high to ensure the scheduling latency
    ///    & responsiveness.
    fn period(&self) -> u64 {
        let base_slice_clks = base_slice_clocks();
        let min_period_clks = min_period_clocks();

        let min_gran_clks = base_slice_clks * u64::from((1 + num_cpus()).ilog2());
        // `+ 1` means including the current running thread.
        (min_gran_clks * (self.threads.len() + 1) as u64).max(min_period_clks)
    }

    /// Pop a thread capable for running.
    ///
    /// If `target_cpu` is specified, this function filters out threads which
    /// are not affined with it.
    fn pop(&mut self, target_cpu: Option<CpuId>) -> Option<Arc<Thread>> {
        let mut front = self.threads.lower_bound_mut(Bound::Unbounded);
        let FairQueueItem(thread) = loop {
            let thread = front.peek_next()?;
            if target_cpu.is_none_or(|cpu| thread.0.atomic_cpu_affinity().load().contains(cpu)) {
                let thread = front.remove_next().unwrap();
                break thread;
            }
            front.next().unwrap();
        };

        self.total_weight -= thread.sched_attr().fair.weight.load(Relaxed);

        Some(thread)
    }
}

impl SchedClassRq for FairClassRq {
    fn enqueue(&mut self, thread: Arc<Thread>) {
        self.threads.insert(FairQueueItem(thread));
    }

    fn is_empty(&mut self) -> bool {
        self.threads.is_empty()
    }

    fn pick_next(&mut self) -> Option<Arc<Thread>> {
        self.pop(None)
    }

    fn update_current(&mut self, now: u64, thread: &Thread, flags: UpdateFlags) -> bool {
        let attr = &thread.sched_attr().fair;
        match flags {
            UpdateFlags::Yield => true,
            UpdateFlags::Tick | UpdateFlags::Wait => {
                // Includes the current running thread.
                let total_weight = self.total_weight + attr.weight.load(Relaxed);
                attr.tick(
                    now,
                    thread.sched_attr().start.load(Relaxed),
                    thread.sched_attr().period_start.load(Relaxed),
                    total_weight,
                    self.period(),
                )
            }
        }
    }
}
