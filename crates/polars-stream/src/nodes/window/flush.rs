use std::collections::{BTreeMap, VecDeque};

use futures::future::join_all;
use parking_lot::Mutex;
use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_ooc::{LeastRecentSpillContext, MostRecentSpillContext, SpillFrame, memory_manager};
use polars_utils::IdxSize;
use rayon::prelude::*;
use tokio::sync::Notify;

use super::evaluate::ROW_ID;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::pipe::PortSender;

/// Consecutive input morsels and the window rows of their input rows.
pub struct OutputRange {
    pub morsels: Vec<SpillFrame>,
    /// The input row of the first row of `morsels`.
    pub first_row: IdxSize,
    pub num_rows: usize,
    /// The estimated size of the morsels and their window frames.
    pub bytes: usize,
    /// Window columns and the input row of each of their rows, in any order. Together they hold
    /// every input row of `morsels` once.
    pub window_frames: Vec<SpillFrame>,
}

pub enum OutputUnit {
    Range(OutputRange),
    /// The sorted rows of a partition, with the window columns appended, and their estimated
    /// size.
    Frame(SpillFrame, usize),
}

impl OutputUnit {
    fn bytes(&self) -> usize {
        match self {
            Self::Range(range) => range.bytes,
            Self::Frame(_, bytes) => *bytes,
        }
    }
}

struct FlushInner {
    /// Units not yet prepared, in output order.
    pending: VecDeque<OutputUnit>,
    /// The index of the first unit of `pending`.
    next_pending: usize,
    /// Prepared output morsels by unit index, with the bytes of the unit.
    prepared: BTreeMap<usize, (VecDeque<DataFrame>, usize)>,
    /// The index of the unit to emit from.
    next_unit: usize,
    /// The bytes of the units prepared or being prepared.
    ahead_bytes: usize,
    num_preparing: usize,
    next_seq: u64,
    /// Set when preparing failed, so that the other tasks stop instead of waiting for it.
    failed: bool,
}

impl FlushInner {
    fn is_drained(&self) -> bool {
        self.pending.is_empty() && self.prepared.is_empty() && self.num_preparing == 0
    }
}

enum Step<'a> {
    Emit(DataFrame, u64),
    Prepare(OutputUnit, usize),
    Wait(tokio::sync::futures::Notified<'a>),
    Stop,
}

/// Emits the output units in order. The sender tasks share one emission cursor, so every
/// execution phase emits a prefix of the output. A task with nothing to emit prepares a unit
/// ahead, up to `max_ahead_bytes` of units prepared or being prepared, and at least one.
pub struct Flush {
    inner: Mutex<FlushInner>,
    notify: Notify,
    max_ahead_bytes: usize,
    /// Keep the stored frames spillable while they wait.
    _spill_ctxs: (LeastRecentSpillContext, MostRecentSpillContext),
}

impl Flush {
    pub fn new(
        units: Vec<OutputUnit>,
        max_ahead_bytes: usize,
        spill_ctxs: (LeastRecentSpillContext, MostRecentSpillContext),
    ) -> Self {
        Self {
            inner: Mutex::new(FlushInner {
                pending: units.into(),
                next_pending: 0,
                prepared: BTreeMap::new(),
                next_unit: 0,
                ahead_bytes: 0,
                num_preparing: 0,
                next_seq: 0,
                failed: false,
            }),
            notify: Notify::new(),
            max_ahead_bytes,
            _spill_ctxs: spill_ctxs,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.inner.lock().is_drained()
    }

    fn next_step(&self, source_token: &SourceToken) -> Step<'_> {
        let mut inner = self.inner.lock();
        if source_token.stop_requested() || inner.failed {
            return Step::Stop;
        }
        let inner = &mut *inner;
        while let Some(mut entry) = inner.prepared.first_entry()
            && *entry.key() == inner.next_unit
        {
            let (dfs, bytes) = entry.get_mut();
            let df = dfs.pop_front();
            if dfs.is_empty() {
                inner.ahead_bytes -= *bytes;
                inner.next_unit += 1;
                entry.remove();
                self.notify.notify_waiters();
            }
            if let Some(df) = df {
                let seq = inner.next_seq;
                inner.next_seq += 1;
                return Step::Emit(df, seq);
            }
        }

        if let Some(unit) = inner.pending.front() {
            let bytes = unit.bytes();
            let nothing_ahead = inner.prepared.is_empty() && inner.num_preparing == 0;
            if nothing_ahead || inner.ahead_bytes + bytes <= self.max_ahead_bytes {
                let unit = inner.pending.pop_front().unwrap();
                let idx = inner.next_pending;
                inner.next_pending += 1;
                inner.ahead_bytes += bytes;
                inner.num_preparing += 1;
                return Step::Prepare(unit, idx);
            }
        }
        if inner.is_drained() {
            return Step::Stop;
        }
        // The next unit to emit is being prepared, which notifies when done.
        Step::Wait(self.notify.notified())
    }

    fn finish_prepare(
        &self,
        idx: usize,
        bytes: usize,
        prepared: PolarsResult<VecDeque<DataFrame>>,
    ) -> PolarsResult<()> {
        let result = {
            let mut inner = self.inner.lock();
            inner.num_preparing -= 1;
            match prepared {
                Ok(dfs) => {
                    inner.prepared.insert(idx, (dfs, bytes));
                    Ok(())
                },
                Err(e) => {
                    inner.failed = true;
                    Err(e)
                },
            }
        };
        self.notify.notify_waiters();
        result
    }

    pub fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        senders: Vec<PortSender>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let source_token = SourceToken::new();
        for mut send in senders {
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let wait_group = WaitGroup::default();
                loop {
                    match self.next_step(&source_token) {
                        Step::Emit(df, seq) => {
                            let mut morsel = Morsel::new_unregistered(
                                df,
                                MorselSeq::new(seq),
                                source_token.clone(),
                            );
                            morsel.set_consume_token(wait_group.token());
                            if send.send(morsel).await.is_err() {
                                return Ok(());
                            }
                            wait_group.wait().await;
                        },
                        Step::Prepare(unit, idx) => {
                            let bytes = unit.bytes();
                            let prepared = prepare(unit).await;
                            self.finish_prepare(idx, bytes, prepared)?;
                        },
                        Step::Wait(notified) => notified.await,
                        Step::Stop => return Ok(()),
                    }
                }
            }));
        }
    }
}

impl Drop for Flush {
    fn drop(&mut self) {
        let units: Vec<_> = self.inner.get_mut().pending.drain(..).collect();
        RAYON.install(|| units.into_par_iter().for_each(drop));
    }
}

/// Loads a unit and returns its output morsels.
async fn prepare(unit: OutputUnit) -> PolarsResult<VecDeque<DataFrame>> {
    let range = match unit {
        OutputUnit::Range(range) => range,
        OutputUnit::Frame(sf, _) => {
            let df = sf.into_df().await;
            let morsel_size = get_ideal_morsel_size();
            return Ok((0..df.height())
                .step_by(morsel_size)
                .map(|offset| df.slice(offset as i64, morsel_size))
                .collect());
        },
    };

    if range.num_rows == 0 {
        return Ok(VecDeque::new());
    }
    let _reservation = memory_manager().reserve(range.bytes as u64).await;
    let dfs = join_all(range.window_frames.into_iter().map(SpillFrame::into_df)).await;
    let mut windows = accumulate_dataframes_vertical_unchecked(dfs);
    let row_ids = windows.drop_in_place(&ROW_ID)?;
    // Taking rows from one chunk is much faster than from many.
    windows.rechunk_mut();
    let row_ids = row_ids.idx()?.rechunk();
    let row_ids = row_ids.downcast_as_array().values();
    assert_eq!(row_ids.len(), range.num_rows);

    // Puts the window rows in input order.
    let mut window_row_for_range_row = vec![0 as IdxSize; range.num_rows];
    for (i, row) in row_ids.iter().enumerate() {
        // SAFETY: the window frames hold each input row of the range once.
        unsafe {
            *window_row_for_range_row.get_unchecked_mut((*row - range.first_row) as usize) =
                i as IdxSize;
        }
    }
    // SAFETY: the window rows are in-bounds.
    let windows = unsafe { windows.take_slice_unchecked(&window_row_for_range_row) };
    drop(window_row_for_range_row);

    let mut out = VecDeque::with_capacity(range.morsels.len());
    let mut offset = 0;
    for sf in range.morsels {
        let mut df = sf.into_df().await;
        let height = df.height();
        if height == 0 {
            continue;
        }
        let columns = windows.slice(offset as i64, height).into_columns();
        // SAFETY: the window columns have the height of the morsel and new names.
        unsafe { df.hstack_mut_unchecked(&columns) };
        offset += height;
        out.push_back(df);
    }
    Ok(out)
}
