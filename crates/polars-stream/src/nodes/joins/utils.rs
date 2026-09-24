use std::collections::BTreeMap;
use std::ops::{Range, RangeBounds};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::runtime::ASYNC;
use polars_core::schema::SchemaRef;
use polars_core::series::Series;
use polars_expr::hash_keys::HashKeys;
use polars_ooc::{ParameterFreeSpillContext, RandomSpillContext, SpillFrame};
use polars_utils::IdxSize;
use polars_utils::aliases::PlRandomState;
use polars_utils::cardinality_sketch::CardinalitySketch;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::range::check_range;

use super::{fold_sample, select_key_columns};
use crate::expression::StreamExpr;
use crate::morsel::Morsel;
use crate::nodes::ExecutionState;
use crate::pipe::PortReceiver;

#[derive(Clone, Debug)]
pub(super) struct SpillFrameSearchBuffer {
    schema: SchemaRef,
    // Use Arc<_> to prevent unspilling the SpillFrames when splitting the DFSB.
    sfs_at_offsets: BTreeMap<usize, Arc<SpillFrame>>,
    spill_ctx: RandomSpillContext,
    total_rows: usize,
    skip_rows: usize,
    frozen: bool,
}

impl SpillFrameSearchBuffer {
    pub(super) fn empty_with_schema(schema: SchemaRef, spill_ctx: RandomSpillContext) -> Self {
        SpillFrameSearchBuffer {
            schema,
            sfs_at_offsets: BTreeMap::new(),
            spill_ctx,
            total_rows: 0,
            skip_rows: 0,
            frozen: false,
        }
    }

    pub(super) fn empty_clone(&self) -> Self {
        Self::empty_with_schema(self.schema.clone(), self.spill_ctx.clone())
    }

    pub(super) fn height(&self) -> usize {
        self.total_rows
    }

    /// Get the offset of the first live row in `sfs_at_offsets` coordinates.
    fn live_start(&self) -> usize {
        let first_offset = match self.sfs_at_offsets.first_key_value() {
            Some((offset, _)) => *offset,
            None => 0,
        };
        first_offset + self.skip_rows
    }

    fn spillframe_at(&self, row_index: usize) -> (&Arc<SpillFrame>, usize) {
        debug_assert!(row_index < self.total_rows);
        let buf_index = self.live_start() + row_index;
        let (frame_offset, frame) = self.sfs_at_offsets.range(..=buf_index).next_back().unwrap();
        (frame, buf_index - frame_offset)
    }

    /// Get the `row_index`th value from the `column`.
    ///
    /// SAFETY: Caller must ensure that `row_index` is within bounds.
    pub(super) async unsafe fn get_unchecked(
        &self,
        column: &str,
        row_index: usize,
    ) -> AnyValue<'static> {
        unsafe { self.get_bypass_validity(column, row_index, false).await }
    }

    /// Get the `row_index`th value from the `column` potentially bypassing its
    /// validity bitmap.
    ///
    /// SAFETY: Caller must ensure that `row_index` is within bounds.
    pub(super) async unsafe fn get_bypass_validity(
        &self,
        column: &str,
        row_index: usize,
        bypass_validity: bool,
    ) -> AnyValue<'static> {
        let (frame, frame_index) = self.spillframe_at(row_index);
        let df = frame.get().await;
        let series = df.column(column).unwrap().as_materialized_series();
        unsafe { series_get_bypass_validity(series, frame_index, bypass_validity) }.into_static()
    }

    pub(super) async fn push_sf(&mut self, sf: SpillFrame) {
        assert!(!self.frozen);
        let added_rows = sf.height();
        let offset = match self.sfs_at_offsets.last_key_value() {
            Some((last_key, last_sf)) => last_key + last_sf.height(),
            None => 0,
        };
        self.spill_ctx.register(&sf).await;
        self.sfs_at_offsets.insert(offset, Arc::new(sf));
        self.total_rows += added_rows;
    }

    pub(super) fn split_at(&mut self, mut at: usize) -> Self {
        at = at.clamp(0, self.total_rows);
        let mut top = self.clone();
        top.total_rows = at;
        top.frozen = true;
        self.skip_rows += at;
        self.total_rows -= at;
        self.gc();
        top
    }

    pub(super) fn slice(mut self, offset: usize, len: usize) -> Self {
        self.skip_rows += offset;
        self.total_rows -= offset;
        self.total_rows = usize::min(self.total_rows, len);
        self.frozen = true;
        self.gc();
        self
    }

    pub(super) async fn into_df(self) -> DataFrame {
        let mut acc = DataFrame::empty_with_schema(&self.schema);
        if self.total_rows == 0 {
            return acc;
        }

        let live_start = self.live_start();
        let live_end = live_start + self.total_rows;
        let first_offset = *self
            .sfs_at_offsets
            .range(..=live_start)
            .next_back()
            .unwrap()
            .0;
        for (_, frame) in self.sfs_at_offsets.range(first_offset..live_end) {
            acc.vstack_mut(&*frame.get().await).unwrap();
        }
        acc.slice((live_start - first_offset) as i64, self.total_rows)
    }

    fn gc(&mut self) {
        while let Some((_, frame)) = self.sfs_at_offsets.first_key_value()
            && self.skip_rows > frame.height()
        {
            let (_, gc_frame) = self.sfs_at_offsets.pop_first().unwrap();
            self.skip_rows -= gc_frame.height();
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    /// Find the index of the first item in the buffer that satisfies `predicate`,
    /// assuming it is first always false and then always true.
    pub(super) async fn binary_search<P, R>(
        &self,
        predicate: P,
        key_col_name: &str,
        range: R,
    ) -> usize
    where
        P: Fn(&AnyValue<'_>) -> bool,
        R: RangeBounds<usize>,
    {
        self.binary_search_binary_offset_bypass_validity(predicate, key_col_name, range, false)
            .await
    }

    /// Find the index of the first item in the buffer that satisfies `predicate`,
    /// assuming it is first always false and then always true.
    pub(super) async fn binary_search_binary_offset_bypass_validity<P, R>(
        &self,
        predicate: P,
        key_col_name: &str,
        range: R,
        binary_offset_bypass_validity: bool,
    ) -> usize
    where
        P: Fn(&AnyValue<'_>) -> bool,
        R: RangeBounds<usize>,
    {
        let Range {
            start: mut lower,
            end: mut upper,
        } = check_range(range, ..self.height());
        while lower < upper {
            let mid = (lower + upper) / 2;
            let (frame, frame_index) = self.spillframe_at(mid);
            let df = frame.get().await;
            let series = df.column(key_col_name).unwrap().as_materialized_series();
            let mid_val = unsafe {
                series_get_bypass_validity(series, frame_index, binary_offset_bypass_validity)
            };
            if predicate(&mid_val) {
                upper = mid;
            } else {
                lower = mid + 1;
            }
        }
        lower
    }

    pub(super) async fn stop_and_buffer_from_pipe(&mut self, port: Option<&mut PortReceiver>) {
        for sf in stop_and_take_pipe_contents(port).await {
            self.push_sf(sf).await;
        }
    }
}

/// Tell the sender to this port to stop, and take everything that is still in the pipe.
pub(super) async fn stop_and_take_pipe_contents(
    port: Option<&mut PortReceiver>,
) -> Vec<SpillFrame> {
    let mut frames = Vec::new();
    let Some(port) = port else {
        return frames;
    };

    while let Ok(morsel) = port.recv().await {
        morsel.source_token().stop();
        let (sf, _, _, _) = morsel.into_inner();
        frames.push(sf);
    }
    frames
}

/// Get value from series bypassing the validity bitmap.
///
/// SAFETY: Caller must ensure that `index` is within bounds of `s`.
unsafe fn series_get_bypass_validity<'a>(
    s: &'a Series,
    index: usize,
    binary_offset_bypass_validity: bool,
) -> AnyValue<'a> {
    debug_assert!(index < s.len());
    if binary_offset_bypass_validity {
        let arr = s.binary_offset().unwrap();
        unsafe { arr.get_any_value_bypass_validity(index) }
    } else {
        unsafe { s.get_unchecked(index) }
    }
}

/// Bytes a hash table takes per key on top of the key itself.
const KEY_SLOT_OVERHEAD: f64 = 16.0;

/// Statistics of one join side's sample, used to estimate the bytes its build
/// keeps.
pub struct JoinSampleStats {
    /// Distinct keys, as a fraction of the rows.
    pub key_ratio: f64,
    pub key_width: f64,
    pub row_width: f64,
}

impl JoinSampleStats {
    /// With a `payload_select` a row is its selected columns and its keys,
    /// otherwise it is the whole frame.
    pub fn from_sample(
        morsels: &[Morsel],
        key_selectors: &[StreamExpr],
        payload_select: Option<&[Option<PlSmallStr>]>,
        null_is_valid: bool,
        random_state: &PlRandomState,
        sample_limit: usize,
        state: &ExecutionState,
    ) -> PolarsResult<Self> {
        if morsels.is_empty() || sample_limit == 0 {
            return Ok(Self {
                key_ratio: 0.0,
                key_width: 0.0,
                row_width: 0.0,
            });
        }
        let ((sketch, key_bytes, row_bytes), rows) = fold_sample(
            morsels,
            sample_limit,
            || (CardinalitySketch::new(), 0usize, 0usize),
            |(mut sketch, key_bytes, row_bytes), df| {
                let keys = ASYNC.block_on(select_key_columns(df, key_selectors, state))?;
                HashKeys::from_df(&keys, random_state.clone(), null_is_valid, false)
                    .sketch_cardinality(&mut sketch);
                let row_size = match payload_select {
                    None => df.estimated_size(),
                    Some(select) => {
                        let payload = df
                            .columns()
                            .iter()
                            .zip(select)
                            .filter(|(_, name)| name.is_some())
                            .map(|(c, _)| c.clone())
                            .collect();
                        let payload = unsafe { DataFrame::new_unchecked(df.height(), payload) };
                        payload.estimated_size() + keys.estimated_size()
                    },
                };
                Ok((
                    sketch,
                    key_bytes + keys.estimated_size(),
                    row_bytes + row_size,
                ))
            },
            |(mut a, ak, ar), (b, bk, br)| {
                a.combine(&b);
                (a, ak + bk, ar + br)
            },
        )?;
        let rows = rows as f64;
        Ok(Self {
            key_ratio: (sketch.estimate() as f64 / rows).min(1.0),
            key_width: key_bytes as f64 / rows,
            row_width: row_bytes as f64 / rows,
        })
    }

    /// Bytes retained when building a table of the distinct keys of `rows` rows.
    pub fn distinct_keys_build_bytes(&self, rows: usize) -> f64 {
        rows as f64 * self.key_ratio * (self.key_width + KEY_SLOT_OVERHEAD)
    }

    /// Bytes retained when building the distinct keys of `rows` rows and
    /// keeping every row.
    pub fn all_rows_build_bytes(&self, rows: usize) -> f64 {
        self.distinct_keys_build_bytes(rows)
            + rows as f64 * (self.row_width + size_of::<IdxSize>() as f64)
    }
}
