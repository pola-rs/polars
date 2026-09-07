use std::collections::BTreeMap;
use std::ops::{Range, RangeBounds};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::schema::SchemaRef;
use polars_core::series::Series;
use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext, SpillFrame};
use polars_utils::range::check_range;

use crate::pipe::PortReceiver;

#[derive(Clone, Debug)]
pub(super) struct SpillFrameSearchBuffer {
    schema: SchemaRef,
    // Use Arc<_> to prevent unspilling the SpillFrames when splitting the DFSB.
    sfs_at_offsets: BTreeMap<usize, Arc<SpillFrame>>,
    spill_ctx: MostRecentSpillContext,
    total_rows: usize,
    skip_rows: usize,
    frozen: bool,
}

impl SpillFrameSearchBuffer {
    pub(super) fn empty_with_schema(schema: SchemaRef, spill_ctx: MostRecentSpillContext) -> Self {
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

    fn spillframe_at(&self, row_index: usize) -> (&Arc<SpillFrame>, usize) {
        debug_assert!(row_index < self.total_rows);
        let first_offset = match self.sfs_at_offsets.first_key_value() {
            Some((offset, _)) => *offset,
            None => 0,
        };
        let buf_index = self.skip_rows + first_offset + row_index;
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
        for frame in self.sfs_at_offsets.values() {
            acc.vstack_mut(&*frame.get().await).unwrap();
        }
        acc.slice(self.skip_rows as i64, self.total_rows)
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
