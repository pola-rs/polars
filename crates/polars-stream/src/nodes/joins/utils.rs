use std::collections::BTreeMap;

use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::schema::SchemaRef;
use polars_core::series::Series;

#[derive(Clone, Debug)]
pub(super) struct DataFrameSearchBuffer {
    schema: SchemaRef,
    dfs_at_offsets: BTreeMap<usize, DataFrame>,
    total_rows: usize,
    skip_rows: usize,
    frozen: bool,
}

impl DataFrameSearchBuffer {
    pub(super) fn empty_with_schema(schema: SchemaRef) -> Self {
        DataFrameSearchBuffer {
            schema,
            dfs_at_offsets: BTreeMap::new(),
            total_rows: 0,
            skip_rows: 0,
            frozen: false,
        }
    }

    pub(super) fn height(&self) -> usize {
        self.total_rows
    }

    /// Get the `row_index`th value from the `column` bypassing its validity bitmap.
    ///
    /// SAFETY: Caller must ensure that `row_index` is within bounds.
    pub(super) unsafe fn get_bypass_validity(
        &self,
        column: &str,
        row_index: usize,
        bypass_validity: bool,
    ) -> AnyValue<'_> {
        debug_assert!(row_index < self.total_rows);
        let first_offset = match self.dfs_at_offsets.first_key_value() {
            Some((offset, _)) => *offset,
            None => 0,
        };
        let buf_index = self.skip_rows + first_offset + row_index;
        let (df_offset, df) = self.dfs_at_offsets.range(..=buf_index).next_back().unwrap();
        let series_index = buf_index - df_offset;
        let series = df.column(column).unwrap().as_materialized_series();
        unsafe { series_get_bypass_validity(series, series_index, bypass_validity) }
    }

    pub(super) fn push_df(&mut self, df: DataFrame) {
        assert!(!self.frozen);
        let added_rows = df.height();
        let offset = match self.dfs_at_offsets.last_key_value() {
            Some((last_key, last_df)) => last_key + last_df.height(),
            None => 0,
        };
        self.dfs_at_offsets.insert(offset, df);
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
        self
    }

    pub(super) fn into_df(self) -> DataFrame {
        let mut acc = DataFrame::empty_with_schema(&self.schema);
        for df in self.dfs_at_offsets.into_values() {
            acc.vstack_mut_owned(df).unwrap();
        }
        acc.slice(self.skip_rows as i64, self.total_rows)
    }

    fn gc(&mut self) {
        while let Some((_, df)) = self.dfs_at_offsets.first_key_value() {
            if self.skip_rows > df.height() {
                let (_, df) = self.dfs_at_offsets.pop_first().unwrap();
                self.skip_rows -= df.height();
            } else {
                break;
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    /// Find the index of the first item in the buffer that satisfies `predicate`,
    /// assuming it is first always false and then always true.
    pub(super) fn binary_search<P>(
        &self,
        predicate: P,
        key_col_name: &str,
        binary_offset_bypass_validity: bool,
    ) -> usize
    where
        P: Fn(&AnyValue<'_>) -> bool,
    {
        let mut lower = 0;
        let mut upper = self.height();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let mid_val = unsafe {
                self.get_bypass_validity(key_col_name, mid, binary_offset_bypass_validity)
            };
            if predicate(&mid_val) {
                upper = mid;
            } else {
                lower = mid + 1;
            }
        }
        lower
    }
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
