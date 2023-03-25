use std::cmp::Ordering;

use polars_error::PolarsResult;
use polars_utils::iter::EnumerateIdxTrait;
use polars_utils::IdxSize;
use smartstring::alias::String as SmartString;

use crate::datatypes::IdxCa;
use crate::frame::DataFrame;
use crate::prelude::sort::_broadcast_descending;
use crate::prelude::sort::arg_sort_multiple::_get_rows_encoded;
use crate::prelude::*;
use crate::utils::NoNull;

#[derive(Eq)]
struct CompareRow<'a> {
    idx: IdxSize,
    bytes: &'a [u8],
}

impl PartialEq for CompareRow<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Ord for CompareRow<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bytes.cmp(other.bytes)
    }
}

impl PartialOrd for CompareRow<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bytes.partial_cmp(other.bytes)
    }
}

impl DataFrame {
    pub fn top_k(
        &self,
        k: usize,
        descending: impl IntoVec<bool>,
        by_column: impl IntoVec<SmartString>,
    ) -> PolarsResult<DataFrame> {
        let by_column = self.select_series(by_column)?;
        let descending = descending.into_vec();
        self.top_k_impl(k, descending, by_column, false)
    }

    pub(crate) fn top_k_impl(
        &self,
        k: usize,
        mut descending: Vec<bool>,
        by_column: Vec<Series>,
        nulls_last: bool,
    ) -> PolarsResult<DataFrame> {
        _broadcast_descending(by_column.len(), &mut descending);
        let encoded = _get_rows_encoded(&by_column, &descending, nulls_last)?;
        let arr = encoded.into_array();
        let mut rows = arr
            .values_iter()
            .enumerate_idx()
            .map(|(idx, bytes)| CompareRow { idx, bytes })
            .collect::<Vec<_>>();

        let sorted = if k >= self.height() {
            &rows
        } else {
            let (lower, _el, _upper) = rows.select_nth_unstable(k);
            lower.sort_unstable();
            &*lower
        };

        let idx: NoNull<IdxCa> = sorted.iter().map(|cmp_row| cmp_row.idx).collect();

        unsafe { Ok(self.take_unchecked(&idx.into_inner())) }
    }
}
