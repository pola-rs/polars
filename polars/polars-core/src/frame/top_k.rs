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
use crate::series::IsSorted;
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
        self.top_k_impl(k, descending, by_column, false, false)
    }

    pub(crate) fn top_k_impl(
        &self,
        k: usize,
        mut descending: Vec<bool>,
        by_column: Vec<Series>,
        nulls_last: bool,
        maintain_order: bool,
    ) -> PolarsResult<DataFrame> {
        eprintln!("In top_k_impl top k impl");
        _broadcast_descending(by_column.len(), &mut descending);
        let encoded = _get_rows_encoded(&by_column, &descending, nulls_last)?;
        let arr = encoded.into_array();
        let mut rows = arr
            .values_iter()
            .enumerate_idx()
            .map(|(idx, bytes)| CompareRow { idx, bytes })
            .collect::<Vec<_>>();

        eprintln!("This is k {} and self.height {}", k, self.height());
        let sorted = if k >= self.height() {
            eprintln!("In top_k_impl k >= self.height()");
            if maintain_order {
                eprintln!("sort stable");
                rows.sort();
            }
            else {
                eprintln!("sort unstable");
                rows.sort_unstable();
            }
            &rows
        } else {
            if maintain_order {
                eprintln!("sort stable lower");
                // todo: maybe there is some more efficient method, comparable to select_nth_unstable
                rows.sort();
                &rows[..k]
            }
            else {

                let (lower, _el, _upper) = rows.select_nth_unstable(k);
                eprintln!("sort unstable lower");
                lower.sort_unstable();
                &*lower
            }
        };

        let idx: NoNull<IdxCa> = sorted.iter().map(|cmp_row| cmp_row.idx).collect();
        eprintln!("idx {:?}", idx.inner);

        let mut df = unsafe { self.take_unchecked(&idx.into_inner()) };
        eprintln!("df {:?}", df);

        let first_descending = descending[0];
        let first_by_column = by_column[0].name().to_string();

        // Mark the first sort column as sorted
        // if the column did not exists it is ok, because we sorted by an expression
        // not present in the dataframe
        let _ = df.apply(&first_by_column, |s| {
            let mut s = s.clone();
            if first_descending {
                s.set_sorted_flag(IsSorted::Descending)
            } else {
                s.set_sorted_flag(IsSorted::Ascending)
            }
            s
        });
        Ok(df)
    }
}
