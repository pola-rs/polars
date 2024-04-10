use std::cmp::Ordering;

use polars_utils::iter::EnumerateIdxTrait;
use smartstring::alias::String as SmartString;

use super::*;
use crate::prelude::sort::_broadcast_descending;
use crate::prelude::sort::arg_sort_multiple::_get_rows_encoded;

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
        Some(self.cmp(other))
    }
}

impl DataFrame {
    pub fn top_k(
        &self,
        k: usize,
        by_column: impl IntoVec<SmartString>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<DataFrame> {
        let by_column = self.select_series(by_column)?;
        self.top_k_impl(k, by_column, sort_options)
    }

    pub(crate) fn top_k_impl(
        &self,
        k: usize,
        by_column: Vec<Series>,
        mut sort_options: SortMultipleOptions,
    ) -> PolarsResult<DataFrame> {
        _broadcast_descending(by_column.len(), &mut sort_options.descending);
        let encoded = _get_rows_encoded(
            &by_column,
            &sort_options.descending,
            sort_options.nulls_last,
        )?;
        let arr = encoded.into_array();
        let mut rows = arr
            .values_iter()
            .enumerate_idx()
            .map(|(idx, bytes)| CompareRow { idx, bytes })
            .collect::<Vec<_>>();

        let sorted = if k >= self.height() {
            match (sort_options.multithreaded, sort_options.maintain_order) {
                (true, true) => POOL.install(|| {
                    rows.par_sort();
                }),
                (true, false) => POOL.install(|| {
                    rows.par_sort_unstable();
                }),
                (false, true) => rows.sort(),
                (false, false) => rows.sort_unstable(),
            }
            &rows
        } else if sort_options.maintain_order {
            // todo: maybe there is some more efficient method, comparable to select_nth_unstable
            if sort_options.multithreaded {
                POOL.install(|| {
                    rows.par_sort();
                })
            } else {
                rows.sort();
            }
            &rows[..k]
        } else {
            // todo: possible multi threaded `select_nth_unstable`?
            let (lower, _el, _upper) = rows.select_nth_unstable(k);
            if sort_options.multithreaded {
                POOL.install(|| {
                    lower.par_sort_unstable();
                })
            } else {
                lower.sort_unstable();
            }
            &*lower
        };

        let idx: NoNull<IdxCa> = sorted.iter().map(|cmp_row| cmp_row.idx).collect();

        let mut df = unsafe { self.take_unchecked(&idx.into_inner()) };

        let first_descending = sort_options.descending[0];
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
