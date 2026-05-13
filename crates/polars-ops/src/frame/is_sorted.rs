use std::cmp::Ordering;

use polars_core::chunked_array::ops::SortOptions;
#[cfg(feature = "dtype-struct")]
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::sort::reorder_tot_cmp;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash, TotalOrd};
use recursive::recursive;

use crate::prelude::*;
use crate::series::ops::rle::rle_lengths_helper_ca;

pub trait DataFrameIsSorted {
    fn is_sorted(
        &self,
        by: &[PlSmallStr],
        descending: &[bool],
        nulls_last: &[bool],
    ) -> PolarsResult<bool>;
}

impl DataFrameIsSorted for DataFrame {
    fn is_sorted(
        &self,
        by: &[PlSmallStr],
        descending: &[bool],
        nulls_last: &[bool],
    ) -> PolarsResult<bool> {
        if by.len() == 0 {
            polars_bail!(InvalidOperation: "by must specify at least one column");
        }
        if descending.len() != by.len() {
            polars_bail!(InvalidOperation: "descending must be of same length as by");
        }
        if nulls_last.len() != by.len() {
            polars_bail!(InvalidOperation: "nulls_last must be of same length as by");
        }

        if let &[ref single_by] = by {
            // Fast path
            let s = self.column(single_by)?.as_materialized_series();
            let options = SortOptions {
                descending: descending[0],
                nulls_last: nulls_last[0],
                ..Default::default()
            };
            return SeriesMethods::is_sorted(s, options);
        }

        let cols = by
            .iter()
            .map(|c| self.column(c).cloned())
            .collect::<PolarsResult<Vec<_>>>()?;

        let mut scratch_vec_pool = (1..cols.len()).map(|_| ScratchVec::default()).collect_vec();
        is_sorted_cols(&cols, descending, nulls_last, &mut scratch_vec_pool)
    }
}

/// Recursively checks whether `cols` are sorted by `(cols[0], cols[1], ...)`.
///
/// All columns are assumed to have the same length. Ties in `cols[0]` trigger
/// a recursive check on the sliced tail `cols[1..]`.
#[recursive]
fn is_sorted_cols(
    cols: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
    scratch_pool: &mut [ScratchVec<Column>],
) -> PolarsResult<bool> {
    let Some(first) = cols.first() else {
        return Ok(true);
    };
    let by_more = &cols[1..];

    // Nested types: row-encode so they compare as plain bytes.
    #[cfg(feature = "dtype-struct")]
    if matches!(first.dtype(), DataType::Struct(_)) {
        let encoded = _get_rows_encoded_ca(
            PlSmallStr::EMPTY,
            &[first.clone()],
            &[descending[0]],
            &[nulls_last[0]],
            false,
        )?;
        let encoded_col = encoded.into_series().into_column();
        let mut new_cols = Vec::with_capacity(cols.len());
        new_cols.push(encoded_col);
        new_cols.extend_from_slice(by_more);
        return is_sorted_cols(&new_cols, descending, nulls_last, scratch_pool);
    }

    let s = first
        .as_materialized_series()
        .to_physical_repr()
        .into_owned();

    match s.dtype() {
        DataType::Boolean => {
            let ca: &BooleanChunked = s.as_ref().as_ref().as_ref();
            is_sorted_ca(ca, by_more, descending, nulls_last, scratch_pool)
        },
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_sorted_ca(ca, by_more, descending, nulls_last, scratch_pool)
            })
        },
        DataType::String => {
            let ca: &StringChunked = s.as_ref().as_ref().as_ref();
            is_sorted_ca(
                &ca.as_binary(),
                by_more,
                descending,
                nulls_last,
                scratch_pool,
            )
        },
        DataType::Binary => {
            let ca: &BinaryChunked = s.as_ref().as_ref().as_ref();
            is_sorted_ca(ca, by_more, descending, nulls_last, scratch_pool)
        },
        DataType::BinaryOffset => {
            let ca: &BinaryOffsetChunked = s.as_ref().as_ref().as_ref();
            is_sorted_ca(ca, by_more, descending, nulls_last, scratch_pool)
        },
        _ => {
            // TODO: allocates a full boolean series and doesn't propagate ties
            // to subsequent columns (see SeriesMethods::is_sorted)
            is_sorted_fallback(first, by_more, descending, nulls_last)
        },
    }
}

/// Typed fast path. Uses RLE of `by` to find run boundaries, checks ordering
/// at each boundary, and recurses into `by_more` for tie runs.
fn is_sorted_ca<'a, T>(
    by: &'a ChunkedArray<T>,
    by_more: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
    scratch_pool: &mut [ScratchVec<Column>],
) -> PolarsResult<bool>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + TotalOrd + ToTotalOrd + Copy,
{
    let mut run_lengths: Vec<IdxSize> = Vec::new();
    rle_lengths_helper_ca(by, &mut run_lengths);

    let mut prev: Option<Option<T::Physical<'a>>> = None;
    let mut pos: usize = 0;

    for rl in run_lengths {
        let rl = rl as usize;
        let start = pos;

        // SAFETY: pos is a valid run-start index derived from rle_lengths_helper_ca.
        let cur = unsafe { by.get_unchecked(start) };

        if let Some(prev) = prev.as_ref() {
            let ord = reorder_tot_cmp(prev, &cur, descending[0], nulls_last[0]);
            debug_assert_ne!(ord, Ordering::Equal);
            if ord == Ordering::Greater {
                return Ok(false);
            }
        }

        if rl > 1 && !by_more.is_empty() {
            let Some((scratch, scratch_pool)) = scratch_pool.split_first_mut() else {
                unreachable!()
            };
            let sliced = scratch.get();
            sliced.extend(by_more.iter().map(|c| c.slice(start as i64, rl)));
            if !is_sorted_cols(&sliced, &descending[1..], &nulls_last[1..], scratch_pool)? {
                return Ok(false);
            }
        }

        prev = Some(cur);
        pos += rl;
    }

    Ok(true)
}

/// General fallback for dtypes not covered by [`is_sorted_ca`].
///
/// Checks that `first` is sorted using adjacent-slice comparison. Does not
/// propagate tie information to `by_more`.
///
/// TODO: allocates a full boolean series and doesn't fail fast
/// (see [`SeriesMethods::is_sorted`]).
fn is_sorted_fallback(
    first: &Column,
    _by_more: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<bool> {
    let s = first.as_materialized_series();
    let options = SortOptions {
        descending: descending[0],
        nulls_last: nulls_last[0],
        multithreaded: true,
        maintain_order: false,
        limit: None,
    };

    // Fast path via sorted flag.
    let null_count = s.null_count();
    if (options.descending
        && (options.nulls_last || null_count == 0)
        && matches!(s.is_sorted_flag(), IsSorted::Descending))
        || (!options.descending
            && (!options.nulls_last || null_count == 0)
            && matches!(s.is_sorted_flag(), IsSorted::Ascending))
    {
        return Ok(true);
    }

    SeriesMethods::is_sorted(s, options)
}
