use std::cmp::Ordering;

use polars_core::chunked_array::ops::SortOptions;
#[cfg(feature = "dtype-struct")]
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
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
            let col = self.column(single_by)?;
            return SeriesMethods::is_sorted(
                col.as_materialized_series(),
                SortOptions::new()
                    .with_order_descending(descending[0])
                    .with_nulls_last(nulls_last[0]),
            );
        }

        if std::env::var("AMBER_DF_IS_SORTED").unwrap() == "row-encode" {
            let by_vec = by
                .iter()
                .map(|name| self.column(name).cloned())
                .try_collect_vec()?;
            let row_encoded =
                _get_rows_encoded_ca(PlSmallStr::EMPTY, &by_vec, descending, nulls_last, false)?;
            let s = Series::new(PlSmallStr::EMPTY, row_encoded);
            return SeriesMethods::is_sorted(&s, SortOptions::new());
        }

        let mut cols: Vec<Column> = Vec::with_capacity(by.len());
        let mut desc: Vec<bool> = Vec::with_capacity(by.len());
        let mut nls: Vec<bool> = Vec::with_capacity(by.len());
        for (idx, c) in by.iter().enumerate() {
            let col = self.column(c)?.clone();
            if should_row_encode_dtype(col.dtype()) {
                let encoded = _get_rows_encoded_ca(
                    c.clone(),
                    &[col.to_physical_repr()],
                    &[descending[idx]],
                    &[nulls_last[idx]],
                    false,
                )?;
                cols.push(encoded.into_series().into_column());
                desc.push(false);
                nls.push(false);
            } else {
                cols.push(col.to_physical_repr());
                desc.push(descending[idx]);
                nls.push(nulls_last[idx]);
            }
        }

        let mut scratch_vec_pool = (1..cols.len()).map(|_| ScratchVec::default()).collect_vec();
        is_sorted_cols(&cols, &desc, &nls, &mut scratch_vec_pool)
    }
}

fn should_row_encode_dtype(dtype: &DataType) -> bool {
    use DataType::*;
    !(matches!(
        dtype,
        Null | Boolean | String | Enum(..) | Binary | BinaryOffset
    ) || dtype.to_physical().is_numeric())
}

/// Recursively checks whether `cols` are sorted by `(cols[0], cols[1], ...)`.
///
/// All columns are assumed to have the same length. Ties in `cols[0]` trigger
/// a recursive check on the sliced tail `cols[1..]`.
#[recursive]
fn is_sorted_cols(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
    scratch_pool: &mut [ScratchVec<Column>],
) -> PolarsResult<bool> {
    let Some((by, by_more)) = by.split_first() else {
        unreachable!()
    };
    if by_more.is_empty() {
        return SeriesMethods::is_sorted(
            by.as_materialized_series(),
            SortOptions::new()
                .with_order_descending(descending[0])
                .with_nulls_last(nulls_last[0]),
        );
    }
    let s = by.as_materialized_series();

    match s.dtype() {
        DataType::Null => {
            is_sorted_cols(&by_more, &descending[1..], &nulls_last[1..], scratch_pool)
        },
        DataType::Boolean => is_sorted_ca(
            s.bool().unwrap(),
            by_more,
            descending,
            nulls_last,
            scratch_pool,
        ),
        dt if dt.to_physical().is_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_sorted_ca(ca, by_more, descending, nulls_last, scratch_pool)
            })
        },
        DataType::String => {
            let ca: &StringChunked = s.str().unwrap();
            is_sorted_ca(
                &ca.as_binary(),
                by_more,
                descending,
                nulls_last,
                scratch_pool,
            )
        },
        #[cfg(feature = "dtype-categorical")]
        dt @ DataType::Enum(..) => {
            with_match_categorical_physical_type!(dt.cat_physical().unwrap(), |$C| {
                type CA = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                let ca = s.as_ref().as_any().downcast_ref::<CA>().unwrap();
                is_sorted_ca(ca, by_more, descending, nulls_last, scratch_pool)
            })
        },
        DataType::Binary => is_sorted_ca(
            s.binary().unwrap(),
            by_more,
            descending,
            nulls_last,
            scratch_pool,
        ),
        DataType::BinaryOffset => is_sorted_ca(
            s.binary_offset().unwrap(),
            by_more,
            descending,
            nulls_last,
            scratch_pool,
        ),
        dt => unreachable!("{dt} data should have been row-encoded"),
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
