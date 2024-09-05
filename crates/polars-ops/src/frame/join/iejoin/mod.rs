mod filtered_bit_array;
mod l1_l2;

use filtered_bit_array::FilteredBitArray;
use l1_l2::*;
use polars_core::chunked_array::ChunkedArray;
use polars_core::datatypes::{IdxCa, NumericNative, PolarsNumericType};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::{with_match_physical_numeric_polars_type, POOL};
use polars_error::{polars_err, PolarsResult};
use polars_utils::binary_search::ExponentialSearch;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::total_ord::{TotalEq};
use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::frame::_finish_join;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InequalityOperator {
    #[default]
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl InequalityOperator {
    fn is_strict(&self) -> bool {
        matches!(self, InequalityOperator::Gt | InequalityOperator::Lt)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IEJoinOptions {
    pub operator1: InequalityOperator,
    pub operator2: InequalityOperator,
}

#[allow(clippy::too_many_arguments)]
fn ie_join_impl_t<T: PolarsNumericType>(
    slice: Option<(i64, usize)>,
    l1_order: IdxCa,
    l2_order: &[IdxSize],
    op1: InequalityOperator,
    op2: InequalityOperator,
    x: Series,
    y_ordered_by_x: Series,
    left_height: usize,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)> {
    // Create a bit array with order corresponding to L1,
    // denoting which entries have been visited while traversing L2.
    let mut bit_array = FilteredBitArray::from_len_zeroed(l1_order.len());

    let mut left_row_idx: Vec<IdxSize> = vec![];
    let mut right_row_idx: Vec<IdxSize> = vec![];

    let slice_end = match slice {
        Some((offset, len)) if offset >= 0 => Some(offset.saturating_add_unsigned(len as u64)),
        _ => None,
    };
    let mut match_count = 0;

    let ca: &ChunkedArray<T> = x.as_ref().as_ref();
    let l1_array = build_l1_array(ca, &l1_order, left_height as IdxSize)?;

    if op2.is_strict() {
        // For strict inequalities, we rely on using a stable sort of l2 so that
        // p values only increase as we traverse a run of equal y values.
        // To handle inclusive comparisons in x and duplicate x values we also need the
        // sort of l1 to be stable, so that the left hand side entries come before the right
        // hand side entries (as we mark visited entries from the right hand side).
        for &p in l2_order {
            match_count += unsafe {
                l1_array.process_entry(
                    p as usize,
                    &mut bit_array,
                    op1,
                    &mut left_row_idx,
                    &mut right_row_idx,
                )
            };

            if slice_end.is_some_and(|end| match_count >= end) {
                break;
            }
        }
    } else {
        let l2_array = build_l2_array(&y_ordered_by_x, l2_order)?;

        // For non-strict inequalities in l2, we need to track runs of equal y values and only
        // check for matches after we reach the end of the run and have marked all rhs entries
        // in the run as visited.
        let mut run_start = 0;

        for i in 0..l2_array.len() {
            // Elide bound checks
            unsafe {
                let item = l2_array.get_unchecked_release(i);
                let p = item.l1_index;
                l1_array.mark_visited(p as usize, &mut bit_array);

                if item.run_end {
                    for l2_item in l2_array.get_unchecked_release(run_start..i + 1) {
                        let p = l2_item.l1_index;
                        match_count += l1_array.process_lhs_entry(
                            p as usize,
                            &bit_array,
                            op1,
                            &mut left_row_idx,
                            &mut right_row_idx,
                        );
                    }

                    run_start = i + 1;

                    if slice_end.is_some_and(|end| match_count >= end) {
                        break;
                    }
                }
            }
        }
    }
    Ok((left_row_idx, right_row_idx))
}

/// Inequality join. Matches rows between two DataFrames using two inequality operators
/// (one of [<, <=, >, >=]).
/// Based on Khayyat et al. 2015, "Lightning Fast and Space Efficient Inequality Joins"
/// and extended to work with duplicate values.
pub fn iejoin(
    left: &DataFrame,
    right: &DataFrame,
    selected_left: Vec<Series>,
    selected_right: Vec<Series>,
    options: &IEJoinOptions,
    suffix: Option<PlSmallStr>,
    slice: Option<(i64, usize)>,
) -> PolarsResult<DataFrame> {
    if selected_left.len() != 2 {
        return Err(
            polars_err!(ComputeError: "IEJoin requires exactly two expressions from the left DataFrame"),
        );
    };
    if selected_right.len() != 2 {
        return Err(
            polars_err!(ComputeError: "IEJoin requires exactly two expressions from the right DataFrame"),
        );
    };

    let op1 = options.operator1;
    let op2 = options.operator2;

    // Determine the sort order based on the comparison operators used.
    // We want to sort L1 so that "x[i] op1 x[j]" is true for j > i,
    // and L2 so that "y[i] op2 y[j]" is true for j < i
    // (except in the case of duplicates and strict inequalities).
    // Note that the algorithms published in Khayyat et al. have incorrect logic for
    // determining whether to sort descending.
    let l1_descending = matches!(op1, InequalityOperator::Gt | InequalityOperator::GtEq);
    let l2_descending = matches!(op2, InequalityOperator::Lt | InequalityOperator::LtEq);

    let mut x = selected_left[0].to_physical_repr().into_owned();
    x.extend(&selected_right[0].to_physical_repr())?;
    // Rechunk because we will gather.
    let x = x.rechunk();

    let mut y = selected_left[1].to_physical_repr().into_owned();
    y.extend(&selected_right[1].to_physical_repr())?;
    // Rechunk because we will gather.
    let y = y.rechunk();

    let l1_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l1_descending);
    // Get ordering of x, skipping any null entries as these cannot be matches
    let l1_order = x
        .arg_sort(l1_sort_options)
        .slice(x.null_count() as i64, x.len() - x.null_count());

    let y_ordered_by_x = unsafe { y.take_unchecked(&l1_order) };
    let l2_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l2_descending);
    // Get the indexes into l1, ordered by y values.
    // l2_order is the same as "p" from Khayyat et al.
    let l2_order = y_ordered_by_x
        .arg_sort(l2_sort_options)
        .slice(
            y_ordered_by_x.null_count() as i64,
            y_ordered_by_x.len() - y_ordered_by_x.null_count(),
        )
        .rechunk();
    let l2_order = l2_order.downcast_get(0).unwrap().values().as_slice();

    let (left_row_idx, right_row_idx) = with_match_physical_numeric_polars_type!(x.dtype(), |$T| {
         ie_join_impl_t::<$T>(
            slice,
            l1_order,
            l2_order,
            op1,
            op2,
            x,
            y_ordered_by_x,
            left.height()
        )
    })?;

    debug_assert_eq!(left_row_idx.len(), right_row_idx.len());
    let left_row_idx = IdxCa::from_vec("".into(), left_row_idx);
    let right_row_idx = IdxCa::from_vec("".into(), right_row_idx);
    let (left_row_idx, right_row_idx) = match slice {
        None => (left_row_idx, right_row_idx),
        Some((offset, len)) => (
            left_row_idx.slice(offset, len),
            right_row_idx.slice(offset, len),
        ),
    };

    let (join_left, join_right) = unsafe {
        POOL.join(
            || left.take_unchecked(&left_row_idx),
            || right.take_unchecked(&right_row_idx),
        )
    };

    _finish_join(join_left, join_right, suffix)
}
