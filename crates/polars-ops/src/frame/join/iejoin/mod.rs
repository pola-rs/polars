mod filtered_bit_array;

use filtered_bit_array::FilteredBitArray;
use polars_core::chunked_array::ChunkedArray;
use polars_core::datatypes::{IdxCa, NumericNative, PolarsNumericType};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::{with_match_physical_numeric_polars_type, POOL};
use polars_error::{polars_err, PolarsResult};
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::total_ord::{TotalEq, TotalOrd};
use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use polars_utils::binary_search::ExponentialSearch;
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

    let mut x = selected_left[0].to_physical_repr().into_owned();
    x.extend(&selected_right[0].to_physical_repr())?;
    // Rechunk because we will gather.
    let x = x.rechunk();

    let mut y = selected_left[1].to_physical_repr().into_owned();
    y.extend(&selected_right[1].to_physical_repr())?;
    // Rechunk because we will gather.
    let y = y.rechunk();

    // Determine the sort order based on the comparison operators used.
    // We want to sort L1 so that "x[i] op1 x[j]" is true for j > i,
    // and L2 so that "y[i] op2 y[j]" is true for j < i
    // (except in the case of duplicates and strict inequalities).
    // Note that the algorithms published in Khayyat et al. have incorrect logic for
    // determining whether to sort descending.
    let l1_descending = matches!(op1, InequalityOperator::Gt | InequalityOperator::GtEq);
    let l2_descending = matches!(op2, InequalityOperator::Lt | InequalityOperator::LtEq);
    // let l1_descending = false;
    // let l2_descending = false;

    let l1_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l1_descending);
    // Get ordering of x, skipping any null entries as these cannot be matches
    let l1_order = x
        .arg_sort(l1_sort_options)
        .slice(x.null_count() as i64, x.len() - x.null_count());

    let l1_array = with_match_physical_numeric_polars_type!(x.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = x.as_ref().as_ref().as_ref();
        build_l1_array(ca, &l1_order, left.height() as IdxSize)
    })?;

    let y_ordered = unsafe { y.take_unchecked(&l1_order) };
    let l2_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l2_descending);
    // Get the indexes into l1, ordered by y values.
    // l2_order is the same as "p" from Khayyat et al.
    let l2_order = y_ordered
        .arg_sort(l2_sort_options)
        .slice(
            y_ordered.null_count() as i64,
            y_ordered.len() - y_ordered.null_count(),
        )
        .rechunk();
    let l2_order = l2_order.downcast_get(0).unwrap().values().as_slice();

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

    if op2.is_strict() {
        // For strict inequalities, we rely on using a stable sort of l2 so that
        // p values only increase as we traverse a run of equal y values.
        // To handle inclusive comparisons in x and duplicate x values we also need the
        // sort of l1 to be stable, so that the left hand side entries come before the right
        // hand side entries (as we mark visited entries from the right hand side).
        for &p in l2_order {
            match_count += l1_array.process_entry(
                p as usize,
                &mut bit_array,
                op1,
                &mut left_row_idx,
                &mut right_row_idx,
            );

            if slice_end.is_some_and(|end| match_count >= end) {
                break;
            }
        }
    } else {
        let l2_array = with_match_physical_numeric_polars_type!(y.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = y_ordered.as_ref().as_ref().as_ref();
            build_l2_array(ca, &l2_order)
        })?;
        // For non-strict inequalities in l2, we need to track runs of equal y values and only
        // check for matches after we reach the end of the run and have marked all rhs entries
        // in the run as visited.
        let mut run_start = 0;
        for i in 0..l2_array.len() {
            let p = l2_array[i].l1_index;
            l1_array.mark_visited(p as usize, &mut bit_array);
            if l2_array[i].run_end {
                for l2_item in l2_array.iter().take(i + 1).skip(run_start) {
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

/// Item in L1 array used in the IEJoin algorithm
#[derive(Clone, Copy, Debug)]
struct L1Item<T> {
    /// 1 based index for entries from the LHS df, or -1 based index for entries from the RHS
    row_index: i64,
    /// X value
    value: T,
}

/// Item in L2 array used in the IEJoin algorithm
#[derive(Clone, Copy, Debug)]
struct L2Item {
    /// Corresponding index into the L1 array of
    l1_index: IdxSize,
    /// Whether this is the end of a run of equal y values
    run_end: bool,
}

trait L1Array {
    fn process_entry(
        &self,
        l1_index: usize,
        bit_array: &mut FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64;

    fn process_lhs_entry(
        &self,
        l1_index: usize,
        bit_array: &FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64;

    fn mark_visited(&self, index: usize, bit_array: &mut FilteredBitArray);
}

/// Find the position in the L1 array where we should begin checking for matches,
/// given the index in L1 corresponding to the current position in L2.
unsafe fn find_search_start_index<T>(
    l1_array: &[L1Item<T>],
    index: usize,
    operator: InequalityOperator,
) -> usize
where
    T: NumericNative,
    T: TotalOrd,
{
    let sub_l1 = l1_array.get_unchecked_release(index..);
    let value = l1_array.get_unchecked_release(index).value;

    match operator {
        InequalityOperator::Gt => {
            sub_l1.partition_point_exponential(|a| a.value.tot_ge(&value)) + index
        },
        InequalityOperator::Lt => {
            sub_l1.partition_point_exponential(|a| a.value.tot_le(&value)) + index
        }
        InequalityOperator::GtEq => {
            sub_l1.partition_point_exponential(|a| value.tot_lt(&a.value)) + index
        }
        InequalityOperator::LtEq => {
            sub_l1.partition_point_exponential(|a| value.tot_gt(&a.value)) + index
        }
    }
}

fn find_matches_in_l1<T>(
    l1_array: &[L1Item<T>],
    l1_index: usize,
    row_index: i64,
    bit_array: &FilteredBitArray,
    op1: InequalityOperator,
    left_row_ids: &mut Vec<IdxSize>,
    right_row_ids: &mut Vec<IdxSize>,
) -> i64
where
    T: NumericNative,
    T: TotalOrd,
{
    debug_assert!(row_index > 0);
    let mut match_count = 0;

    // This entry comes from the left hand side DataFrame.
    // Find all following entries in L1 (meaning they satisfy the first operator)
    // that have already been visited (so satisfy the second operator).
    // Because we use a stable sort for l2, we know that we won't find any
    // matches for duplicate y values when traversing forwards in l1.
    let start_index = unsafe { find_search_start_index(l1_array, l1_index, op1) };
    bit_array.on_set_bits_from(start_index, |set_bit: usize| {
        // SAFETY
        // set bit is within bounds.
        let right_row_index = unsafe { l1_array.get_unchecked_release(set_bit) }.row_index;
        debug_assert!(right_row_index < 0);
        left_row_ids.push((row_index - 1) as IdxSize);
        right_row_ids.push((-right_row_index) as IdxSize - 1);
        match_count += 1;
    });

    match_count
}

impl<T> L1Array for Vec<L1Item<T>>
where
    T: NumericNative,
{
    fn process_entry(
        &self,
        l1_index: usize,
        bit_array: &mut FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64 {
        let row_index = self[l1_index].row_index;
        let from_lhs = row_index > 0;
        if from_lhs {
            find_matches_in_l1(
                self,
                l1_index,
                row_index,
                bit_array,
                op1,
                left_row_ids,
                right_row_ids,
            )
        } else {
            bit_array.set_bit(l1_index);
            0
        }
    }

    fn process_lhs_entry(
        &self,
        l1_index: usize,
        bit_array: &FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64 {
        let row_index = self[l1_index].row_index;
        let from_lhs = row_index > 0;
        if from_lhs {
            find_matches_in_l1(
                self,
                l1_index,
                row_index,
                bit_array,
                op1,
                left_row_ids,
                right_row_ids,
            )
        } else {
            0
        }
    }

    fn mark_visited(&self, index: usize, bit_array: &mut FilteredBitArray) {
        let from_lhs = self[index].row_index > 0;
        // We only mark RHS entries as visited,
        // so that we don't try to match LHS entries with other LHS entries.
        if !from_lhs {
            bit_array.set_bit(index);
        }
    }
}

/// Create a vector of L1 items from the array of LHS x values concatenated with RHS x values
/// and their ordering.
fn build_l1_array<T>(
    ca: &ChunkedArray<T>,
    order: &IdxCa,
    right_df_offset: IdxSize,
) -> PolarsResult<Box<dyn L1Array>>
where
    T: PolarsNumericType,
{
    assert_eq!(order.null_count(), 0);
    assert_eq!(ca.chunks().len(), 1);
    let arr = ca.downcast_get(0).unwrap();
    // Even if there are nulls, they will not be selected by order.
    let values = arr.values().as_slice();

    let mut array: Vec<L1Item<T::Native>> = Vec::with_capacity(ca.len());

    for order_arr in order.downcast_iter() {
        for index in order_arr.values().as_slice().iter().copied() {
            debug_assert!(arr.get(index as usize).is_some());
            let value = unsafe { *values.get_unchecked(index as usize) };
            let row_index = if index < right_df_offset {
                // Row from LHS
                index as i64 + 1
            } else {
                // Row from RHS
                -((index - right_df_offset) as i64) - 1
            };
            array.push(L1Item { row_index, value });
        }
    }

    Ok(Box::new(array))
}

/// Create a vector of L2 items from the array of y values ordered according to the L1 order,
/// and their ordering. We don't need to store actual y values but only track whether we're at
/// the end of a run of equal values.
fn build_l2_array<T>(ca: &ChunkedArray<T>, order: &[IdxSize]) -> PolarsResult<Vec<L2Item>>
where
    T: PolarsNumericType,
    T::Native: TotalOrd,
{
    assert_eq!(ca.chunks().len(), 1);

    let mut array = Vec::with_capacity(ca.len());
    let mut prev_index = 0;
    let mut prev_value = T::Native::default();

    let arr = ca.downcast_get(0).unwrap();
    // Even if there are nulls, they will not be selected by order.
    let values = arr.values().as_slice();

    for (i, l1_index) in order.iter().copied().enumerate() {
        debug_assert!(arr.get(l1_index as usize).is_some());
        let value = unsafe { *values.get_unchecked(l1_index as usize) };
        if i > 0 {
            array.push(L2Item {
                l1_index: prev_index,
                run_end: value.tot_ne(&prev_value),
            });
        }
        prev_index = l1_index;
        prev_value = value;
    }
    if !order.is_empty() {
        array.push(L2Item {
            l1_index: prev_index,
            run_end: true,
        });
    }
    Ok(array)
}
