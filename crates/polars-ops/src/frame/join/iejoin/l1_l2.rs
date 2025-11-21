#![allow(unsafe_op_in_unsafe_fn)]
use polars_core::chunked_array::ChunkedArray;
use polars_core::datatypes::{IdxCa, PolarsNumericType};
use polars_core::prelude::Series;
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::total_ord::TotalOrd;

use super::*;

/// Create a vector of L1 items from the array of LHS x values concatenated with RHS x values
/// and their ordering.
pub(super) fn build_l1_array<T>(
    ca: &ChunkedArray<T>,
    order: &IdxCa,
    right_df_offset: IdxSize,
) -> PolarsResult<Vec<L1Item<T::Native>>>
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

    Ok(array)
}

pub(super) fn build_l2_array(s: &Series, order: &[IdxSize]) -> PolarsResult<Vec<L2Item>> {
    with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
        build_l2_array_impl::<$T>(s.as_ref().as_ref(), order)
    })
}

/// Create a vector of L2 items from the array of y values ordered according to the L1 order,
/// and their ordering. We don't need to store actual y values but only track whether we're at
/// the end of a run of equal values.
fn build_l2_array_impl<T>(ca: &ChunkedArray<T>, order: &[IdxSize]) -> PolarsResult<Vec<L2Item>>
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

/// Item in L1 array used in the IEJoin algorithm
#[derive(Clone, Copy, Debug)]
pub(super) struct L1Item<T> {
    /// 1 based index for entries from the LHS df, or -1 based index for entries from the RHS
    pub(super) row_index: i64,
    /// X value
    pub(super) value: T,
}

/// Item in L2 array used in the IEJoin algorithm
#[derive(Clone, Copy, Debug)]
pub(super) struct L2Item {
    /// Corresponding index into the L1 array of
    pub(super) l1_index: IdxSize,
    /// Whether this is the end of a run of equal y values
    pub(super) run_end: bool,
}

pub(super) trait L1Array {
    unsafe fn process_entry(
        &self,
        l1_index: usize,
        bit_array: &mut FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64;

    unsafe fn process_lhs_entry(
        &self,
        l1_index: usize,
        bit_array: &FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64;

    unsafe fn mark_visited(&self, index: usize, bit_array: &mut FilteredBitArray);
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
    let sub_l1 = l1_array.get_unchecked(index..);
    let value = l1_array.get_unchecked(index).value;

    match operator {
        InequalityOperator::Gt => {
            sub_l1.partition_point_exponential(|a| a.value.tot_ge(&value)) + index
        },
        InequalityOperator::Lt => {
            sub_l1.partition_point_exponential(|a| a.value.tot_le(&value)) + index
        },
        InequalityOperator::GtEq => {
            sub_l1.partition_point_exponential(|a| value.tot_lt(&a.value)) + index
        },
        InequalityOperator::LtEq => {
            sub_l1.partition_point_exponential(|a| value.tot_gt(&a.value)) + index
        },
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
    unsafe {
        bit_array.on_set_bits_from(start_index, |set_bit: usize| {
            // SAFETY
            // set bit is within bounds.
            let right_row_index = l1_array.get_unchecked(set_bit).row_index;
            debug_assert!(right_row_index < 0);
            left_row_ids.push((row_index - 1) as IdxSize);
            right_row_ids.push((-right_row_index) as IdxSize - 1);
            match_count += 1;
        })
    };

    match_count
}

impl<T> L1Array for Vec<L1Item<T>>
where
    T: NumericNative,
{
    unsafe fn process_entry(
        &self,
        l1_index: usize,
        bit_array: &mut FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64 {
        let row_index = self.get_unchecked(l1_index).row_index;
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
            bit_array.set_bit_unchecked(l1_index);
            0
        }
    }

    unsafe fn process_lhs_entry(
        &self,
        l1_index: usize,
        bit_array: &FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut Vec<IdxSize>,
        right_row_ids: &mut Vec<IdxSize>,
    ) -> i64 {
        let row_index = self.get_unchecked(l1_index).row_index;
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

    unsafe fn mark_visited(&self, index: usize, bit_array: &mut FilteredBitArray) {
        let from_lhs = self.get_unchecked(index).row_index > 0;
        // We only mark RHS entries as visited,
        // so that we don't try to match LHS entries with other LHS entries.
        if !from_lhs {
            bit_array.set_bit_unchecked(index);
        }
    }
}
