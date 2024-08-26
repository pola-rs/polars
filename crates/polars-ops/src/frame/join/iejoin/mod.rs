mod filtered_bit_array;

use std::cmp::min;

use filtered_bit_array::FilteredBitArray;
use polars_core::chunked_array::ChunkedArray;
use polars_core::datatypes::{IdxCa, IdxType, NumericNative, PolarsNumericType};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::{polars_err, PolarsResult};
use polars_utils::total_ord::{TotalEq, TotalOrd};
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
    suffix: Option<&str>,
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

    let mut y = selected_left[1].to_physical_repr().into_owned();
    y.extend(&selected_right[1].to_physical_repr())?;

    // Determine the sort order based on the comparison operators used.
    // We want to sort L1 so that "x[i] op1 x[j]" is true for j > i,
    // and L2 so that "y[i] op2 y[j]" is true for j < i
    // (except in the case of duplicates and strict inequalities).
    // Note that the algorithms published in Khayyat et al. have incorrect logic for
    // determining whether to sort descending.
    let l1_descending = matches!(op1, InequalityOperator::Gt | InequalityOperator::GtEq);
    let l2_descending = matches!(op2, InequalityOperator::Lt | InequalityOperator::LtEq);

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

    let y_ordered = y.take(&l1_order)?;
    let l2_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l2_descending);
    // Get the indexes into l1, ordered by y values.
    // l2_order is the same as "p" from Khayyat et al.
    let l2_order = y_ordered.arg_sort(l2_sort_options).slice(
        y_ordered.null_count() as i64,
        y_ordered.len() - y_ordered.null_count(),
    );

    // Create a bit array with order corresponding to L1,
    // denoting which entries have been visited while traversing L2.
    let mut bit_array = FilteredBitArray::from_len_zeroed(l1_order.len());

    let mut left_row_ids_builder = PrimitiveChunkedBuilder::<IdxType>::new("left_indices", 0);
    let mut right_row_ids_builder = PrimitiveChunkedBuilder::<IdxType>::new("right_indices", 0);

    let slice_end = match slice {
        Some((offset, len)) if offset >= 0 => Some(offset.saturating_add_unsigned(len as u64)),
        _ => None,
    };
    let mut match_count = 0;

    if is_strict(op2) {
        // For strict inequalities, we rely on using a stable sort of l2 so that
        // p values only increase as we traverse a run of equal y values.
        // To handle inclusive comparisons in x and duplicate x values we also need the
        // sort of l1 to be stable, so that the left hand side entries come before the right
        // hand side entries (as we mark visited entries from the right hand side).
        for p in l2_order.into_no_null_iter() {
            match_count += l1_array.process_entry(
                p as usize,
                &mut bit_array,
                op1,
                &mut left_row_ids_builder,
                &mut right_row_ids_builder,
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
                        &mut left_row_ids_builder,
                        &mut right_row_ids_builder,
                    );
                }

                run_start = i + 1;

                if slice_end.is_some_and(|end| match_count >= end) {
                    break;
                }
            }
        }
    }

    let left_rows = left_row_ids_builder.finish();
    let right_rows = right_row_ids_builder.finish();

    debug_assert_eq!(left_rows.len(), right_rows.len());
    let (left_rows, right_rows) = match slice {
        None => (left_rows, right_rows),
        Some((offset, len)) => (left_rows.slice(offset, len), right_rows.slice(offset, len)),
    };

    let join_left = left.take(&left_rows)?;
    let join_right = right.take(&right_rows)?;

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
        left_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
        right_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
    ) -> i64;

    fn process_lhs_entry(
        &self,
        l1_index: usize,
        bit_array: &FilteredBitArray,
        op1: InequalityOperator,
        left_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
        right_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
    ) -> i64;

    fn mark_visited(&self, index: usize, bit_array: &mut FilteredBitArray);
}

/// Find the position in the L1 array where we should begin checking for matches,
/// given the index in L1 corresponding to the current position in L2.
fn find_search_start_index<T>(
    l1_array: &[L1Item<T>],
    index: usize,
    operator: InequalityOperator,
) -> usize
where
    T: NumericNative,
    T: TotalOrd,
{
    let value = l1_array[index].value;

    if is_strict(operator) {
        // Search forward until we find a value not equal to the current x value
        let mut left_bound = index;
        let mut right_bound = index + 1;
        let mut step_size = 1;
        while right_bound < l1_array.len() && l1_array[right_bound].value.tot_eq(&value) {
            left_bound = right_bound;
            right_bound = min(right_bound + step_size, l1_array.len());
            step_size *= 2;
        }
        // Now binary search to find the first value not equal to the current x value
        while right_bound - left_bound > 1 {
            let mid = left_bound + (right_bound - left_bound) / 2;
            if l1_array[mid].value.tot_eq(&value) {
                left_bound = mid;
            } else {
                right_bound = mid;
            }
        }
        right_bound
    } else {
        // Search backwards to find the first value equal to the current x value
        let mut left_bound = if index > 0 { index - 1 } else { 0 };
        let mut right_bound = index;
        let mut step_size = 1;
        while left_bound > 0 && l1_array[left_bound].value.tot_eq(&value) {
            right_bound = left_bound;
            left_bound = if left_bound > step_size {
                left_bound - step_size
            } else {
                0
            };
            step_size *= 2
        }

        if l1_array[left_bound].value.tot_eq(&value) {
            return left_bound;
        }

        // Now binary search to find the first value equal to the current x value
        while right_bound - left_bound > 1 {
            let mid = left_bound + (right_bound - left_bound) / 2;
            if l1_array[mid].value.tot_eq(&value) {
                right_bound = mid;
            } else {
                left_bound = mid;
            }
        }
        right_bound
    }
}

fn find_matches_in_l1<T>(
    l1_array: &[L1Item<T>],
    l1_index: usize,
    row_index: i64,
    bit_array: &FilteredBitArray,
    op1: InequalityOperator,
    left_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
    right_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
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
    let start_index = find_search_start_index(l1_array, l1_index, op1);
    bit_array.on_set_bits_from(start_index, |set_bit: usize| {
        let right_row_index = l1_array[set_bit].row_index;
        debug_assert!(right_row_index < 0);
        left_row_ids.append_value((row_index - 1) as IdxSize);
        right_row_ids.append_value((-right_row_index) as IdxSize - 1);
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
        left_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
        right_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
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
        left_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
        right_row_ids: &mut PrimitiveChunkedBuilder<IdxType>,
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
    let mut array: Vec<L1Item<T::Native>> = Vec::with_capacity(ca.len());
    for index in order.into_no_null_iter() {
        let value = ca
            .get(index as usize)
            // Nulls should have been skipped over
            .ok_or_else(|| polars_err!(ComputeError: "Unexpected null value in IEJoin data"))?;
        let row_index = if index < right_df_offset {
            // Row from LHS
            index as i64 + 1
        } else {
            // Row from RHS
            -((index - right_df_offset) as i64) - 1
        };
        array.push(L1Item { row_index, value });
    }
    Ok(Box::new(array))
}

/// Create a vector of L2 items from the array of y values ordered according to the L1 order,
/// and their ordering. We don't need to store actual y values but only track whether we're at
/// the end of a run of equal values.
fn build_l2_array<T>(ca: &ChunkedArray<T>, order: &IdxCa) -> PolarsResult<Vec<L2Item>>
where
    T: PolarsNumericType,
    T::Native: TotalOrd,
{
    let mut array = Vec::with_capacity(ca.len());
    let mut prev_index = 0;
    let mut prev_value = T::Native::default();
    for (i, l1_index) in order.into_no_null_iter().enumerate() {
        let value = ca
            .get(l1_index as usize)
            // Nulls should have been skipped over
            .ok_or_else(|| polars_err!(ComputeError: "Unexpected null value in IEJoin data"))?;
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

fn is_strict(operator: InequalityOperator) -> bool {
    matches!(operator, InequalityOperator::Gt | InequalityOperator::Lt)
}
