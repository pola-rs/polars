//! Contains operators to sort individual and slices of [`Array`]s.
use std::cmp::Ordering;

use crate::array::ord;
use crate::compute::take;
use crate::datatypes::*;
use crate::error::{Error, Result};
use crate::offset::Offset;
use crate::{array::*, types::Index};

mod binary;
mod boolean;
mod common;
mod lex_sort;
mod primitive;
mod utf8;

pub mod row;
pub(crate) use lex_sort::build_compare;
pub use lex_sort::{lexsort, lexsort_to_indices, lexsort_to_indices_impl, SortColumn};

macro_rules! dyn_sort {
    ($ty:ty, $array:expr, $cmp:expr, $options:expr, $limit:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .unwrap();
        Ok(Box::new(primitive::sort_by::<$ty, _>(
            &array, $cmp, $options, $limit,
        )))
    }};
}

/// Sort the [`Array`] using [`SortOptions`].
///
/// Performs an unstable sort on values and indices. Nulls are ordered according to the `nulls_first` flag in `options`.
/// Floats are sorted using IEEE 754 totalOrder
/// # Errors
/// Errors if the [`DataType`] is not supported.
pub fn sort(
    values: &dyn Array,
    options: &SortOptions,
    limit: Option<usize>,
) -> Result<Box<dyn Array>> {
    match values.data_type() {
        DataType::Int8 => dyn_sort!(i8, values, ord::total_cmp, options, limit),
        DataType::Int16 => dyn_sort!(i16, values, ord::total_cmp, options, limit),
        DataType::Int32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(IntervalUnit::YearMonth) => {
            dyn_sort!(i32, values, ord::total_cmp, options, limit)
        }
        DataType::Int64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, None)
        | DataType::Duration(_) => dyn_sort!(i64, values, ord::total_cmp, options, limit),
        DataType::UInt8 => dyn_sort!(u8, values, ord::total_cmp, options, limit),
        DataType::UInt16 => dyn_sort!(u16, values, ord::total_cmp, options, limit),
        DataType::UInt32 => dyn_sort!(u32, values, ord::total_cmp, options, limit),
        DataType::UInt64 => dyn_sort!(u64, values, ord::total_cmp, options, limit),
        DataType::Float32 => dyn_sort!(f32, values, ord::total_cmp_f32, options, limit),
        DataType::Float64 => dyn_sort!(f64, values, ord::total_cmp_f64, options, limit),
        _ => {
            let indices = sort_to_indices::<u64>(values, options, limit)?;
            take::take(values, &indices)
        }
    }
}

// partition indices into valid and null indices
fn partition_validity<I: Index>(array: &dyn Array) -> (Vec<I>, Vec<I>) {
    let length = array.len();
    let indices = (0..length).map(|x| I::from_usize(x).unwrap());
    if let Some(validity) = array.validity() {
        indices.partition(|index| validity.get_bit(index.to_usize()))
    } else {
        (indices.collect(), vec![])
    }
}

macro_rules! dyn_sort_indices {
    ($index:ty, $ty:ty, $array:expr, $cmp:expr, $options:expr, $limit:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .unwrap();
        Ok(primitive::indices_sorted_unstable_by::<$index, $ty, _>(
            &array, $cmp, $options, $limit,
        ))
    }};
}

/// Sort elements from `values` into a non-nullable [`PrimitiveArray`] of indices that sort `values`.
pub fn sort_to_indices<I: Index>(
    values: &dyn Array,
    options: &SortOptions,
    limit: Option<usize>,
) -> Result<PrimitiveArray<I>> {
    match values.data_type() {
        DataType::Boolean => {
            let (v, n) = partition_validity(values);
            Ok(boolean::sort_boolean(
                values.as_any().downcast_ref().unwrap(),
                v,
                n,
                options,
                limit,
            ))
        }
        DataType::Int8 => dyn_sort_indices!(I, i8, values, ord::total_cmp, options, limit),
        DataType::Int16 => dyn_sort_indices!(I, i16, values, ord::total_cmp, options, limit),
        DataType::Int32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(IntervalUnit::YearMonth) => {
            dyn_sort_indices!(I, i32, values, ord::total_cmp, options, limit)
        }
        DataType::Int64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, None)
        | DataType::Duration(_) => {
            dyn_sort_indices!(I, i64, values, ord::total_cmp, options, limit)
        }
        DataType::UInt8 => dyn_sort_indices!(I, u8, values, ord::total_cmp, options, limit),
        DataType::UInt16 => dyn_sort_indices!(I, u16, values, ord::total_cmp, options, limit),
        DataType::UInt32 => dyn_sort_indices!(I, u32, values, ord::total_cmp, options, limit),
        DataType::UInt64 => dyn_sort_indices!(I, u64, values, ord::total_cmp, options, limit),
        DataType::Float32 => dyn_sort_indices!(I, f32, values, ord::total_cmp_f32, options, limit),
        DataType::Float64 => dyn_sort_indices!(I, f64, values, ord::total_cmp_f64, options, limit),
        DataType::Utf8 => Ok(utf8::indices_sorted_unstable_by::<I, i32>(
            values.as_any().downcast_ref().unwrap(),
            options,
            limit,
        )),
        DataType::LargeUtf8 => Ok(utf8::indices_sorted_unstable_by::<I, i64>(
            values.as_any().downcast_ref().unwrap(),
            options,
            limit,
        )),
        DataType::Binary => Ok(binary::indices_sorted_unstable_by::<I, i32>(
            values.as_any().downcast_ref().unwrap(),
            options,
            limit,
        )),
        DataType::LargeBinary => Ok(binary::indices_sorted_unstable_by::<I, i64>(
            values.as_any().downcast_ref().unwrap(),
            options,
            limit,
        )),
        DataType::List(field) => {
            let (v, n) = partition_validity(values);
            match &field.data_type {
                DataType::Int8 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::Int16 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::Int32 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::Int64 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt8 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt16 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt32 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt64 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                t => Err(Error::NotYetImplemented(format!(
                    "Sort not supported for list type {t:?}"
                ))),
            }
        }
        DataType::LargeList(field) => {
            let (v, n) = partition_validity(values);
            match field.data_type() {
                DataType::Int8 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::Int16 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::Int32 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::Int64 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::UInt8 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::UInt16 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::UInt32 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                DataType::UInt64 => Ok(sort_list::<I, i64>(values, v, n, options, limit)),
                t => Err(Error::NotYetImplemented(format!(
                    "Sort not supported for list type {t:?}"
                ))),
            }
        }
        DataType::FixedSizeList(field, _) => {
            let (v, n) = partition_validity(values);
            match field.data_type() {
                DataType::Int8 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::Int16 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::Int32 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::Int64 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt8 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt16 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt32 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                DataType::UInt64 => Ok(sort_list::<I, i32>(values, v, n, options, limit)),
                t => Err(Error::NotYetImplemented(format!(
                    "Sort not supported for list type {t:?}"
                ))),
            }
        }
        DataType::Dictionary(key_type, value_type, _) => match value_type.as_ref() {
            DataType::Utf8 => Ok(sort_dict::<I, i32>(values, key_type, options, limit)),
            DataType::LargeUtf8 => Ok(sort_dict::<I, i64>(values, key_type, options, limit)),
            t => Err(Error::NotYetImplemented(format!(
                "Sort not supported for dictionary type with keys {t:?}"
            ))),
        },
        t => Err(Error::NotYetImplemented(format!(
            "Sort not supported for data type {t:?}"
        ))),
    }
}

fn sort_dict<I: Index, O: Offset>(
    values: &dyn Array,
    key_type: &IntegerType,
    options: &SortOptions,
    limit: Option<usize>,
) -> PrimitiveArray<I> {
    match_integer_type!(key_type, |$T| {
        utf8::indices_sorted_unstable_by_dictionary::<I, $T, O>(
            values.as_any().downcast_ref().unwrap(),
            options,
            limit,
        )
    })
}

/// Checks if an array of type `datatype` can be sorted
///
/// # Examples
/// ```
/// use arrow2::compute::sort::can_sort;
/// use arrow2::datatypes::{DataType};
///
/// let data_type = DataType::Int8;
/// assert_eq!(can_sort(&data_type), true);
///
/// let data_type = DataType::LargeBinary;
/// assert_eq!(can_sort(&data_type), true)
/// ```
pub fn can_sort(data_type: &DataType) -> bool {
    match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(_)
        | DataType::Int64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, None)
        | DataType::Duration(_)
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::LargeBinary => true,
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            matches!(
                field.data_type(),
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
            )
        }
        DataType::Dictionary(_, value_type, _) => {
            matches!(*value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8)
        }
        _ => false,
    }
}

/// Options that define how sort kernels should behave
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SortOptions {
    /// Whether to sort in descending order
    pub descending: bool,
    /// Whether to sort nulls first
    pub nulls_first: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            descending: false,
            // default to nulls first to match spark's behavior
            nulls_first: true,
        }
    }
}

fn sort_list<I, O>(
    values: &dyn Array,
    value_indices: Vec<I>,
    null_indices: Vec<I>,
    options: &SortOptions,
    limit: Option<usize>,
) -> PrimitiveArray<I>
where
    I: Index,
    O: Offset,
{
    let mut valids: Vec<(I, Box<dyn Array>)> = values
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .map_or_else(
            || {
                let values = values.as_any().downcast_ref::<ListArray<O>>().unwrap();
                value_indices
                    .iter()
                    .copied()
                    .map(|index| (index, values.value(index.to_usize())))
                    .collect()
            },
            |values| {
                value_indices
                    .iter()
                    .copied()
                    .map(|index| (index, values.value(index.to_usize())))
                    .collect()
            },
        );

    if !options.descending {
        valids.sort_by(|a, b| cmp_array(a.1.as_ref(), b.1.as_ref()))
    } else {
        valids.sort_by(|a, b| cmp_array(b.1.as_ref(), a.1.as_ref()))
    }

    let values = valids.iter().map(|tuple| tuple.0);

    let mut values = if options.nulls_first {
        null_indices.into_iter().chain(values).collect::<Vec<I>>()
    } else {
        values.chain(null_indices.into_iter()).collect::<Vec<I>>()
    };

    values.truncate(limit.unwrap_or(values.len()));

    let data_type = I::PRIMITIVE.into();
    PrimitiveArray::<I>::new(data_type, values.into(), None)
}

/// Compare two `Array`s based on the ordering defined in [ord](crate::array::ord).
fn cmp_array(a: &dyn Array, b: &dyn Array) -> Ordering {
    let cmp_op = ord::build_compare(a, b).unwrap();
    let length = a.len().min(b.len());

    for i in 0..length {
        let result = cmp_op(i, i);
        if result != Ordering::Equal {
            return result;
        }
    }
    a.len().cmp(&b.len())
}
