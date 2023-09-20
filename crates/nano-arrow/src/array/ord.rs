//! Contains functions and function factories to order values within arrays.

use std::cmp::Ordering;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{Error, Result};
use crate::offset::Offset;
use crate::types::NativeType;

/// Compare the values at two arbitrary indices in two arrays.
pub type DynComparator = Box<dyn Fn(usize, usize) -> Ordering + Send + Sync>;

/// implements comparison using IEEE 754 total ordering for f32
// Original implementation from https://doc.rust-lang.org/std/primitive.f32.html#method.total_cmp
// TODO to change to use std when it becomes stable
#[inline]
pub fn total_cmp_f32(l: &f32, r: &f32) -> std::cmp::Ordering {
    let mut left = l.to_bits() as i32;
    let mut right = r.to_bits() as i32;

    left ^= (((left >> 31) as u32) >> 1) as i32;
    right ^= (((right >> 31) as u32) >> 1) as i32;

    left.cmp(&right)
}

/// implements comparison using IEEE 754 total ordering for f64
// Original implementation from https://doc.rust-lang.org/std/primitive.f64.html#method.total_cmp
// TODO to change to use std when it becomes stable
#[inline]
pub fn total_cmp_f64(l: &f64, r: &f64) -> std::cmp::Ordering {
    let mut left = l.to_bits() as i64;
    let mut right = r.to_bits() as i64;

    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;

    left.cmp(&right)
}

/// Total order of all native types whose Rust implementation
/// that support total order.
#[inline]
pub fn total_cmp<T>(l: &T, r: &T) -> std::cmp::Ordering
where
    T: NativeType + Ord,
{
    l.cmp(r)
}

fn compare_primitives<T: NativeType + Ord>(left: &dyn Array, right: &dyn Array) -> DynComparator {
    let left = left
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .clone();
    let right = right
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .clone();
    Box::new(move |i, j| total_cmp(&left.value(i), &right.value(j)))
}

fn compare_boolean(left: &dyn Array, right: &dyn Array) -> DynComparator {
    let left = left
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .clone();
    let right = right
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .clone();
    Box::new(move |i, j| left.value(i).cmp(&right.value(j)))
}

fn compare_f32(left: &dyn Array, right: &dyn Array) -> DynComparator {
    let left = left
        .as_any()
        .downcast_ref::<PrimitiveArray<f32>>()
        .unwrap()
        .clone();
    let right = right
        .as_any()
        .downcast_ref::<PrimitiveArray<f32>>()
        .unwrap()
        .clone();
    Box::new(move |i, j| total_cmp_f32(&left.value(i), &right.value(j)))
}

fn compare_f64(left: &dyn Array, right: &dyn Array) -> DynComparator {
    let left = left
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .unwrap()
        .clone();
    let right = right
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .unwrap()
        .clone();
    Box::new(move |i, j| total_cmp_f64(&left.value(i), &right.value(j)))
}

fn compare_string<O: Offset>(left: &dyn Array, right: &dyn Array) -> DynComparator {
    let left = left
        .as_any()
        .downcast_ref::<Utf8Array<O>>()
        .unwrap()
        .clone();
    let right = right
        .as_any()
        .downcast_ref::<Utf8Array<O>>()
        .unwrap()
        .clone();
    Box::new(move |i, j| left.value(i).cmp(right.value(j)))
}

fn compare_binary<O: Offset>(left: &dyn Array, right: &dyn Array) -> DynComparator {
    let left = left
        .as_any()
        .downcast_ref::<BinaryArray<O>>()
        .unwrap()
        .clone();
    let right = right
        .as_any()
        .downcast_ref::<BinaryArray<O>>()
        .unwrap()
        .clone();
    Box::new(move |i, j| left.value(i).cmp(right.value(j)))
}

fn compare_dict<K>(left: &DictionaryArray<K>, right: &DictionaryArray<K>) -> Result<DynComparator>
where
    K: DictionaryKey,
{
    let left_keys = left.keys().values().clone();
    let right_keys = right.keys().values().clone();

    let comparator = build_compare(left.values().as_ref(), right.values().as_ref())?;

    Ok(Box::new(move |i: usize, j: usize| {
        // safety: all dictionaries keys are guaranteed to be castable to usize
        let key_left = unsafe { left_keys[i].as_usize() };
        let key_right = unsafe { right_keys[j].as_usize() };
        (comparator)(key_left, key_right)
    }))
}

macro_rules! dyn_dict {
    ($key:ty, $lhs:expr, $rhs:expr) => {{
        let lhs = $lhs.as_any().downcast_ref().unwrap();
        let rhs = $rhs.as_any().downcast_ref().unwrap();
        compare_dict::<$key>(lhs, rhs)?
    }};
}

/// returns a comparison function that compares values at two different slots
/// between two [`Array`].
/// # Example
/// ```
/// use arrow2::array::{ord::build_compare, PrimitiveArray};
///
/// # fn main() -> arrow2::error::Result<()> {
/// let array1 = PrimitiveArray::from_slice([1, 2]);
/// let array2 = PrimitiveArray::from_slice([3, 4]);
///
/// let cmp = build_compare(&array1, &array2)?;
///
/// // 1 (index 0 of array1) is smaller than 4 (index 1 of array2)
/// assert_eq!(std::cmp::Ordering::Less, (cmp)(0, 1));
/// # Ok(())
/// # }
/// ```
/// # Error
/// The arrays' [`DataType`] must be equal and the types must have a natural order.
// This is a factory of comparisons.
pub fn build_compare(left: &dyn Array, right: &dyn Array) -> Result<DynComparator> {
    use DataType::*;
    use IntervalUnit::*;
    use TimeUnit::*;
    Ok(match (left.data_type(), right.data_type()) {
        (a, b) if a != b => {
            return Err(Error::InvalidArgumentError(
                "Can't compare arrays of different types".to_string(),
            ));
        },
        (Boolean, Boolean) => compare_boolean(left, right),
        (UInt8, UInt8) => compare_primitives::<u8>(left, right),
        (UInt16, UInt16) => compare_primitives::<u16>(left, right),
        (UInt32, UInt32) => compare_primitives::<u32>(left, right),
        (UInt64, UInt64) => compare_primitives::<u64>(left, right),
        (Int8, Int8) => compare_primitives::<i8>(left, right),
        (Int16, Int16) => compare_primitives::<i16>(left, right),
        (Int32, Int32)
        | (Date32, Date32)
        | (Time32(Second), Time32(Second))
        | (Time32(Millisecond), Time32(Millisecond))
        | (Interval(YearMonth), Interval(YearMonth)) => compare_primitives::<i32>(left, right),
        (Int64, Int64)
        | (Date64, Date64)
        | (Time64(Microsecond), Time64(Microsecond))
        | (Time64(Nanosecond), Time64(Nanosecond))
        | (Timestamp(Second, None), Timestamp(Second, None))
        | (Timestamp(Millisecond, None), Timestamp(Millisecond, None))
        | (Timestamp(Microsecond, None), Timestamp(Microsecond, None))
        | (Timestamp(Nanosecond, None), Timestamp(Nanosecond, None))
        | (Duration(Second), Duration(Second))
        | (Duration(Millisecond), Duration(Millisecond))
        | (Duration(Microsecond), Duration(Microsecond))
        | (Duration(Nanosecond), Duration(Nanosecond)) => compare_primitives::<i64>(left, right),
        (Float32, Float32) => compare_f32(left, right),
        (Float64, Float64) => compare_f64(left, right),
        (Decimal(_, _), Decimal(_, _)) => compare_primitives::<i128>(left, right),
        (Utf8, Utf8) => compare_string::<i32>(left, right),
        (LargeUtf8, LargeUtf8) => compare_string::<i64>(left, right),
        (Binary, Binary) => compare_binary::<i32>(left, right),
        (LargeBinary, LargeBinary) => compare_binary::<i64>(left, right),
        (Dictionary(key_type_lhs, ..), Dictionary(key_type_rhs, ..)) => {
            match (key_type_lhs, key_type_rhs) {
                (IntegerType::UInt8, IntegerType::UInt8) => dyn_dict!(u8, left, right),
                (IntegerType::UInt16, IntegerType::UInt16) => dyn_dict!(u16, left, right),
                (IntegerType::UInt32, IntegerType::UInt32) => dyn_dict!(u32, left, right),
                (IntegerType::UInt64, IntegerType::UInt64) => dyn_dict!(u64, left, right),
                (IntegerType::Int8, IntegerType::Int8) => dyn_dict!(i8, left, right),
                (IntegerType::Int16, IntegerType::Int16) => dyn_dict!(i16, left, right),
                (IntegerType::Int32, IntegerType::Int32) => dyn_dict!(i32, left, right),
                (IntegerType::Int64, IntegerType::Int64) => dyn_dict!(i64, left, right),
                (lhs, _) => {
                    return Err(Error::InvalidArgumentError(format!(
                        "Dictionaries do not support keys of type {lhs:?}"
                    )))
                },
            }
        },
        (lhs, _) => {
            return Err(Error::InvalidArgumentError(format!(
                "The data type type {lhs:?} has no natural order"
            )))
        },
    })
}
