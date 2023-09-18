use crate::bitmap::utils::{BitChunkIterExact, BitChunksExact};
use crate::datatypes::{DataType, PhysicalType, PrimitiveType};
use crate::error::{Error, Result};
use crate::offset::Offset;
use crate::scalar::*;
use crate::types::simd::*;
use crate::types::NativeType;
use crate::{
    array::{Array, BinaryArray, BooleanArray, PrimitiveArray, Utf8Array},
    bitmap::Bitmap,
};
use multiversion::multiversion;

/// Trait describing a type describing multiple lanes with an order relationship
/// consistent with the same order of `T`.
pub trait SimdOrd<T> {
    /// The minimum value
    const MIN: T;
    /// The maximum value
    const MAX: T;
    /// reduce itself to the minimum
    fn max_element(self) -> T;
    /// reduce itself to the maximum
    fn min_element(self) -> T;
    /// lane-wise maximum between two instances
    fn max_lane(self, x: Self) -> Self;
    /// lane-wise minimum between two instances
    fn min_lane(self, x: Self) -> Self;
    /// returns a new instance with all lanes equal to `MIN`
    fn new_min() -> Self;
    /// returns a new instance with all lanes equal to `MAX`
    fn new_max() -> Self;
}

#[multiversion(targets = "simd")]
fn nonnull_min_primitive<T>(values: &[T]) -> T
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    let chunks = values.chunks_exact(T::Simd::LANES);
    let remainder = chunks.remainder();

    let chunk_reduced = chunks.fold(T::Simd::new_min(), |acc, chunk| {
        let chunk = T::Simd::from_chunk(chunk);
        acc.min_lane(chunk)
    });

    let remainder = T::Simd::from_incomplete_chunk(remainder, T::Simd::MAX);
    let reduced = chunk_reduced.min_lane(remainder);

    reduced.min_element()
}

#[multiversion(targets = "simd")]
fn null_min_primitive_impl<T, I>(values: &[T], mut validity_masks: I) -> T
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
    I: BitChunkIterExact<<<T as Simd>::Simd as NativeSimd>::Chunk>,
{
    let mut chunks = values.chunks_exact(T::Simd::LANES);

    let chunk_reduced = chunks.by_ref().zip(validity_masks.by_ref()).fold(
        T::Simd::new_min(),
        |acc, (chunk, validity_chunk)| {
            let chunk = T::Simd::from_chunk(chunk);
            let mask = <T::Simd as NativeSimd>::Mask::from_chunk(validity_chunk);
            let chunk = chunk.select(mask, T::Simd::new_min());
            acc.min_lane(chunk)
        },
    );

    let remainder = T::Simd::from_incomplete_chunk(chunks.remainder(), T::Simd::MAX);
    let mask = <T::Simd as NativeSimd>::Mask::from_chunk(validity_masks.remainder());
    let remainder = remainder.select(mask, T::Simd::new_min());
    let reduced = chunk_reduced.min_lane(remainder);

    reduced.min_element()
}

/// # Panics
/// iff `values.len() != bitmap.len()` or the operation overflows.
fn null_min_primitive<T>(values: &[T], bitmap: &Bitmap) -> T
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    let (slice, offset, length) = bitmap.as_slice();
    if offset == 0 {
        let validity_masks = BitChunksExact::<<T::Simd as NativeSimd>::Chunk>::new(slice, length);
        null_min_primitive_impl(values, validity_masks)
    } else {
        let validity_masks = bitmap.chunks::<<T::Simd as NativeSimd>::Chunk>();
        null_min_primitive_impl(values, validity_masks)
    }
}

/// # Panics
/// iff `values.len() != bitmap.len()` or the operation overflows.
fn null_max_primitive<T>(values: &[T], bitmap: &Bitmap) -> T
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    let (slice, offset, length) = bitmap.as_slice();
    if offset == 0 {
        let validity_masks = BitChunksExact::<<T::Simd as NativeSimd>::Chunk>::new(slice, length);
        null_max_primitive_impl(values, validity_masks)
    } else {
        let validity_masks = bitmap.chunks::<<T::Simd as NativeSimd>::Chunk>();
        null_max_primitive_impl(values, validity_masks)
    }
}

#[multiversion(targets = "simd")]
fn nonnull_max_primitive<T>(values: &[T]) -> T
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    let chunks = values.chunks_exact(T::Simd::LANES);
    let remainder = chunks.remainder();

    let chunk_reduced = chunks.fold(T::Simd::new_max(), |acc, chunk| {
        let chunk = T::Simd::from_chunk(chunk);
        acc.max_lane(chunk)
    });

    let remainder = T::Simd::from_incomplete_chunk(remainder, T::Simd::MIN);
    let reduced = chunk_reduced.max_lane(remainder);

    reduced.max_element()
}

#[multiversion(targets = "simd")]
fn null_max_primitive_impl<T, I>(values: &[T], mut validity_masks: I) -> T
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
    I: BitChunkIterExact<<<T as Simd>::Simd as NativeSimd>::Chunk>,
{
    let mut chunks = values.chunks_exact(T::Simd::LANES);

    let chunk_reduced = chunks.by_ref().zip(validity_masks.by_ref()).fold(
        T::Simd::new_max(),
        |acc, (chunk, validity_chunk)| {
            let chunk = T::Simd::from_chunk(chunk);
            let mask = <T::Simd as NativeSimd>::Mask::from_chunk(validity_chunk);
            let chunk = chunk.select(mask, T::Simd::new_max());
            acc.max_lane(chunk)
        },
    );

    let remainder = T::Simd::from_incomplete_chunk(chunks.remainder(), T::Simd::MIN);
    let mask = <T::Simd as NativeSimd>::Mask::from_chunk(validity_masks.remainder());
    let remainder = remainder.select(mask, T::Simd::new_max());
    let reduced = chunk_reduced.max_lane(remainder);

    reduced.max_element()
}

/// Returns the minimum value in the array, according to the natural order.
/// For floating point arrays any NaN values are considered to be greater than any other non-null value
pub fn min_primitive<T>(array: &PrimitiveArray<T>) -> Option<T>
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    let null_count = array.null_count();

    // Includes case array.len() == 0
    if null_count == array.len() {
        return None;
    }
    let values = array.values();

    Some(if let Some(validity) = array.validity() {
        null_min_primitive(values, validity)
    } else {
        nonnull_min_primitive(values)
    })
}

/// Returns the maximum value in the array, according to the natural order.
/// For floating point arrays any NaN values are considered to be greater than any other non-null value
pub fn max_primitive<T>(array: &PrimitiveArray<T>) -> Option<T>
where
    T: NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    let null_count = array.null_count();

    // Includes case array.len() == 0
    if null_count == array.len() {
        return None;
    }
    let values = array.values();

    Some(if let Some(validity) = array.validity() {
        null_max_primitive(values, validity)
    } else {
        nonnull_max_primitive(values)
    })
}

/// Helper to compute min/max of [`BinaryArray`] and [`Utf8Array`]
macro_rules! min_max_binary_utf8 {
    ($array: expr, $cmp: expr) => {
        if $array.null_count() == $array.len() {
            None
        } else if $array.validity().is_some() {
            $array
                .iter()
                .reduce(|v1, v2| match (v1, v2) {
                    (None, v2) => v2,
                    (v1, None) => v1,
                    (Some(v1), Some(v2)) => {
                        if $cmp(v1, v2) {
                            Some(v2)
                        } else {
                            Some(v1)
                        }
                    }
                })
                .unwrap_or(None)
        } else {
            $array
                .values_iter()
                .reduce(|v1, v2| if $cmp(v1, v2) { v2 } else { v1 })
        }
    };
}

/// Returns the maximum value in the binary array, according to the natural order.
pub fn max_binary<O: Offset>(array: &BinaryArray<O>) -> Option<&[u8]> {
    min_max_binary_utf8!(array, |a, b| a < b)
}

/// Returns the minimum value in the binary array, according to the natural order.
pub fn min_binary<O: Offset>(array: &BinaryArray<O>) -> Option<&[u8]> {
    min_max_binary_utf8!(array, |a, b| a > b)
}

/// Returns the maximum value in the string array, according to the natural order.
pub fn max_string<O: Offset>(array: &Utf8Array<O>) -> Option<&str> {
    min_max_binary_utf8!(array, |a, b| a < b)
}

/// Returns the minimum value in the string array, according to the natural order.
pub fn min_string<O: Offset>(array: &Utf8Array<O>) -> Option<&str> {
    min_max_binary_utf8!(array, |a, b| a > b)
}

/// Returns the minimum value in the boolean array.
///
/// ```
/// use arrow2::{
///   array::BooleanArray,
///   compute::aggregate::min_boolean,
/// };
///
/// let a = BooleanArray::from(vec![Some(true), None, Some(false)]);
/// assert_eq!(min_boolean(&a), Some(false))
/// ```
pub fn min_boolean(array: &BooleanArray) -> Option<bool> {
    // short circuit if all nulls / zero length array
    let null_count = array.null_count();
    if null_count == array.len() {
        None
    } else if null_count == 0 {
        Some(array.values().unset_bits() == 0)
    } else {
        // Note the min bool is false (0), so short circuit as soon as we see it
        array
            .iter()
            .find(|&b| b == Some(false))
            .flatten()
            .or(Some(true))
    }
}

/// Returns the maximum value in the boolean array
///
/// ```
/// use arrow2::{
///   array::BooleanArray,
///   compute::aggregate::max_boolean,
/// };
///
/// let a = BooleanArray::from(vec![Some(true), None, Some(false)]);
/// assert_eq!(max_boolean(&a), Some(true))
/// ```
pub fn max_boolean(array: &BooleanArray) -> Option<bool> {
    // short circuit if all nulls / zero length array
    let null_count = array.null_count();
    if null_count == array.len() {
        None
    } else if null_count == 0 {
        Some(array.values().unset_bits() < array.len())
    } else {
        // Note the max bool is true (1), so short circuit as soon as we see it
        array
            .iter()
            .find(|&b| b == Some(true))
            .flatten()
            .or(Some(false))
    }
}

macro_rules! dyn_generic {
    ($array_ty:ty, $scalar_ty:ty, $array:expr, $f:ident) => {{
        let array = $array.as_any().downcast_ref::<$array_ty>().unwrap();
        Box::new(<$scalar_ty>::new($f(array)))
    }};
}

macro_rules! with_match_primitive_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use crate::datatypes::PrimitiveType::*;
    match $key_type {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        Int128 => __with_ty__! { i128 },
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        _ => return Err(Error::InvalidArgumentError(format!(
            "`min` and `max` operator do not support primitive `{:?}`",
            $key_type,
        ))),
    }
})}

/// Returns the maximum of [`Array`]. The scalar is null when all elements are null.
/// # Error
/// Errors iff the type does not support this operation.
pub fn max(array: &dyn Array) -> Result<Box<dyn Scalar>> {
    Ok(match array.data_type().to_physical_type() {
        PhysicalType::Boolean => dyn_generic!(BooleanArray, BooleanScalar, array, max_boolean),
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let data_type = array.data_type().clone();
            let array = array.as_any().downcast_ref().unwrap();
            Box::new(PrimitiveScalar::<$T>::new(data_type, max_primitive::<$T>(array)))
        }),
        PhysicalType::Utf8 => dyn_generic!(Utf8Array<i32>, Utf8Scalar<i32>, array, max_string),
        PhysicalType::LargeUtf8 => dyn_generic!(Utf8Array<i64>, Utf8Scalar<i64>, array, max_string),
        PhysicalType::Binary => {
            dyn_generic!(BinaryArray<i32>, BinaryScalar<i32>, array, max_binary)
        }
        PhysicalType::LargeBinary => {
            dyn_generic!(BinaryArray<i64>, BinaryScalar<i64>, array, min_binary)
        }
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "The `max` operator does not support type `{:?}`",
                array.data_type(),
            )))
        }
    })
}

/// Returns the minimum of [`Array`]. The scalar is null when all elements are null.
/// # Error
/// Errors iff the type does not support this operation.
pub fn min(array: &dyn Array) -> Result<Box<dyn Scalar>> {
    Ok(match array.data_type().to_physical_type() {
        PhysicalType::Boolean => dyn_generic!(BooleanArray, BooleanScalar, array, min_boolean),
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let data_type = array.data_type().clone();
            let array = array.as_any().downcast_ref().unwrap();
            Box::new(PrimitiveScalar::<$T>::new(data_type, min_primitive::<$T>(array)))
        }),
        PhysicalType::Utf8 => dyn_generic!(Utf8Array<i32>, Utf8Scalar<i32>, array, min_string),
        PhysicalType::LargeUtf8 => dyn_generic!(Utf8Array<i64>, Utf8Scalar<i64>, array, min_string),
        PhysicalType::Binary => {
            dyn_generic!(BinaryArray<i32>, BinaryScalar<i32>, array, min_binary)
        }
        PhysicalType::LargeBinary => {
            dyn_generic!(BinaryArray<i64>, BinaryScalar<i64>, array, min_binary)
        }
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "The `max` operator does not support type `{:?}`",
                array.data_type(),
            )))
        }
    })
}

/// Whether [`min`] supports `data_type`
pub fn can_min(data_type: &DataType) -> bool {
    let physical = data_type.to_physical_type();
    if let PhysicalType::Primitive(primitive) = physical {
        use PrimitiveType::*;
        matches!(
            primitive,
            Int8 | Int16 | Int64 | Int128 | UInt8 | UInt16 | UInt32 | UInt64 | Float32 | Float64
        )
    } else {
        use PhysicalType::*;
        matches!(physical, Boolean | Utf8 | LargeUtf8 | Binary | LargeBinary)
    }
}

/// Whether [`max`] supports `data_type`
pub fn can_max(data_type: &DataType) -> bool {
    can_min(data_type)
}
