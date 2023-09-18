use std::ops::Add;

use multiversion::multiversion;

use crate::bitmap::utils::{BitChunkIterExact, BitChunksExact};
use crate::datatypes::{DataType, PhysicalType, PrimitiveType};
use crate::error::{Error, Result};
use crate::scalar::*;
use crate::types::simd::*;
use crate::types::NativeType;
use crate::{
    array::{Array, PrimitiveArray},
    bitmap::Bitmap,
};

/// Object that can reduce itself to a number. This is used in the context of SIMD to reduce
/// a MD (e.g. `[f32; 16]`) into a single number (`f32`).
pub trait Sum<T> {
    /// Reduces this element to a single value.
    fn simd_sum(self) -> T;
}

#[multiversion(targets = "simd")]
/// Compute the sum of a slice
pub fn sum_slice<T>(values: &[T]) -> T
where
    T: NativeType + Simd + Add<Output = T> + std::iter::Sum<T>,
    T::Simd: Sum<T> + Add<Output = T::Simd>,
{
    let (head, simd_vals, tail) = T::Simd::align(values);

    let mut reduced = T::Simd::from_incomplete_chunk(&[], T::default());
    for chunk in simd_vals {
        reduced = reduced + *chunk;
    }

    reduced.simd_sum() + head.iter().copied().sum() + tail.iter().copied().sum()
}

/// # Panics
/// iff `values.len() != bitmap.len()` or the operation overflows.
#[multiversion(targets = "simd")]
fn null_sum_impl<T, I>(values: &[T], mut validity_masks: I) -> T
where
    T: NativeType + Simd,
    T::Simd: Add<Output = T::Simd> + Sum<T>,
    I: BitChunkIterExact<<<T as Simd>::Simd as NativeSimd>::Chunk>,
{
    let mut chunks = values.chunks_exact(T::Simd::LANES);

    let sum = chunks.by_ref().zip(validity_masks.by_ref()).fold(
        T::Simd::default(),
        |acc, (chunk, validity_chunk)| {
            let chunk = T::Simd::from_chunk(chunk);
            let mask = <T::Simd as NativeSimd>::Mask::from_chunk(validity_chunk);
            let selected = chunk.select(mask, T::Simd::default());
            acc + selected
        },
    );

    let remainder = T::Simd::from_incomplete_chunk(chunks.remainder(), T::default());
    let mask = <T::Simd as NativeSimd>::Mask::from_chunk(validity_masks.remainder());
    let remainder = remainder.select(mask, T::Simd::default());
    let reduced = sum + remainder;

    reduced.simd_sum()
}

/// # Panics
/// iff `values.len() != bitmap.len()` or the operation overflows.
fn null_sum<T>(values: &[T], bitmap: &Bitmap) -> T
where
    T: NativeType + Simd,
    T::Simd: Add<Output = T::Simd> + Sum<T>,
{
    let (slice, offset, length) = bitmap.as_slice();
    if offset == 0 {
        let validity_masks = BitChunksExact::<<T::Simd as NativeSimd>::Chunk>::new(slice, length);
        null_sum_impl(values, validity_masks)
    } else {
        let validity_masks = bitmap.chunks::<<T::Simd as NativeSimd>::Chunk>();
        null_sum_impl(values, validity_masks)
    }
}

/// Returns the sum of values in the array.
///
/// Returns `None` if the array is empty or only contains null values.
pub fn sum_primitive<T>(array: &PrimitiveArray<T>) -> Option<T>
where
    T: NativeType + Simd + Add<Output = T> + std::iter::Sum<T>,
    T::Simd: Add<Output = T::Simd> + Sum<T>,
{
    let null_count = array.null_count();

    if null_count == array.len() {
        return None;
    }

    match array.validity() {
        None => Some(sum_slice(array.values())),
        Some(bitmap) => Some(null_sum(array.values(), bitmap)),
    }
}

/// Whether [`sum`] supports `data_type`
pub fn can_sum(data_type: &DataType) -> bool {
    if let PhysicalType::Primitive(primitive) = data_type.to_physical_type() {
        use PrimitiveType::*;
        matches!(
            primitive,
            Int8 | Int16 | Int64 | Int128 | UInt8 | UInt16 | UInt32 | UInt64 | Float32 | Float64
        )
    } else {
        false
    }
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
            "`sum` operator do not support primitive `{:?}`",
            $key_type,
        ))),
    }
})}

/// Returns the sum of all elements in `array` as a [`Scalar`] of the same physical
/// and logical types as `array`.
/// # Error
/// Errors iff the operation is not supported.
pub fn sum(array: &dyn Array) -> Result<Box<dyn Scalar>> {
    Ok(match array.data_type().to_physical_type() {
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let data_type = array.data_type().clone();
            let array = array.as_any().downcast_ref().unwrap();
            Box::new(PrimitiveScalar::new(data_type, sum_primitive::<$T>(array)))
        }),
        _ => {
            return Err(Error::InvalidArgumentError(format!(
                "The `sum` operator does not support type `{:?}`",
                array.data_type(),
            )))
        }
    })
}
