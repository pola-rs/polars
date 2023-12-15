use std::simd::{
    LaneCount, Mask, Simd, SimdCast, SimdElement, SimdFloat, SimdInt, SimdUint, StdFloat,
    SupportedLaneCount, ToBitMask,
};

use multiversion::multiversion;
use num_traits::ToPrimitive;
use polars_utils::float::IsFloat;

use crate::array::{Array, PrimitiveArray};
use crate::bitmap::utils::{BitChunkIterExact, BitChunksExact};
use crate::bitmap::Bitmap;
use crate::datatypes::PhysicalType::Primitive;
use crate::types::NativeType;
use crate::with_match_primitive_type;

// TODO! try to remove this if we can cast again directly
pub trait SimdCastPl<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn cast_custom<U: SimdCast>(self) -> Simd<U, N>;
}

macro_rules! impl_cast_custom {
    ($_type:ty) => {
        impl<const N: usize> SimdCastPl<N> for Simd<$_type, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            fn cast_custom<U: SimdCast>(self) -> Simd<U, N> {
                self.cast::<U>()
            }
        }
    };
}

impl_cast_custom!(u8);
impl_cast_custom!(u16);
impl_cast_custom!(u32);
impl_cast_custom!(u64);
impl_cast_custom!(i8);
impl_cast_custom!(i16);
impl_cast_custom!(i32);
impl_cast_custom!(i64);
impl_cast_custom!(f32);
impl_cast_custom!(f64);

#[multiversion(targets = "simd")]
fn nonnull_sum_as_f64<T>(values: &[T]) -> f64
where
    T: NativeType + SimdElement + ToPrimitive + SimdCast,
    Simd<T, 8>: SimdCastPl<8>,
{
    // we choose 8 as that the maximum size of f64x8 -> 512bit wide
    const LANES: usize = 8;
    let (head, simd_vals, tail) = unsafe { values.align_to::<Simd<T, LANES>>() };

    let mut reduced: Simd<f64, LANES> = Simd::splat(0.0);
    for chunk in simd_vals {
        reduced += chunk.cast_custom::<f64>();
    }

    unsafe {
        reduced.reduce_sum()
            + head
                .iter()
                .map(|v| v.to_f64().unwrap_unchecked())
                .sum::<f64>()
            + tail
                .iter()
                .map(|v| v.to_f64().unwrap_unchecked())
                .sum::<f64>()
    }
}

#[multiversion(targets = "simd")]
fn null_sum_as_f64_impl<T, I>(values: &[T], mut validity_masks: I) -> f64
where
    T: NativeType + SimdElement + ToPrimitive + IsFloat + SimdCast,
    I: BitChunkIterExact<u8>,
    Simd<T, 8>: SimdCastPl<8>,
{
    const LANES: usize = 8;
    let mut chunks = values.chunks_exact(LANES);
    let min_one = Simd::<f64, LANES>::splat(-1.0);
    let min_one_i64 = Simd::<i64, LANES>::splat(-1);

    let sum = chunks.by_ref().zip(validity_masks.by_ref()).fold(
        Simd::<f64, LANES>::splat(0.0),
        |acc, (chunk, validity_chunk)| {
            // safety: exact size chunks
            let chunk: [T; LANES] = unsafe { chunk.try_into().unwrap_unchecked() };
            let chunk = Simd::from(chunk).cast_custom::<f64>();

            // construct [bools]
            let mask = Mask::<i8, LANES>::from_bitmask(validity_chunk);
            // let's say we have mask
            //      [true, false, true, true]
            // a cast to int gives:
            //      [-1, 0, -1, -1]
            // multiply by -1
            //     [1, 0, 1, 1]
            // and then use that as mask to multiply
            // the chunk.

            if T::is_float() {
                // there can be NaNs masked out by validity
                // so our strategy if multiplying by zero doesn't work on floats
                // we transmute to i64 and multiply by 1s and 0s. This works as the multiply
                // by 1 doesn't change the bits and the multiply by 0 gives 0 which has the same
                // bit repr in f64 and i64
                unsafe {
                    let chunk_i64 =
                        std::mem::transmute::<Simd<f64, LANES>, Simd<i64, LANES>>(chunk);
                    let mask_mul = mask.to_int().cast::<i64>() * min_one_i64;

                    // mask out and transmute back
                    std::mem::transmute::<_, Simd<f64, LANES>>(chunk_i64 * mask_mul) + acc
                }
            } else {
                // cast true to -1 and false to 0 so we multiply with -1 to get a branchless mask
                let mask_mul = mask.to_int().cast::<f64>() * min_one;
                // eg. null values are multiplied with 0
                // and valid value with 1
                chunk.mul_add(mask_mul, acc)
            }
        },
    );
    let mut sum = sum.reduce_sum();

    for (v, valid) in chunks
        .remainder()
        .iter()
        .zip(validity_masks.remainder_iter())
    {
        unsafe {
            sum += (valid as u8 as f64) * v.to_f64().unwrap_unchecked();
        }
    }
    sum
}

fn null_sum_as_f64<T>(values: &[T], bitmap: &Bitmap) -> f64
where
    T: NativeType + SimdElement + ToPrimitive + IsFloat + SimdCast,
    Simd<T, 8>: SimdCastPl<8>,
{
    let (slice, offset, length) = bitmap.as_slice();
    if offset == 0 {
        let validity_masks = BitChunksExact::<u8>::new(slice, length);
        null_sum_as_f64_impl(values, validity_masks)
    } else {
        let validity_masks = bitmap.chunks::<u8>();
        null_sum_as_f64_impl(values, validity_masks)
    }
}

pub fn sum_as_f64(values: &dyn Array) -> f64 {
    if let Primitive(primitive) = values.data_type().to_physical_type() {
        with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = values.as_any().downcast_ref().unwrap();
            if arr.null_count() == 0 {
                nonnull_sum_as_f64(arr.values())
            } else {
                let validity = arr.validity().unwrap();
                null_sum_as_f64(arr.values().as_slice(), validity)
            }
        })
    } else {
        unreachable!()
    }
}
