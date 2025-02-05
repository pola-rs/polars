use std::ops::Add;
#[cfg(feature = "simd")]
use std::simd::prelude::*;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::bitmask::BitMask;
use arrow::types::NativeType;
use num_traits::Zero;

macro_rules! wrapping_impl {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline(always)]
            fn wrapping_add(&self, v: &Self) -> Self {
                <$t>::$method(*self, *v)
            }
        }
    };
}

/// Performs addition that wraps around on overflow.
///
/// Differs from num::WrappingAdd in that this is also implemented for floats.
pub trait WrappingAdd: Sized {
    /// Wrapping (modular) addition. Computes `self + other`, wrapping around at
    /// the boundary of the type.
    fn wrapping_add(&self, v: &Self) -> Self;
}

wrapping_impl!(WrappingAdd, wrapping_add, u8);
wrapping_impl!(WrappingAdd, wrapping_add, u16);
wrapping_impl!(WrappingAdd, wrapping_add, u32);
wrapping_impl!(WrappingAdd, wrapping_add, u64);
wrapping_impl!(WrappingAdd, wrapping_add, usize);
wrapping_impl!(WrappingAdd, wrapping_add, u128);

wrapping_impl!(WrappingAdd, wrapping_add, i8);
wrapping_impl!(WrappingAdd, wrapping_add, i16);
wrapping_impl!(WrappingAdd, wrapping_add, i32);
wrapping_impl!(WrappingAdd, wrapping_add, i64);
wrapping_impl!(WrappingAdd, wrapping_add, isize);
wrapping_impl!(WrappingAdd, wrapping_add, i128);

wrapping_impl!(WrappingAdd, add, f32);
wrapping_impl!(WrappingAdd, add, f64);

#[cfg(feature = "simd")]
const STRIPE: usize = 16;

fn wrapping_sum_with_mask_scalar<T: Zero + WrappingAdd + Copy>(vals: &[T], mask: &BitMask) -> T {
    assert!(vals.len() == mask.len());
    vals.iter()
        .enumerate()
        .map(|(i, x)| {
            // No filter but rather select of 0 for cmov opt.
            if mask.get(i) {
                *x
            } else {
                T::zero()
            }
        })
        .fold(T::zero(), |a, b| a.wrapping_add(&b))
}

#[cfg(not(feature = "simd"))]
impl<T> WrappingSum for T
where
    T: NativeType + WrappingAdd + Zero,
{
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter()
            .copied()
            .fold(T::zero(), |a, b| a.wrapping_add(&b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        wrapping_sum_with_mask_scalar(vals, mask)
    }
}

#[cfg(feature = "simd")]
impl<T> WrappingSum for T
where
    T: NativeType + WrappingAdd + Zero + crate::SimdPrimitive,
{
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter()
            .copied()
            .fold(T::zero(), |a, b| a.wrapping_add(&b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        assert!(vals.len() == mask.len());
        let remainder = vals.len() % STRIPE;
        let (rest, main) = vals.split_at(remainder);
        let (rest_mask, main_mask) = mask.split_at(remainder);
        let zero: Simd<T, STRIPE> = Simd::default();

        let vsum = main
            .chunks_exact(STRIPE)
            .enumerate()
            .map(|(i, a)| {
                let m: Mask<_, STRIPE> = main_mask.get_simd(i * STRIPE);
                m.select(Simd::from_slice(a), zero)
            })
            .fold(zero, |a, b| {
                let a = a.to_array();
                let b = b.to_array();
                Simd::from_array(std::array::from_fn(|i| a[i].wrapping_add(&b[i])))
            });

        let mainsum = vsum
            .to_array()
            .into_iter()
            .fold(T::zero(), |a, b| a.wrapping_add(&b));

        // TODO: faster remainder.
        let restsum = wrapping_sum_with_mask_scalar(rest, &rest_mask);
        mainsum.wrapping_add(&restsum)
    }
}

#[cfg(feature = "simd")]
impl WrappingSum for u128 {
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter().copied().fold(0, |a, b| a.wrapping_add(b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        wrapping_sum_with_mask_scalar(vals, mask)
    }
}

#[cfg(feature = "simd")]
impl WrappingSum for i128 {
    fn wrapping_sum(vals: &[Self]) -> Self {
        vals.iter().copied().fold(0, |a, b| a.wrapping_add(b))
    }

    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self {
        wrapping_sum_with_mask_scalar(vals, mask)
    }
}

pub trait WrappingSum: Sized {
    fn wrapping_sum(vals: &[Self]) -> Self;
    fn wrapping_sum_with_validity(vals: &[Self], mask: &BitMask) -> Self;
}

pub fn wrapping_sum_arr<T>(arr: &PrimitiveArray<T>) -> T
where
    T: NativeType + WrappingSum,
{
    let validity = arr.validity().filter(|_| arr.null_count() > 0);
    if let Some(mask) = validity {
        WrappingSum::wrapping_sum_with_validity(arr.values(), &BitMask::from_bitmap(mask))
    } else {
        WrappingSum::wrapping_sum(arr.values())
    }
}
