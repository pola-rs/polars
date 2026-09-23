use std::ops::Add;

use num_traits::Zero;
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::bitmap::bitmask::BitMask;
use polars_arrow::types::NativeType;
use polars_utils::float16::pf16;

use crate::float_sum::{FloatSum, sum_arr_as_f64};
use crate::sum::{WrappingAdd, wrapping_sum_arr_upcast};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntMeanRounding {
    Floor,
    Trunc,
}

/// Exact mean of `count > 0` integers summing to `sum`, multiplied by `scale`.
///
/// With `scale == 1` the result lies between the minimum and maximum of the inputs.
pub fn int_mean(sum: i128, count: usize, scale: i64, rounding: IntMeanRounding) -> i64 {
    let num = sum * scale as i128;
    let count = count as i128;
    let mean = match rounding {
        IntMeanRounding::Floor => num.div_euclid(count),
        IntMeanRounding::Trunc => num / count,
    };
    mean as i64
}

/// 256-bit accumulator for exactly summing 128-bit integers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I256Acc(pub ethnum::I256);

impl Add for I256Acc {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Zero for I256Acc {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.0 == ethnum::I256::ZERO
    }
}

impl WrappingAdd for I256Acc {
    fn wrapping_add(&self, v: &Self) -> Self {
        *self + *v
    }
}

impl From<i128> for I256Acc {
    fn from(v: i128) -> Self {
        Self(v.into())
    }
}

impl From<u128> for I256Acc {
    fn from(v: u128) -> Self {
        Self(v.into())
    }
}

pub trait MeanAcc: Copy + Default + Send + Sync + 'static + WrappingAdd {
    fn into_f64(self) -> f64;
    /// Sum of `count` copies of `self`.
    fn mul_count(self, count: usize) -> Self;
    /// `None` if the accumulator is a float or does not fit in an `i128`.
    fn try_into_i128(self) -> Option<i128>;
}

/// Same result as `x as f64`, which is a slow library call on common targets.
#[inline]
pub fn i128_to_f64(x: i128) -> f64 {
    if let Ok(v) = i64::try_from(x) {
        return v as f64;
    }
    let abs = x.unsigned_abs();
    // Keep the top 64 bits and fold the discarded bits into the lowest one. That bit lies
    // below the rounding position of an f64, so round-to-nearest-even is unaffected.
    let shift = 64 - abs.leading_zeros();
    let sticky = (abs & ((1u128 << shift) - 1) != 0) as u64;
    let top = ((abs >> shift) as u64) | sticky;
    let scale = f64::from_bits(((1023 + shift) as u64) << 52);
    let out = top as f64 * scale;
    if x < 0 { -out } else { out }
}

impl MeanAcc for i128 {
    fn into_f64(self) -> f64 {
        i128_to_f64(self)
    }

    fn mul_count(self, count: usize) -> Self {
        self.wrapping_mul(count as i128)
    }

    fn try_into_i128(self) -> Option<i128> {
        Some(self)
    }
}

impl MeanAcc for I256Acc {
    fn into_f64(self) -> f64 {
        self.0.as_f64()
    }

    fn mul_count(self, count: usize) -> Self {
        Self(self.0.wrapping_mul((count as u128).into()))
    }

    fn try_into_i128(self) -> Option<i128> {
        self.0.try_into().ok()
    }
}

impl MeanAcc for f64 {
    fn into_f64(self) -> f64 {
        self
    }

    fn mul_count(self, count: usize) -> Self {
        self * count as f64
    }

    fn try_into_i128(self) -> Option<i128> {
        None
    }
}

/// Sums values for a mean: exactly for integers, with the float kernels for floats.
pub trait MeanSum: NativeType {
    type Acc: MeanAcc;

    fn to_mean_acc(self) -> Self::Acc;

    fn sum_slice(vals: &[Self]) -> Self::Acc;
    fn sum_arr(arr: &PrimitiveArray<Self>) -> Self::Acc;
}

/// An integer of at most 64 bits, split as `hi * 2^SHIFT + lo` so that `hi` and `lo` can be
/// summed exactly in 64-bit lanes over [`SPLIT_BLOCK`] values, which vectorizes.
trait SplitInt: Copy {
    const SHIFT: u32;
    fn split(self) -> (i64, u64);
}

macro_rules! impl_split_int {
    (narrow; $($t:ty),*) => {
        $(
            impl SplitInt for $t {
                const SHIFT: u32 = 0;

                #[inline(always)]
                fn split(self) -> (i64, u64) {
                    (self as i64, 0)
                }
            }
        )*
    };
}

impl_split_int!(narrow; u8, u16, u32, i8, i16, i32);

impl SplitInt for i64 {
    const SHIFT: u32 = 32;

    #[inline(always)]
    fn split(self) -> (i64, u64) {
        (self >> 32, (self as u64) & 0xFFFF_FFFF)
    }
}

impl SplitInt for u64 {
    const SHIFT: u32 = 32;

    #[inline(always)]
    fn split(self) -> (i64, u64) {
        ((self >> 32) as i64, self & 0xFFFF_FFFF)
    }
}

/// Both halves are below 2^32 in magnitude, so their block sums stay below 2^56.
const SPLIT_BLOCK: usize = 1 << 24;

#[inline(always)]
fn combine_split<T: SplitInt>(hi: i64, lo: u64) -> i128 {
    ((hi as i128) << T::SHIFT) + lo as i128
}

#[inline(always)]
fn split_sum_block<T: SplitInt>(block: &[T]) -> i128 {
    let (hi, lo) = block.iter().fold((0i64, 0u64), |(hi, lo), v| {
        let (h, l) = v.split();
        (hi.wrapping_add(h), lo.wrapping_add(l))
    });
    combine_split::<T>(hi, lo)
}

fn split_sum<T: SplitInt>(vals: &[T]) -> i128 {
    if vals.len() <= SPLIT_BLOCK {
        return split_sum_block(vals);
    }
    vals.chunks(SPLIT_BLOCK).map(split_sum_block).sum()
}

fn split_sum_masked<T: SplitInt>(vals: &[T], mask: BitMask<'_>) -> i128 {
    assert!(vals.len() == mask.len());
    vals.chunks(SPLIT_BLOCK)
        .enumerate()
        .map(|(b, block)| {
            let offset = b * SPLIT_BLOCK;
            let (mut hi, mut lo) = (0i64, 0u64);
            let (words, rest) = block.as_chunks::<32>();
            for (i, word) in words.iter().enumerate() {
                let bits = mask.get_u32(offset + i * 32);
                for (j, v) in word.iter().enumerate() {
                    // All ones if valid, zero otherwise, to stay branch-free.
                    let keep = (((bits >> j) & 1) as i64).wrapping_neg();
                    let (h, l) = v.split();
                    hi = hi.wrapping_add(h & keep);
                    lo = lo.wrapping_add(l & keep as u64);
                }
            }
            let rest_offset = offset + words.len() * 32;
            for (j, v) in rest.iter().enumerate() {
                if mask.get(rest_offset + j) {
                    let (h, l) = v.split();
                    hi = hi.wrapping_add(h);
                    lo = lo.wrapping_add(l);
                }
            }
            combine_split::<T>(hi, lo)
        })
        .sum()
}

macro_rules! impl_split_mean_sum {
    ($($t:ty),*) => {
        $(
            impl MeanSum for $t {
                type Acc = i128;

                fn to_mean_acc(self) -> i128 {
                    self.into()
                }

                fn sum_slice(vals: &[Self]) -> i128 {
                    split_sum(vals)
                }

                fn sum_arr(arr: &PrimitiveArray<Self>) -> i128 {
                    match arr.validity().filter(|_| arr.null_count() > 0) {
                        Some(validity) => {
                            split_sum_masked(arr.values(), BitMask::from_bitmap(validity))
                        },
                        None => split_sum(arr.values()),
                    }
                }
            }
        )*
    };
}

impl_split_mean_sum!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! impl_wide_mean_sum {
    ($($t:ty),*) => {
        $(
            impl MeanSum for $t {
                type Acc = I256Acc;

                fn to_mean_acc(self) -> I256Acc {
                    self.into()
                }

                fn sum_slice(vals: &[Self]) -> I256Acc {
                    vals.iter()
                        .fold(I256Acc::zero(), |a, b| a.wrapping_add(&(*b).into()))
                }

                fn sum_arr(arr: &PrimitiveArray<Self>) -> I256Acc {
                    wrapping_sum_arr_upcast::<Self, I256Acc>(arr)
                }
            }
        )*
    };
}

impl_wide_mean_sum!(i128, u128);

macro_rules! impl_float_mean_sum {
    ($($t:ty),*) => {
        $(
            impl MeanSum for $t {
                type Acc = f64;

                fn to_mean_acc(self) -> f64 {
                    self.into()
                }

                fn sum_slice(vals: &[Self]) -> f64 {
                    FloatSum::sum(vals)
                }

                fn sum_arr(arr: &PrimitiveArray<Self>) -> f64 {
                    sum_arr_as_f64(arr)
                }
            }
        )*
    };
}

impl_float_mean_sum!(pf16, f32, f64);
