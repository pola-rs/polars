use std::ops::Add;

use num_traits::Zero;
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::bitmap::bitmask::BitMask;
use polars_arrow::types::NativeType;
use polars_utils::float16::pf16;

use crate::float_sum::{FloatSum, sum_arr_as_f64};
use crate::sum::WrappingAdd;

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
    fn sum_iter(vals: impl Iterator<Item = Self>) -> Self::Acc {
        vals.fold(Self::Acc::default(), |a, v| {
            a.wrapping_add(&v.to_mean_acc())
        })
    }
    fn sum_arr(arr: &PrimitiveArray<Self>) -> Self::Acc;

    /// `sum_slice(vals).into_f64() / len`, `None` if `vals` is empty.
    fn mean_slice(vals: &[Self]) -> Option<f64> {
        (!vals.is_empty()).then(|| Self::sum_slice(vals).into_f64() / vals.len() as f64)
    }
}

/// An integer split into 64-bit lanes of at most 32 significant bits each, so that the lanes
/// can be summed exactly over [`SPLIT_BLOCK`] values, which vectorizes.
trait SplitInt: Copy {
    type Lanes: Copy + Default;
    type Acc: MeanAcc + Add<Output = Self::Acc> + Zero;

    /// Adds `self` to `lanes` if `keep` is all ones, nothing if it is zero.
    fn add_to(self, lanes: Self::Lanes, keep: u64) -> Self::Lanes;
    fn combine(lanes: Self::Lanes) -> Self::Acc;
}

/// Up to this many values, both lanes of an at most 64-bit split sum are exact in an `f64`.
const SPLIT_F64_LEN: usize = 1 << 21;

trait SplitF64: SplitInt {
    /// `combine(lanes) as f64` for the lanes of at most [`SPLIT_F64_LEN`] values.
    fn lanes_to_f64(lanes: Self::Lanes) -> f64;
}

macro_rules! impl_split_int {
    (narrow; $($t:ty),*) => {
        $(
            impl SplitInt for $t {
                type Lanes = (i64, u64);
                type Acc = i128;

                #[inline(always)]
                fn add_to(self, (hi, lo): (i64, u64), keep: u64) -> (i64, u64) {
                    (hi.wrapping_add(self as i64 & keep as i64), lo)
                }

                #[inline(always)]
                fn combine((hi, _): (i64, u64)) -> i128 {
                    hi as i128
                }
            }

            impl SplitF64 for $t {
                #[inline(always)]
                fn lanes_to_f64((hi, _): (i64, u64)) -> f64 {
                    hi as f64
                }
            }
        )*
    };
    (wide; $($t:ty),*) => {
        $(
            impl SplitInt for $t {
                /// `hi * 2^32 + lo`.
                type Lanes = (i64, u64);
                type Acc = i128;

                #[inline(always)]
                fn add_to(self, (hi, lo): (i64, u64), keep: u64) -> (i64, u64) {
                    let h = (self >> 32) as i64;
                    let l = self as u64 & 0xFFFF_FFFF;
                    (hi.wrapping_add(h & keep as i64), lo.wrapping_add(l & keep))
                }

                #[inline(always)]
                fn combine((hi, lo): (i64, u64)) -> i128 {
                    ((hi as i128) << 32) + lo as i128
                }
            }

            impl SplitF64 for $t {
                /// Scaling by a power of two is exact, so the only rounding is in the addition.
                #[inline(always)]
                fn lanes_to_f64((hi, lo): (i64, u64)) -> f64 {
                    hi as f64 * 4294967296.0 + lo as f64
                }
            }
        )*
    };
    (x128; $($t:ty),*) => {
        $(
            impl SplitInt for $t {
                /// Limbs of 32 bits, most significant first.
                type Lanes = (i64, u64, u64, u64);
                type Acc = I256Acc;

                #[inline(always)]
                fn add_to(self, (a, b, c, d): Self::Lanes, keep: u64) -> Self::Lanes {
                    let w = (self >> 96) as i64;
                    let x = (self >> 64) as u64 & 0xFFFF_FFFF;
                    let y = (self >> 32) as u64 & 0xFFFF_FFFF;
                    let z = self as u64 & 0xFFFF_FFFF;
                    (
                        a.wrapping_add(w & keep as i64),
                        b.wrapping_add(x & keep),
                        c.wrapping_add(y & keep),
                        d.wrapping_add(z & keep),
                    )
                }

                #[inline(always)]
                fn combine((a, b, c, d): Self::Lanes) -> I256Acc {
                    let hi = ethnum::I256::from(((a as i128) << 32) + b as i128);
                    let lo = ethnum::I256::from(((c as i128) << 32) + d as i128);
                    I256Acc((hi << 64) + lo)
                }
            }
        )*
    };
}

impl_split_int!(narrow; u8, u16, u32, i8, i16, i32);
impl_split_int!(wide; u64, i64);
impl_split_int!(x128; u128, i128);

/// Every lane is below 2^32 in magnitude, so its block sum stays below 2^56.
const SPLIT_BLOCK: usize = 1 << 24;

#[inline(always)]
fn split_lanes<T: SplitInt>(vals: impl Iterator<Item = T>) -> T::Lanes {
    vals.fold(T::Lanes::default(), |lanes, v| v.add_to(lanes, u64::MAX))
}

fn split_sum<T: SplitInt>(vals: &[T]) -> T::Acc {
    if vals.len() <= SPLIT_BLOCK {
        return T::combine(split_lanes(vals.iter().copied()));
    }
    vals.chunks(SPLIT_BLOCK)
        .map(|block| T::combine(split_lanes(block.iter().copied())))
        .fold(T::Acc::zero(), Add::add)
}

fn split_sum_iter<T: SplitInt>(mut vals: impl Iterator<Item = T>) -> T::Acc {
    let mut acc = T::Acc::zero();
    loop {
        let mut n = 0;
        let lanes = vals
            .by_ref()
            .take(SPLIT_BLOCK)
            .fold(T::Lanes::default(), |lanes, v| {
                n += 1;
                v.add_to(lanes, u64::MAX)
            });
        acc = acc + T::combine(lanes);
        if n < SPLIT_BLOCK {
            return acc;
        }
    }
}

fn split_sum_f64<T: SplitF64>(vals: &[T]) -> f64 {
    debug_assert!(vals.len() <= SPLIT_F64_LEN);
    T::lanes_to_f64(split_lanes(vals.iter().copied()))
}

fn split_sum_masked<T: SplitInt>(vals: &[T], mask: BitMask<'_>) -> T::Acc {
    assert!(vals.len() == mask.len());
    vals.chunks(SPLIT_BLOCK)
        .enumerate()
        .map(|(b, block)| {
            let offset = b * SPLIT_BLOCK;
            let mut lanes = T::Lanes::default();
            let (words, rest) = block.as_chunks::<32>();
            for (i, word) in words.iter().enumerate() {
                let bits = mask.get_u32(offset + i * 32);
                for (j, v) in word.iter().enumerate() {
                    // All ones if valid, zero otherwise, to stay branch-free.
                    let keep = ((bits >> j) & 1) as u64;
                    lanes = v.add_to(lanes, keep.wrapping_neg());
                }
            }
            let rest_offset = offset + words.len() * 32;
            for (j, v) in rest.iter().enumerate() {
                let keep = mask.get(rest_offset + j) as u64;
                lanes = v.add_to(lanes, keep.wrapping_neg());
            }
            T::combine(lanes)
        })
        .fold(T::Acc::zero(), Add::add)
}

macro_rules! impl_split_mean_sum {
    ($acc:ty; $($t:tt),*) => {
        $(
            impl MeanSum for $t {
                type Acc = $acc;

                fn to_mean_acc(self) -> Self::Acc {
                    self.into()
                }

                fn sum_slice(vals: &[Self]) -> Self::Acc {
                    split_sum(vals)
                }

                fn sum_arr(arr: &PrimitiveArray<Self>) -> Self::Acc {
                    match arr.validity().filter(|_| arr.null_count() > 0) {
                        Some(validity) => {
                            split_sum_masked(arr.values(), BitMask::from_bitmap(validity))
                        },
                        None => split_sum(arr.values()),
                    }
                }

                impl_split_mean_sum!(@extra $t);
            }
        )*
    };
    (@extra u128) => { impl_split_mean_sum!(@sum_iter); };
    (@extra i128) => { impl_split_mean_sum!(@sum_iter); };
    // Gathered 128-bit values are also cheaper to sum in lanes than in 256 bits.
    (@sum_iter) => {
        fn sum_iter(vals: impl Iterator<Item = Self>) -> Self::Acc {
            split_sum_iter(vals)
        }
    };
    (@extra $t:tt) => {
        // Called once per list, so it must inline across crates.
        #[inline]
        fn mean_slice(vals: &[Self]) -> Option<f64> {
            let sum = match vals.len() {
                0 => return None,
                // Skips the setup of the vectorized loop, which dominates for a single value.
                1 => <$t as SplitF64>::lanes_to_f64(vals[0].add_to(Default::default(), u64::MAX)),
                n if n <= SPLIT_F64_LEN => split_sum_f64(vals),
                _ => i128_to_f64(split_sum(vals)),
            };
            Some(sum / vals.len() as f64)
        }
    };
}

impl_split_mean_sum!(i128; u8, u16, u32, u64, i8, i16, i32, i64);
impl_split_mean_sum!(I256Acc; u128, i128);

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
