use std::ops::{Add, AddAssign, Sub, SubAssign};

use bytemuck::Zeroable;
use num_traits::{NumCast, ToPrimitive};
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
#[inline]
pub fn int_mean(sum: i128, count: usize, scale: i64, rounding: IntMeanRounding) -> i64 {
    let num = sum * scale as i128;
    let count = count as i128;
    let mean = match rounding {
        IntMeanRounding::Floor => num.div_euclid(count),
        IntMeanRounding::Trunc => num / count,
    };
    mean as i64
}

/// Wrapping 256-bit accumulator for exactly summing 128-bit integers.
///
/// Unlike the Decimal256 storage type `i256`, it has the arithmetic that sums and sliding windows
/// need.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I256Acc(pub ethnum::I256);

// SAFETY: all-zero bits are the value 0.
unsafe impl Zeroable for I256Acc {}

impl Add for I256Acc {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for I256Acc {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl AddAssign for I256Acc {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for I256Acc {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl WrappingAdd for I256Acc {
    #[inline]
    fn wrapping_add(&self, v: &Self) -> Self {
        *self + *v
    }
}

impl From<i128> for I256Acc {
    #[inline]
    fn from(v: i128) -> Self {
        Self(v.into())
    }
}

impl From<u128> for I256Acc {
    #[inline]
    fn from(v: u128) -> Self {
        Self(v.into())
    }
}

impl ToPrimitive for I256Acc {
    fn to_i64(&self) -> Option<i64> {
        self.0.try_into().ok()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.try_into().ok()
    }

    fn to_i128(&self) -> Option<i128> {
        self.0.try_into().ok()
    }

    fn to_u128(&self) -> Option<u128> {
        self.0.try_into().ok()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(i256_to_f64(self.0))
    }
}

impl NumCast for I256Acc {
    #[inline]
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        let v: Option<ethnum::I256> = n.to_i128().map(Into::into);
        v.or_else(|| n.to_u128().map(Into::into)).map(Self)
    }
}

pub trait MeanAcc: Copy + Default + Send + Sync + 'static + WrappingAdd {
    fn into_f64(self) -> f64;

    /// `Some` for the exact accumulator of integers of at most 64 bits.
    #[inline]
    fn try_into_i128(self) -> Option<i128> {
        None
    }
}

/// Same result as `x as f64`, which is a slow library call on common targets.
#[inline]
fn i128_to_f64(x: i128) -> f64 {
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
    #[inline]
    fn into_f64(self) -> f64 {
        i128_to_f64(self)
    }

    #[inline]
    fn try_into_i128(self) -> Option<i128> {
        Some(self)
    }
}

/// Same result as `x as f64` would be, which `ethnum::I256::as_f64` is not: it rounds the two
/// 128-bit halves separately.
fn i256_to_f64(x: ethnum::I256) -> f64 {
    if let Ok(v) = i128::try_from(x) {
        return i128_to_f64(v);
    }
    // As in `i128_to_f64`; here `abs >= 2^127`, so `64 <= shift <= 192`.
    let abs = x.unsigned_abs();
    let shift = 192 - abs.leading_zeros();
    let sticky = (abs.trailing_zeros() < shift) as u64;
    let top = (abs >> shift).as_u64() | sticky;
    let scale = f64::from_bits(((1023 + shift) as u64) << 52);
    let out = top as f64 * scale;
    if x.is_negative() { -out } else { out }
}

impl MeanAcc for I256Acc {
    #[inline]
    fn into_f64(self) -> f64 {
        i256_to_f64(self.0)
    }
}

impl MeanAcc for f64 {
    #[inline]
    fn into_f64(self) -> f64 {
        self
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
    #[inline]
    fn mean_slice(vals: &[Self]) -> Option<f64> {
        (!vals.is_empty()).then(|| Self::sum_slice(vals).into_f64() / vals.len() as f64)
    }
}

/// An integer split into 64-bit lanes of at most 32 significant bits each, so that the lanes
/// can be summed exactly over [`SPLIT_BLOCK`] values, which vectorizes.
trait SplitInt: Copy {
    type Lanes: Copy + Default;
    type Acc: MeanAcc;

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
                type Lanes = i64;
                type Acc = i128;

                #[inline(always)]
                fn add_to(self, lanes: i64, keep: u64) -> i64 {
                    lanes.wrapping_add(self as i64 & keep as i64)
                }

                #[inline(always)]
                fn combine(lanes: i64) -> i128 {
                    lanes as i128
                }
            }

            impl SplitF64 for $t {
                #[inline(always)]
                fn lanes_to_f64(lanes: i64) -> f64 {
                    lanes as f64
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
        .fold(T::Acc::default(), |a, b| a.wrapping_add(&b))
}

fn split_sum_iter<T: SplitInt>(mut vals: impl Iterator<Item = T>) -> T::Acc {
    let mut acc = T::Acc::default();
    loop {
        let mut n = 0;
        let lanes = vals
            .by_ref()
            .take(SPLIT_BLOCK)
            .fold(T::Lanes::default(), |lanes, v| {
                n += 1;
                v.add_to(lanes, u64::MAX)
            });
        acc = acc.wrapping_add(&T::combine(lanes));
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
        .fold(T::Acc::default(), |a, b| a.wrapping_add(&b))
}

macro_rules! impl_split_mean_sum {
    ($acc:ty, $to_acc:expr, $extra:tt; $($t:ty),*) => {
        $(impl_split_mean_sum!(@one $acc, $to_acc, $extra, $t);)*
    };
    (@one $acc:ty, $to_acc:expr, { $($extra:item)* }, $t:ty) => {
            impl MeanSum for $t {
                type Acc = $acc;

                #[inline]
                fn to_mean_acc(self) -> $acc {
                    $to_acc(self)
                }

                fn sum_slice(vals: &[Self]) -> $acc {
                    split_sum(vals)
                }

                fn sum_arr(arr: &PrimitiveArray<Self>) -> $acc {
                    match arr.validity().filter(|_| arr.null_count() > 0) {
                        Some(validity) => {
                            split_sum_masked(arr.values(), BitMask::from_bitmap(validity))
                        },
                        None => split_sum(arr.values()),
                    }
                }

                $($extra)*
            }
    };
}

impl_split_mean_sum!(i128, Into::into, {
    // Called once per list, so it must inline across crates.
    #[inline]
    fn mean_slice(vals: &[Self]) -> Option<f64> {
        let sum = match vals.len() {
            0 => return None,
            // Skips the setup of the vectorized loop, which dominates for a single value.
            1 => Self::lanes_to_f64(vals[0].add_to(Default::default(), u64::MAX)),
            n if n <= SPLIT_F64_LEN => split_sum_f64(vals),
            _ => i128_to_f64(split_sum(vals)),
        };
        Some(sum / vals.len() as f64)
    }
}; u8, u16, u32, u64, i8, i16, i32, i64);

impl_split_mean_sum!(I256Acc, Into::into, {
    // Gathered 128-bit values are also cheaper to sum in lanes than in 256 bits.
    fn sum_iter(vals: impl Iterator<Item = Self>) -> I256Acc {
        split_sum_iter(vals)
    }
}; u128, i128);

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
