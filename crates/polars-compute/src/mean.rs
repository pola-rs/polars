use std::ops::Add;

use num_traits::Zero;
use polars_arrow::array::PrimitiveArray;
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

impl MeanAcc for i128 {
    fn into_f64(self) -> f64 {
        self as f64
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

macro_rules! impl_int_mean_sum {
    ($acc:ty; $($t:ty),*) => {
        $(
            impl MeanSum for $t {
                type Acc = $acc;

                fn to_mean_acc(self) -> $acc {
                    self.into()
                }

                fn sum_slice(vals: &[Self]) -> Self::Acc {
                    vals.iter()
                        .fold(<$acc>::zero(), |a, b| WrappingAdd::wrapping_add(&a, &(*b).into()))
                }

                fn sum_arr(arr: &PrimitiveArray<Self>) -> Self::Acc {
                    wrapping_sum_arr_upcast::<Self, $acc>(arr)
                }
            }
        )*
    };
}

impl_int_mean_sum!(i128; u8, u16, u32, u64, i8, i16, i32, i64);
impl_int_mean_sum!(I256Acc; i128, u128);

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
