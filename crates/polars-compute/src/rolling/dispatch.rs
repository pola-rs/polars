//! The rolling kernels as a column reaches them: in whatever representation it holds its chunk.

use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use arrow::legacy::error::PolarsResult;
use arrow::types::NativeType;
use num_traits::{Bounded, Float, Num, NumCast, One, Zero};
use polars_array::{PlArray, PlPrimitiveArray, StaticArray};
use polars_utils::float::IsFloat;

use super::{RollingFnParams, no_nulls, nulls, quantile_filter};

/// Defines an entry point that resolves the representation and forks to the two implementations.
macro_rules! rolling_dispatch {
    (
        $(#[$meta:meta])*
        $name:ident($($arg:ident: $ty:ty),* $(,)?)
        where T: $($bound:tt)*
    ) => {
        $(#[$meta])*
        pub fn $name<T>(
            arr: &PlPrimitiveArray<T>,
            window_size: usize,
            min_periods: usize,
            center: bool,
            $($arg: $ty,)*
        ) -> PolarsResult<Box<dyn PlArray>>
        where
            T: $($bound)*,
        {
            // Nothing is laid out here: whether any element is null is a count, and each of the
            // two implementations resolves the representation itself, writing out only what its
            // own reader cannot take as it stands.
            match arr.as_no_nulls() {
                Some(no_nulls) => no_nulls::$name(
                    no_nulls,
                    window_size,
                    min_periods,
                    center,
                    $($arg,)*
                ),
                None => Ok(nulls::$name(
                    arr,
                    window_size,
                    min_periods,
                    center,
                    $($arg,)*
                )),
            }
        }
    };
}

rolling_dispatch!(
    /// The sum of each window of `arr`.
    rolling_sum(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType
        + std::iter::Sum
        + NumCast
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + AddAssign
        + SubAssign
        + IsFloat
        + Num
        + PartialOrd
);

rolling_dispatch!(
    /// The mean of each window of `arr`.
    rolling_mean(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType
        + Float
        + std::iter::Sum<T>
        + SubAssign
        + AddAssign
        + IsFloat
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + NumCast
        + Div<Output = T>
);

rolling_dispatch!(
    /// The smallest element of each window of `arr`.
    rolling_min(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T> + Num
);

rolling_dispatch!(
    /// The largest element of each window of `arr`.
    rolling_max(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType
        + PartialOrd
        + IsFloat
        + Bounded
        + NumCast
        + Mul<Output = T>
        + Num
        + std::iter::Sum
        + AddAssign
);

rolling_dispatch!(
    /// The quantile of each window of `arr`, per `params`.
    rolling_quantile(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType
        + IsFloat
        + Float
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + One
        + Zero
        + quantile_filter::SealedRolling
        + PartialOrd
        + Sub<Output = T>
);

rolling_dispatch!(
    /// The variance of each window of `arr`, per `params`.
    rolling_var(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType
        + Float
        + IsFloat
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + AddAssign
);

rolling_dispatch!(
    /// The rank of the last element of each window of `arr`, per `params`.
    rolling_rank(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    where T: NativeType + Num
);

rolling_dispatch!(
    /// The skew of each window of `arr`, per `params`.
    rolling_skew(params: Option<RollingFnParams>)
    where T: NativeType
        + Float
        + IsFloat
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + AddAssign
);

rolling_dispatch!(
    /// The kurtosis of each window of `arr`, per `params`.
    rolling_kurtosis(params: Option<RollingFnParams>)
    where T: NativeType
        + Float
        + IsFloat
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + AddAssign
);
