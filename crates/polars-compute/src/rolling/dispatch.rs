//! The rolling kernels as a column reaches them: in whatever representation it holds its chunk.

use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use num_traits::{Bounded, Float, Num, NumCast, One, Zero};
use polars_array::concatenate::concatenate;
use polars_array::{PlArray, PlBitmap, PlPrimitiveArray, StaticArray};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::legacy::error::PolarsResult;
use polars_arrow::types::NativeType;
use polars_utils::float::IsFloat;

use super::{RollingFnParams, RollingRankMethod, no_nulls, nulls, quantile_filter};

/// The answer a kernel gives over a chunk that repeats one value, out of a run over one window.
#[inline(never)]
fn repeated_chunk_answer<T, F>(
    arr: &PlPrimitiveArray<T>,
    window_size: usize,
    center: bool,
    run: F,
) -> Option<PolarsResult<Box<dyn PlArray>>>
where
    T: NativeType,
    F: FnOnce(&PlPrimitiveArray<T>) -> PolarsResult<Box<dyn PlArray>>,
{
    if center || window_size == 0 {
        return None;
    }
    let len = arr.len();
    if len <= window_size {
        return None;
    }
    let head = if let Some(no_nulls) = arr.as_no_nulls() {
        PlPrimitiveArray::new_scalar(no_nulls.scalar_value_ignore_validity()?, window_size)
    } else if arr.null_count() == len {
        PlPrimitiveArray::new_full_null(window_size)
    } else {
        return None;
    };

    let head = match run(&head) {
        Ok(head) => head,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(spread_over(&*head, len)))
}

/// The answer over `len` elements that a `head` over one window's worth of them stands for.
fn spread_over(head: &dyn PlArray, len: usize) -> Box<dyn PlArray> {
    let full = head.len() - 1;
    let partial = head.sliced(0, full);

    if partial.null_count() == full {
        if head.is_null(full) {
            return head.new_full_null_like_self(len);
        }
        let answer = head.new_from_index(full, len);
        if full == 0 {
            return answer;
        }
        let mut mask = BitmapBuilder::with_capacity(len);
        mask.extend_constant(full, false);
        mask.extend_constant(len - full, true);
        return answer.with_validity(Some(PlBitmap::from_bitmap(mask.freeze())));
    }

    let tail = head.new_from_index(full, len - full);
    concatenate(&[&*partial, &*tail])
        .unwrap_or_else(|_| unreachable!("a kernel's answer concatenated with itself"))
}

/// Defines an entry point that resolves the representation and forks to the two implementations.
macro_rules! rolling_dispatch {
    (
        $(#[$meta:meta])*
        $name:ident($($arg:ident: $ty:ty),* $(,)?)
        steady_when: $steady:expr,
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
            if $steady {
                let repeated = repeated_chunk_answer(arr, window_size, center, |head| {
                    walk(head, window_size, min_periods, center, $($arg,)*)
                });
                if let Some(answer) = repeated {
                    return answer;
                }
            }

            return walk(arr, window_size, min_periods, center, $($arg,)*);

            fn walk<T>(
                arr: &PlPrimitiveArray<T>,
                window_size: usize,
                min_periods: usize,
                center: bool,
                $($arg: $ty,)*
            ) -> PolarsResult<Box<dyn PlArray>>
            where
                T: $($bound)*,
            {
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
        }
    };
}

rolling_dispatch!(
    /// The sum of each window of `arr`.
    rolling_sum(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    steady_when: !T::is_float(),
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
    steady_when: false,
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
    steady_when: true,

    where T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T> + Num
);

rolling_dispatch!(
    /// The largest element of each window of `arr`.
    rolling_max(weights: Option<&[f64]>, params: Option<RollingFnParams>)
    steady_when: true,

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
    steady_when: true,

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
    steady_when: true,

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
    steady_when: !matches!(
        params,
        Some(RollingFnParams::Rank { method: RollingRankMethod::Random, .. })
    ),

    where T: NativeType + Num
);

rolling_dispatch!(
    /// The skew of each window of `arr`, per `params`.
    rolling_skew(params: Option<RollingFnParams>)
    steady_when: true,

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
    steady_when: true,

    where T: NativeType
        + Float
        + IsFloat
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + AddAssign
);
