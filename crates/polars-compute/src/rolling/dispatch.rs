//! The rolling kernels as a column reaches them: in whatever representation it holds its chunk.
//!
//! Each entry point here takes a `PlPrimitiveArray` as it stands and settles the one question the
//! implementations below disagree on — whether any element is null — then hands the chunk over to
//! [`no_nulls`] or [`nulls`], whose signatures say which of the two they were written for. Neither
//! side asks its caller for a representation: each resolves its own, so a chunk is never written
//! out on the way to a kernel that could have read it as it was.

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

#[cfg(test)]
mod tests {
    use polars_array::PlBitmap;

    use super::*;
    use crate::rolling::elements_of;

    /// A mask that is present but leaves no element null takes the no-nulls path, and its values
    /// buffer reaches the kernel as the very allocation it was — the mask is never written out.
    #[test]
    fn a_repeated_set_mask_neither_slows_nor_reallocates() {
        let values = vec![1.0f64, 5.0, 3.0, 4.0];
        let plain = PlPrimitiveArray::from_vec(values.clone());
        let masked =
            PlPrimitiveArray::from_vec(values).with_validity(Some(PlBitmap::new_scalar(true, 4)));

        assert!(masked.as_no_nulls().is_some());
        assert_eq!(
            masked.flat_values().unwrap().as_ptr(),
            masked.to_flat_values().as_ptr(),
            "the values were copied on the way to the kernel",
        );

        let of = |arr| elements_of::<f64>(&*rolling_max(arr, 2, 2, false, None, None).unwrap());
        assert_eq!(of(&masked), of(&plain));
        assert_eq!(of(&plain), [None, Some(5.0), Some(5.0), Some(4.0)]);
    }

    /// A chunk that repeats one value answers as the written-out chunk does, on both paths.
    #[test]
    fn a_repeated_value_rolls_like_the_written_out_chunk() {
        let repeated = PlPrimitiveArray::new_scalar(2.0f64, 5);
        let written_out = PlPrimitiveArray::from_vec(vec![2.0f64; 5]);

        let of = |arr| elements_of::<f64>(&*rolling_sum(arr, 3, 2, false, None, None).unwrap());
        assert_eq!(of(&repeated), of(&written_out));
        assert_eq!(
            of(&repeated),
            [None, Some(4.0), Some(6.0), Some(6.0), Some(6.0)]
        );

        // A repeated unset bit leaves every element null, and every window short of `min_periods`.
        let all_null = PlPrimitiveArray::new_scalar(2.0f64, 5)
            .with_validity(Some(PlBitmap::new_scalar(false, 5)));
        assert!(
            elements_of::<f64>(&*rolling_sum(&all_null, 3, 2, false, None, None).unwrap())
                .iter()
                .all(Option::is_none)
        );
    }
}
