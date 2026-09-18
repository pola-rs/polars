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

/// The answer a kernel gives over a chunk that repeats one value, out of a short run of its own.
///
/// Every window of such a chunk holds the same elements as soon as it is full, so a kernel whose
/// answer for one full window stands for every later one only has to run over the first
/// `window_size` elements: what it answers at the last of them is what it answers from there to
/// the end, and what it answers before then is the partial windows' answers, which that same
/// short run has already computed.
///
/// Kept out of line: it is a whole second kernel run, reached once per chunk and never at all
/// where the elements differ, and inlining it into the entry point grows that function enough to
/// change what the optimizer does with the walk beside it -- which cost `rolling_sum` over floats
/// 2.5x on the flat path this is not even reached from.
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
    // A centered window shrinks against the end of the chunk as well as against its start, so
    // what it answers there is not what it answers in the middle.
    if center || window_size == 0 {
        return None;
    }
    // The short run would be the whole chunk; there is nothing left over for its answer to stand
    // for.
    let len = arr.len();
    if len <= window_size {
        return None;
    }
    // Only a chunk whose elements are all the same one has windows that all hold the same
    // elements: either the values repeat and nothing is null, or every element is null and what
    // the values are does not matter.
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
    // The last element of the head is the first one a full window answers for, and every element
    // after it reads the same window.
    let full = head.len() - 1;
    let partial = head.sliced(0, full);

    if partial.null_count() == full {
        // Nothing the partial windows answered is an element anyone reads, so the answer keeps
        // one value and one mask for all of `len`: whatever the full window answers, held null
        // until the first window that is full.
        if head.is_null(full) {
            return head.new_full_null_like_self(len);
        }
        let answer = head.new_from_index(full, len);
        if full == 0 {
            // A window of one is full at the first element, so no element is held null.
            return answer;
        }
        let mut mask = BitmapBuilder::with_capacity(len);
        mask.extend_constant(full, false);
        mask.extend_constant(len - full, true);
        return answer.with_validity(Some(PlBitmap::from_bitmap(mask.freeze())));
    }

    // The partial windows answered elements of their own, so those are laid out and the rest of
    // the chunk repeats the first full window's answer after them.
    let tail = head.new_from_index(full, len - full);
    concatenate(&[&*partial, &*tail])
        .unwrap_or_else(|_| unreachable!("a kernel's answer concatenated with itself"))
}

/// Defines an entry point that resolves the representation and forks to the two implementations.
macro_rules! rolling_dispatch {
    (
        $(#[$meta:meta])*
        $name:ident($($arg:ident: $ty:ty),* $(,)?)
        // Whether what this kernel answers for one full window is what it answers for every
        // later one, over a chunk that repeats a single value.
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

            // The run over every window of the chunk, which the short run over a repeating one
            // reaches as well. It is a function of its own rather than a call back into the entry
            // point above, which would make that entry point recursive for no reason.
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
                // Nothing is laid out here: whether any element is null is a count, and each of
                // the two implementations resolves the representation itself, writing out only
                // what its own reader cannot take as it stands.
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
    // A running sum takes the element leaving each window back out of it. Over floats that
    // drifts from window to window, so no one window's answer stands for the rest and the run
    // has to be walked; over integers adding and subtracting the one value the chunk repeats
    // lands back on the same sum every time, which is what makes the answer steady there.
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
    // As `rolling_sum`, which it divides.
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

#[cfg(test)]
mod test {
    use polars_utils::IdxSize;

    use super::*;
    use crate::rolling::{
        RollingQuantileParams, RollingRankMethod, RollingVarParams, elements_of, flat_chunk,
    };

    const LEN: usize = 37;
    const VALUE: f64 = 1.25;

    /// The same chunk in both representations: one value repeated, and that value written out.
    fn repeated_and_flat() -> (PlPrimitiveArray<f64>, PlPrimitiveArray<f64>) {
        (
            PlPrimitiveArray::new_scalar(VALUE, LEN),
            flat_chunk(vec![VALUE; LEN], None),
        )
    }

    /// The same all-null chunk in both representations.
    fn null_repeated_and_flat() -> (PlPrimitiveArray<f64>, PlPrimitiveArray<f64>) {
        (
            PlPrimitiveArray::new_full_null(LEN),
            flat_chunk(
                vec![VALUE; LEN],
                Some(PlBitmap::from_bitmap(
                    polars_arrow::bitmap::Bitmap::new_zeroed(LEN),
                )),
            ),
        )
    }

    /// Every kernel answers a repeated chunk exactly what it answers the same values written out.
    #[test]
    fn repeated_chunk_answers_as_the_flat_one() {
        for (repeated, flat) in [repeated_and_flat(), null_repeated_and_flat()] {
            same_as_flat(&repeated, &flat);
        }
    }

    fn same_as_flat(repeated: &PlPrimitiveArray<f64>, flat: &PlPrimitiveArray<f64>) {
        let quantile = Some(RollingFnParams::Quantile(RollingQuantileParams {
            prob: 0.37,
            method: crate::rolling::QuantileMethod::Linear,
        }));
        let var = Some(RollingFnParams::Var(RollingVarParams { ddof: 1 }));

        for window_size in [1, 2, 5, LEN - 1, LEN, LEN + 3] {
            for min_periods in [1, 2, window_size] {
                let min_periods = min_periods.min(window_size);
                for center in [false, true] {
                    macro_rules! same {
                        ($name:ident, $out:ty, $params:expr) => {
                            let a =
                                $name(repeated, window_size, min_periods, center, None, $params)
                                    .unwrap();
                            let b = $name(flat, window_size, min_periods, center, None, $params)
                                .unwrap();
                            assert_eq!(
                                elements_of::<$out>(&*a),
                                elements_of::<$out>(&*b),
                                "{} w={window_size} m={min_periods} c={center}",
                                stringify!($name),
                            );
                        };
                    }
                    same!(rolling_min, f64, None);
                    same!(rolling_max, f64, None);
                    same!(rolling_quantile, f64, quantile);
                    same!(rolling_var, f64, var);
                    same!(rolling_sum, f64, None);
                    same!(rolling_mean, f64, None);

                    for method in [
                        RollingRankMethod::Average,
                        RollingRankMethod::Min,
                        RollingRankMethod::Max,
                        RollingRankMethod::Dense,
                    ] {
                        let params = Some(RollingFnParams::Rank { method, seed: None });
                        let a =
                            rolling_rank(repeated, window_size, min_periods, center, None, params)
                                .unwrap();
                        let b = rolling_rank(flat, window_size, min_periods, center, None, params)
                            .unwrap();
                        match method {
                            RollingRankMethod::Average => assert_eq!(
                                elements_of::<f64>(&*a),
                                elements_of::<f64>(&*b),
                                "rank {method:?} w={window_size} m={min_periods} c={center}",
                            ),
                            _ => assert_eq!(
                                elements_of::<IdxSize>(&*a),
                                elements_of::<IdxSize>(&*b),
                                "rank {method:?} w={window_size} m={min_periods} c={center}",
                            ),
                        }
                    }

                    let a = rolling_skew(repeated, window_size, min_periods, center, None).unwrap();
                    let b = rolling_skew(flat, window_size, min_periods, center, None).unwrap();
                    assert_eq!(
                        format!("{:?}", elements_of::<f64>(&*a)),
                        format!("{:?}", elements_of::<f64>(&*b)),
                        "rolling_skew w={window_size} m={min_periods} c={center}",
                    );
                    let a =
                        rolling_kurtosis(repeated, window_size, min_periods, center, None).unwrap();
                    let b = rolling_kurtosis(flat, window_size, min_periods, center, None).unwrap();
                    assert_eq!(
                        format!("{:?}", elements_of::<f64>(&*a)),
                        format!("{:?}", elements_of::<f64>(&*b)),
                        "rolling_kurtosis w={window_size} m={min_periods} c={center}",
                    );
                }
            }
        }
    }

    /// The answer to a repeated chunk keeps the value it repeats and writes out only the mask.
    #[test]
    fn repeated_chunk_answer_stays_repeated() {
        let repeated = PlPrimitiveArray::new_scalar(VALUE, LEN);
        let answer = rolling_min(&repeated, 4, 4, false, None, None).unwrap();
        let answer = answer
            .as_any()
            .downcast_ref::<PlPrimitiveArray<f64>>()
            .unwrap();
        assert_eq!(answer.len(), LEN);
        assert!(answer.values_are_scalar());
        assert_eq!(answer.null_count(), 3);
    }
}
