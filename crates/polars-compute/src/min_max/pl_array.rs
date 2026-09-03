//! The min/max kernels over the arrays of `polars-array`.
//!
//! A chunk that is entirely stored in the scalar representation is its own extremum: the one
//! element every element of it repeats is read once, in `O(1)`, whether the caller asks for the
//! minimum or the maximum.

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use polars_array::arrow::bridge::ToArrow;
use polars_array::{
    PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlPrimitiveArray, PlUtf8ViewArray,
    StaticArray,
};
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;
use crate::boolean::{all, any};

/// Folds the non-null elements of `arr` with `f`, reading a scalar chunk as the one element it
/// repeats.
fn reduce_values<'a, A, F>(arr: &'a A, f: F) -> Option<A::ValueT<'a>>
where
    A: StaticArray,
    F: Fn(A::ValueT<'a>, A::ValueT<'a>) -> A::ValueT<'a>,
{
    // Every element of a scalar chunk is the one element it repeats, which is therefore its own
    // extremum; a chunk that repeats a null has no extremum at all.
    if let Some(value) = arr.scalar_value() {
        return value;
    }

    if arr.has_nulls() {
        arr.iter().flatten().reduce(f)
    } else {
        arr.values_iter().reduce(f)
    }
}

/// As [`reduce_values`], folding the minimum and the maximum in one pass.
fn reduce_min_max<'a, A, F>(arr: &'a A, f: F) -> Option<(A::ValueT<'a>, A::ValueT<'a>)>
where
    A: StaticArray,
    F: Fn(
        (A::ValueT<'a>, A::ValueT<'a>),
        (A::ValueT<'a>, A::ValueT<'a>),
    ) -> (A::ValueT<'a>, A::ValueT<'a>),
{
    if let Some(value) = arr.scalar_value() {
        return value.map(|value| (value.clone(), value));
    }

    let pair = |value: A::ValueT<'a>| (value.clone(), value);
    if arr.has_nulls() {
        arr.iter().flatten().map(pair).reduce(f)
    } else {
        arr.values_iter().map(pair).reduce(f)
    }
}

fn min_max_ignore_nan<T: MinMax>((cur_min, cur_max): (T, T), (min, max): (T, T)) -> (T, T) {
    (
        MinMax::min_ignore_nan(cur_min, min),
        MinMax::max_ignore_nan(cur_max, max),
    )
}

fn min_max_propagate_nan<T: MinMax>((cur_min, cur_max): (T, T), (min, max): (T, T)) -> (T, T) {
    (
        MinMax::min_propagate_nan(cur_min, min),
        MinMax::max_propagate_nan(cur_max, max),
    )
}

/// The values of `arr` laid out one per element, for the Arrow kernel to read.
fn to_flat_arrow<T: NativeType>(arr: &PlPrimitiveArray<T>) -> PrimitiveArray<T> {
    PlPrimitiveArray::to_arrow(&arr.to_flat())
}

/// A chunk that repeats a single value answers in `O(1)`; anything else crosses over to the Arrow
/// kernel, which is the vectorized one — see `simd`.
impl<T> MinMaxKernel for PlPrimitiveArray<T>
where
    T: NativeType + MinMax,
    PrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    type Scalar<'a> = T;

    fn min_ignore_nan_kernel(&self) -> Option<T> {
        match self.scalar_value() {
            Some(value) => value,
            None => to_flat_arrow(self).min_ignore_nan_kernel(),
        }
    }

    fn max_ignore_nan_kernel(&self) -> Option<T> {
        match self.scalar_value() {
            Some(value) => value,
            None => to_flat_arrow(self).max_ignore_nan_kernel(),
        }
    }

    fn min_max_ignore_nan_kernel(&self) -> Option<(T, T)> {
        match self.scalar_value() {
            Some(value) => value.map(|value| (value, value)),
            None => to_flat_arrow(self).min_max_ignore_nan_kernel(),
        }
    }

    fn min_propagate_nan_kernel(&self) -> Option<T> {
        match self.scalar_value() {
            Some(value) => value,
            None => to_flat_arrow(self).min_propagate_nan_kernel(),
        }
    }

    fn max_propagate_nan_kernel(&self) -> Option<T> {
        match self.scalar_value() {
            Some(value) => value,
            None => to_flat_arrow(self).max_propagate_nan_kernel(),
        }
    }

    fn min_max_propagate_nan_kernel(&self) -> Option<(T, T)> {
        match self.scalar_value() {
            Some(value) => value.map(|value| (value, value)),
            None => to_flat_arrow(self).min_max_propagate_nan_kernel(),
        }
    }
}

/// `false` orders before `true`, so the minimum is the conjunction of the non-null values and the
/// maximum is their disjunction: both already read a scalar chunk in `O(1)`.
impl MinMaxKernel for PlBooleanArray {
    type Scalar<'a> = bool;

    fn min_ignore_nan_kernel(&self) -> Option<bool> {
        all(self)
    }

    fn max_ignore_nan_kernel(&self) -> Option<bool> {
        any(self)
    }

    #[inline(always)]
    fn min_propagate_nan_kernel(&self) -> Option<bool> {
        self.min_ignore_nan_kernel()
    }

    #[inline(always)]
    fn max_propagate_nan_kernel(&self) -> Option<bool> {
        self.max_ignore_nan_kernel()
    }
}

/// The byte-ordered arrays, whose elements no kernel compares faster than one at a time.
macro_rules! impl_min_max_kernel {
    ($($A:ty => $S:ty),* $(,)?) => {
        $(
        impl MinMaxKernel for $A {
            type Scalar<'a> = $S;

            fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                reduce_values(self, MinMax::min_ignore_nan)
            }

            fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                reduce_values(self, MinMax::max_ignore_nan)
            }

            fn min_max_ignore_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
                reduce_min_max(self, min_max_ignore_nan)
            }

            fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                reduce_values(self, MinMax::min_propagate_nan)
            }

            fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>> {
                reduce_values(self, MinMax::max_propagate_nan)
            }

            fn min_max_propagate_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
                reduce_min_max(self, min_max_propagate_nan)
            }
        }
        )*
    };
}

impl_min_max_kernel! {
    PlBinaryViewArray => &'a [u8],
    PlUtf8ViewArray => &'a str,
    PlBinaryArray => &'a [u8],
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;

    /// `length` copies of `value`, marked by `validity`, in both representations.
    fn repeated<T: NativeType>(
        value: T,
        validity: Option<&Bitmap>,
        length: usize,
    ) -> [PlPrimitiveArray<T>; 2] {
        let scalar =
            PlPrimitiveArray::new_scalar(value, length).with_validity_broadcast(validity.cloned());
        let flat = PlPrimitiveArray::from_vec(vec![value; length])
            .with_validity_broadcast(validity.cloned());
        assert_eq!(scalar, flat);
        [scalar, flat]
    }

    /// Every element of a chunk that repeats one value is that value, which is therefore both its
    /// minimum and its maximum — read in `O(1)` where the chunk is stored that way.
    #[test]
    fn a_repeated_value_is_its_own_extremum() {
        for length in [0, 1, 3, 65] {
            for valid in 0..=length {
                let mask: Bitmap = (0..length).map(|i| i < valid).collect();
                for validity in [None, Some(&mask)] {
                    let [scalar, flat] = repeated(7i32, validity, length);
                    let count = validity.map_or(length, |_| valid);
                    let expected = (count > 0).then_some(7);

                    assert_eq!(scalar.min_ignore_nan_kernel(), flat.min_ignore_nan_kernel());
                    assert_eq!(scalar.max_ignore_nan_kernel(), flat.max_ignore_nan_kernel());
                    assert_eq!(
                        scalar.min_ignore_nan_kernel(),
                        expected,
                        "min of {scalar:?}"
                    );
                    assert_eq!(
                        scalar.max_ignore_nan_kernel(),
                        expected,
                        "max of {scalar:?}"
                    );
                    assert_eq!(
                        scalar.min_max_ignore_nan_kernel(),
                        expected.map(|value| (value, value)),
                    );
                    assert_eq!(
                        scalar.min_max_propagate_nan_kernel(),
                        flat.min_max_propagate_nan_kernel(),
                    );
                }
            }
        }
    }

    /// A chunk of nothing but NaNs answers the same whichever representation it is in. It is
    /// built here rather than through `repeated`, whose check that the two hold the same elements
    /// no NaN ever passes.
    #[test]
    fn a_repeated_nan_reads_the_same_either_way() {
        let scalar = PlPrimitiveArray::new_scalar(f64::NAN, 8);
        let flat = PlPrimitiveArray::from_vec(vec![f64::NAN; 8]);

        let is_nan = |value: Option<f64>| value.map(f64::is_nan);
        assert_eq!(
            is_nan(scalar.min_ignore_nan_kernel()),
            is_nan(flat.min_ignore_nan_kernel()),
        );
        assert_eq!(
            is_nan(scalar.max_ignore_nan_kernel()),
            is_nan(flat.max_ignore_nan_kernel()),
        );
        assert_eq!(
            is_nan(scalar.min_propagate_nan_kernel()),
            is_nan(flat.min_propagate_nan_kernel()),
        );
        assert_eq!(
            is_nan(scalar.max_propagate_nan_kernel()),
            is_nan(flat.max_propagate_nan_kernel()),
        );
    }

    /// A chunk laid out one value per element folds its non-null values as it always has.
    #[test]
    fn null_elements_are_passed_over() {
        let arr = PlPrimitiveArray::from_iter([Some(3i32), None, Some(-1), Some(9)]);

        assert_eq!(arr.min_ignore_nan_kernel(), Some(-1));
        assert_eq!(arr.max_ignore_nan_kernel(), Some(9));
        assert_eq!(arr.min_max_ignore_nan_kernel(), Some((-1, 9)));

        let all_null = PlPrimitiveArray::<i32>::new_full_null(4);
        assert_eq!(all_null.min_ignore_nan_kernel(), None);
        assert_eq!(all_null.max_ignore_nan_kernel(), None);
    }

    /// `false` orders before `true`, and a repeated bit is its own extremum.
    #[test]
    fn booleans_order_false_first() {
        let arr = PlBooleanArray::from_iter([Some(true), None, Some(false)]);
        assert_eq!(arr.min_ignore_nan_kernel(), Some(false));
        assert_eq!(arr.max_ignore_nan_kernel(), Some(true));

        let scalar = PlBooleanArray::new_scalar(true, 100);
        assert_eq!(scalar.min_ignore_nan_kernel(), Some(true));
        assert_eq!(scalar.max_ignore_nan_kernel(), Some(true));
    }

    /// The byte-ordered arrays fold their elements, and read a repeated one once.
    #[test]
    fn byte_arrays_order_lexicographically() {
        let arr = PlUtf8ViewArray::from_iter([Some("pear"), None, Some("apple"), Some("fig")]);
        assert_eq!(arr.min_ignore_nan_kernel(), Some("apple"));
        assert_eq!(arr.max_ignore_nan_kernel(), Some("pear"));
        assert_eq!(arr.min_max_ignore_nan_kernel(), Some(("apple", "pear")));

        let scalar = PlBinaryViewArray::new_scalar(b"fig", 100);
        assert_eq!(scalar.min_ignore_nan_kernel(), Some(&b"fig"[..]));
        assert_eq!(scalar.max_ignore_nan_kernel(), Some(&b"fig"[..]));
    }
}
