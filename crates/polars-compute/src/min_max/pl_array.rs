//! The min/max kernels over the arrays of `polars-array`.

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_array::{
    PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlPrimitiveArray, PlUtf8ViewArray,
    StaticArray,
};
use polars_utils::min_max::MinMax;

use super::MinMaxKernel;
use crate::boolean::{all, any};

/// Folds the non-null elements of `arr` with `f`, reading a scalar chunk as the one it repeats.
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

/// What is left of a primitive chunk for a kernel to reduce.
enum Values<'a, T> {
    /// The one value every element of the chunk holds, at least one of which is not null.
    Repeated(T),
    /// The values, one per element, and the mask that says which of them are there at all.
    Flat(&'a [T], Option<&'a Bitmap>),
}

/// What `arr` leaves for a kernel to reduce, or `None` where it leaves nothing.
fn values_of<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Option<Values<'_, T>> {
    // A chunk with nothing but nulls in it, an empty one included, has no extremum.
    if arr.null_count() == arr.len() {
        return None;
    }

    // Every element is the one value the buffer holds, and at least one element is not null, so
    // that value is both the minimum and the maximum — read here in `O(1)`.
    if let Some(value) = arr.scalar_values() {
        return Some(Values::Repeated(value));
    }
    let values = arr.flat_values().unwrap();

    // A mask that is set everywhere marks nothing, and one that is unset everywhere left no
    // element to reduce, which the null count has already answered — so a scalar mask says
    // nothing either way, and an absent one says nothing at all.
    let validity = arr.validity().and_then(|validity| validity.flat_bitmap());

    Some(Values::Flat(values.as_slice(), validity))
}

/// Reduces `arr` to its extremum, with `flat` over a flat chunk and `repeated` over a scalar one.
pub(super) fn reduce_flat<T, R, F, G>(arr: &PlPrimitiveArray<T>, repeated: F, flat: G) -> Option<R>
where
    T: NativeType,
    F: FnOnce(T) -> R,
    G: FnOnce(&[T], Option<&Bitmap>) -> Option<R>,
{
    match values_of(arr)? {
        Values::Repeated(value) => Some(repeated(value)),
        Values::Flat(values, validity) => flat(values, validity),
    }
}

/// Folds the elements of `values` that `validity` marks as being there with `f`.
pub(super) fn fold_flat<T: Copy, F>(values: &[T], validity: Option<&Bitmap>, f: F) -> Option<T>
where
    F: Fn(T, T) -> T,
{
    match validity {
        None => values.iter().copied().reduce(f),
        Some(validity) => values
            .iter()
            .zip(validity.iter())
            .filter_map(|(value, valid)| valid.then_some(*value))
            .reduce(f),
    }
}

/// As [`fold_flat`], folding the minimum and the maximum in one pass.
pub(super) fn fold_flat_min_max<T: Copy, F>(
    values: &[T],
    validity: Option<&Bitmap>,
    f: F,
) -> Option<(T, T)>
where
    F: Fn((T, T), (T, T)) -> (T, T),
{
    let pair = |value: T| (value, value);
    match validity {
        None => values.iter().copied().map(pair).reduce(f),
        Some(validity) => values
            .iter()
            .zip(validity.iter())
            .filter_map(|(value, valid)| valid.then_some(pair(*value)))
            .reduce(f),
    }
}

pub(super) fn min_max_ignore_nan<T: MinMax>(
    (cur_min, cur_max): (T, T),
    (min, max): (T, T),
) -> (T, T) {
    (
        MinMax::min_ignore_nan(cur_min, min),
        MinMax::max_ignore_nan(cur_max, max),
    )
}

pub(super) fn min_max_propagate_nan<T: MinMax>(
    (cur_min, cur_max): (T, T),
    (min, max): (T, T),
) -> (T, T) {
    (
        MinMax::min_propagate_nan(cur_min, min),
        MinMax::max_propagate_nan(cur_max, max),
    )
}

/// `false` orders before `true`, so the minimum is the conjunction and the maximum the disjunction.
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
