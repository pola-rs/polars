//! The min/max kernels over the arrays of `polars-array`.
//!
//! A chunk that repeats a single value is its own extremum: the one value every element of it
//! holds is read once, in `O(1)`, whether the caller asks for the minimum or the maximum. Neither
//! the run nor a validity mask that repeats one bit is ever written out one slot per element to
//! reach a kernel — see [`reduce_flat`].

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_array::{
    ArrayRepr, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlPrimitiveArray, PlUtf8ViewArray,
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

/// What is left of a primitive chunk for a kernel to reduce.
enum Values<'a, T> {
    /// The one value every element of the chunk holds, at least one of which is not null. It is
    /// its own extremum, so no kernel ever sees it.
    Repeated(T),
    /// The values, one per element, and the mask that says which of them are there at all.
    Flat(&'a [T], Option<&'a Bitmap>),
}

/// What `arr` leaves for a kernel to reduce, or `None` where it leaves nothing: a chunk of no
/// elements, or one whose every element is null, has no extremum.
///
/// Neither of the two buffers is written out one slot per element to get here. A values buffer
/// that holds a single slot is handed back as the [`Values::Repeated`] value it stands for,
/// however the validity mask is stored; and a mask that holds a single bit is either set — in
/// which case it marks nothing and is dropped — or unset, in which case there was no element left
/// to reduce and the null count above has already answered.
fn values_of<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Option<Values<'_, T>> {
    // A chunk with nothing but nulls in it, an empty one included, has no extremum.
    if arr.null_count() == arr.len() {
        return None;
    }

    // Every element is the one value the buffer holds, and at least one element is not null, so
    // that value is both the minimum and the maximum — read here in `O(1)`.
    let values = match arr.values_repr() {
        ArrayRepr::Scalar(value) => return Some(Values::Repeated(value)),
        ArrayRepr::Flat(values) => values,
    };

    // A mask that is set everywhere marks nothing, and one that is unset everywhere left no
    // element to reduce, which the null count has already answered — so a scalar mask says
    // nothing either way, and an absent one says nothing at all.
    let validity = arr.validity().and_then(|validity| validity.repr().flat());

    Some(Values::Flat(values.as_slice(), validity))
}

/// Reduces `arr` to its extremum, folding the elements it lays out one per slot with `flat` and
/// reading the one value it repeats, where that is all it holds, through `repeated`.
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

    /// The two mixed representations reach the kernels without either buffer being written out,
    /// and read exactly as the chunk that holds the same elements one slot per element
    /// throughout: a repeated value under a mask laid out one bit per element, and values laid
    /// out one per element under a mask that repeats one bit.
    #[test]
    fn a_mixed_representation_reads_like_a_flat_one() {
        for length in [1, 3, 8, 17, 64, 65] {
            // A repeated value under a flat mask. The value is its own extremum however many of
            // the elements it marks are left, so long as one of them is.
            for valid in 0..=length {
                let mask: Bitmap = (0..length).map(|i| i < valid).collect();
                let repeated =
                    PlPrimitiveArray::new_scalar(7i32, length).with_validity(Some(mask.clone()));
                let flat = PlPrimitiveArray::from_vec(vec![7i32; length]).with_validity(Some(mask));

                assert!(repeated.values_are_scalar());
                assert_eq!(repeated, flat);
                assert_eq!(
                    repeated.min_max_ignore_nan_kernel(),
                    flat.min_max_ignore_nan_kernel(),
                    "{length} copies of 7, {valid} of them valid",
                );
                assert_eq!(
                    repeated.min_max_propagate_nan_kernel(),
                    flat.min_max_propagate_nan_kernel(),
                );
            }

            // Values laid out one per element under a mask that repeats one bit, which either
            // marks nothing or leaves no element to reduce at all.
            for bit in [false, true] {
                let values: Vec<i32> = (0..length as i32).rev().collect();
                let one_bit = PlPrimitiveArray::from_vec(values.clone())
                    .with_validity_broadcast(Some(Bitmap::new_with_value(bit, 1)));
                let flat = PlPrimitiveArray::from_vec(values)
                    .with_validity(Some(Bitmap::new_with_value(bit, length)));

                assert!(one_bit.validity_is_scalar());
                assert_eq!(one_bit, flat);
                assert_eq!(
                    one_bit.min_ignore_nan_kernel(),
                    flat.min_ignore_nan_kernel(),
                    "{length} values under a repeated {bit}",
                );
                assert_eq!(
                    one_bit.max_ignore_nan_kernel(),
                    flat.max_ignore_nan_kernel()
                );
                assert_eq!(
                    one_bit.min_max_ignore_nan_kernel(),
                    flat.min_max_ignore_nan_kernel(),
                );
            }
        }
    }

    /// As above, over floats, whose kernels part ways over a NaN: a mask that repeats one bit
    /// leaves the NaN to be folded away or carried out just as a flat one does.
    #[test]
    fn a_mixed_representation_reads_the_same_over_nan() {
        let values = vec![2.5f64, f64::NAN, -1.0, 4.0, f64::NAN, 0.0, 7.5];

        for bit in [false, true] {
            let one_bit = PlPrimitiveArray::from_vec(values.clone())
                .with_validity_broadcast(Some(Bitmap::new_with_value(bit, 1)));
            let flat = PlPrimitiveArray::from_vec(values.clone())
                .with_validity(Some(Bitmap::new_with_value(bit, values.len())));

            assert!(one_bit.validity_is_scalar());
            let is_nan = |value: Option<f64>| value.map(f64::is_nan);
            assert_eq!(
                one_bit.min_ignore_nan_kernel(),
                flat.min_ignore_nan_kernel(),
                "a repeated {bit} over {values:?}",
            );
            assert_eq!(
                one_bit.max_ignore_nan_kernel(),
                flat.max_ignore_nan_kernel()
            );
            assert_eq!(
                is_nan(one_bit.min_propagate_nan_kernel()),
                is_nan(flat.min_propagate_nan_kernel()),
            );
            assert_eq!(
                is_nan(one_bit.max_propagate_nan_kernel()),
                is_nan(flat.max_propagate_nan_kernel()),
            );
        }

        // A repeated NaN under a flat mask is the one value there is, whichever kernel asks.
        let repeated = PlPrimitiveArray::new_scalar(f64::NAN, 4)
            .with_validity(Some([true, false, true, false].into_iter().collect()));
        assert!(repeated.min_ignore_nan_kernel().is_some_and(f64::is_nan));
        assert!(repeated.max_propagate_nan_kernel().is_some_and(f64::is_nan));

        // And one that marks nothing leaves no element at all.
        let all_null = PlPrimitiveArray::new_scalar(f64::NAN, 4)
            .with_validity_broadcast(Some(Bitmap::new_with_value(false, 1)));
        assert_eq!(all_null.min_ignore_nan_kernel(), None);
        assert_eq!(all_null.max_propagate_nan_kernel(), None);
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
