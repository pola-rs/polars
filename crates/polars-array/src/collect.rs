//! Collecting an iterator of elements into an array.
//!
//! [`FromIterator`] is what collects an iterator into one *named* array — [`PlPrimitiveArray`],
//! [`PlBooleanArray`], [`PlBinaryViewArray`]. It is of no use to a kernel that is generic over the
//! array it produces: the element type of such a kernel is an associated type of the array, so
//! `A: FromIterator<A::ValueT<'_>>` is not a bound that can be written down. [`ArrayFromIter`] is
//! that same collect with the array as the generic parameter and the element type as the trait's,
//! which is a bound the caller can name, and [`ArrayCollectIterExt`] hangs it off the iterator so
//! that it reads like [`Iterator::collect`].
//!
//! The fallible variants collect an iterator of [`Result`]s, returning the first error instead of
//! the array; the `_trusted` variants take an iterator whose length can be
//! [trusted](arrow::trusted_len::TrustedLen), which lets an implementation that can do better
//! knowing the exact length do so.
//!
//! # What can be collected
//!
//! Every array whose elements carry their own value has an implementation over the values and one
//! over the optional values — an iterator of `T` fills a fully valid array, and an iterator of
//! `Option<T>` writes a validity mask. Both leave the array [flat](crate::Flat); collecting cannot
//! produce the scalar representation, which is what [`new_scalar`](PlPrimitiveArray::new_scalar)
//! and friends are for.
//!
//! The nested arrays — [`PlListArray`](crate::PlListArray),
//! [`PlFixedSizeListArray`](crate::PlFixedSizeListArray), [`PlStructArray`](crate::PlStructArray) —
//! and [`PlNullArray`](crate::PlNullArray) have no implementation. Unlike
//! [`ArrayFromIterDtype`](arrow::array::ArrayFromIterDtype) of the Arrow arrays, there is no dtype
//! here to say what an empty iterator should build, and an element of one of those arrays is not a
//! value that could be appended on its own: it is a range of the values array, or a row across the
//! field arrays. Those are built with a builder — see [`crate::builder`] — or, when the pieces are
//! laid out already, by the constructors of the arrays themselves.
//!
//! # Example
//! ```
//! use polars_array::collect::{ArrayCollectIterExt, ArrayFromIter};
//! use polars_array::{PlBinaryViewArray, PlPrimitiveArray, StaticArray};
//!
//! /// The length of every element of `array`, in whichever array holds those lengths.
//! fn lengths<A: StaticArray + ArrayFromIter<Option<u32>>>(array: &PlBinaryViewArray) -> A {
//!     array.iter().map(|v| Some(v?.len() as u32)).collect_arr()
//! }
//!
//! let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None].into_iter().collect();
//!
//! let lengths: PlPrimitiveArray<u32> = lengths(&array);
//! assert_eq!(lengths.get(0), Some(3));
//! assert_eq!(lengths.get(1), None);
//! ```

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_buffer::Buffer;

use crate::static_array::StaticArray;
use crate::{PlBinaryViewArray, PlBooleanArray, PlPrimitiveArray};

/// An array that can be collected from an iterator of `T`.
///
/// This is [`FromIterator`] with the array in the generic parameter rather than the trait, which
/// is what lets a caller that is generic over the array it builds name the bound; see the
/// [module docs](self) for why that matters and for which arrays implement it.
///
/// The four methods are one collect each: infallible or fallible, over an iterator of unknown
/// length or one whose length can be trusted. Only the first is required — the others have
/// defaults that fall back to it — and only the ones an array can do better on are overridden, so
/// the trusted variants are not by themselves a promise of a faster path.
///
/// # Example
/// ```
/// use polars_array::collect::ArrayFromIter;
/// use polars_array::PlPrimitiveArray;
///
/// let array = PlPrimitiveArray::arr_from_iter([Some(1i32), None]);
/// assert_eq!(array.len(), 2);
/// assert_eq!(array.null_count(), 1);
///
/// // The fallible collect stops at the first error.
/// let failed = PlPrimitiveArray::<i32>::try_arr_from_iter([Ok(1), Err("nope")]);
/// assert_eq!(failed.unwrap_err(), "nope");
/// ```
pub trait ArrayFromIter<T>: Sized {
    /// Collects `iter` into an array of its elements, in order.
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self;

    /// Collects an iterator whose length can be trusted into an array of its elements, in order.
    #[inline(always)]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter(iter)
    }

    /// Collects `iter` into an array of its elements, in order, returning the first error instead.
    ///
    /// The iterator is not walked past an error, so what follows one is never evaluated.
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E>;

    /// Collects an iterator whose length can be trusted, returning the first error instead.
    ///
    /// The iterator is not walked past an error, so what follows one is never evaluated — the
    /// trusted length is the length of the iterator, not of the array this returns.
    #[inline(always)]
    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
        I::IntoIter: TrustedLen,
    {
        Self::try_arr_from_iter(iter)
    }
}

/// [`ArrayFromIter`] as a method on the iterator, the way [`Iterator::collect`] reads.
///
/// Every iterator implements this for every array, so `collect_arr` is available wherever
/// `collect` is; which array is built is the type parameter, inferred from the context the same
/// way `collect` infers its collection.
///
/// # Example
/// ```
/// use polars_array::collect::ArrayCollectIterExt;
/// use polars_array::{PlBooleanArray, PlPrimitiveArray};
///
/// let squares: PlPrimitiveArray<i32> = (1..=4).map(|x| x * x).collect_arr();
/// assert_eq!(squares.values().as_slice(), [1, 4, 9, 16]);
///
/// let parity: PlBooleanArray = squares.values_iter().map(|x| x % 2 == 0).collect_arr();
/// assert_eq!(parity.values_iter().collect::<Vec<_>>(), [false, true, false, true]);
/// ```
pub trait ArrayCollectIterExt<A: StaticArray>: Iterator + Sized {
    /// Collects this iterator into an array of its elements, in order.
    #[inline(always)]
    fn collect_arr(self) -> A
    where
        A: ArrayFromIter<Self::Item>,
    {
        A::arr_from_iter(self)
    }

    /// Collects this iterator, whose length can be trusted, into an array of its elements.
    #[inline(always)]
    fn collect_arr_trusted(self) -> A
    where
        A: ArrayFromIter<Self::Item>,
        Self: TrustedLen,
    {
        A::arr_from_iter_trusted(self)
    }

    /// Collects this iterator of [`Result`]s, returning the first error instead of the array.
    #[inline(always)]
    fn try_collect_arr<U, E>(self) -> Result<A, E>
    where
        A: ArrayFromIter<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        A::try_arr_from_iter(self)
    }

    /// Collects this iterator of [`Result`]s, whose length can be trusted, returning the first
    /// error instead of the array.
    #[inline(always)]
    fn try_collect_arr_trusted<U, E>(self) -> Result<A, E>
    where
        A: ArrayFromIter<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        A::try_arr_from_iter_trusted(self)
    }
}

impl<A: StaticArray, I: Iterator> ArrayCollectIterExt<A> for I {}

// ---------------
// Implementations
// ---------------
//
// The infallible collects are the `FromIterator` implementations of the arrays, which take their
// capacity from the lower bound of the size hint — the exact length when the iterator is
// `TrustedLen`. There is therefore nothing left for the trusted variants to do, and none of them
// is overridden.

impl<T: NativeType> ArrayFromIter<T> for PlPrimitiveArray<T> {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }

    #[inline]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E> {
        let values: Vec<T> = iter.into_iter().collect::<Result<_, E>>()?;
        Ok(Self::from_vec(values))
    }
}

impl<T: NativeType> ArrayFromIter<Option<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        iter.into_iter().collect()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = Vec::with_capacity(lower);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for item in iter {
            let item = item?;
            // The value of a null element is undetermined, so it is left at the default.
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        Ok(Self::new(
            Buffer::from(values),
            length,
            validity.into_opt_validity(),
        ))
    }
}

impl ArrayFromIter<bool> for PlBooleanArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        iter.into_iter().collect()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<bool, E>>>(iter: I) -> Result<Self, E> {
        let iter = iter.into_iter();
        let mut values = BitmapBuilder::with_capacity(iter.size_hint().0);

        for item in iter {
            values.push(item?);
        }

        Ok(Self::from_values(values.freeze()))
    }
}

impl ArrayFromIter<Option<bool>> for PlBooleanArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        iter.into_iter().collect()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<bool>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = BitmapBuilder::with_capacity(lower);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for item in iter {
            let item = item?;
            // The value of a null element is undetermined, so it is left at the default.
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        Ok(Self::new(
            values.freeze(),
            length,
            validity.into_opt_validity(),
        ))
    }
}

/// The values a [`PlBinaryViewArray`] can be collected from: the byte slices, and the strings,
/// owned or borrowed.
///
/// This is not [`AsRef<[u8]>`] because that would leave the implementation over the values and the
/// one over the optional values overlapping: nothing stops a downstream crate from implementing
/// `AsRef<[u8]>` for `Option<&[u8]>`, so the compiler has to assume it might. A trait that is
/// private to this crate cannot grow such an implementation, which is what keeps the two apart. It
/// also lets a [`Cow<str>`] be collected, which is not [`AsRef<[u8]>`] either.
trait IntoBytes {
    /// What this turns into, which is the byte slice itself for everything but a [`Cow<str>`].
    type AsRefT: AsRef<[u8]>;

    fn into_bytes(self) -> Self::AsRefT;
}

/// The values that are already [`AsRef<[u8]>`], and so are their own bytes.
trait TrivialIntoBytes: AsRef<[u8]> {}

impl<T: TrivialIntoBytes> IntoBytes for T {
    type AsRefT = Self;

    #[inline(always)]
    fn into_bytes(self) -> Self {
        self
    }
}

impl TrivialIntoBytes for Vec<u8> {}
impl TrivialIntoBytes for Cow<'_, [u8]> {}
impl TrivialIntoBytes for &[u8] {}
impl TrivialIntoBytes for String {}
impl TrivialIntoBytes for &str {}

impl<'a> IntoBytes for Cow<'a, str> {
    type AsRefT = Cow<'a, [u8]>;

    #[inline]
    fn into_bytes(self) -> Cow<'a, [u8]> {
        match self {
            Cow::Borrowed(s) => Cow::Borrowed(s.as_bytes()),
            Cow::Owned(s) => Cow::Owned(s.into_bytes()),
        }
    }
}

impl<V: IntoBytes> ArrayFromIter<V> for PlBinaryViewArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self::from_values_iter(iter.into_iter().map(IntoBytes::into_bytes))
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<V, E>>>(iter: I) -> Result<Self, E> {
        // A view is written over the data buffers it points at, so the values are laid out first
        // and written out in one pass once they are known to be there.
        let values = iter
            .into_iter()
            .map(|value| Ok(value?.into_bytes()))
            .collect::<Result<Vec<_>, E>>()?;

        Ok(Self::from_values_iter(values))
    }
}

impl<V: IntoBytes> ArrayFromIter<Option<V>> for PlBinaryViewArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        iter.into_iter()
            .map(|value| value.map(IntoBytes::into_bytes))
            .collect()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<V>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        // As above: the values are laid out before any view is written.
        let values = iter
            .into_iter()
            .map(|value| Ok(value?.map(IntoBytes::into_bytes)))
            .collect::<Result<Vec<_>, E>>()?;

        Ok(values.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_collects_values_and_optional_values() {
        let values: PlPrimitiveArray<i32> = [1, 2, 3].into_iter().collect_arr();
        assert_eq!(values.values().as_slice(), [1, 2, 3]);
        assert_eq!(values.null_count(), 0);

        let options: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect_arr();
        assert_eq!(options.iter().collect::<Vec<_>>(), [Some(1), None, Some(3)]);

        // Both collects lay out one slot per element.
        assert!(values.is_flat() && options.is_flat());
    }

    #[test]
    fn boolean_collects_values_and_optional_values() {
        let values: PlBooleanArray = [true, false].into_iter().collect_arr();
        assert_eq!(values.values_iter().collect::<Vec<_>>(), [true, false]);
        assert_eq!(values.null_count(), 0);

        let options: PlBooleanArray = [Some(true), None].into_iter().collect_arr();
        assert_eq!(options.iter().collect::<Vec<_>>(), [Some(true), None]);

        assert!(values.is_flat() && options.is_flat());
    }

    #[test]
    fn binview_collects_values_and_optional_values() {
        let values: PlBinaryViewArray = [b"foo".as_slice(), b"bar"].into_iter().collect_arr();
        assert_eq!(values.value(0), b"foo");
        assert_eq!(values.null_count(), 0);

        let options: PlBinaryViewArray = [Some(b"foo".as_slice()), None].into_iter().collect_arr();
        assert_eq!(options.get(0), Some(b"foo".as_slice()));
        assert_eq!(options.get(1), None);

        assert!(values.is_flat() && options.is_flat());
    }

    /// Every value a [`PlBinaryViewArray`] can be collected from holds the same bytes, owned or
    /// borrowed, string or not.
    #[test]
    fn binview_collects_every_kind_of_value() {
        fn collected<V: IntoBytes>(value: V) -> PlBinaryViewArray {
            std::iter::once(Some(value)).collect_arr()
        }

        let expected: PlBinaryViewArray = std::iter::once(b"foo".as_slice()).collect_arr();

        assert_eq!(collected(b"foo".as_slice()), expected);
        assert_eq!(collected(b"foo".to_vec()), expected);
        assert_eq!(collected(Cow::Borrowed(b"foo".as_slice())), expected);
        assert_eq!(collected("foo"), expected);
        assert_eq!(collected(String::from("foo")), expected);
        assert_eq!(collected(Cow::Borrowed("foo")), expected);
        assert_eq!(collected(Cow::<str>::Owned(String::from("foo"))), expected);
    }

    #[test]
    fn a_fallible_collect_returns_the_array_when_nothing_fails() {
        let array: PlPrimitiveArray<i32> = [Ok::<_, ()>(1), Ok(2)]
            .into_iter()
            .try_collect_arr()
            .unwrap();
        assert_eq!(array.values().as_slice(), [1, 2]);

        let array: PlPrimitiveArray<i32> = [Ok::<_, ()>(Some(1)), Ok(None)]
            .into_iter()
            .try_collect_arr()
            .unwrap();
        assert_eq!(array.iter().collect::<Vec<_>>(), [Some(1), None]);

        let array: PlBooleanArray = [Ok::<_, ()>(true), Ok(false)]
            .into_iter()
            .try_collect_arr()
            .unwrap();
        assert_eq!(array.values_iter().collect::<Vec<_>>(), [true, false]);

        let array: PlBooleanArray = [Ok::<_, ()>(Some(true)), Ok(None)]
            .into_iter()
            .try_collect_arr()
            .unwrap();
        assert_eq!(array.iter().collect::<Vec<_>>(), [Some(true), None]);

        let array: PlBinaryViewArray = [Ok::<_, ()>(b"foo".as_slice()), Ok(b"bar")]
            .into_iter()
            .try_collect_arr()
            .unwrap();
        assert_eq!(array.value(1), b"bar");

        let array: PlBinaryViewArray = [Ok::<_, ()>(Some(b"foo".as_slice())), Ok(None)]
            .into_iter()
            .try_collect_arr()
            .unwrap();
        assert_eq!(
            array.iter().collect::<Vec<_>>(),
            [Some(b"foo".as_slice()), None],
        );
    }

    /// Every fallible collect returns the first error, and none of them walks the iterator past
    /// it.
    #[test]
    fn a_fallible_collect_stops_at_the_first_error() {
        /// The error of collecting `[Ok(value), Err("nope"), Ok(value)]`, and how many of those
        /// three items were pulled from the iterator.
        fn failed<A, T: Clone>(value: T) -> (&'static str, usize)
        where
            A: StaticArray + ArrayFromIter<T>,
        {
            let mut pulled = 0;
            let result: Result<A, &str> = [Ok(value.clone()), Err("nope"), Ok(value)]
                .into_iter()
                .inspect(|_| pulled += 1)
                .try_collect_arr();

            (result.err().unwrap(), pulled)
        }

        assert_eq!(failed::<PlPrimitiveArray<i32>, _>(1), ("nope", 2));
        assert_eq!(failed::<PlPrimitiveArray<i32>, _>(Some(1)), ("nope", 2));
        assert_eq!(failed::<PlBooleanArray, _>(true), ("nope", 2));
        assert_eq!(failed::<PlBooleanArray, _>(Some(true)), ("nope", 2));
        assert_eq!(
            failed::<PlBinaryViewArray, _>(b"foo".as_slice()),
            ("nope", 2),
        );
        assert_eq!(
            failed::<PlBinaryViewArray, _>(Some(b"foo".as_slice())),
            ("nope", 2),
        );
    }

    /// The trusted collects build the same arrays as the untrusted ones; the trusted length is
    /// only ever an optimization.
    #[test]
    fn a_trusted_collect_agrees_with_an_untrusted_one() {
        let trusted: PlPrimitiveArray<i32> = (1..4).collect_arr_trusted();
        let untrusted: PlPrimitiveArray<i32> = (1..4).collect_arr();
        assert_eq!(trusted, untrusted);

        let trusted: PlPrimitiveArray<i32> = vec![Some(1), None].into_iter().collect_arr_trusted();
        let untrusted: PlPrimitiveArray<i32> = vec![Some(1), None].into_iter().collect_arr();
        assert_eq!(trusted, untrusted);

        let trusted: PlBooleanArray = vec![true, false].into_iter().collect_arr_trusted();
        let untrusted: PlBooleanArray = vec![true, false].into_iter().collect_arr();
        assert_eq!(trusted, untrusted);

        let trusted: PlBinaryViewArray = vec![b"foo".as_slice()].into_iter().collect_arr_trusted();
        let untrusted: PlBinaryViewArray = vec![b"foo".as_slice()].into_iter().collect_arr();
        assert_eq!(trusted, untrusted);

        let array: PlPrimitiveArray<i32> = vec![Ok::<_, ()>(1), Ok(2)]
            .into_iter()
            .try_collect_arr_trusted()
            .unwrap();
        assert_eq!(array.values().as_slice(), [1, 2]);

        let failed: Result<PlBooleanArray, &str> = vec![Ok(true), Err("nope")]
            .into_iter()
            .try_collect_arr_trusted();
        assert_eq!(failed.unwrap_err(), "nope");
    }

    #[test]
    fn an_empty_iterator_collects_an_empty_array() {
        let primitive: PlPrimitiveArray<i32> = std::iter::empty::<Option<i32>>().collect_arr();
        let boolean: PlBooleanArray = std::iter::empty::<bool>().collect_arr();
        let binview: PlBinaryViewArray = std::iter::empty::<&[u8]>().collect_arr();

        assert!(primitive.is_empty() && boolean.is_empty() && binview.is_empty());
    }

    /// What the traits are for: a kernel that names the array it builds as a type parameter, which
    /// [`FromIterator`] cannot express over an element type that is the array's own.
    #[test]
    fn collecting_is_generic_over_the_array() {
        /// The elements of `array` that are not null, in an array of the same type.
        fn compacted<A>(array: &A) -> A
        where
            A: StaticArray + for<'a> ArrayFromIter<A::ValueT<'a>>,
        {
            array.iter().flatten().collect_arr()
        }

        let array: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect_arr();
        assert_eq!(compacted(&array).values().as_slice(), [1, 3]);

        let array: PlBooleanArray = [Some(true), None].into_iter().collect_arr();
        assert_eq!(compacted(&array).len(), 1);

        let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None].into_iter().collect_arr();
        assert_eq!(compacted(&array).value(0), b"foo");
    }
}
