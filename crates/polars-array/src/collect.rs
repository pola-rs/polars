//! Collecting an iterator of elements into an array.

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_buffer::Buffer;

use crate::static_array::StaticArray;
use crate::{PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlPrimitiveArray, PlUtf8ViewArray};

/// An array that can be collected from an iterator of `T`.
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
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E>;

    /// Collects an iterator whose length can be trusted, returning the first error instead.
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

/// An array that can be collected from the [zeroable stand-ins](StaticArray::ZeroableValueT) for
/// its elements.
pub trait ZeroableArrayFromIter:
    StaticArray + for<'a> ArrayFromIter<Self::ZeroableValueT<'a>>
{
    /// Collects `iter` into an array of its elements, in order.
    #[inline(always)]
    fn arr_from_zeroable_iter<'a, I>(iter: I) -> Self
    where
        Self: 'a,
        I: IntoIterator<Item = Self::ZeroableValueT<'a>>,
    {
        Self::arr_from_iter(iter)
    }

    /// Collects an iterator whose length can be trusted into an array of its elements, in order.
    #[inline(always)]
    fn arr_from_zeroable_iter_trusted<'a, I>(iter: I) -> Self
    where
        Self: 'a,
        I: IntoIterator<Item = Self::ZeroableValueT<'a>>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter_trusted(iter)
    }
}

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

/// The values a [`PlBinaryArray`] or a [`PlBinaryViewArray`] can be collected from: the byte
/// slices, and the strings, owned or borrowed.
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

impl<V: IntoBytes> ArrayFromIter<V> for PlBinaryArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self::from_values_iter(iter.into_iter().map(IntoBytes::into_bytes))
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<V, E>>>(iter: I) -> Result<Self, E> {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(lower + 1);
        offsets.push(0);

        for value in iter {
            bytes.extend_from_slice(value?.into_bytes().as_ref());
            offsets.push(bytes.len() as u64);
        }

        let length = offsets.len() - 1;
        // SAFETY: the offsets are the ends of the values appended so far: ordered, one per element
        // plus the end of the last, ending at the length of the bytes they were built over.
        Ok(
            unsafe {
                Self::new_unchecked(Buffer::from(bytes), Buffer::from(offsets), length, None)
            },
        )
    }
}

impl<V: IntoBytes> ArrayFromIter<Option<V>> for PlBinaryArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        iter.into_iter()
            .map(|value| value.map(IntoBytes::into_bytes))
            .collect()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<V>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(lower + 1);
        offsets.push(0);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for value in iter {
            let value = value?;
            // The value of a null element is undetermined, so nothing is written out for it.
            if let Some(value) = value {
                bytes.extend_from_slice(value.into_bytes().as_ref());
                offsets.push(bytes.len() as u64);
                validity.push(true);
            } else {
                offsets.push(bytes.len() as u64);
                validity.push(false);
            }
        }

        let length = offsets.len() - 1;
        // SAFETY: as above, and the mask holds one bit per element.
        Ok(unsafe {
            Self::new_unchecked(
                Buffer::from(bytes),
                Buffer::from(offsets),
                length,
                validity.into_opt_validity(),
            )
        })
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

/// The values a [`PlUtf8ViewArray`] can be collected from: the strings, owned or borrowed.
trait IntoUtf8Bytes: Sized {}

impl IntoUtf8Bytes for &str {}
impl IntoUtf8Bytes for String {}
impl IntoUtf8Bytes for Cow<'_, str> {}

impl<V: IntoUtf8Bytes> ArrayFromIter<V> for PlUtf8ViewArray
where
    PlBinaryViewArray: ArrayFromIter<V>,
{
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        // SAFETY: `IntoUtf8Bytes` says every value collected was a string.
        unsafe { Self::from_binview_unchecked(PlBinaryViewArray::arr_from_iter(iter)) }
    }

    #[inline]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<V, E>>>(iter: I) -> Result<Self, E> {
        let bytes = PlBinaryViewArray::try_arr_from_iter(iter)?;
        // SAFETY: as above.
        Ok(unsafe { Self::from_binview_unchecked(bytes) })
    }
}

impl<V: IntoUtf8Bytes> ArrayFromIter<Option<V>> for PlUtf8ViewArray
where
    PlBinaryViewArray: ArrayFromIter<Option<V>>,
{
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        // SAFETY: `IntoUtf8Bytes` says every value collected was a string.
        unsafe { Self::from_binview_unchecked(PlBinaryViewArray::arr_from_iter(iter)) }
    }

    #[inline]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<V>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let bytes = PlBinaryViewArray::try_arr_from_iter(iter)?;
        // SAFETY: as above.
        Ok(unsafe { Self::from_binview_unchecked(bytes) })
    }
}

// The collects above under another name: the zeroable stand-in for an element of one of these
// four is the element type itself or an `Option` of it, so there is nothing left for the marker
// to do.
impl<T: NativeType> ZeroableArrayFromIter for PlPrimitiveArray<T> {}
impl ZeroableArrayFromIter for PlBooleanArray {}
impl ZeroableArrayFromIter for PlBinaryArray {}
impl ZeroableArrayFromIter for PlBinaryViewArray {}
// The zeroable stand-in for a `&str` is `Option<&str>`, which is what the collect above takes.
impl ZeroableArrayFromIter for PlUtf8ViewArray {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_collects_values_and_optional_values() {
        let values: PlPrimitiveArray<i32> = [1, 2, 3].into_iter().collect_arr();
        assert_eq!(values.flat_values().unwrap().as_slice(), [1, 2, 3]);
        assert_eq!(values.null_count(), 0);

        let options: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect_arr();
        assert_eq!(options.iter().collect::<Vec<_>>(), [Some(1), None, Some(3)]);

        // Both collects lay out one slot per element.
        assert!(values.is_flat() && options.is_flat());
    }

    /// Every fallible collect returns the first error, and none of them walks the iterator past it.
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
        assert_eq!(failed::<PlBinaryArray, _>(b"foo".as_slice()), ("nope", 2));
        assert_eq!(
            failed::<PlBinaryArray, _>(Some(b"foo".as_slice())),
            ("nope", 2),
        );
        assert_eq!(
            failed::<PlBinaryViewArray, _>(b"foo".as_slice()),
            ("nope", 2),
        );
        assert_eq!(
            failed::<PlBinaryViewArray, _>(Some(b"foo".as_slice())),
            ("nope", 2),
        );
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
        assert_eq!(compacted(&array).flat_values().unwrap().as_slice(), [1, 3]);

        let array: PlBooleanArray = [Some(true), None].into_iter().collect_arr();
        assert_eq!(compacted(&array).len(), 1);

        let array: PlBinaryArray = [Some(b"foo".as_slice()), None].into_iter().collect_arr();
        assert_eq!(compacted(&array).value(0), b"foo");

        let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None].into_iter().collect_arr();
        assert_eq!(compacted(&array).value(0), b"foo");
    }
}
