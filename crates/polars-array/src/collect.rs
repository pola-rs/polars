//! Collecting an iterator of elements into an array.

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_utils::vec::PushUnchecked;

use crate::bitmap::PlBitmap;
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

    /// Collects this iterator of [`Result`]s, returning the first error instead of the array.
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

/// An array collectable from the [zeroable stand-ins](StaticArray::ZeroableValueT) for elements.
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
// capacity from the lower bound of the size hint. Reserving the room up front is not the same as
// not checking for it: `Vec`'s extend still compares the length against the capacity once per
// element, and drives the iterator by `next()`, so an iterator that resolves its representation
// in `fold` never gets to. The trusted variants below write into the reserved room directly and
// through `fold`, which is worth a third of the work on a cheap element — see
// [`vec_from_trusted_len_iter`].

/// Collects `iter` into a `Vec`, writing straight into the room reserved for its elements.
///
/// Two things the safe collect cannot do. The capacity is not compared against the length once
/// per element, since the iterator's length is trusted to be the room reserved for it. And the
/// elements are taken by [`Iterator::for_each`], which is [`Iterator::fold`]: an iterator over
/// values that are either flat or scalar resolves which it is in `fold`, once, where `next()`
/// leaves the branch in the caller's loop.
///
/// Over a million elements of a cheap unary kernel (`dt.weekday`), the two together are 26
/// instructions per element against 17.
#[inline]
fn vec_from_trusted_len_iter<T, I>(iter: I) -> Vec<T>
where
    I: IntoIterator<Item = T>,
    I::IntoIter: TrustedLen,
{
    let iter = iter.into_iter();
    let length = iter
        .size_hint()
        .1
        .expect("a trusted-length iterator knows how many elements it has left");

    let mut values = Vec::with_capacity(length);
    // SAFETY: room for `length` elements was just reserved, and a `TrustedLen` iterator yields
    // exactly as many as its size hint says.
    unsafe {
        let mut next: *mut T = values.as_mut_ptr();
        iter.for_each(|value| {
            next.write(value);
            next = next.add(1);
        });
        values.set_len(length);
    }

    values
}

impl<T: NativeType> ArrayFromIter<T> for PlPrimitiveArray<T> {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        Self::from_vec(vec_from_trusted_len_iter(iter))
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

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let length = iter
            .size_hint()
            .1
            .expect("a trusted-length iterator knows how many elements it has left");

        // The value of a null element is undetermined, so it is left at the default; the mask is
        // built alongside, into room reserved with the values. Both are reserved once, ahead of
        // the walk, and written to unchecked: the iterator yields exactly as many items as the
        // room holds.
        //
        // This one walks the iterator itself rather than handing it to `vec_from_trusted_len_iter`
        // as a `map` that pushes the bit on the way past. Folding a mask builder through a closure
        // keeps its word and its bit count live across the whole walk, which halves the loop's
        // instructions-per-cycle: gathering a million elements out of a chunk with a mask cost 2.5
        // times the cycles of the same gather without one, for 1.3 times the instructions.
        let mut validity = BitmapBuilder::with_capacity(length);
        let mut values = Vec::with_capacity(length);
        // SAFETY: room for `length` values and as many bits was just reserved, and a `TrustedLen`
        // iterator yields exactly as many items as its size hint says.
        unsafe {
            for item in iter {
                values.push_unchecked(item.unwrap_or_default());
                validity.push_unchecked(item.is_some());
            }
        }

        Self::new(
            Buffer::from(values),
            length,
            validity.into_opt_validity().map(PlBitmap::from_bitmap),
        )
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
            validity.into_opt_validity().map(PlBitmap::from_bitmap),
        ))
    }
}

/// Collects `iter` into a bitmap a word at a time, touching the builder once per 64 elements.
fn collect_bitmap<I: Iterator<Item = bool>>(mut iter: I) -> BitmapBuilder {
    let mut builder = BitmapBuilder::with_capacity(iter.size_hint().0);

    loop {
        builder.reserve(64);

        let mut word = 0u64;
        let mut length = 0;
        while length < 64 {
            let Some(value) = iter.next() else { break };
            word |= (value as u64) << length;
            length += 1;
        }

        // SAFETY: room for a whole word was just reserved, `length` is at most 64, and the bits
        // above it were never written.
        unsafe { builder.push_word_with_len_unchecked(word, length) };

        if length < 64 {
            return builder;
        }
    }
}

impl ArrayFromIter<bool> for PlBooleanArray {
    fn arr_from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self::from_values(collect_bitmap(iter.into_iter()).freeze())
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
    fn arr_from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = BitmapBuilder::with_capacity(lower);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for item in iter {
            // The value of a null element is undetermined, so it is left at the default.
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        Self::new(
            values.freeze(),
            length,
            validity.into_opt_validity().map(PlBitmap::from_bitmap),
        )
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
            validity.into_opt_validity().map(PlBitmap::from_bitmap),
        ))
    }
}

/// The values a [`PlBinaryArray`] or [`PlBinaryViewArray`] can be collected from.
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
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
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
mod test {
    use super::*;

    /// The trusted collect writes into reserved room and drives the iterator by `fold`, so it is
    /// checked against the safe one it overrides — including over a scalar values iterator, where
    /// the `fold` it goes through is the one that resolves the representation.
    #[test]
    fn trusted_collect_answers_as_the_safe_one_does() {
        for length in [0usize, 1, 2, 7, 64, 65, 1000] {
            let flat = PlPrimitiveArray::from_vec((0..length as i64).collect::<Vec<_>>());
            let scalar = PlPrimitiveArray::new_scalar(7i64, length);

            for source in [&flat, &scalar] {
                let values: PlPrimitiveArray<i64> =
                    source.values_iter().map(|v| v * 2).collect_arr();
                let trusted: PlPrimitiveArray<i64> =
                    source.values_iter().map(|v| v * 2).collect_arr_trusted();
                assert_eq!(values.len(), length);
                assert_eq!(
                    values.values_iter().collect::<Vec<_>>(),
                    trusted.values_iter().collect::<Vec<_>>()
                );

                // The `Option` collect builds the mask alongside the values.
                let elements: PlPrimitiveArray<i64> = source
                    .values_iter()
                    .map(|v| (v % 3 != 0).then_some(v))
                    .collect_arr();
                let trusted: PlPrimitiveArray<i64> = source
                    .values_iter()
                    .map(|v| (v % 3 != 0).then_some(v))
                    .collect_arr_trusted();
                assert_eq!(trusted.len(), length);
                assert_eq!(trusted.null_count(), elements.null_count());
                assert_eq!(
                    elements.iter().collect::<Vec<_>>(),
                    trusted.iter().collect::<Vec<_>>()
                );
            }
        }
    }
}
