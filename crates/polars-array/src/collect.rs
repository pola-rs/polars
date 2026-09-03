//! Collecting an iterator of elements into an array.

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_buffer::Buffer;

use crate::static_array::StaticArray;
use crate::{PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlPrimitiveArray, PlUtf8ViewArray};

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
/// assert_eq!(squares.flat_values().unwrap().as_slice(), [1, 4, 9, 16]);
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

/// An array that can be collected from the [zeroable stand-ins](StaticArray::ZeroableValueT) for
/// its elements.
///
/// This is [`ArrayFromIter`] over [`StaticArray::ZeroableValueT`], which is a bound a caller
/// cannot name without writing out the higher-ranked projection. Every array that can be
/// collected at all implements it — the zeroable stand-in for an element is the element type
/// itself or an [`Option`] of it, and those are what such an array is collected from already — so
/// it is [`from_zeroable_vec`](arrow::array::StaticArray::from_zeroable_vec) of the Arrow arrays,
/// minus the dtype the arrays of this crate do not carry.
///
/// This is what a kernel that walks an array element by element collects: it fills one slot per
/// element, leaves the slots it has no value for zeroed, and puts the mask that says which those
/// were on the array afterwards with
/// [`with_validity_typed`](StaticArray::with_validity_typed). Which is to say that the values a
/// zeroed slot collects as are unspecified — [`None`] collects as a null, a zeroed number as a
/// zero — and it is the mask, not the value, that makes the element null.
///
/// # Example
/// ```
/// use arrow::bitmap::BitmapBuilder;
/// use bytemuck::Zeroable;
/// use polars_array::collect::ZeroableArrayFromIter;
/// use polars_array::{PlBinaryViewArray, PlPrimitiveArray, StaticArray};
///
/// /// The elements of `array` at `indices`, with an out-of-bounds index standing for a null.
/// fn opt_gather<A: ZeroableArrayFromIter>(array: &A, indices: &[usize]) -> A {
///     let mut validity = BitmapBuilder::with_capacity(indices.len());
///
///     let values: Vec<A::ZeroableValueT<'_>> = indices
///         .iter()
///         .map(|&i| {
///             let value = (i < array.len()).then(|| array.get(i)).flatten();
///             validity.push(value.is_some());
///             // The slot of an element there is no value for is left zeroed.
///             value.map_or_else(Zeroable::zeroed, Into::into)
///         })
///         .collect();
///
///     A::arr_from_zeroable_iter(values).with_validity_typed(validity.into_opt_validity())
/// }
///
/// let array: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
/// let gathered = opt_gather(&array, &[2, 9, 0, 1]);
/// assert_eq!(gathered.iter().collect::<Vec<_>>(), [Some(3), None, Some(1), None]);
///
/// // An element that is null is one there is no value for either, so its slot is zeroed too.
/// let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None].into_iter().collect();
/// let gathered = opt_gather(&array, &[0, 9, 1]);
/// assert_eq!(gathered.iter().collect::<Vec<_>>(), [Some(b"foo".as_slice()), None, None]);
/// ```
pub trait ZeroableArrayFromIter:
    StaticArray + for<'a> ArrayFromIter<Self::ZeroableValueT<'a>>
{
    /// Collects `iter` into an array of its elements, in order.
    ///
    /// This is [`ArrayFromIter::arr_from_iter`] over the zeroable values, named so that it can be
    /// reached without the projection spelled out.
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
///
/// This is the marker that keeps that array's invariant across a collect. The bytes reach the
/// inner [`PlBinaryViewArray`] through the same [`IntoBytes`] conversion any byte string does, and
/// it is membership of this trait — a `&str`, a [`String`], a [`Cow<str>`](Cow), and nothing else
/// — that says they were a string to begin with. It is private to this crate for the same reason
/// [`IntoBytes`] is, and so that nothing downstream can add a value to it that is not one.
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
