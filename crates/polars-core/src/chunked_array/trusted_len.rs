use std::borrow::Borrow;

use arrow::legacy::trusted_len::{FromIteratorReversed, TrustedLenPush};

use crate::chunked_array::from_iterator::PolarsAsRef;
use crate::prelude::*;
use crate::utils::{FromTrustedLenIterator, NoNull};

impl<T> FromTrustedLenIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        // SAFETY: iter is TrustedLen.
        let iter = iter.into_iter();
        let arr = unsafe {
            PrimitiveArray::from_trusted_len_iter_unchecked(iter)
                .to(T::get_dtype().to_arrow(CompatLevel::newest()))
        };
        arr.into()
    }
}

// NoNull is only a wrapper needed for specialization.
impl<T> FromTrustedLenIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    // We use Vec because it is way faster than Arrows builder. We can do this
    // because we know we don't have null values.
    fn from_iter_trusted_length<I: IntoIterator<Item = T::Native>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        // SAFETY: iter is TrustedLen.
        let iter = iter.into_iter();
        let values = unsafe { Vec::from_trusted_len_iter_unchecked(iter) }.into();
        let arr = PrimitiveArray::new(T::get_dtype().to_arrow(CompatLevel::newest()), values, None);
        NoNull::new(arr.into())
    }
}

impl<T> FromIteratorReversed<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = Option<T::Native>>>(iter: I) -> Self {
        let arr: PrimitiveArray<T::Native> = iter.collect_reversed();
        arr.into()
    }
}

impl<T> FromIteratorReversed<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = T::Native>>(iter: I) -> Self {
        let arr: PrimitiveArray<T::Native> = iter.collect_reversed();
        NoNull::new(arr.into())
    }
}

impl FromIteratorReversed<Option<bool>> for BooleanChunked {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = Option<bool>>>(iter: I) -> Self {
        let arr: BooleanArray = iter.collect_reversed();
        arr.into()
    }
}

impl FromIteratorReversed<bool> for NoNull<BooleanChunked> {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = bool>>(iter: I) -> Self {
        let arr: BooleanArray = iter.collect_reversed();
        NoNull::new(arr.into())
    }
}

impl<Ptr> FromTrustedLenIterator<Ptr> for ListChunked
where
    Ptr: Borrow<Series>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl FromTrustedLenIterator<Option<Series>> for ListChunked {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<Series>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl FromTrustedLenIterator<Option<bool>> for ChunkedArray<BooleanType> {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let arr: BooleanArray = iter.collect_trusted();
        arr.into()
    }
}

impl FromTrustedLenIterator<bool> for BooleanChunked {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let arr: BooleanArray = iter.collect_trusted();
        arr.into()
    }
}

impl FromTrustedLenIterator<bool> for NoNull<BooleanChunked> {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}
impl<Ptr> FromTrustedLenIterator<Ptr> for StringChunked
where
    Ptr: PolarsAsRef<str>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl<Ptr> FromTrustedLenIterator<Option<Ptr>> for StringChunked
where
    Ptr: AsRef<str>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl<Ptr> FromTrustedLenIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl<Ptr> FromTrustedLenIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl<Ptr> FromTrustedLenIterator<Ptr> for BinaryOffsetChunked
where
    Ptr: PolarsAsRef<[u8]>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let arr = BinaryArray::from_iter_values(iter.into_iter());
        ChunkedArray::with_chunk("", arr)
    }
}

impl<Ptr> FromTrustedLenIterator<Option<Ptr>> for BinaryOffsetChunked
where
    Ptr: AsRef<[u8]>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let arr = BinaryArray::from_iter(iter);
        ChunkedArray::with_chunk("", arr)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> FromTrustedLenIterator<Option<T>> for ObjectChunked<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_reverse_collect() {
        let ca: NoNull<Int32Chunked> = (0..5).collect_reversed();
        let arr = ca.downcast_iter().next().unwrap();
        let s = arr.values().as_slice();
        assert_eq!(s, &[4, 3, 2, 1, 0]);

        let ca: Int32Chunked = (0..5)
            .map(|val| match val % 2 == 0 {
                true => Some(val),
                false => None,
            })
            .collect_reversed();
        assert_eq!(Vec::from(&ca), &[Some(4), None, Some(2), None, Some(0)]);
    }
}
