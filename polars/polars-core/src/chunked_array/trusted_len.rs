use crate::chunked_array::upstream_traits::PolarsAsRef;
use crate::prelude::*;
use crate::utils::{CustomIterTools, FromTrustedLenIterator, NoNull};
use arrow::array::{ArrayData, BooleanArray, PrimitiveArray};
use arrow::buffer::Buffer;
use std::borrow::Borrow;

impl<T> FromTrustedLenIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let len = iter.size_hint().0;

        let arr: PrimitiveArray<T> = unsafe {
            let arr = PrimitiveArray::from_trusted_len_iter(iter);
            assert_eq!(arr.len(), len);
            arr
        };
        ChunkedArray::new_from_chunks("", vec![Arc::new(arr)])
    }
}

// NoNull is only a wrapper needed for specialization
impl<T> FromTrustedLenIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsPrimitiveType,
{
    // We use AlignedVec because it is way faster than Arrows builder. We can do this because we
    // know we don't have null values.
    fn from_iter_trusted_length<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let len = iter.size_hint().0;
        let buffer = unsafe { Buffer::from_trusted_len_iter(iter) };

        let data = ArrayData::new(T::DATA_TYPE, len, None, None, 0, vec![buffer], vec![]);
        NoNull::new(ChunkedArray::new_from_chunks(
            "",
            vec![Arc::new(PrimitiveArray::<T>::from(data))],
        ))
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

impl<Ptr> FromTrustedLenIterator<Option<Ptr>> for ListChunked
where
    Ptr: Borrow<Series>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
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

        Self::new_from_chunks("", vec![Arc::new(arr)])
    }
}

impl FromTrustedLenIterator<bool> for BooleanChunked {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let arr: BooleanArray = iter.collect_trusted();

        Self::new_from_chunks("", vec![Arc::new(arr)])
    }
}

impl FromTrustedLenIterator<bool> for NoNull<BooleanChunked> {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}
impl<Ptr> FromTrustedLenIterator<Ptr> for Utf8Chunked
where
    Ptr: PolarsAsRef<str>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

impl<Ptr> FromTrustedLenIterator<Option<Ptr>> for Utf8Chunked
where
    Ptr: AsRef<str>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> FromTrustedLenIterator<Option<T>> for ObjectChunked<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}
