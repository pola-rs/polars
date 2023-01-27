use std::borrow::Borrow;

use arrow::bitmap::MutableBitmap;
use polars_arrow::bit_util::{set_bit_raw, unset_bit_raw};
use polars_arrow::trusted_len::{FromIteratorReversed, PushUnchecked};

use crate::chunked_array::upstream_traits::PolarsAsRef;
use crate::prelude::*;
use crate::utils::{CustomIterTools, FromTrustedLenIterator, NoNull};

impl<T> FromTrustedLenIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let arr = unsafe {
            PrimitiveArray::from_trusted_len_iter_unchecked(iter).to(T::get_dtype().to_arrow())
        };
        unsafe { ChunkedArray::from_chunks("", vec![Box::new(arr)]) }
    }
}

// NoNull is only a wrapper needed for specialization
impl<T> FromTrustedLenIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    // We use Vec because it is way faster than Arrows builder. We can do this because we
    // know we don't have null values.
    fn from_iter_trusted_length<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let values = unsafe { Vec::from_trusted_len_iter_unchecked(iter) }.into();
        let arr = PrimitiveArray::new(T::get_dtype().to_arrow(), values, None);

        unsafe { NoNull::new(ChunkedArray::from_chunks("", vec![Box::new(arr)])) }
    }
}

impl<T> FromIteratorReversed<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = Option<T::Native>>>(iter: I) -> Self {
        let size = iter.size_hint().1.unwrap();

        let mut vals: Vec<T::Native> = Vec::with_capacity(size);
        let mut validity = MutableBitmap::with_capacity(size);
        validity.extend_constant(size, true);
        let validity_ptr = validity.as_slice().as_ptr() as *mut u8;
        unsafe {
            // set to end of buffer
            let mut ptr = vals.as_mut_ptr().add(size);
            let mut offset = size;

            iter.for_each(|opt_item| {
                offset -= 1;
                ptr = ptr.sub(1);
                match opt_item {
                    Some(item) => {
                        std::ptr::write(ptr, item);
                    }
                    None => {
                        std::ptr::write(ptr, T::Native::default());
                        unset_bit_raw(validity_ptr, offset)
                    }
                }
            });
            vals.set_len(size)
        }
        let arr = PrimitiveArray::new(
            T::get_dtype().to_arrow(),
            vals.into(),
            Some(validity.into()),
        );
        unsafe { ChunkedArray::from_chunks("", vec![Box::new(arr)]) }
    }
}

impl FromIteratorReversed<Option<bool>> for BooleanChunked {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = Option<bool>>>(iter: I) -> Self {
        let size = iter.size_hint().1.unwrap();

        let vals = MutableBitmap::from_len_zeroed(size);
        let mut validity = MutableBitmap::with_capacity(size);
        validity.extend_constant(size, true);
        let validity_ptr = validity.as_slice().as_ptr() as *mut u8;
        let vals_ptr = vals.as_slice().as_ptr() as *mut u8;
        unsafe {
            let mut offset = size;

            iter.for_each(|opt_item| {
                offset -= 1;
                match opt_item {
                    Some(item) => {
                        if item {
                            // set value
                            // validity bit is already true
                            set_bit_raw(vals_ptr, offset);
                        }
                    }
                    None => {
                        // unset validity bit
                        unset_bit_raw(validity_ptr, offset)
                    }
                }
            });
        }
        let arr = BooleanArray::new(ArrowDataType::Boolean, vals.into(), Some(validity.into()));
        unsafe { ChunkedArray::from_chunks("", vec![Box::new(arr)]) }
    }
}

impl<T> FromIteratorReversed<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = T::Native>>(iter: I) -> Self {
        let size = iter.size_hint().1.unwrap();

        let mut vals: Vec<T::Native> = Vec::with_capacity(size);
        unsafe {
            // set to end of buffer
            let mut ptr = vals.as_mut_ptr().add(size);

            iter.for_each(|item| {
                ptr = ptr.sub(1);
                std::ptr::write(ptr, item);
            });
            vals.set_len(size)
        }
        let arr = PrimitiveArray::new(T::get_dtype().to_arrow(), vals.into(), None);
        unsafe { NoNull::new(ChunkedArray::from_chunks("", vec![Box::new(arr)])) }
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

        unsafe { Self::from_chunks("", vec![Box::new(arr)]) }
    }
}

impl FromTrustedLenIterator<bool> for BooleanChunked {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let arr: BooleanArray = iter.collect_trusted();

        unsafe { Self::from_chunks("", vec![Box::new(arr)]) }
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

#[cfg(feature = "dtype-binary")]
impl<Ptr> FromTrustedLenIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]>,
{
    fn from_iter_trusted_length<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        let iter = iter.into_iter();
        iter.collect()
    }
}

#[cfg(feature = "dtype-binary")]
impl<Ptr> FromTrustedLenIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]>,
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::CustomIterTools;

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
