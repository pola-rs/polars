use crate::array::PrimitiveArray;
use crate::bitmap::utils::set_bit_unchecked;
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::legacy::trusted_len::{FromIteratorReversed, TrustedLenPush};
use crate::trusted_len::{TrustMyLength, TrustedLen};
use crate::types::NativeType;

pub trait CustomIterTools: Iterator {
    /// Turn any iterator in a trusted length iterator
    ///
    /// # Safety
    /// The given length must be correct.
    #[inline]
    unsafe fn trust_my_length(self, length: usize) -> TrustMyLength<Self, Self::Item>
    where
        Self: Sized,
    {
        unsafe { TrustMyLength::new(self, length) }
    }

    fn collect_trusted<T: FromTrustedLenIterator<Self::Item>>(self) -> T
    where
        Self: Sized + TrustedLen,
    {
        FromTrustedLenIterator::from_iter_trusted_length(self)
    }

    fn collect_reversed<T: FromIteratorReversed<Self::Item>>(self) -> T
    where
        Self: Sized + TrustedLen,
    {
        FromIteratorReversed::from_trusted_len_iter_rev(self)
    }
}

pub trait CustomIterToolsSized: Iterator + Sized {}

impl<T: ?Sized> CustomIterTools for T where T: Iterator {}

pub trait FromTrustedLenIterator<A>: Sized {
    fn from_iter_trusted_length<T: IntoIterator<Item = A>>(iter: T) -> Self
    where
        T::IntoIter: TrustedLen;
}

impl<T> FromTrustedLenIterator<T> for Vec<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = T>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let len = iter.size_hint().0;
        let mut v = Vec::with_capacity(len);
        v.extend_trusted_len(iter);
        v
    }
}

impl<T: NativeType> FromTrustedLenIterator<Option<T>> for PrimitiveArray<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        unsafe { PrimitiveArray::from_trusted_len_iter_unchecked(iter) }
    }
}

impl<T: NativeType> FromTrustedLenIterator<T> for PrimitiveArray<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = T>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        unsafe { PrimitiveArray::from_trusted_len_values_iter_unchecked(iter) }
    }
}

impl<T> FromIteratorReversed<T> for Vec<T> {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = T>>(iter: I) -> Self {
        unsafe {
            let len = iter.size_hint().1.unwrap();
            let mut out: Vec<T> = Vec::with_capacity(len);
            let mut idx = len;
            for x in iter {
                debug_assert!(idx > 0);
                idx -= 1;
                out.as_mut_ptr().add(idx).write(x);
            }
            debug_assert!(idx == 0);
            out.set_len(len);
            out
        }
    }
}

impl<T: NativeType> FromIteratorReversed<T> for PrimitiveArray<T> {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = T>>(iter: I) -> Self {
        let vals: Vec<T> = iter.collect_reversed();
        PrimitiveArray::new(ArrowDataType::from(T::PRIMITIVE), vals.into(), None)
    }
}

impl<T: NativeType> FromIteratorReversed<Option<T>> for PrimitiveArray<T> {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = Option<T>>>(iter: I) -> Self {
        let size = iter.size_hint().1.unwrap();

        let mut vals: Vec<T> = Vec::with_capacity(size);
        let mut validity = MutableBitmap::with_capacity(size);
        validity.extend_constant(size, true);
        let validity_slice = validity.as_mut_slice();
        unsafe {
            // Set to end of buffer.
            let mut ptr = vals.as_mut_ptr().add(size);
            let mut offset = size;

            iter.for_each(|opt_item| {
                offset -= 1;
                ptr = ptr.sub(1);
                match opt_item {
                    Some(item) => {
                        std::ptr::write(ptr, item);
                    },
                    None => {
                        std::ptr::write(ptr, T::default());
                        set_bit_unchecked(validity_slice, offset, false);
                    },
                }
            });
            vals.set_len(size)
        }
        PrimitiveArray::new(
            ArrowDataType::from(T::PRIMITIVE),
            vals.into(),
            Some(validity.into()),
        )
    }
}
