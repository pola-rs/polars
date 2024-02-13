use std::borrow::Borrow;

use crate::array::PrimitiveArray;
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::legacy::bit_util::unset_bit_raw;
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
        TrustMyLength::new(self, length)
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

    fn all_equal(&mut self) -> bool
    where
        Self: Sized,
        Self::Item: PartialEq,
    {
        match self.next() {
            None => true,
            Some(a) => self.all(|x| a == x),
        }
    }

    fn fold_options<A, B, F>(&mut self, mut start: B, mut f: F) -> Option<B>
    where
        Self: Iterator<Item = Option<A>>,
        F: FnMut(B, A) -> B,
    {
        for elt in self {
            match elt {
                Some(v) => start = f(start, v),
                None => return None,
            }
        }
        Some(start)
    }

    fn contains<Q>(&mut self, query: &Q) -> bool
    where
        Self: Sized,
        Self::Item: Borrow<Q>,
        Q: PartialEq,
    {
        self.any(|x| x.borrow() == query)
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

impl<T: NativeType> FromIteratorReversed<T> for PrimitiveArray<T> {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = T>>(iter: I) -> Self {
        let size = iter.size_hint().1.unwrap();

        let mut vals: Vec<T> = Vec::with_capacity(size);
        unsafe {
            // Set to end of buffer.
            let mut ptr = vals.as_mut_ptr().add(size);

            iter.for_each(|item| {
                ptr = ptr.sub(1);
                std::ptr::write(ptr, item);
            });
            vals.set_len(size)
        }
        PrimitiveArray::new(ArrowDataType::from(T::PRIMITIVE), vals.into(), None)
    }
}

impl<T: NativeType> FromIteratorReversed<Option<T>> for PrimitiveArray<T> {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = Option<T>>>(iter: I) -> Self {
        let size = iter.size_hint().1.unwrap();

        let mut vals: Vec<T> = Vec::with_capacity(size);
        let mut validity = MutableBitmap::with_capacity(size);
        validity.extend_constant(size, true);
        let validity_ptr = validity.as_slice().as_ptr() as *mut u8;
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
                        unset_bit_raw(validity_ptr, offset)
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
