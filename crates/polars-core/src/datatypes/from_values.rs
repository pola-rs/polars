use std::borrow::Cow;
use std::error::Error;

use arrow::array::{BinaryArray, BooleanArray, MutablePrimitiveArray, PrimitiveArray, Utf8Array};
use polars_arrow::array::utf8::{BinaryFromIter, Utf8FromIter};
use polars_arrow::trusted_len::TrustedLen;

use crate::prelude::StaticArray;

pub trait ArrayFromElementIter
where
    Self: Sized,
{
    type ArrayType: StaticArray;

    fn array_from_iter<I: TrustedLen<Item = Option<Self>>>(iter: I) -> Self::ArrayType;

    fn array_from_values_iter<I: TrustedLen<Item = Self>>(iter: I) -> Self::ArrayType;

    fn try_array_from_iter<E: Error, I: TrustedLen<Item = Result<Option<Self>, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E>;
}

impl ArrayFromElementIter for bool {
    type ArrayType = BooleanArray;

    fn array_from_iter<I: TrustedLen<Item = Option<Self>>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { BooleanArray::from_trusted_len_iter_unchecked(iter) }
    }

    fn array_from_values_iter<I: TrustedLen<Item = Self>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { BooleanArray::from_trusted_len_values_iter_unchecked(iter) }
    }

    fn try_array_from_iter<E: Error, I: TrustedLen<Item = Result<Option<Self>, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { BooleanArray::try_from_trusted_len_iter_unchecked(iter) }
    }
}

macro_rules! impl_primitive {
    ($tp:ty) => {
        impl ArrayFromElementIter for $tp {
            type ArrayType = PrimitiveArray<Self>;

            fn array_from_iter<I: TrustedLen<Item = Option<Self>>>(iter: I) -> Self::ArrayType {
                // SAFETY: guarded by `TrustedLen` trait
                unsafe { PrimitiveArray::from_trusted_len_iter_unchecked(iter) }
            }

            fn array_from_values_iter<I: TrustedLen<Item = Self>>(iter: I) -> Self::ArrayType {
                // SAFETY: guarded by `TrustedLen` trait
                unsafe { PrimitiveArray::from_trusted_len_values_iter_unchecked(iter) }
            }
            fn try_array_from_iter<E: Error, I: TrustedLen<Item = Result<Option<Self>, E>>>(
                iter: I,
            ) -> Result<Self::ArrayType, E> {
                // SAFETY: guarded by `TrustedLen` trait
                unsafe {
                    Ok(MutablePrimitiveArray::try_from_trusted_len_iter_unchecked(iter)?.into())
                }
            }
        }
    };
}

impl_primitive!(u8);
impl_primitive!(u16);
impl_primitive!(u32);
impl_primitive!(u64);
impl_primitive!(i8);
impl_primitive!(i16);
impl_primitive!(i32);
impl_primitive!(i64);
impl_primitive!(f32);
impl_primitive!(f64);

impl ArrayFromElementIter for &str {
    type ArrayType = Utf8Array<i64>;

    fn array_from_iter<I: TrustedLen<Item = Option<Self>>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { Utf8Array::from_trusted_len_iter_unchecked(iter) }
    }

    fn array_from_values_iter<I: TrustedLen<Item = Self>>(iter: I) -> Self::ArrayType {
        let len = iter.size_hint().0;
        Utf8Array::from_values_iter(iter, len, len * 24)
    }
    fn try_array_from_iter<E: Error, I: TrustedLen<Item = Result<Option<Self>, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { Utf8Array::try_from_trusted_len_iter_unchecked(iter) }
    }
}

impl ArrayFromElementIter for Cow<'_, str> {
    type ArrayType = Utf8Array<i64>;

    fn array_from_iter<I: TrustedLen<Item = Option<Self>>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { Utf8Array::from_trusted_len_iter_unchecked(iter) }
    }

    fn array_from_values_iter<I: TrustedLen<Item = Self>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        let len = iter.size_hint().0;
        Utf8Array::from_values_iter(iter, len, len * 24)
    }
    fn try_array_from_iter<E: Error, I: TrustedLen<Item = Result<Option<Self>, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { Utf8Array::try_from_trusted_len_iter_unchecked(iter) }
    }
}

impl ArrayFromElementIter for Cow<'_, [u8]> {
    type ArrayType = BinaryArray<i64>;

    fn array_from_iter<I: TrustedLen<Item = Option<Self>>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { BinaryArray::from_trusted_len_iter_unchecked(iter) }
    }

    fn array_from_values_iter<I: TrustedLen<Item = Self>>(iter: I) -> Self::ArrayType {
        // SAFETY: guarded by `TrustedLen` trait
        let len = iter.size_hint().0;
        BinaryArray::from_values_iter(iter, len, len * 24)
    }
    fn try_array_from_iter<E: Error, I: TrustedLen<Item = Result<Option<Self>, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        // SAFETY: guarded by `TrustedLen` trait
        unsafe { BinaryArray::try_from_trusted_len_iter_unchecked(iter) }
    }
}
