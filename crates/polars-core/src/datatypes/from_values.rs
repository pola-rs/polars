use std::borrow::Cow;
use std::error::Error;

use arrow::array::{
    BinaryArray, BooleanArray, MutableBinaryArray, MutableBinaryValuesArray, MutablePrimitiveArray,
    MutableUtf8Array, MutableUtf8ValuesArray, PrimitiveArray, Utf8Array,
};
use arrow::bitmap::Bitmap;
use polars_arrow::array::utf8::{BinaryFromIter, Utf8FromIter};
use polars_arrow::prelude::FromData;
use polars_arrow::trusted_len::TrustedLen;

use crate::datatypes::NumericNative;
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

    fn try_array_from_values_iter<E: Error, I: TrustedLen<Item = Result<Self, E>>>(
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
    fn try_array_from_values_iter<E: Error, I: TrustedLen<Item = Result<Self, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        // SAFETY: guarded by `TrustedLen` trait
        let values = unsafe { Bitmap::try_from_trusted_len_iter_unchecked(iter) }?;
        Ok(BooleanArray::from_data_default(values, None))
    }
}

impl<T> ArrayFromElementIter for T
where
    T: NumericNative,
{
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
        unsafe { Ok(MutablePrimitiveArray::try_from_trusted_len_iter_unchecked(iter)?.into()) }
    }
    fn try_array_from_values_iter<E: Error, I: TrustedLen<Item = Result<Self, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        let values: Vec<_> = iter.collect::<Result<Vec<_>, _>>()?;
        Ok(PrimitiveArray::from_vec(values))
    }
}

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
        let len = iter.size_hint().0;
        let mut mutable = MutableUtf8Array::<i64>::with_capacities(len, len * 24);
        mutable.extend_fallible(iter)?;
        Ok(mutable.into())
    }

    fn try_array_from_values_iter<E: Error, I: TrustedLen<Item = Result<Self, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        let len = iter.size_hint().0;
        let mut mutable = MutableUtf8ValuesArray::<i64>::with_capacities(len, len * 24);
        mutable.extend_fallible(iter)?;
        Ok(mutable.into())
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
        let len = iter.size_hint().0;
        let mut mutable = MutableUtf8Array::<i64>::with_capacities(len, len * 24);
        mutable.extend_fallible(iter)?;
        Ok(mutable.into())
    }

    fn try_array_from_values_iter<E: Error, I: TrustedLen<Item = Result<Self, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        let len = iter.size_hint().0;
        let mut mutable = MutableUtf8ValuesArray::<i64>::with_capacities(len, len * 24);
        mutable.extend_fallible(iter)?;
        Ok(mutable.into())
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
        let len = iter.size_hint().0;
        let mut mutable = MutableBinaryArray::<i64>::with_capacities(len, len * 24);
        mutable.extend_fallible(iter)?;
        Ok(mutable.into())
    }

    fn try_array_from_values_iter<E: Error, I: TrustedLen<Item = Result<Self, E>>>(
        iter: I,
    ) -> Result<Self::ArrayType, E> {
        let len = iter.size_hint().0;
        let mut mutable = MutableBinaryValuesArray::<i64>::with_capacities(len, len * 24);
        mutable.extend_fallible(iter)?;
        Ok(mutable.into())
    }
}
