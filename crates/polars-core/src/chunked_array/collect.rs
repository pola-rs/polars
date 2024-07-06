//! Methods for collecting into a ChunkedArray.
//!
//! For types that don't have dtype parameters:
//! iter.(try_)collect_ca(_trusted) (name)
//!
//! For all types:
//! iter.(try_)collect_ca(_trusted)_like (other_df)  Copies name/dtype from other_df
//! iter.(try_)collect_ca(_trusted)_with_dtype (name, df)
//!
//! The try variants work on iterators of Results, the trusted variants do not
//! check the length of the iterator.

use std::sync::Arc;

use arrow::trusted_len::TrustedLen;

use crate::chunked_array::ChunkedArray;
use crate::datatypes::{
    ArrayCollectIterExt, ArrayFromIter, ArrayFromIterDtype, DataType, Field, PolarsDataType,
};
use crate::prelude::CompatLevel;

pub trait ChunkedCollectIterExt<T: PolarsDataType>: Iterator + Sized {
    #[inline]
    fn collect_ca_with_dtype(self, name: &str, dtype: DataType) -> ChunkedArray<T>
    where
        T::Array: ArrayFromIterDtype<Self::Item>,
    {
        let field = Arc::new(Field::new(name, dtype.clone()));
        let arr = self.collect_arr_with_dtype(field.dtype.to_arrow(CompatLevel::newest()));
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    #[inline]
    fn collect_ca_like(self, name_dtype_src: &ChunkedArray<T>) -> ChunkedArray<T>
    where
        T::Array: ArrayFromIterDtype<Self::Item>,
    {
        let field = Arc::clone(&name_dtype_src.field);
        let arr = self.collect_arr_with_dtype(field.dtype.to_arrow(CompatLevel::newest()));
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    #[inline]
    fn collect_ca_trusted_with_dtype(self, name: &str, dtype: DataType) -> ChunkedArray<T>
    where
        T::Array: ArrayFromIterDtype<Self::Item>,
        Self: TrustedLen,
    {
        let field = Arc::new(Field::new(name, dtype.clone()));
        let arr = self.collect_arr_trusted_with_dtype(field.dtype.to_arrow(CompatLevel::newest()));
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    #[inline]
    fn collect_ca_trusted_like(self, name_dtype_src: &ChunkedArray<T>) -> ChunkedArray<T>
    where
        T::Array: ArrayFromIterDtype<Self::Item>,
        Self: TrustedLen,
    {
        let field = Arc::clone(&name_dtype_src.field);
        let arr = self.collect_arr_trusted_with_dtype(field.dtype.to_arrow(CompatLevel::newest()));
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    #[inline]
    fn try_collect_ca_with_dtype<U, E>(
        self,
        name: &str,
        dtype: DataType,
    ) -> Result<ChunkedArray<T>, E>
    where
        T::Array: ArrayFromIterDtype<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        let field = Arc::new(Field::new(name, dtype.clone()));
        let arr = self.try_collect_arr_with_dtype(field.dtype.to_arrow(CompatLevel::newest()))?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_collect_ca_like<U, E>(
        self,
        name_dtype_src: &ChunkedArray<T>,
    ) -> Result<ChunkedArray<T>, E>
    where
        T::Array: ArrayFromIterDtype<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        let field = Arc::clone(&name_dtype_src.field);
        let arr = self.try_collect_arr_with_dtype(field.dtype.to_arrow(CompatLevel::newest()))?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_collect_ca_trusted_with_dtype<U, E>(
        self,
        name: &str,
        dtype: DataType,
    ) -> Result<ChunkedArray<T>, E>
    where
        T::Array: ArrayFromIterDtype<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        let field = Arc::new(Field::new(name, dtype.clone()));
        let arr =
            self.try_collect_arr_trusted_with_dtype(field.dtype.to_arrow(CompatLevel::newest()))?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_collect_ca_trusted_like<U, E>(
        self,
        name_dtype_src: &ChunkedArray<T>,
    ) -> Result<ChunkedArray<T>, E>
    where
        T::Array: ArrayFromIterDtype<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        let field = Arc::clone(&name_dtype_src.field);
        let arr =
            self.try_collect_arr_trusted_with_dtype(field.dtype.to_arrow(CompatLevel::newest()))?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }
}

impl<T: PolarsDataType, I: Iterator> ChunkedCollectIterExt<T> for I {}

pub trait ChunkedCollectInferIterExt<T: PolarsDataType>: Iterator + Sized {
    #[inline]
    fn collect_ca(self, name: &str) -> ChunkedArray<T>
    where
        T::Array: ArrayFromIter<Self::Item>,
    {
        let field = Arc::new(Field::new(name, T::get_dtype()));
        let arr = self.collect_arr();
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    #[inline]
    fn collect_ca_trusted(self, name: &str) -> ChunkedArray<T>
    where
        T::Array: ArrayFromIter<Self::Item>,
        Self: TrustedLen,
    {
        let field = Arc::new(Field::new(name, T::get_dtype()));
        let arr = self.collect_arr_trusted();
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    #[inline]
    fn try_collect_ca<U, E>(self, name: &str) -> Result<ChunkedArray<T>, E>
    where
        T::Array: ArrayFromIter<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        let field = Arc::new(Field::new(name, T::get_dtype()));
        let arr = self.try_collect_arr()?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_collect_ca_trusted<U, E>(self, name: &str) -> Result<ChunkedArray<T>, E>
    where
        T::Array: ArrayFromIter<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        let field = Arc::new(Field::new(name, T::get_dtype()));
        let arr = self.try_collect_arr_trusted()?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }
}

impl<T: PolarsDataType, I: Iterator> ChunkedCollectInferIterExt<T> for I {}
