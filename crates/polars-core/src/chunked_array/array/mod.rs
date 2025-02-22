//! Special fixed-size-list utility methods

mod iterator;

use std::borrow::Cow;

use either::Either;

use crate::prelude::*;

impl ArrayChunked {
    /// Get the inner data type of the fixed size list.
    pub fn inner_dtype(&self) -> &DataType {
        match self.dtype() {
            DataType::Array(dt, _size) => dt.as_ref(),
            _ => unreachable!(),
        }
    }

    pub fn width(&self) -> usize {
        match self.dtype() {
            DataType::Array(_dt, size) => *size,
            _ => unreachable!(),
        }
    }

    /// # Safety
    /// The caller must ensure that the logical type given fits the physical type of the array.
    pub unsafe fn to_logical(&mut self, inner_dtype: DataType) {
        debug_assert_eq!(&inner_dtype.to_physical(), self.inner_dtype());
        let width = self.width();
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::Array(Box::new(inner_dtype), width))
    }

    /// Convert the datatype of the array into the physical datatype.
    pub fn to_physical_repr(&self) -> Cow<ArrayChunked> {
        let Cow::Owned(physical_repr) = self.get_inner().to_physical_repr() else {
            return Cow::Borrowed(self);
        };

        let chunk_len_validity_iter =
            if physical_repr.chunks().len() == 1 && self.chunks().len() > 1 {
                // Physical repr got rechunked, rechunk our validity as well.
                Either::Left(std::iter::once((self.len(), self.rechunk_validity())))
            } else {
                // No rechunking, expect the same number of chunks.
                assert_eq!(self.chunks().len(), physical_repr.chunks().len());
                Either::Right(
                    self.chunks()
                        .iter()
                        .map(|c| (c.len(), c.validity().cloned())),
                )
            };

        let width = self.width();
        let chunks: Vec<_> = chunk_len_validity_iter
            .zip(physical_repr.into_chunks())
            .map(|((len, validity), values)| {
                FixedSizeListArray::new(
                    ArrowDataType::FixedSizeList(
                        Box::new(ArrowField::new(
                            PlSmallStr::from_static("item"),
                            values.dtype().clone(),
                            true,
                        )),
                        width,
                    ),
                    len,
                    values,
                    validity,
                )
                .to_boxed()
            })
            .collect();

        let name = self.name().clone();
        let dtype = DataType::Array(Box::new(self.inner_dtype().to_physical()), width);
        Cow::Owned(unsafe { ArrayChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Convert a non-logical [`ArrayChunked`] back into a logical [`ArrayChunked`] without casting.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn from_physical_unchecked(&self, to_inner_dtype: DataType) -> PolarsResult<Self> {
        debug_assert!(!self.inner_dtype().is_logical());

        let chunks = self
            .downcast_iter()
            .map(|chunk| chunk.values())
            .cloned()
            .collect();

        let inner = unsafe {
            Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, chunks, self.inner_dtype())
        };
        let inner = unsafe { inner.from_physical_unchecked(&to_inner_dtype) }?;

        let chunks: Vec<_> = self
            .downcast_iter()
            .zip(inner.into_chunks())
            .map(|(chunk, values)| {
                FixedSizeListArray::new(
                    ArrowDataType::FixedSizeList(
                        Box::new(ArrowField::new(
                            PlSmallStr::from_static("item"),
                            values.dtype().clone(),
                            true,
                        )),
                        self.width(),
                    ),
                    chunk.len(),
                    values,
                    chunk.validity().cloned(),
                )
                .to_boxed()
            })
            .collect();

        let name = self.name().clone();
        let dtype = DataType::Array(Box::new(to_inner_dtype), self.width());
        Ok(unsafe { Self::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Get the inner values as `Series`
    pub fn get_inner(&self) -> Series {
        let chunks: Vec<_> = self.downcast_iter().map(|c| c.values().clone()).collect();

        // SAFETY: Data type of arrays matches because they are chunks from the same array.
        unsafe {
            Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, self.inner_dtype())
        }
    }

    /// Ignore the list indices and apply `func` to the inner type as [`Series`].
    pub fn apply_to_inner(
        &self,
        func: &dyn Fn(Series) -> PolarsResult<Series>,
    ) -> PolarsResult<ArrayChunked> {
        // Rechunk or the generated Series will have wrong length.
        let ca = self.rechunk();
        let field = self
            .inner_dtype()
            .to_arrow_field(PlSmallStr::from_static("item"), CompatLevel::newest());

        let chunks = ca.downcast_iter().map(|arr| {
            let elements = unsafe {
                Series::_try_from_arrow_unchecked_with_md(
                    self.name().clone(),
                    vec![(*arr.values()).clone()],
                    &field.dtype,
                    field.metadata.as_deref(),
                )
                .unwrap()
            };

            let expected_len = elements.len();
            let out: Series = func(elements)?;
            polars_ensure!(
                out.len() == expected_len,
                ComputeError: "the function should apply element-wise, it removed elements instead"
            );
            let out = out.rechunk();
            let values = out.chunks()[0].clone();

            let inner_dtype = FixedSizeListArray::default_datatype(
                out.dtype().to_arrow(CompatLevel::newest()),
                ca.width(),
            );
            let arr =
                FixedSizeListArray::new(inner_dtype, arr.len(), values, arr.validity().cloned());
            Ok(arr)
        });

        ArrayChunked::try_from_chunk_iter(self.name().clone(), chunks)
    }

    /// Recurse nested types until we are at the leaf array.
    pub fn get_leaf_array(&self) -> Series {
        let mut current = self.get_inner();
        while let Some(child_array) = current.try_array() {
            current = child_array.get_inner();
        }
        current
    }
}
