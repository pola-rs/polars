//! Special list utility methods
pub(super) mod iterator;

use std::borrow::Cow;

use crate::prelude::*;

impl ListChunked {
    /// Get the inner data type of the list.
    pub fn inner_dtype(&self) -> &DataType {
        match self.dtype() {
            DataType::List(dt) => dt.as_ref(),
            _ => unreachable!(),
        }
    }

    pub fn set_inner_dtype(&mut self, dtype: DataType) {
        assert_eq!(dtype.to_physical(), self.inner_dtype().to_physical());
        let field = Arc::make_mut(&mut self.field);
        field.coerce(DataType::List(Box::new(dtype)));
    }

    pub fn set_fast_explode(&mut self) {
        self.set_fast_explode_list(true)
    }

    pub fn _can_fast_explode(&self) -> bool {
        self.get_fast_explode_list()
    }

    /// Set the logical type of the [`ListChunked`].
    ///
    /// # Safety
    /// The caller must ensure that the logical type given fits the physical type of the array.
    pub unsafe fn to_logical(&mut self, inner_dtype: DataType) {
        debug_assert_eq!(&inner_dtype.to_physical(), self.inner_dtype());
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::List(Box::new(inner_dtype)))
    }

    /// Convert the datatype of the list into the physical datatype.
    pub fn to_physical_repr(&self) -> Cow<ListChunked> {
        let Cow::Owned(physical_repr) = self.get_inner().to_physical_repr() else {
            return Cow::Borrowed(self);
        };

        let ca = if physical_repr.chunks().len() == 1 && self.chunks().len() > 1 {
            // Physical repr got rechunked, rechunk self as well.
            self.rechunk()
        } else {
            Cow::Borrowed(self)
        };

        assert_eq!(ca.chunks().len(), physical_repr.chunks().len());

        let chunks: Vec<_> = ca
            .downcast_iter()
            .zip(physical_repr.into_chunks())
            .map(|(chunk, values)| {
                LargeListArray::new(
                    ArrowDataType::LargeList(Box::new(ArrowField::new(
                        PlSmallStr::from_static("item"),
                        values.dtype().clone(),
                        true,
                    ))),
                    chunk.offsets().clone(),
                    values,
                    chunk.validity().cloned(),
                )
                .to_boxed()
            })
            .collect();

        let name = self.name().clone();
        let dtype = DataType::List(Box::new(self.inner_dtype().to_physical()));
        Cow::Owned(unsafe { ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Convert a non-logical [`ListChunked`] back into a logical [`ListChunked`] without casting.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn from_physical_unchecked(
        &self,
        to_inner_dtype: DataType,
    ) -> PolarsResult<ListChunked> {
        debug_assert!(!self.inner_dtype().is_logical());

        let inner_chunks = self
            .downcast_iter()
            .map(|chunk| chunk.values())
            .cloned()
            .collect();

        let inner = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                inner_chunks,
                self.inner_dtype(),
            )
        };
        let inner = unsafe { inner.from_physical_unchecked(&to_inner_dtype) }?;

        let chunks: Vec<_> = self
            .downcast_iter()
            .zip(inner.into_chunks())
            .map(|(chunk, values)| {
                LargeListArray::new(
                    ArrowDataType::LargeList(Box::new(ArrowField::new(
                        PlSmallStr::from_static("item"),
                        values.dtype().clone(),
                        true,
                    ))),
                    chunk.offsets().clone(),
                    values,
                    chunk.validity().cloned(),
                )
                .to_boxed()
            })
            .collect();

        let name = self.name().clone();
        let dtype = DataType::List(Box::new(to_inner_dtype));
        Ok(unsafe { ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Get the inner values as [`Series`], ignoring the list offsets.
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
    ) -> PolarsResult<ListChunked> {
        // generated Series will have wrong length otherwise.
        let ca = self.rechunk();
        let arr = ca.downcast_as_array();

        // SAFETY:
        // Inner dtype is passed correctly
        let elements = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                vec![arr.values().clone()],
                ca.inner_dtype(),
            )
        };

        let expected_len = elements.len();
        let out: Series = func(elements)?;
        polars_ensure!(
            out.len() == expected_len,
            ComputeError: "the function should apply element-wise, it removed elements instead"
        );
        let out = out.rechunk();
        let values = out.chunks()[0].clone();

        let inner_dtype = LargeListArray::default_datatype(values.dtype().clone());
        let arr = LargeListArray::new(
            inner_dtype,
            (*arr.offsets()).clone(),
            values,
            arr.validity().cloned(),
        );

        // SAFETY: arr's inner dtype is derived from out dtype.
        Ok(unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                ca.name().clone(),
                vec![Box::new(arr)],
                DataType::List(Box::new(out.dtype().clone())),
            )
        })
    }

    pub fn rechunk_and_trim_to_normalized_offsets(&self) -> Self {
        Self::with_chunk(
            self.name().clone(),
            self.rechunk()
                .downcast_get(0)
                .unwrap()
                .trim_to_normalized_offsets_recursive(),
        )
    }
}
