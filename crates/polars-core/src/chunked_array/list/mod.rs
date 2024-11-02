//! Special list utility methods
pub(super) mod iterator;

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
        let arr = ca.downcast_iter().next().unwrap();

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
