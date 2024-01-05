//! Special fixed-size-list utility methods

mod iterator;

use crate::prelude::*;

impl ArrayChunked {
    /// Get the inner data type of the fixed size list.
    pub fn inner_dtype(&self) -> DataType {
        match self.dtype() {
            DataType::Array(dt, _size) => *dt.clone(),
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
        debug_assert_eq!(inner_dtype.to_physical(), self.inner_dtype());
        let width = self.width();
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::Array(Box::new(inner_dtype), width))
    }

    /// Get the inner values as `Series`
    pub fn get_inner(&self) -> Series {
        let ca = self.rechunk();
        let field = self.inner_dtype().to_arrow_field(self.name(), true);
        let arr = ca.downcast_iter().next().unwrap();
        unsafe { Series::_try_from_arrow_unchecked(&field, vec![(arr.values()).clone()]).unwrap() }
    }

    /// Ignore the list indices and apply `func` to the inner type as [`Series`].
    pub fn apply_to_inner(
        &self,
        func: &dyn Fn(Series) -> PolarsResult<Series>,
    ) -> PolarsResult<ArrayChunked> {
        // Rechunk or the generated Series will have wrong length.
        let ca = self.rechunk();

        let field = self.inner_dtype().to_arrow_field(self.name(), true);

        let chunks = ca.downcast_iter().map(|arr| {
            let elements = unsafe {
                Series::_try_from_arrow_unchecked(&field, vec![(*arr.values()).clone()]).unwrap()
            };

            let expected_len = elements.len();
            let out: Series = func(elements)?;
            polars_ensure!(
                out.len() == expected_len,
                ComputeError: "the function should apply element-wise, it removed elements instead"
            );
            let out = out.rechunk();
            let values = out.chunks()[0].clone();

            let inner_dtype =
                FixedSizeListArray::default_datatype(out.dtype().to_arrow(true), ca.width());
            let arr = FixedSizeListArray::new(inner_dtype, values, arr.validity().cloned());
            Ok(arr)
        });

        ArrayChunked::try_from_chunk_iter(self.name(), chunks)
    }
}
