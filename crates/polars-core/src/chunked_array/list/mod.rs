//! Special list utility methods
pub(super) mod iterator;

use crate::chunked_array::Settings;
use crate::prelude::*;

impl ListChunked {
    /// Get the inner data type of the list.
    pub fn inner_dtype(&self) -> DataType {
        match self.dtype() {
            DataType::List(dt) => *dt.clone(),
            _ => unreachable!(),
        }
    }

    pub fn set_inner_dtype(&mut self, dtype: DataType) {
        assert_eq!(dtype.to_physical(), self.inner_dtype().to_physical());
        let field = Arc::make_mut(&mut self.field);
        field.coerce(DataType::List(Box::new(dtype)));
    }
    pub fn set_fast_explode(&mut self) {
        self.bit_settings.insert(Settings::FAST_EXPLODE_LIST)
    }
    pub(crate) fn unset_fast_explode(&mut self) {
        self.bit_settings.remove(Settings::FAST_EXPLODE_LIST)
    }

    pub fn _can_fast_explode(&self) -> bool {
        self.bit_settings.contains(Settings::FAST_EXPLODE_LIST)
    }

    /// Set the logical type of the [`ListChunked`].
    ///
    /// # Safety
    /// The caller must ensure that the logical type given fits the physical type of the array.
    pub unsafe fn to_logical(&mut self, inner_dtype: DataType) {
        debug_assert_eq!(inner_dtype.to_physical(), self.inner_dtype());
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::List(Box::new(inner_dtype)))
    }

    /// Get the inner values as [`Series`], ignoring the list offsets.
    pub fn get_inner(&self) -> Series {
        let chunks: Vec<_> = self.downcast_iter().map(|c| c.values().clone()).collect();

        // SAFETY: Data type of arrays matches because they are chunks from the same array.
        unsafe { Series::from_chunks_and_dtype_unchecked(self.name(), chunks, &self.inner_dtype()) }
    }

    /// Returns an iterator over the offsets of this chunked array.
    ///
    /// The offsets are returned as though the array consisted of a single chunk.
    pub fn iter_offsets(&self) -> impl Iterator<Item = i64> + '_ {
        let mut offsets = self.downcast_iter().map(|arr| arr.offsets());

        let offsets_first_chunk = offsets.next().unwrap();

        // Make sure the offsets in the other chunks continue where the previous chunk left off.
        let mut prev = *offsets_first_chunk.last();
        let offsets_rest = offsets.flat_map(move |buf| {
            let adjust = prev - buf.first();
            prev = buf.last() + adjust;
            buf.iter().map(move |v| v + adjust).skip(1)
        });

        offsets_first_chunk.iter().copied().chain(offsets_rest)
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
                self.name(),
                vec![arr.values().clone()],
                &ca.inner_dtype(),
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

        let inner_dtype = LargeListArray::default_datatype(values.data_type().clone());
        let arr = LargeListArray::new(
            inner_dtype,
            (*arr.offsets()).clone(),
            values,
            arr.validity().cloned(),
        );

        // SAFETY: arr's inner dtype is derived from out dtype.
        Ok(unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                ca.name(),
                vec![Box::new(arr)],
                DataType::List(Box::new(out.dtype().clone())),
            )
        })
    }
}
