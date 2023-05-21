use std::ptr::NonNull;

use super::*;
use crate::chunked_array::list::iterator::AmortizedListIter;
use crate::series::unstable::ArrayBox;

impl FixedSizeListChunked {
    /// This is an iterator over a ListChunked that save allocations.
    /// A Series is:
    ///     1. Arc<ChunkedArray>
    ///     ChunkedArray is:
    ///         2. Vec< 3. ArrayRef>
    ///
    /// The ArrayRef we indicated with 3. will be updated during iteration.
    /// The Series will be pinned in memory, saving an allocation for
    /// 1. Arc<..>
    /// 2. Vec<...>
    ///
    /// # Warning
    /// Though memory safe in the sense that it will not read unowned memory, UB, or memory leaks
    /// this function still needs precautions. The returned should never be cloned or taken longer
    /// than a single iteration, as every call on `next` of the iterator will change the contents of
    /// that Series.
    #[cfg(feature = "private")]
    pub fn amortized_iter(&self) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        self.amortized_iter_with_name("")
    }

    #[cfg(feature = "private")]
    pub fn amortized_iter_with_name(
        &self,
        name: &str,
    ) -> AmortizedListIter<impl Iterator<Item = Option<ArrayBox>> + '_> {
        // we create the series container from the inner array
        // so that the container has the proper dtype.
        let arr = self.downcast_iter().next().unwrap();
        let inner_values = arr.values();

        let inner_dtype = self.inner_dtype();
        let iter_dtype = match inner_dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => inner_dtype.to_physical(),
            // TODO: figure out how to deal with physical/logical distinction
            // physical primitives like time, date etc. work
            // physical nested need more
            _ => inner_dtype.clone(),
        };

        // Safety:
        // inner type passed as physical type
        let series_container = unsafe {
            Box::new(Series::from_chunks_and_dtype_unchecked(
                name,
                vec![inner_values.clone()],
                &iter_dtype,
            ))
        };

        let ptr = series_container.array_ref(0) as *const ArrayRef as *mut ArrayRef;

        AmortizedListIter::new(
            self.len(),
            series_container,
            NonNull::new(ptr).unwrap(),
            self.downcast_iter().flat_map(|arr| arr.iter()),
            inner_dtype,
        )
    }
}
