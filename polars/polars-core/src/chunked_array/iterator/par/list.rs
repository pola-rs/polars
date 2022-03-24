use crate::prelude::*;
use rayon::prelude::*;

unsafe fn idx_to_array(idx: usize, arr: &ListArray<i64>, dtype: &DataType) -> Option<Series> {
    if arr.is_valid(idx) {
        Some(Arc::from(arr.value_unchecked(idx)))
            .map(|arr: ArrayRef| Series::from_chunks_and_dtype_unchecked("", vec![arr], dtype))
    } else {
        None
    }
}

impl ListChunked {
    // Get a parallel iterator over the [`Series`] in this [`ListChunked`].
    pub fn par_iter(&self) -> impl ParallelIterator<Item = Option<Series>> + '_ {
        self.chunks
            .par_iter()
            .map(move |arr| {
                let dtype = self.inner_dtype();
                // Safety:
                // guarded by the type system
                let arr = &**arr;
                let arr = unsafe { &*(arr as *const dyn Array as *const ListArray<i64>) };
                (0..arr.len())
                    .into_par_iter()
                    .map(move |idx| unsafe { idx_to_array(idx, arr, &dtype) })
            })
            .flatten()
    }
}
