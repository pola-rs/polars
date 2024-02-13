use rayon::prelude::*;

use crate::prelude::*;

#[inline]
unsafe fn idx_to_str(idx: usize, arr: &Utf8ViewArray) -> Option<&str> {
    if arr.is_valid(idx) {
        Some(arr.value_unchecked(idx))
    } else {
        None
    }
}

impl StringChunked {
    pub fn par_iter_indexed(&self) -> impl IndexedParallelIterator<Item = Option<&str>> {
        assert_eq!(self.chunks.len(), 1);
        let arr = &*self.chunks[0];

        // SAFETY:
        // guarded by the type system
        let arr = unsafe { &*(arr as *const dyn Array as *const Utf8ViewArray) };
        (0..arr.len())
            .into_par_iter()
            .map(move |idx| unsafe { idx_to_str(idx, arr) })
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        self.chunks.par_iter().flat_map(move |arr| {
            // SAFETY:
            // guarded by the type system
            let arr = &**arr;
            let arr = unsafe { &*(arr as *const dyn Array as *const Utf8ViewArray) };
            (0..arr.len())
                .into_par_iter()
                .map(move |idx| unsafe { idx_to_str(idx, arr) })
        })
    }
}
