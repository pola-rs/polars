use rayon::prelude::*;

use crate::prelude::*;

#[inline]
unsafe fn idx_to_str(idx: usize, arr: &PlUtf8ViewArray) -> Option<&str> {
    if arr.is_valid(idx) {
        Some(arr.value_unchecked(idx))
    } else {
        None
    }
}

impl StringChunked {
    pub fn par_iter_indexed(&self) -> impl IndexedParallelIterator<Item = Option<&str>> {
        assert_eq!(self.chunks.len(), 1);
        // SAFETY: the elements of a `StringChunked` are valid UTF-8.
        let arr = unsafe {
            PlUtf8ViewArray::from_binview_ref_unchecked(
                self.chunks[0]
                    .as_any()
                    .downcast_ref::<PlBinaryViewArray>()
                    .unwrap(),
            )
        };
        (0..arr.len())
            .into_par_iter()
            .map(move |idx| unsafe { idx_to_str(idx, arr) })
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        self.chunks.par_iter().flat_map(move |arr| {
            // SAFETY: the elements of a `StringChunked` are valid UTF-8.
            let arr = unsafe {
                PlUtf8ViewArray::from_binview_ref_unchecked(
                    arr.as_any().downcast_ref::<PlBinaryViewArray>().unwrap(),
                )
            };
            (0..arr.len())
                .into_par_iter()
                .map(move |idx| unsafe { idx_to_str(idx, arr) })
        })
    }
}
