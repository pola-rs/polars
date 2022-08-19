use super::*;

impl CategoricalChunked {
    pub fn full_null(name: &str, length: usize) -> CategoricalChunked {
        let cats = UInt32Chunked::full_null(name, length);

        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                cats,
                Arc::new(RevMapping::default()),
            )
        }
    }
}
