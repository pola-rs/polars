use super::*;

impl CategoricalChunked {
    pub fn full_null(
        name: &str,
        is_enum: bool,
        length: usize,
        ordering: CategoricalOrdering,
    ) -> CategoricalChunked {
        let cats = UInt32Chunked::full_null(name, length);

        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                cats,
                Arc::new(RevMapping::default()),
                is_enum,
                ordering,
            )
        }
    }
}
