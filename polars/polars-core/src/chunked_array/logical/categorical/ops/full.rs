use super::*;

impl CategoricalChunked {
    pub fn full_null(name: &str, length: usize) -> CategoricalChunked {
        let cats = UInt32Chunked::full_null(name, length);
        CategoricalChunked::from_cats_and_rev_map(cats, Arc::new(RevMapping::default()))
    }
}
