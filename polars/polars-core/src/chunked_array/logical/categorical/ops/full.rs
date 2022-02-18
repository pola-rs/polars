use super::*;

impl CategoricalChunked {
    pub fn full_null(name: &str, length: usize) -> CategoricalChunked {
        // TODO! implement proper, can be faster
        use crate::chunked_array::categorical::CategoricalChunkedBuilder;
        let mut builder = CategoricalChunkedBuilder::new(name, length);
        let iter = (0..length).map(|_| None);
        builder.drain_iter(iter);
        builder.finish()
    }
}
