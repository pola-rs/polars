use crate::chunked_array::ChunkedArray;

#[macro_use]
mod macros;
pub mod boolean;
pub mod list;
pub mod numeric;
pub mod utf8;

impl<T> ChunkedArray<T> {
    /// Helper function for parallel iterators. It computes the chunk index and the index inside that chunk, for right
    /// indexes in many chunk parallel iterators.
    ///
    /// An iterator has two indexes, a left and a right index, which do not overlap as if both indexes point to the
    /// same index means that the iterator is consumed.
    /// In order to avoid overlapping the right index is always one position ahead of the last chunk index, then:
    /// - left chunk indexes: goes from 0 to `chunk.len() - 1`.
    /// - right chunk indexes: goes from 1 to `chunk.len()`.
    ///
    /// The goal of this function is to translate an index into the expected one by a right index. That means, that
    /// any right index pointing to index 0 in a chunk  different from the first it is translated to `chunk.len()`
    /// of the previous chunk.
    ///
    /// # Index
    ///
    /// right_index: The position of the right index to translate into `right_chunk_index` and `index_in_right_chunk`.
    ///    Applying the needed conversions for right indexes.
    ///
    /// # Returns
    ///
    /// A tuple `(right_chunk_index, index_in_right_chunk)`.
    /// - right_chunk_index: The chunk in which is located the `right_index`.
    /// - index_in_right_chunk: The index in the chunk where is located the `right_index`.
    fn right_index_to_chunked_index(&self, right_index: usize) -> (usize, usize) {
        let (chunk_idx, index_remainder) = self.index_to_chunked_index(right_index);
        if index_remainder == 0 && right_index > 0 {
            let chunk_idx = chunk_idx - 1;
            let index_remainder = self.chunks[chunk_idx].len();
            (chunk_idx, index_remainder)
        } else {
            (chunk_idx, index_remainder)
        }
    }
}
