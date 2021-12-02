use super::*;

impl<T> ChunkLen for ChunkedArray<T> {
    /// Combined length of all the chunks.
    #[inline]
    fn len(&self) -> usize {
        match self.chunks.len() {
            // fast path
            1 => self.chunks[0].len(),
            _ => self.chunks.iter().fold(0, |acc, arr| acc + arr.len()),
        }
    }
}
