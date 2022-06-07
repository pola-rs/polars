use super::*;
use crate::chunked_array::ops::append::new_chunks;
use crate::series::IsSorted;

impl CategoricalChunked {
    pub fn append(&mut self, other: &Self) -> Result<()> {
        let new_rev_map = self.merge_categorical_map(other)?;
        self.set_rev_map(new_rev_map, false);

        let len = self.len();
        new_chunks(&mut self.logical.chunks, &other.logical().chunks, len);
        self.logical.set_sorted2(IsSorted::Not);
        Ok(())
    }
}
