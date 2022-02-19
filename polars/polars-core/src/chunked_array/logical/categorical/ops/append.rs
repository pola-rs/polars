use super::*;
use crate::chunked_array::ops::append::new_chunks;

impl CategoricalChunked {
    pub fn append(&mut self, other: &Self) -> Result<()> {
        let (rev_map_l, rev_map_r) = (self.get_rev_map(), other.get_rev_map());
        // first assertion checks if the global string cache is equal,
        // the second checks if we append a slice from this array to self
        if !rev_map_l.same_src(rev_map_r) && !Arc::ptr_eq(rev_map_l, rev_map_r) {
            return Err(PolarsError::ComputeError(
                "Appending categorical data can only be done if they are made under the same global string cache. \
            Consider using a global string cache.".into()
            ));
        }

        let new_rev_map = self.merge_categorical_map(other);
        self.set_rev_map(new_rev_map, false);

        let len = self.len();
        new_chunks(&mut self.logical.chunks, &other.logical().chunks, len);
        Ok(())
    }
}
