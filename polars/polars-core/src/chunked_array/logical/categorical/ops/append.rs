use super::*;
use crate::chunked_array::ops::append::new_chunks;
use crate::series::IsSorted;

impl CategoricalChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        if self.logical.null_count() == self.len() && other.logical.null_count() == other.len() {
            let len = self.len();
            self.logical_mut().length += other.len() as IdxSize;
            new_chunks(&mut self.logical.chunks, &other.logical().chunks, len);
            return Ok(());
        }
        let is_local_different_source =
            match (self.get_rev_map().as_ref(), other.get_rev_map().as_ref()) {
                (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => !std::ptr::eq(arr_l, arr_r),
                _ => false,
            };

        if is_local_different_source {
            return Err(PolarsError::ComputeError("Cannot concat Categoricals coming from a different source. Consider setting a global StringCache.".into()));
        } else {
            let len = self.len();
            let new_rev_map = self.merge_categorical_map(other)?;
            unsafe { self.set_rev_map(new_rev_map, false) };

            self.logical_mut().length += other.len() as IdxSize;
            new_chunks(&mut self.logical.chunks, &other.logical().chunks, len);
        }
        self.logical.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}
