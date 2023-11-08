use polars_error::constants::LENGTH_LIMIT_MSG;

use super::*;
use crate::chunked_array::ops::append::new_chunks;
use crate::series::IsSorted;

impl CategoricalChunked {
    fn set_lengths(&mut self, other: &Self) {
        let length_self = &mut self.physical_mut().length;
        *length_self = length_self
            .checked_add(other.len() as IdxSize)
            .expect(LENGTH_LIMIT_MSG);
        self.physical_mut().null_count += other.null_count() as IdxSize;
    }

    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        if self.physical.null_count() == self.len() && other.physical.null_count() == other.len() {
            let len = self.len();
            self.set_lengths(other);
            new_chunks(&mut self.physical.chunks, &other.physical().chunks, len);
            return Ok(());
        }
        let is_local_different_source =
            match (self.get_rev_map().as_ref(), other.get_rev_map().as_ref()) {
                (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => !std::ptr::eq(arr_l, arr_r),
                _ => false,
            };

        if is_local_different_source {
            polars_bail!(string_cache_mismatch);
        } else {
            let len = self.len();
            let new_rev_map = self._merge_categorical_map(other)?;
            unsafe { self.set_rev_map(new_rev_map, false) };

            self.set_lengths(other);
            new_chunks(&mut self.physical.chunks, &other.physical().chunks, len);
        }
        self.physical.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}
