use polars_error::constants::LENGTH_LIMIT_MSG;

use super::*;
use crate::chunked_array::ops::append::new_chunks;

struct CategoricalAppend;

impl CategoricalMergeOperation for CategoricalAppend {
    fn finish(self, lhs: &UInt32Chunked, rhs: &UInt32Chunked) -> PolarsResult<UInt32Chunked> {
        let mut lhs_mut = lhs.clone();
        lhs_mut.append(rhs)?;
        Ok(lhs_mut)
    }
}

impl CategoricalChunked {
    fn set_lengths(&mut self, other: &Self) {
        let length_self = &mut self.physical_mut().length;
        *length_self = length_self
            .checked_add(other.len())
            .expect(LENGTH_LIMIT_MSG);

        assert!(
            IdxSize::try_from(*length_self).is_ok(),
            "{}",
            LENGTH_LIMIT_MSG
        );
        self.physical_mut().null_count += other.null_count();
    }

    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        polars_ensure!(!self.is_enum() || self.dtype() == other.dtype(), append);

        // fast path all nulls
        if self.physical.null_count() == self.len() && other.physical.null_count() == other.len() {
            let len = self.len();
            self.set_lengths(other);
            new_chunks(&mut self.physical.chunks, &other.physical().chunks, len);
            return Ok(());
        }

        if self.is_enum() {
            self.physical_mut().append(other.physical())?;
        } else {
            let mut new_self = call_categorical_merge_operation(self, other, CategoricalAppend)?;
            std::mem::swap(self, &mut new_self);
        }
        Ok(())
    }

    pub fn append_owned(&mut self, other: Self) -> PolarsResult<()> {
        if (self.is_enum() && other.is_enum())
            || (self.get_rev_map().is_global() && other.get_rev_map().is_global())
        {
            if self.get_rev_map().is_global() {
                let mut rev_map_merger = GlobalRevMapMerger::new(self.get_rev_map().clone());
                rev_map_merger.merge_map(other.get_rev_map())?;

                // SAFETY: We just merged the revmaps and this is the global one, so it the indices
                // are the same.
                unsafe { self.set_rev_map(rev_map_merger.finish(), false) };
            }

            // In these cases we can just append the physicals. We don't have to do any categorical
            // merging or anything.
            self.physical_mut().append_owned(other.into_physical())?;
            Ok(())
        } else {
            self.append(&other)
        }
    }
}
