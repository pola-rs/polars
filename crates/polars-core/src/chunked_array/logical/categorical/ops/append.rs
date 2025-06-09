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

    pub fn take(&mut self) -> Self {
        Self {
            physical: Logical {
                phys: core::mem::take(&mut self.physical.phys),
                dtype: self.physical.dtype.clone(),
                _phantom: PhantomData,
            },
            bit_settings: self.bit_settings.clone(),
        }
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
        // @TODO: Move the implementation to append_owned and make append dispatch here.
        self.append(&other)
    }
}
