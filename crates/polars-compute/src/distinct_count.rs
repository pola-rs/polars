use arrow::array::{Array, BooleanArray};

/// Kernel to calculate the number of unique non-null elements
pub trait DistinctCountKernel {
    /// Calculate the number of unique non-null elements in [`Self`]
    fn distinct_count(&self) -> usize;
}

impl DistinctCountKernel for BooleanArray {
    fn distinct_count(&self) -> usize {
        if self.len() - self.null_count() == 0 {
            return 0;
        }

        let unset_bits = self.values().unset_bits();
        2 - usize::from(unset_bits == 0 || unset_bits == self.values().len())
    }
}
