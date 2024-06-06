use arrow::array::{Array, BooleanArray};

/// Kernel to calculate the number of unique non-null elements
pub trait DistinctCountKernel {
    /// Calculate the number of unique non-null elements in [`Self`]
    fn distinct_count(&self) -> usize;
}

impl DistinctCountKernel for BooleanArray {
    fn distinct_count(&self) -> usize {
        let null_count = self.null_count();
        if self.len() - null_count == 0 {
            return 0;
        }

        if null_count == 0 {
            let unset_bits = self.values().unset_bits();
            2 - usize::from(unset_bits == 0 || unset_bits == self.values().len())
        } else {
            let set_bits = (self.values() & self.validity().unwrap()).set_bits();
            2 - usize::from(set_bits == 0 || set_bits == self.values().len() - null_count)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_distinct_count() {
        let test_cases = vec![
            (BooleanArray::from(vec![]), 0),
            (BooleanArray::from(vec![None, None]), 0),
            (BooleanArray::from(vec![Some(true), Some(true)]), 1),
            (BooleanArray::from(vec![Some(false), Some(false)]), 1),
            (BooleanArray::from(vec![Some(true), Some(false)]), 2),
            (BooleanArray::from(vec![Some(true), None]), 1),
            (BooleanArray::from(vec![Some(false), None]), 1),
            (BooleanArray::from(vec![Some(true), Some(false), None]), 2),
        ];

        for (input, expected) in test_cases {
            assert_eq!(input.distinct_count(), expected);
        }
    }
}
