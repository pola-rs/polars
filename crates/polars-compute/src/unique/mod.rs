use arrow::array::Array;

/// Kernel to calculate the number of unique elements
pub trait UniqueKernel: Array {
    /// Calculate the set of unique elements in `fst` and `others` and fold the result into one
    /// array.
    fn unique_fold<'a>(fst: &'a Self, others: impl Iterator<Item = &'a Self>) -> Self;

    /// Calculate the set of unique elements in [`Self`] where we have no further information about
    /// `self`.
    fn unique(&self) -> Self;

    /// Calculate the set of unique elements in [`Self`] where `self` is sorted.
    fn unique_sorted(&self) -> Self;

    /// Calculate the number of unique elements in [`Self`]
    ///
    /// A null is also considered a unique value
    fn n_unique(&self) -> usize;

    /// Calculate the number of unique non-null elements in [`Self`]
    fn n_unique_non_null(&self) -> usize;
}

mod boolean;
