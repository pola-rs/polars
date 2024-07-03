use arrow::array::Array;

/// Kernel to calculate the number of unique elements where the elements are already sorted.
pub trait SortedUniqueKernel: Array {
    /// Calculate the set of unique elements in `fst` and `others` and fold the result into one
    /// array.
    fn unique_fold<'a>(fst: &'a Self, others: impl Iterator<Item = &'a Self>) -> Self;

    /// Calculate the set of unique elements in [`Self`] where we have no further information about
    /// `self`.
    fn unique(&self) -> Self;

    /// Calculate the number of unique elements in [`Self`]
    ///
    /// A null is also considered a unique value
    fn n_unique(&self) -> usize;

    /// Calculate the number of unique non-null elements in [`Self`]
    fn n_unique_non_null(&self) -> usize;
}

/// Optimized kernel to calculate the unique elements of an array.
///
/// This kernel is a specialized for where all values are known to be in some small range of
/// values. In this case, you can usually get by with a bitset and bit-arithmetic instead of using
/// vectors and hashsets. Consequently, this kernel is usually called when further information is
/// known about the underlying array.
///
/// This trait is not implemented directly on the `Array` as with many other kernels. Rather, it is
/// implemented on a `State` struct to which `Array`s can be appended. This allows for sharing of
/// `State` between many chunks and allows for different implementations for the same array (e.g. a
/// maintain order and no maintain-order variant).
pub trait RangedUniqueKernel {
    type Array: Array;

    /// Returns whether all the values in the whole range are in the state
    fn has_seen_all(&self) -> bool;

    /// Append an `Array`'s values to the `State`
    fn append(&mut self, array: &Self::Array);

    /// Consume the state to get the unique elements
    fn finalize_unique(self) -> Self::Array;
    /// Consume the state to get the number of unique elements including null
    fn finalize_n_unique(self) -> usize;
    /// Consume the state to get the number of unique elements excluding null
    fn finalize_n_unique_non_null(self) -> usize;
}

/// A generic unique kernel that selects the generally applicable unique kernel for an `Array`.
pub trait GenericUniqueKernel {
    /// Calculate the set of unique elements
    fn unique(&self) -> Self;
    /// Calculate the number of unique elements including null
    fn n_unique(&self) -> usize;
    /// Calculate the number of unique elements excluding null
    fn n_unique_non_null(&self) -> usize;
}

mod boolean;
mod primitive;

pub use boolean::BooleanUniqueKernelState;
pub use primitive::PrimitiveRangedUniqueState;
