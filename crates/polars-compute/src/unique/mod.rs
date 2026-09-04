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
    /// The array of `polars-array` whose elements are appended to the state, and which the unique
    /// ones come back in.
    type Array;

    /// Returns whether all the values in the whole range are in the state
    fn has_seen_all(&self) -> bool;

    /// Append an `Array`'s values to the `State`
    fn append(&mut self, array: &Self::Array);

    /// Append another state into the `State`
    fn append_state(&mut self, other: &Self);

    /// Consume the state to get the unique elements
    fn finalize_unique(self) -> Self::Array;
    /// Consume the state to get the number of unique elements including null
    fn finalize_n_unique(&self) -> usize;
    /// Consume the state to get the number of unique elements excluding null
    fn finalize_n_unique_non_null(&self) -> usize;
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
mod distinct;

pub use boolean::BooleanUniqueKernelState;
pub use distinct::{AmortizedUnique, amortized_unique_like};
