//! Contains the operator [`limit`].

use crate::array::Array;

/// Returns the [`Array`] limited by `num_elements`.
///
/// Limit performs a zero-copy slice of the array, and is a convenience method on slice
/// where:
/// * it performs a bounds-check on the array
/// * it slices from offset 0
pub fn limit(array: &dyn Array, num_elements: usize) -> Box<dyn Array> {
    let lim = num_elements.min(array.len());
    array.sliced(0, lim)
}
