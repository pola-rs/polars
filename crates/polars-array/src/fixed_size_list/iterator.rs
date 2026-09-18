//! The iterators of a [`PlFixedSizeListArray`](super::PlFixedSizeListArray).

use crate::nested::{NestedIter, NestedValuesIter, Stride};

/// Iterator over the elements of a fixed size list array, ignoring validity.
pub type PlFixedSizeListValuesIter<'a> = NestedValuesIter<'a, Stride>;

/// Iterator over the optional elements of a fixed size list array.
pub type PlFixedSizeListIter<'a> = NestedIter<'a, Stride>;
