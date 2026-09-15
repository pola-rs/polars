//! The iterators of a [`PlListArray`](super::PlListArray).

use crate::nested::{NestedIter, NestedValuesIter, Offsets};

/// Iterator over the elements of a [`PlListArray`](super::PlListArray), ignoring validity.
pub type PlListValuesIter<'a> = NestedValuesIter<'a, Offsets<'a>>;

/// Iterator over the optional elements of a [`PlListArray`](super::PlListArray).
pub type PlListIter<'a> = NestedIter<'a, Offsets<'a>>;
