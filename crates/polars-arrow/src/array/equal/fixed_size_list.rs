use crate::array::{Array, FixedSizeListArray};

pub(super) fn equal(lhs: &FixedSizeListArray, rhs: &FixedSizeListArray) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
