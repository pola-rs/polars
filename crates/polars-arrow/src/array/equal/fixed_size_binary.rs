use crate::array::{Array, FixedSizeBinaryArray};

pub(super) fn equal(lhs: &FixedSizeBinaryArray, rhs: &FixedSizeBinaryArray) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
