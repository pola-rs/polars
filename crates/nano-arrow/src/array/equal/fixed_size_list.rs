use crate::array::{Array, FixedSizeListArray};

pub(super) fn equal(lhs: &FixedSizeListArray, rhs: &FixedSizeListArray) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
