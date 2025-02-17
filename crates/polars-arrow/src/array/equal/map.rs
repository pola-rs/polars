use crate::array::{Array, MapArray};

pub(super) fn equal(lhs: &MapArray, rhs: &MapArray) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
