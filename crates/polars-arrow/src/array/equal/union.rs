use crate::array::{Array, UnionArray};

pub(super) fn equal(lhs: &UnionArray, rhs: &UnionArray) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
