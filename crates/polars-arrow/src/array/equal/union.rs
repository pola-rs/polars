use crate::array::{Array, UnionArray};

pub(super) fn equal(lhs: &UnionArray, rhs: &UnionArray) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
