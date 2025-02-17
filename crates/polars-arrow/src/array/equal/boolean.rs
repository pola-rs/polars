use crate::array::BooleanArray;

pub(super) fn equal(lhs: &BooleanArray, rhs: &BooleanArray) -> bool {
    lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
