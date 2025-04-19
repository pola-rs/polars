use crate::array::BinaryArray;
use crate::offset::Offset;

pub(super) fn equal<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
