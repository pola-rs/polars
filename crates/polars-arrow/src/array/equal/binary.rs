use crate::array::BinaryArray;
use crate::offset::Offset;

pub(super) fn equal<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
