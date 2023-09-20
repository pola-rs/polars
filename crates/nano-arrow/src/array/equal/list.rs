use crate::array::{Array, ListArray};
use crate::offset::Offset;

pub(super) fn equal<O: Offset>(lhs: &ListArray<O>, rhs: &ListArray<O>) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
