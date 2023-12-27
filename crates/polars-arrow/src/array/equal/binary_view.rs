use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::array::Array;

pub(super) fn equal<T: ViewType>(
    lhs: &BinaryViewArrayGeneric<T>,
    rhs: &BinaryViewArrayGeneric<T>,
) -> bool {
    lhs.data_type() == rhs.data_type() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
