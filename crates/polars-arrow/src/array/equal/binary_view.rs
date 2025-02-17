use crate::array::binview::{BinaryViewArrayGeneric, ViewType};
use crate::array::Array;

pub(super) fn equal<T: ViewType + ?Sized>(
    lhs: &BinaryViewArrayGeneric<T>,
    rhs: &BinaryViewArrayGeneric<T>,
) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
