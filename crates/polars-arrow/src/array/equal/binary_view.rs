use crate::array::Array;
use crate::array::binview::{BinaryViewArrayGeneric, ViewType};

pub(super) fn equal<T: ViewType + ?Sized>(
    lhs: &BinaryViewArrayGeneric<T>,
    rhs: &BinaryViewArrayGeneric<T>,
) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
