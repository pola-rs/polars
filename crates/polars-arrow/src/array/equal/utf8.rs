use crate::array::Utf8Array;
use crate::offset::Offset;

pub(super) fn equal<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> bool {
    lhs.dtype() == rhs.dtype() && lhs.len() == rhs.len() && lhs.iter().eq(rhs.iter())
}
