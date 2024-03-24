use super::{BinaryArray, MutableBinaryArray};
use crate::offset::Offset;

impl<O: Offset, P: AsRef<[u8]>> FromIterator<Option<P>> for BinaryArray<O> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<P>>>(iter: I) -> Self {
        MutableBinaryArray::<O>::from_iter(iter).into()
    }
}
