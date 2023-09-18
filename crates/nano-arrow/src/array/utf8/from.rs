use std::iter::FromIterator;

use crate::offset::Offset;

use super::{MutableUtf8Array, Utf8Array};

impl<O: Offset, P: AsRef<str>> FromIterator<Option<P>> for Utf8Array<O> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<P>>>(iter: I) -> Self {
        MutableUtf8Array::<O>::from_iter(iter).into()
    }
}
