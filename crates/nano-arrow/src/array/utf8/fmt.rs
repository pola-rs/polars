use std::fmt::{Debug, Formatter, Result, Write};

use crate::offset::Offset;

use super::super::fmt::write_vec;
use super::Utf8Array;

pub fn write_value<O: Offset, W: Write>(array: &Utf8Array<O>, index: usize, f: &mut W) -> Result {
    write!(f, "{}", array.value(index))
}

impl<O: Offset> Debug for Utf8Array<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, f);

        let head = if O::IS_LARGE {
            "LargeUtf8Array"
        } else {
            "Utf8Array"
        };
        write!(f, "{head}")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
