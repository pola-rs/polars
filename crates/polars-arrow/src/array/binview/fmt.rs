use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::write_vec;
use super::BinaryViewArrayGeneric;
use crate::array::binview::ViewType;
use crate::array::{Array, BinaryViewArray, Utf8ViewArray};

pub fn write_value<'a, T: ViewType + ?Sized, W: Write>(
    array: &'a BinaryViewArrayGeneric<T>,
    index: usize,
    f: &mut W,
) -> Result
where
    &'a T: Debug,
{
    let bytes = array.value(index).to_bytes();
    let writer = |f: &mut W, index| write!(f, "{}", bytes[index]);

    write_vec(f, writer, None, bytes.len(), "None", false)
}

impl Debug for BinaryViewArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, f);
        write!(f, "BinaryViewArray")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}

impl Debug for Utf8ViewArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write!(f, "{}", self.value(index));
        write!(f, "Utf8ViewArray")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
