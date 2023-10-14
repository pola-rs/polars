use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::write_vec;
use super::FixedSizeBinaryArray;

pub fn write_value<W: Write>(array: &FixedSizeBinaryArray, index: usize, f: &mut W) -> Result {
    let values = array.value(index);
    let writer = |f: &mut W, index| write!(f, "{}", values[index]);

    write_vec(f, writer, None, values.len(), "None", false)
}

impl Debug for FixedSizeBinaryArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, f);

        write!(f, "{:?}", self.data_type)?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
