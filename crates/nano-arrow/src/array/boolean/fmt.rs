use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::write_vec;
use super::BooleanArray;

pub fn write_value<W: Write>(array: &BooleanArray, index: usize, f: &mut W) -> Result {
    write!(f, "{}", array.value(index))
}

impl Debug for BooleanArray {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, f);

        write!(f, "BooleanArray")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
