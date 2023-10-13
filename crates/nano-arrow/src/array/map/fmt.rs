use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::{get_display, write_vec};
use super::MapArray;

pub fn write_value<W: Write>(
    array: &MapArray,
    index: usize,
    null: &'static str,
    f: &mut W,
) -> Result {
    let values = array.value(index);
    let writer = |f: &mut W, index| get_display(values.as_ref(), null)(f, index);
    write_vec(f, writer, None, values.len(), null, false)
}

impl Debug for MapArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, "None", f);

        write!(f, "MapArray")?;
        write_vec(f, writer, self.validity.as_ref(), self.len(), "None", false)
    }
}
