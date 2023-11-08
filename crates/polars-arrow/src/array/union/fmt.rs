use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::{get_display, write_vec};
use super::UnionArray;

pub fn write_value<W: Write>(
    array: &UnionArray,
    index: usize,
    null: &'static str,
    f: &mut W,
) -> Result {
    let (field, index) = array.index(index);

    get_display(array.fields()[field].as_ref(), null)(f, index)
}

impl Debug for UnionArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, "None", f);

        write!(f, "UnionArray")?;
        write_vec(f, writer, None, self.len(), "None", false)
    }
}
