use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::{get_display, write_map, write_vec};
use super::StructArray;

pub fn write_value<W: Write>(
    array: &StructArray,
    index: usize,
    null: &'static str,
    f: &mut W,
) -> Result {
    let writer = |f: &mut W, _index| {
        for (i, (field, column)) in array.fields().iter().zip(array.values()).enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            let writer = get_display(column.as_ref(), null);
            write!(f, "{}: ", field.name)?;
            writer(f, index)?;
        }
        Ok(())
    };

    write_map(f, writer, None, 1, null, false)
}

impl Debug for StructArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, "None", f);

        write!(f, "StructArray")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
