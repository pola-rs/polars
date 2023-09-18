use std::fmt::{Debug, Formatter, Result, Write};

use crate::array::Array;

use super::super::fmt::{get_display, write_vec};
use super::{DictionaryArray, DictionaryKey};

pub fn write_value<K: DictionaryKey, W: Write>(
    array: &DictionaryArray<K>,
    index: usize,
    null: &'static str,
    f: &mut W,
) -> Result {
    let keys = array.keys();
    let values = array.values();

    if keys.is_valid(index) {
        let key = array.key_value(index);
        get_display(values.as_ref(), null)(f, key)
    } else {
        write!(f, "{null}")
    }
}

impl<K: DictionaryKey> Debug for DictionaryArray<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, "None", f);

        write!(f, "DictionaryArray")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
