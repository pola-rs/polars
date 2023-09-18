use csv::ByteRecord;

use crate::{
    array::Array,
    chunk::Chunk,
    datatypes::{DataType, Field},
    error::Result,
};

use super::super::read_utils::{
    deserialize_batch as deserialize_batch_gen, deserialize_column as deserialize_column_gen,
    ByteRecordGeneric,
};

impl ByteRecordGeneric for ByteRecord {
    #[inline]
    fn get(&self, index: usize) -> Option<&[u8]> {
        self.get(index)
    }
}

/// Deserializes `column` of `rows` into an [`Array`] of [`DataType`] `datatype`.
pub fn deserialize_column(
    rows: &[ByteRecord],
    column: usize,
    datatype: DataType,
    line_number: usize,
) -> Result<Box<dyn Array>> {
    deserialize_column_gen(rows, column, datatype, line_number)
}

/// Deserializes rows [`ByteRecord`] into a [`Chunk`].
/// Note that this is a convenience function: column deserialization
/// is trivially parallelizable (e.g. rayon).
pub fn deserialize_batch<F>(
    rows: &[ByteRecord],
    fields: &[Field],
    projection: Option<&[usize]>,
    line_number: usize,
    deserialize_column: F,
) -> Result<Chunk<Box<dyn Array>>>
where
    F: Fn(&[ByteRecord], usize, DataType, usize) -> Result<Box<dyn Array>>,
{
    deserialize_batch_gen(rows, fields, projection, line_number, deserialize_column)
}
