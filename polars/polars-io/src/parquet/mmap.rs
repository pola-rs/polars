use arrow::datatypes::Field;
use arrow::io::parquet::read::{
    column_iter_to_arrays, get_field_columns, ArrayIter, BasicDecompressor, ColumnChunkMetaData,
    PageReader,
};

use super::*;

/// memory maps all columns that are part of the parquet field `field_name`
pub(super) fn mmap_columns<'a>(
    file: &'a [u8],
    columns: &'a [ColumnChunkMetaData],
    field_name: &str,
) -> Vec<(&'a ColumnChunkMetaData, &'a [u8])> {
    get_field_columns(columns, field_name)
        .into_iter()
        .map(|meta| _mmap_single_column(file, meta))
        .collect()
}

fn _mmap_single_column<'a>(
    file: &'a [u8],
    meta: &'a ColumnChunkMetaData,
) -> (&'a ColumnChunkMetaData, &'a [u8]) {
    let (start, len) = meta.byte_range();
    let chunk = &file[start as usize..(start + len) as usize];
    (meta, chunk)
}

// similar to arrow2 serializer, except this accepts a slice instead of a vec.
// this allows use to memory map
pub(super) fn to_deserializer<'a>(
    columns: Vec<(&ColumnChunkMetaData, &'a [u8])>,
    field: Field,
    num_rows: usize,
    chunk_size: Option<usize>,
) -> ArrowResult<ArrayIter<'a>> {
    let chunk_size = chunk_size.unwrap_or(usize::MAX).min(num_rows);

    let (columns, types): (Vec<_>, Vec<_>) = columns
        .into_iter()
        .map(|(column_meta, chunk)| {
            let pages = PageReader::new(
                std::io::Cursor::new(chunk),
                column_meta,
                std::sync::Arc::new(|_, _| true),
                vec![],
                usize::MAX,
            );
            (
                BasicDecompressor::new(pages, vec![]),
                &column_meta.descriptor().descriptor.primitive_type,
            )
        })
        .unzip();

    column_iter_to_arrays(columns, types, field, Some(chunk_size), num_rows)
}
