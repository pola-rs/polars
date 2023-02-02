use arrow::datatypes::Field;
use arrow::io::parquet::read::{
    column_iter_to_arrays, get_field_columns, ArrayIter, BasicDecompressor, ColumnChunkMetaData,
    PageReader,
};
#[cfg(feature = "async")]
use polars_core::datatypes::PlHashMap;

use super::*;

/// Store columns data in two scenarios:
/// 1. a local memory mapped file
/// 2. data fetched from cloud storage on demand, in this case
///     a. the key in the hashmap is the start in the file
///     b. the value in the hashmap is the actual data.
///
/// For the fetched case we use a two phase approach:
///    a. identify all the needed columns
///    b. asynchronously fetch them in parallel, for example using object_store
///    c. store the data in this data structure
///    d. when all the data is available deserialize on multiple threads, for example using rayon
pub enum ColumnStore<'a> {
    Local(&'a [u8]),
    #[cfg(feature = "async")]
    Fetched(PlHashMap<u64, Vec<u8>>),
}

/// For local files memory maps all columns that are part of the parquet field `field_name`.
/// For cloud files the relevant memory regions should have been prefetched.
pub(super) fn mmap_columns<'a>(
    store: &'a ColumnStore,
    columns: &'a [ColumnChunkMetaData],
    field_name: &str,
) -> Vec<(&'a ColumnChunkMetaData, &'a [u8])> {
    get_field_columns(columns, field_name)
        .into_iter()
        .map(|meta| _mmap_single_column(store, meta))
        .collect()
}

fn _mmap_single_column<'a>(
    store: &'a ColumnStore,
    meta: &'a ColumnChunkMetaData,
) -> (&'a ColumnChunkMetaData, &'a [u8]) {
    let (start, len) = meta.byte_range();
    let chunk = match store {
        ColumnStore::Local(file) => &file[start as usize..(start + len) as usize],
        #[cfg(all(feature = "async", feature = "parquet"))]
        ColumnStore::Fetched(fetched) => {
            let entry = fetched.get(&start).unwrap_or_else(|| {
                panic!(
                    "mmap_columns: column with start {start} must be prefetched in ColumnStore.\n"
                )
            });
            entry.as_slice()
        }
    };
    (meta, chunk)
}

// similar to arrow2 serializer, except this accepts a slice instead of a vec.
// this allows us to memory map
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
