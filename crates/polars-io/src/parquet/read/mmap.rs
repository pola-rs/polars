use std::io::Cursor;

use polars_arrow::array::{Array, DictionaryArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::Field;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_parquet::read::{
    BasicDecompressor, ColumnChunkMetadata, Filter, PageReader, PhysicalType,
    column_iter_to_arrays, decode_dictionary_codes,
};
use polars_utils::mem::prefetch::prefetch_l2;

/// Store columns data in two scenarios:
/// 1. a local memory mapped file
/// 2. data fetched from cloud storage on demand, in this case
///    a. the key in the hashmap is the start in the file
///    b. the value in the hashmap is the actual data.
///
/// For the fetched case we use a two phase approach:
///   a. identify all the needed columns
///   b. asynchronously fetch them in parallel, for example using object_store
///   c. store the data in this data structure
///   d. when all the data is available deserialize on multiple threads, for example using rayon
pub enum ColumnStore {
    Local(Buffer<u8>),
}

/// For local files memory maps all columns that are part of the parquet field `field_name`.
/// For cloud files the relevant memory regions should have been prefetched.
pub(super) fn mmap_columns<'a>(
    store: &'a ColumnStore,
    field_columns: &'a [&ColumnChunkMetadata],
) -> PolarsResult<Vec<(&'a ColumnChunkMetadata, Buffer<u8>)>> {
    field_columns
        .iter()
        .map(|meta| _mmap_single_column(store, meta))
        .collect()
}

fn _mmap_single_column<'a>(
    store: &'a ColumnStore,
    meta: &'a ColumnChunkMetadata,
) -> PolarsResult<(&'a ColumnChunkMetadata, Buffer<u8>)> {
    let byte_range = meta.byte_range()?;
    let chunk = match store {
        ColumnStore::Local(mem_slice) => mem_slice
            .clone()
            .sliced(byte_range.start as usize..byte_range.end as usize),
    };
    Ok((meta, chunk))
}

// similar to arrow2 serializer, except this accepts a slice instead of a vec.
// this allows us to memory map
pub fn to_deserializer(
    columns: Vec<(&ColumnChunkMetadata, Buffer<u8>)>,
    field: Field,
    filter: Option<Filter>,
) -> PolarsResult<(Vec<Box<dyn Array>>, Bitmap)> {
    let (columns, types) = columns
        .into_iter()
        .map(|(column_meta, chunk)| {
            // Advise fetching the data for the column chunk
            prefetch_l2(&chunk);

            let pages = PageReader::new(Cursor::new(chunk), column_meta, vec![], usize::MAX)?;
            PolarsResult::Ok((
                BasicDecompressor::new(pages, vec![]),
                &column_meta.descriptor().descriptor.primitive_type,
            ))
        })
        .collect::<PolarsResult<(Vec<_>, Vec<_>)>>()?;

    column_iter_to_arrays(columns, types, field, filter)
}

/// Decodes a flat String column chunk as codes into its dictionary, if it has a dictionary page,
/// every data page is dictionary-encoded and `accept(num_rows, dict_len, dict_bytes)` holds for
/// the number of rows the filter keeps (all of the chunk's for a predicate, whose result isn't
/// known yet) and the number and uncompressed size of the dictionary values.
/// Returns `None` otherwise, without decoding anything.
pub fn to_dictionary_codes(
    column_meta: &ColumnChunkMetadata,
    chunk: Buffer<u8>,
    field_name: &str,
    filter: Option<Filter>,
    accept: impl FnOnce(usize, usize, usize) -> bool,
) -> PolarsResult<Option<(Vec<DictionaryArray<u32>>, Bitmap)>> {
    // Only a BYTE_ARRAY column's dictionary holds length-prefixed strings.
    if column_meta
        .descriptor()
        .descriptor
        .primitive_type
        .physical_type
        != PhysicalType::ByteArray
    {
        return Ok(None);
    }
    let pages = PageReader::new(Cursor::new(chunk.clone()), column_meta, vec![], usize::MAX)?;
    let num_rows = match &filter {
        Some(Filter::Range(range)) => range.len().min(pages.total_num_values()),
        Some(Filter::Mask(mask)) => mask.set_bits(),
        Some(Filter::Predicate(_)) | None => pages.total_num_values(),
    };
    match pages.dictionary_only_len()? {
        Some((dict_len, dict_bytes)) if accept(num_rows, dict_len, dict_bytes) => {},
        _ => return Ok(None),
    }

    prefetch_l2(&chunk);
    let pages = PageReader::new(Cursor::new(chunk), column_meta, vec![], usize::MAX)?;
    let decoded =
        decode_dictionary_codes(field_name, BasicDecompressor::new(pages, vec![]), filter)?;
    Ok(Some(decoded))
}
