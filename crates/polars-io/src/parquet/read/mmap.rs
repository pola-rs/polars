use arrow::array::Array;
use arrow::bitmap::Bitmap;
use arrow::datatypes::Field;
#[cfg(feature = "async")]
use bytes::Bytes;
#[cfg(feature = "async")]
use polars_core::datatypes::PlHashMap;
use polars_error::PolarsResult;
use polars_parquet::read::{
    column_iter_to_arrays, BasicDecompressor, ColumnChunkMetadata, Filter, PageReader,
};
use polars_utils::mmap::{MemReader, MemSlice};

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
pub enum ColumnStore {
    Local(MemSlice),
    #[cfg(feature = "async")]
    Fetched(PlHashMap<u64, Bytes>),
}

/// For local files memory maps all columns that are part of the parquet field `field_name`.
/// For cloud files the relevant memory regions should have been prefetched.
pub(super) fn mmap_columns<'a>(
    store: &'a ColumnStore,
    field_columns: &'a [&ColumnChunkMetadata],
) -> Vec<(&'a ColumnChunkMetadata, MemSlice)> {
    field_columns
        .iter()
        .map(|meta| _mmap_single_column(store, meta))
        .collect()
}

fn _mmap_single_column<'a>(
    store: &'a ColumnStore,
    meta: &'a ColumnChunkMetadata,
) -> (&'a ColumnChunkMetadata, MemSlice) {
    let byte_range = meta.byte_range();
    let chunk = match store {
        ColumnStore::Local(mem_slice) => {
            mem_slice.slice(byte_range.start as usize..byte_range.end as usize)
        },
        #[cfg(all(feature = "async", feature = "parquet"))]
        ColumnStore::Fetched(fetched) => {
            let entry = fetched.get(&byte_range.start).unwrap_or_else(|| {
                panic!(
                    "mmap_columns: column with start {} must be prefetched in ColumnStore.\n",
                    byte_range.start
                )
            });
            MemSlice::from_bytes(entry.clone())
        },
    };
    (meta, chunk)
}

// similar to arrow2 serializer, except this accepts a slice instead of a vec.
// this allows us to memory map
pub fn to_deserializer(
    columns: Vec<(&ColumnChunkMetadata, MemSlice)>,
    field: Field,
    filter: Option<Filter>,
) -> PolarsResult<(Box<dyn Array>, Bitmap)> {
    let (columns, types): (Vec<_>, Vec<_>) = columns
        .into_iter()
        .map(|(column_meta, chunk)| {
            // Advise fetching the data for the column chunk
            chunk.prefetch();

            let pages = PageReader::new(MemReader::new(chunk), column_meta, vec![], usize::MAX);
            (
                BasicDecompressor::new(pages, vec![]),
                &column_meta.descriptor().descriptor.primitive_type,
            )
        })
        .unzip();

    column_iter_to_arrays(columns, types, field, filter)
}
