mod column;
mod compression;
mod indexes;
pub mod levels;
mod metadata;
mod page;
#[cfg(feature = "async")]
mod stream;

use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

pub use column::*;
pub use compression::{decompress, BasicDecompressor, Decompressor};
pub use indexes::{read_columns_indexes, read_pages_locations};
pub use metadata::{deserialize_metadata, read_metadata, read_metadata_with_size};
#[cfg(feature = "async")]
pub use page::{get_page_stream, get_page_stream_from_column_start};
pub use page::{IndexedPageReader, PageFilter, PageIterator, PageMetaData, PageReader};
#[cfg(feature = "async")]
pub use stream::read_metadata as read_metadata_async;

use crate::parquet::error::Result;
use crate::parquet::metadata::{ColumnChunkMetaData, FileMetaData, RowGroupMetaData};

/// Filters row group metadata to only those row groups,
/// for which the predicate function returns true
pub fn filter_row_groups(
    metadata: &FileMetaData,
    predicate: &dyn Fn(&RowGroupMetaData, usize) -> bool,
) -> FileMetaData {
    let mut filtered_row_groups = Vec::<RowGroupMetaData>::new();
    for (i, row_group_metadata) in metadata.row_groups.iter().enumerate() {
        if predicate(row_group_metadata, i) {
            filtered_row_groups.push(row_group_metadata.clone());
        }
    }
    let mut metadata = metadata.clone();
    metadata.row_groups = filtered_row_groups;
    metadata
}

/// Returns a new [`PageReader`] by seeking `reader` to the begining of `column_chunk`.
pub fn get_page_iterator<R: Read + Seek>(
    column_chunk: &ColumnChunkMetaData,
    mut reader: R,
    pages_filter: Option<PageFilter>,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> Result<PageReader<R>> {
    let pages_filter = pages_filter.unwrap_or_else(|| Arc::new(|_, _| true));

    let (col_start, _) = column_chunk.byte_range();
    reader.seek(SeekFrom::Start(col_start))?;
    Ok(PageReader::new(
        reader,
        column_chunk,
        pages_filter,
        scratch,
        max_page_size,
    ))
}

/// Returns all [`ColumnChunkMetaData`] associated to `field_name`.
/// For non-nested types, this returns an iterator with a single column
pub fn get_field_columns<'a>(
    columns: &'a [ColumnChunkMetaData],
    field_name: &'a str,
) -> impl Iterator<Item = &'a ColumnChunkMetaData> {
    columns
        .iter()
        .filter(move |x| x.descriptor().path_in_schema[0] == field_name)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;
    use crate::parquet::tests::get_path;
    use crate::parquet::FallibleStreamingIterator;

    #[test]
    fn basic() -> Result<()> {
        let mut testdata = get_path();
        testdata.push("alltypes_plain.parquet");
        let mut file = File::open(testdata).unwrap();

        let metadata = read_metadata(&mut file)?;

        let row_group = 0;
        let column = 0;
        let column_metadata = &metadata.row_groups[row_group].columns()[column];
        let buffer = vec![];
        let mut iter = get_page_iterator(column_metadata, &mut file, None, buffer, 1024 * 1024)?;

        let dict = iter.next().unwrap().unwrap();
        assert_eq!(dict.num_values(), 0);
        let page = iter.next().unwrap().unwrap();
        assert_eq!(page.num_values(), 8);
        Ok(())
    }

    #[test]
    fn reuse_buffer() -> Result<()> {
        let mut testdata = get_path();
        testdata.push("alltypes_plain.snappy.parquet");
        let mut file = File::open(testdata).unwrap();

        let metadata = read_metadata(&mut file)?;

        let row_group = 0;
        let column = 0;
        let column_metadata = &metadata.row_groups[row_group].columns()[column];
        let buffer = vec![0];
        let iterator = get_page_iterator(column_metadata, &mut file, None, buffer, 1024 * 1024)?;

        let buffer = vec![];
        let mut iterator = Decompressor::new(iterator, buffer);

        let _dict = iterator.next()?.unwrap();
        let _page = iterator.next()?.unwrap();

        assert!(iterator.next()?.is_none());
        let (a, b) = iterator.into_buffers();
        assert_eq!(a.len(), 11); // note: compressed is higher in this example.
        assert_eq!(b.len(), 9);

        Ok(())
    }

    #[test]
    fn reuse_buffer_decompress() -> Result<()> {
        let mut testdata = get_path();
        testdata.push("alltypes_plain.parquet");
        let mut file = File::open(testdata).unwrap();

        let metadata = read_metadata(&mut file)?;

        let row_group = 0;
        let column = 0;
        let column_metadata = &metadata.row_groups[row_group].columns()[column];
        let buffer = vec![1];
        let iterator = get_page_iterator(column_metadata, &mut file, None, buffer, 1024 * 1024)?;

        let buffer = vec![];
        let mut iterator = Decompressor::new(iterator, buffer);

        // dict
        iterator.next()?.unwrap();
        // page
        iterator.next()?.unwrap();

        assert!(iterator.next()?.is_none());
        let (a, b) = iterator.into_buffers();

        assert_eq!(a.len(), 11);
        assert_eq!(b.len(), 0); // the decompressed buffer is never used because it is always swapped with the other buffer.

        Ok(())
    }

    #[test]
    fn column_iter() -> Result<()> {
        let mut testdata = get_path();
        testdata.push("alltypes_plain.parquet");
        let mut file = File::open(testdata).unwrap();

        let metadata = read_metadata(&mut file)?;

        let row_group = 0;
        let column = 0;
        let column_metadata = &metadata.row_groups[row_group].columns()[column];
        let iter: Vec<_> =
            get_page_iterator(column_metadata, &mut file, None, vec![], usize::MAX)?.collect();

        let field = metadata.schema().fields()[0].clone();
        let mut iter = ReadColumnIterator::new(field, vec![(iter, column_metadata.clone())]);

        loop {
            match iter.advance()? {
                State::Some(mut new_iter) => {
                    if let Some((pages, _descriptor)) = new_iter.get() {
                        let mut iterator = BasicDecompressor::new(pages, vec![]);
                        while let Some(_page) = iterator.next()? {
                            // do something with it
                        }
                        let _internal_buffer = iterator.into_inner();
                    }
                    iter = new_iter;
                },
                State::Finished(_buffer) => {
                    assert!(_buffer.is_empty()); // data is uncompressed => buffer is always moved
                    break;
                },
            }
        }
        Ok(())
    }

    #[test]
    fn basics_column_iterator() -> Result<()> {
        let mut testdata = get_path();
        testdata.push("alltypes_plain.parquet");
        let mut file = File::open(testdata).unwrap();

        let metadata = read_metadata(&mut file)?;

        let mut iter = ColumnIterator::new(
            file,
            metadata.row_groups[0].columns().to_vec(),
            None,
            vec![],
            usize::MAX, // we trust the file is correct
        );

        loop {
            match iter.advance()? {
                State::Some(mut new_iter) => {
                    if let Some((pages, _descriptor)) = new_iter.get() {
                        let mut iterator = BasicDecompressor::new(pages, vec![]);
                        while let Some(_page) = iterator.next()? {
                            // do something with it
                        }
                        let _internal_buffer = iterator.into_inner();
                    }
                    iter = new_iter;
                },
                State::Finished(_buffer) => {
                    assert!(_buffer.is_empty()); // data is uncompressed => buffer is always moved
                    break;
                },
            }
        }
        Ok(())
    }
}
