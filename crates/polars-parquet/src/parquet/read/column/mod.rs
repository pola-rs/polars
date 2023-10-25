use std::io::{Read, Seek};
use std::vec::IntoIter;

use crate::error::Error;
use crate::metadata::{ColumnChunkMetaData, RowGroupMetaData};
use crate::page::CompressedPage;
use crate::schema::types::ParquetType;

use super::{get_field_columns, get_page_iterator, PageFilter, PageReader};

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
mod stream;

/// Returns a [`ColumnIterator`] of column chunks corresponding to `field`.
///
/// Contrarily to [`get_page_iterator`] that returns a single iterator of pages, this iterator
/// iterates over columns, one by one, and returns a [`PageReader`] per column.
/// For primitive fields (e.g. `i64`), [`ColumnIterator`] yields exactly one column.
/// For complex fields, it yields multiple columns.
/// `max_page_size` is the maximum number of bytes allowed.
pub fn get_column_iterator<R: Read + Seek>(
    reader: R,
    row_group: &RowGroupMetaData,
    field_name: &str,
    page_filter: Option<PageFilter>,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> ColumnIterator<R> {
    let columns = get_field_columns(row_group.columns(), field_name)
        .cloned()
        .collect::<Vec<_>>();

    ColumnIterator::new(reader, columns, page_filter, scratch, max_page_size)
}

/// State of [`MutStreamingIterator`].
#[derive(Debug)]
pub enum State<T> {
    /// Iterator still has elements
    Some(T),
    /// Iterator finished
    Finished(Vec<u8>),
}

/// A special kind of fallible streaming iterator where `advance` consumes the iterator.
pub trait MutStreamingIterator: Sized {
    type Item;
    type Error;

    fn advance(self) -> std::result::Result<State<Self>, Self::Error>;
    fn get(&mut self) -> Option<&mut Self::Item>;
}

/// A [`MutStreamingIterator`] that reads column chunks one by one,
/// returning a [`PageReader`] per column.
pub struct ColumnIterator<R: Read + Seek> {
    reader: Option<R>,
    columns: Vec<ColumnChunkMetaData>,
    page_filter: Option<PageFilter>,
    current: Option<(PageReader<R>, ColumnChunkMetaData)>,
    scratch: Vec<u8>,
    max_page_size: usize,
}

impl<R: Read + Seek> ColumnIterator<R> {
    /// Returns a new [`ColumnIterator`]
    /// `max_page_size` is the maximum allowed page size
    pub fn new(
        reader: R,
        mut columns: Vec<ColumnChunkMetaData>,
        page_filter: Option<PageFilter>,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        columns.reverse();
        Self {
            reader: Some(reader),
            scratch,
            columns,
            page_filter,
            current: None,
            max_page_size,
        }
    }
}

impl<R: Read + Seek> MutStreamingIterator for ColumnIterator<R> {
    type Item = (PageReader<R>, ColumnChunkMetaData);
    type Error = Error;

    fn advance(mut self) -> Result<State<Self>, Error> {
        let (reader, scratch) = if let Some((iter, _)) = self.current {
            iter.into_inner()
        } else {
            (self.reader.unwrap(), self.scratch)
        };
        if self.columns.is_empty() {
            return Ok(State::Finished(scratch));
        };
        let column = self.columns.pop().unwrap();

        let iter = get_page_iterator(
            &column,
            reader,
            self.page_filter.clone(),
            scratch,
            self.max_page_size,
        )?;
        let current = Some((iter, column));
        Ok(State::Some(Self {
            reader: None,
            columns: self.columns,
            page_filter: self.page_filter,
            current,
            scratch: vec![],
            max_page_size: self.max_page_size,
        }))
    }

    fn get(&mut self) -> Option<&mut Self::Item> {
        self.current.as_mut()
    }
}

/// A [`MutStreamingIterator`] of pre-read column chunks
#[derive(Debug)]
pub struct ReadColumnIterator {
    field: ParquetType,
    chunks: Vec<(Vec<Result<CompressedPage, Error>>, ColumnChunkMetaData)>,
    current: Option<(IntoIter<Result<CompressedPage, Error>>, ColumnChunkMetaData)>,
}

impl ReadColumnIterator {
    /// Returns a new [`ReadColumnIterator`]
    pub fn new(
        field: ParquetType,
        chunks: Vec<(Vec<Result<CompressedPage, Error>>, ColumnChunkMetaData)>,
    ) -> Self {
        Self {
            field,
            chunks,
            current: None,
        }
    }
}

impl MutStreamingIterator for ReadColumnIterator {
    type Item = (IntoIter<Result<CompressedPage, Error>>, ColumnChunkMetaData);
    type Error = Error;

    fn advance(mut self) -> Result<State<Self>, Error> {
        if self.chunks.is_empty() {
            return Ok(State::Finished(vec![]));
        }
        self.current = self
            .chunks
            .pop()
            .map(|(pages, meta)| (pages.into_iter(), meta));
        Ok(State::Some(Self {
            field: self.field,
            chunks: self.chunks,
            current: self.current,
        }))
    }

    fn get(&mut self) -> Option<&mut Self::Item> {
        self.current.as_mut()
    }
}

/// Reads all columns that are part of the parquet field `field_name`
/// # Implementation
/// This operation is IO-bounded `O(C)` where C is the number of columns associated to
/// the field (one for non-nested types)
/// It reads the columns sequentially. Use [`read_column`] to fork this operation to multiple
/// readers.
pub fn read_columns<'a, R: Read + Seek>(
    reader: &mut R,
    columns: &'a [ColumnChunkMetaData],
    field_name: &'a str,
) -> Result<Vec<(&'a ColumnChunkMetaData, Vec<u8>)>, Error> {
    get_field_columns(columns, field_name)
        .map(|column| read_column(reader, column).map(|c| (column, c)))
        .collect()
}

/// Reads a column chunk into memory
/// This operation is IO-bounded and allocates the column's `compressed_size`.
pub fn read_column<R>(reader: &mut R, column: &ColumnChunkMetaData) -> Result<Vec<u8>, Error>
where
    R: Read + Seek,
{
    let (start, length) = column.byte_range();
    reader.seek(std::io::SeekFrom::Start(start))?;

    let mut chunk = vec![];
    chunk.try_reserve(length as usize)?;
    reader.by_ref().take(length).read_to_end(&mut chunk)?;
    Ok(chunk)
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use stream::{read_column_async, read_columns_async};
