use std::vec::IntoIter;

use polars_utils::idx_vec::UnitVec;

use super::{get_page_iterator, MemReader, PageReader};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetadata, RowGroupMetadata};
use crate::parquet::page::CompressedPage;
use crate::parquet::schema::types::ParquetType;

/// Returns a [`ColumnIterator`] of column chunks corresponding to `field`.
///
/// Contrarily to [`get_page_iterator`] that returns a single iterator of pages, this iterator
/// iterates over columns, one by one, and returns a [`PageReader`] per column.
/// For primitive fields (e.g. `i64`), [`ColumnIterator`] yields exactly one column.
/// For complex fields, it yields multiple columns.
/// `max_page_size` is the maximum number of bytes allowed.
pub fn get_column_iterator<'a>(
    reader: MemReader,
    row_group: &'a RowGroupMetadata,
    field_name: &str,
    max_page_size: usize,
) -> ColumnIterator<'a> {
    let columns = row_group
        .columns_under_root_iter(field_name)
        .unwrap()
        .rev()
        .collect::<UnitVec<_>>();
    ColumnIterator::new(reader, columns, max_page_size)
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
pub struct ColumnIterator<'a> {
    reader: MemReader,
    columns: UnitVec<&'a ColumnChunkMetadata>,
    max_page_size: usize,
}

impl<'a> ColumnIterator<'a> {
    /// Returns a new [`ColumnIterator`]
    /// `max_page_size` is the maximum allowed page size
    pub fn new(
        reader: MemReader,
        columns: UnitVec<&'a ColumnChunkMetadata>,
        max_page_size: usize,
    ) -> Self {
        Self {
            reader,
            columns,
            max_page_size,
        }
    }
}

impl<'a> Iterator for ColumnIterator<'a> {
    type Item = ParquetResult<(PageReader, &'a ColumnChunkMetadata)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.columns.is_empty() {
            return None;
        };
        let column = self.columns.pop().unwrap();

        let iter =
            match get_page_iterator(column, self.reader.clone(), Vec::new(), self.max_page_size) {
                Err(e) => return Some(Err(e)),
                Ok(v) => v,
            };
        Some(Ok((iter, column)))
    }
}

/// A [`MutStreamingIterator`] of pre-read column chunks
#[derive(Debug)]
pub struct ReadColumnIterator {
    field: ParquetType,
    chunks: Vec<(
        Vec<Result<CompressedPage, ParquetError>>,
        ColumnChunkMetadata,
    )>,
    current: Option<(
        IntoIter<Result<CompressedPage, ParquetError>>,
        ColumnChunkMetadata,
    )>,
}

impl ReadColumnIterator {
    /// Returns a new [`ReadColumnIterator`]
    pub fn new(
        field: ParquetType,
        chunks: Vec<(
            Vec<Result<CompressedPage, ParquetError>>,
            ColumnChunkMetadata,
        )>,
    ) -> Self {
        Self {
            field,
            chunks,
            current: None,
        }
    }
}

impl MutStreamingIterator for ReadColumnIterator {
    type Item = (
        IntoIter<Result<CompressedPage, ParquetError>>,
        ColumnChunkMetadata,
    );
    type Error = ParquetError;

    fn advance(mut self) -> Result<State<Self>, ParquetError> {
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
