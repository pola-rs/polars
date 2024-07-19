use std::collections::VecDeque;
use std::io::{Seek, SeekFrom};

use polars_utils::mmap::{MemReader, MemSlice};

use super::reader::{finish_page, read_page_header, PageMetaData};
use crate::parquet::error::ParquetError;
use crate::parquet::indexes::{FilteredPage, Interval};
use crate::parquet::metadata::{ColumnChunkMetaData, Descriptor};
use crate::parquet::page::{CompressedDictPage, CompressedPage, ParquetPageHeader};
use crate::parquet::parquet_bridge::Compression;

#[derive(Debug, Clone, Copy)]
enum State {
    MaybeDict,
    Data,
}

/// A fallible [`Iterator`] of [`CompressedPage`]. This iterator leverages page indexes
/// to skip pages that are not needed. Consequently, the pages from this
/// iterator always have [`Some`] [`crate::parquet::page::CompressedDataPage::selected_rows()`]
pub struct IndexedPageReader {
    // The source
    reader: MemReader,

    column_start: u64,
    compression: Compression,

    // used to deserialize dictionary pages and attach the descriptor to every read page
    descriptor: Descriptor,

    // buffer to read the whole page [header][data] into memory
    buffer: Vec<u8>,

    // buffer to store the data [data] and reuse across pages
    data_buffer: Vec<u8>,

    pages: VecDeque<FilteredPage>,

    state: State,
}

fn read_page(
    reader: &mut MemReader,
    start: u64,
    length: usize,
) -> Result<(ParquetPageHeader, MemSlice), ParquetError> {
    // seek to the page
    reader.seek(SeekFrom::Start(start))?;

    let start_position = reader.position();

    // deserialize [header]
    let page_header = read_page_header(reader, 1024 * 1024)?;
    let header_size = reader.position() - start_position;

    // copy [data]
    let data = reader.read_slice(length - header_size);

    Ok((page_header, data))
}

fn read_dict_page(
    reader: &mut MemReader,
    start: u64,
    length: usize,
    compression: Compression,
    descriptor: &Descriptor,
) -> Result<CompressedDictPage, ParquetError> {
    let (page_header, data) = read_page(reader, start, length)?;

    let page = finish_page(page_header, data, compression, descriptor, None)?;
    if let CompressedPage::Dict(page) = page {
        Ok(page)
    } else {
        Err(ParquetError::oos(
            "The first page is not a dictionary page but it should",
        ))
    }
}

impl IndexedPageReader {
    /// Returns a new [`IndexedPageReader`].
    pub fn new(
        reader: MemReader,
        column: &ColumnChunkMetaData,
        pages: Vec<FilteredPage>,
        buffer: Vec<u8>,
        data_buffer: Vec<u8>,
    ) -> Self {
        Self::new_with_page_meta(reader, column.into(), pages, buffer, data_buffer)
    }

    /// Returns a new [`IndexedPageReader`] with [`PageMetaData`].
    pub fn new_with_page_meta(
        reader: MemReader,
        column: PageMetaData,
        pages: Vec<FilteredPage>,
        buffer: Vec<u8>,
        data_buffer: Vec<u8>,
    ) -> Self {
        let pages = pages.into_iter().collect();
        Self {
            reader,
            column_start: column.column_start,
            compression: column.compression,
            descriptor: column.descriptor,
            buffer,
            data_buffer,
            pages,
            state: State::MaybeDict,
        }
    }

    /// consumes self into the reader and the two internal buffers
    pub fn into_inner(self) -> (MemReader, Vec<u8>, Vec<u8>) {
        (self.reader, self.buffer, self.data_buffer)
    }

    fn read_page(
        &mut self,
        start: u64,
        length: usize,
        selected_rows: Vec<Interval>,
    ) -> Result<CompressedPage, ParquetError> {
        let (page_header, data) = read_page(&mut self.reader, start, length)?;

        finish_page(
            page_header,
            data,
            self.compression,
            &self.descriptor,
            Some(selected_rows),
        )
    }

    fn read_dict(&mut self) -> Option<Result<CompressedPage, ParquetError>> {
        // a dictionary page exists iff the first data page is not at the start of
        // the column
        let (start, length) = match self.pages.front() {
            Some(page) => {
                let length = (page.start - self.column_start) as usize;
                if length > 0 {
                    (self.column_start, length)
                } else {
                    return None;
                }
            },
            None => return None,
        };

        let maybe_page = read_dict_page(
            &mut self.reader,
            start,
            length,
            self.compression,
            &self.descriptor,
        );
        Some(maybe_page.map(CompressedPage::Dict))
    }
}

impl Iterator for IndexedPageReader {
    type Item = Result<CompressedPage, ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            State::MaybeDict => {
                self.state = State::Data;
                if let Some(dict) = self.read_dict() {
                    Some(dict)
                } else {
                    self.next()
                }
            },
            State::Data => {
                if let Some(page) = self.pages.pop_front() {
                    if page.selected_rows.is_empty() {
                        self.next()
                    } else {
                        Some(self.read_page(page.start, page.length, page.selected_rows))
                    }
                } else {
                    None
                }
            },
        }
    }
}
