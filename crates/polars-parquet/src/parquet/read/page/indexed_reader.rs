use std::collections::VecDeque;
use std::io::{Cursor, Read, Seek, SeekFrom};

use super::reader::{finish_page, read_page_header, PageMetaData};
use crate::parquet::error::Error;
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
pub struct IndexedPageReader<R: Read + Seek> {
    // The source
    reader: R,

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

fn read_page<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    length: usize,
    buffer: &mut Vec<u8>,
    data: &mut Vec<u8>,
) -> Result<ParquetPageHeader, Error> {
    // seek to the page
    reader.seek(SeekFrom::Start(start))?;

    // read [header][data] to buffer
    buffer.clear();
    buffer.try_reserve(length)?;
    reader.by_ref().take(length as u64).read_to_end(buffer)?;

    // deserialize [header]
    let mut reader = Cursor::new(buffer);
    let page_header = read_page_header(&mut reader, 1024 * 1024)?;
    let header_size = reader.stream_position().unwrap() as usize;
    let buffer = reader.into_inner();

    // copy [data]
    data.clear();
    data.extend_from_slice(&buffer[header_size..]);
    Ok(page_header)
}

fn read_dict_page<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    length: usize,
    buffer: &mut Vec<u8>,
    data: &mut Vec<u8>,
    compression: Compression,
    descriptor: &Descriptor,
) -> Result<CompressedDictPage, Error> {
    let page_header = read_page(reader, start, length, buffer, data)?;

    let page = finish_page(page_header, data, compression, descriptor, None)?;
    if let CompressedPage::Dict(page) = page {
        Ok(page)
    } else {
        Err(Error::oos(
            "The first page is not a dictionary page but it should",
        ))
    }
}

impl<R: Read + Seek> IndexedPageReader<R> {
    /// Returns a new [`IndexedPageReader`].
    pub fn new(
        reader: R,
        column: &ColumnChunkMetaData,
        pages: Vec<FilteredPage>,
        buffer: Vec<u8>,
        data_buffer: Vec<u8>,
    ) -> Self {
        Self::new_with_page_meta(reader, column.into(), pages, buffer, data_buffer)
    }

    /// Returns a new [`IndexedPageReader`] with [`PageMetaData`].
    pub fn new_with_page_meta(
        reader: R,
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
    pub fn into_inner(self) -> (R, Vec<u8>, Vec<u8>) {
        (self.reader, self.buffer, self.data_buffer)
    }

    fn read_page(
        &mut self,
        start: u64,
        length: usize,
        selected_rows: Vec<Interval>,
    ) -> Result<CompressedPage, Error> {
        // it will be read - take buffer
        let mut data = std::mem::take(&mut self.data_buffer);

        let page_header = read_page(&mut self.reader, start, length, &mut self.buffer, &mut data)?;

        finish_page(
            page_header,
            &mut data,
            self.compression,
            &self.descriptor,
            Some(selected_rows),
        )
    }

    fn read_dict(&mut self) -> Option<Result<CompressedPage, Error>> {
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

        // it will be read - take buffer
        let mut data = std::mem::take(&mut self.data_buffer);

        let maybe_page = read_dict_page(
            &mut self.reader,
            start,
            length,
            &mut self.buffer,
            &mut data,
            self.compression,
            &self.descriptor,
        );
        Some(maybe_page.map(CompressedPage::Dict))
    }
}

impl<R: Read + Seek> Iterator for IndexedPageReader<R> {
    type Item = Result<CompressedPage, Error>;

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
