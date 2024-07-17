use std::collections::VecDeque;
use std::io::SeekFrom;
use std::marker::PhantomData;

use super::reader::{finish_page, read_page_header, PageMetaData};
use super::ReadSliced;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::indexes::{FilteredPage, Interval};
use crate::parquet::metadata::{ColumnChunkMetaData, Descriptor};
use crate::parquet::page::{
    CompressedDictPage, CompressedPage, CowBuffer, DictPage, ParquetPageHeader,
};
use crate::parquet::parquet_bridge::Compression;
use crate::parquet::read::compression::decompress_buffer;

/// A fallible [`Iterator`] of [`CompressedPage`]. This iterator leverages page indexes
/// to skip pages that are not needed. Consequently, the pages from this
/// iterator always have [`Some`] [`crate::parquet::page::CompressedDataPage::selected_rows()`]
pub struct IndexedPageReader<'a, R: ReadSliced<'a>> {
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

    dict: Option<DictPage>,

    _pd: PhantomData<&'a ()>,
}

fn read_page<'a, R: ReadSliced<'a>>(
    reader: &mut R,
    start: u64,
    length: usize,
) -> Result<(ParquetPageHeader, CowBuffer<'a>), ParquetError> {
    // seek to the page
    reader.seek(SeekFrom::Start(start))?;

    // deserialize [header]
    let page_header = read_page_header(reader, 1024 * 1024)?;

    // read [header][data] to buffer
    let buffer = reader.read_sliced(length)?;

    Ok((page_header, buffer))
}

fn read_dict_page<'a, R: ReadSliced<'a>>(
    reader: &mut R,
    start: u64,
    length: usize,
    compression: Compression,
    descriptor: &Descriptor,
) -> Result<CompressedDictPage<'a>, ParquetError> {
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

impl<'a, R: ReadSliced<'a>> IndexedPageReader<'a, R> {
    /// Returns a new [`IndexedPageReader`].
    pub fn new(
        reader: R,
        column: &ColumnChunkMetaData,
        pages: Vec<FilteredPage>,
        buffer: Vec<u8>,
        data_buffer: Vec<u8>,
    ) -> ParquetResult<Self> {
        Self::new_with_page_meta(reader, column.into(), pages, buffer, data_buffer)
    }

    /// Returns a new [`IndexedPageReader`] with [`PageMetaData`].
    pub fn new_with_page_meta(
        reader: R,
        column: PageMetaData,
        pages: Vec<FilteredPage>,
        buffer: Vec<u8>,
        data_buffer: Vec<u8>,
    ) -> ParquetResult<Self> {
        let pages = pages.into_iter().collect();

        Self {
            reader,
            column_start: column.column_start,
            compression: column.compression,
            descriptor: column.descriptor,
            buffer,
            data_buffer,
            pages,
            dict: None,
            _pd: PhantomData,
        }
        .with_dict()
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
    ) -> Result<CompressedPage<'a>, ParquetError> {
        let (page_header, data) = read_page(&mut self.reader, start, length)?;

        finish_page(
            page_header,
            data,
            self.compression,
            &self.descriptor,
            Some(selected_rows),
        )
    }

    fn with_dict(mut self) -> Result<Self, ParquetError> {
        // a dictionary page exists iff the first data page is not at the start of
        // the column
        let opt_compressed_page_dims = self.pages.front().and_then(|page| {
            let length = (page.start - self.column_start) as usize;
            (length > 0).then_some((self.column_start, length))
        });
        let Some((start, length)) = opt_compressed_page_dims else {
            return Ok(self);
        };

        let compressed_dict_page = read_dict_page(
            &mut self.reader,
            start,
            length,
            self.compression,
            &self.descriptor,
        )?;

        let num_values = compressed_dict_page.num_values;
        let is_sorted = compressed_dict_page.is_sorted;

        // @NOTE: We could check whether something was actually decompressed here and just reuse
        // the buffer, but it requires a lot of lifetiming in the rest of the code.
        let mut buffer = Vec::new();
        decompress_buffer(&mut CompressedPage::Dict(compressed_dict_page), &mut buffer)?;
        let dict = DictPage::new(buffer, num_values, is_sorted);

        self.dict = Some(dict);

        Ok(self)
    }
}

impl<'a, R: ReadSliced<'a>> Iterator for IndexedPageReader<'a, R> {
    type Item = Result<CompressedPage<'a>, ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let page = self.pages.pop_front()?;

        if page.selected_rows.is_empty() {
            self.next()
        } else {
            Some(self.read_page(page.start, page.length, page.selected_rows))
        }
    }
}
