use std::convert::TryInto;
use std::io::Read;
use std::sync::Arc;

use parquet_format_safe::thrift::protocol::TCompactInputProtocol;

use super::PageIterator;
use crate::parquet::compression::Compression;
use crate::parquet::error::{Error, Result};
use crate::parquet::indexes::Interval;
use crate::parquet::metadata::{ColumnChunkMetaData, Descriptor};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, DataPageHeader, PageType,
    ParquetPageHeader,
};
use crate::parquet::parquet_bridge::Encoding;

/// This meta is a small part of [`ColumnChunkMetaData`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageMetaData {
    /// The start offset of this column chunk in file.
    pub column_start: u64,
    /// The number of values in this column chunk.
    pub num_values: i64,
    /// Compression type
    pub compression: Compression,
    /// The descriptor of this parquet column
    pub descriptor: Descriptor,
}

impl PageMetaData {
    /// Returns a new [`PageMetaData`].
    pub fn new(
        column_start: u64,
        num_values: i64,
        compression: Compression,
        descriptor: Descriptor,
    ) -> Self {
        Self {
            column_start,
            num_values,
            compression,
            descriptor,
        }
    }
}

impl From<&ColumnChunkMetaData> for PageMetaData {
    fn from(column: &ColumnChunkMetaData) -> Self {
        Self {
            column_start: column.byte_range().0,
            num_values: column.num_values(),
            compression: column.compression(),
            descriptor: column.descriptor().descriptor.clone(),
        }
    }
}

/// Type declaration for a page filter
pub type PageFilter = Arc<dyn Fn(&Descriptor, &DataPageHeader) -> bool + Send + Sync>;

/// A fallible [`Iterator`] of [`CompressedDataPage`]. This iterator reads pages back
/// to back until all pages have been consumed.
/// The pages from this iterator always have [`None`] [`crate::parquet::page::CompressedDataPage::selected_rows()`] since
/// filter pushdown is not supported without a
/// pre-computed [page index](https://github.com/apache/parquet-format/blob/master/PageIndex.md).
pub struct PageReader<R: Read> {
    // The source
    reader: R,

    compression: Compression,

    // The number of values we have seen so far.
    seen_num_values: i64,

    // The number of total values in this column chunk.
    total_num_values: i64,

    pages_filter: PageFilter,

    descriptor: Descriptor,

    // The currently allocated buffer.
    pub(crate) scratch: Vec<u8>,

    // Maximum page size (compressed or uncompressed) to limit allocations
    max_page_size: usize,
}

impl<R: Read> PageReader<R> {
    /// Returns a new [`PageReader`].
    ///
    /// It assumes that the reader has been `sought` (`seek`) to the beginning of `column`.
    /// The parameter `max_header_size`
    pub fn new(
        reader: R,
        column: &ColumnChunkMetaData,
        pages_filter: PageFilter,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        Self::new_with_page_meta(reader, column.into(), pages_filter, scratch, max_page_size)
    }

    /// Create a a new [`PageReader`] with [`PageMetaData`].
    ///
    /// It assumes that the reader has been `sought` (`seek`) to the beginning of `column`.
    pub fn new_with_page_meta(
        reader: R,
        reader_meta: PageMetaData,
        pages_filter: PageFilter,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        Self {
            reader,
            total_num_values: reader_meta.num_values,
            compression: reader_meta.compression,
            seen_num_values: 0,
            descriptor: reader_meta.descriptor,
            pages_filter,
            scratch,
            max_page_size,
        }
    }

    /// Returns the reader and this Readers' interval buffer
    pub fn into_inner(self) -> (R, Vec<u8>) {
        (self.reader, self.scratch)
    }
}

impl<R: Read> PageIterator for PageReader<R> {
    fn swap_buffer(&mut self, scratch: &mut Vec<u8>) {
        std::mem::swap(&mut self.scratch, scratch)
    }
}

impl<R: Read> Iterator for PageReader<R> {
    type Item = Result<CompressedPage>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = std::mem::take(&mut self.scratch);
        let maybe_maybe_page = next_page(self, &mut buffer).transpose();
        if let Some(ref maybe_page) = maybe_maybe_page {
            if let Ok(CompressedPage::Data(page)) = maybe_page {
                // check if we should filter it (only valid for data pages)
                let to_consume = (self.pages_filter)(&self.descriptor, page.header());
                if !to_consume {
                    self.scratch = std::mem::take(&mut buffer);
                    return self.next();
                }
            }
        } else {
            // no page => we take back the buffer
            self.scratch = std::mem::take(&mut buffer);
        }
        maybe_maybe_page
    }
}

/// Reads Page header from Thrift.
pub(super) fn read_page_header<R: Read>(
    reader: &mut R,
    max_size: usize,
) -> Result<ParquetPageHeader> {
    let mut prot = TCompactInputProtocol::new(reader, max_size);
    let page_header = ParquetPageHeader::read_from_in_protocol(&mut prot)?;
    Ok(page_header)
}

/// This function is lightweight and executes a minimal amount of work so that it is IO bounded.
// Any un-necessary CPU-intensive tasks SHOULD be executed on individual pages.
fn next_page<R: Read>(
    reader: &mut PageReader<R>,
    buffer: &mut Vec<u8>,
) -> Result<Option<CompressedPage>> {
    if reader.seen_num_values >= reader.total_num_values {
        return Ok(None);
    };
    build_page(reader, buffer)
}

pub(super) fn build_page<R: Read>(
    reader: &mut PageReader<R>,
    buffer: &mut Vec<u8>,
) -> Result<Option<CompressedPage>> {
    let page_header = read_page_header(&mut reader.reader, reader.max_page_size)?;

    reader.seen_num_values += get_page_header(&page_header)?
        .map(|x| x.num_values() as i64)
        .unwrap_or_default();

    let read_size: usize = page_header.compressed_page_size.try_into()?;

    if read_size > reader.max_page_size {
        return Err(Error::WouldOverAllocate);
    }

    buffer.clear();
    buffer.try_reserve(read_size)?;
    let bytes_read = reader
        .reader
        .by_ref()
        .take(read_size as u64)
        .read_to_end(buffer)?;

    if bytes_read != read_size {
        return Err(Error::oos(
            "The page header reported the wrong page size".to_string(),
        ));
    }

    finish_page(
        page_header,
        buffer,
        reader.compression,
        &reader.descriptor,
        None,
    )
    .map(Some)
}

pub(super) fn finish_page(
    page_header: ParquetPageHeader,
    data: &mut Vec<u8>,
    compression: Compression,
    descriptor: &Descriptor,
    selected_rows: Option<Vec<Interval>>,
) -> Result<CompressedPage> {
    let type_ = page_header.type_.try_into()?;
    let uncompressed_page_size = page_header.uncompressed_page_size.try_into()?;
    match type_ {
        PageType::DictionaryPage => {
            let dict_header = page_header.dictionary_page_header.as_ref().ok_or_else(|| {
                Error::oos(
                    "The page header type is a dictionary page but the dictionary header is empty",
                )
            })?;
            let is_sorted = dict_header.is_sorted.unwrap_or(false);

            // move the buffer to `dict_page`
            let page = CompressedDictPage::new(
                std::mem::take(data),
                compression,
                uncompressed_page_size,
                dict_header.num_values.try_into()?,
                is_sorted,
            );

            Ok(CompressedPage::Dict(page))
        },
        PageType::DataPage => {
            let header = page_header.data_page_header.ok_or_else(|| {
                Error::oos("The page header type is a v1 data page but the v1 data header is empty")
            })?;

            Ok(CompressedPage::Data(CompressedDataPage::new_read(
                DataPageHeader::V1(header),
                std::mem::take(data),
                compression,
                uncompressed_page_size,
                descriptor.clone(),
                selected_rows,
            )))
        },
        PageType::DataPageV2 => {
            let header = page_header.data_page_header_v2.ok_or_else(|| {
                Error::oos("The page header type is a v2 data page but the v2 data header is empty")
            })?;

            Ok(CompressedPage::Data(CompressedDataPage::new_read(
                DataPageHeader::V2(header),
                std::mem::take(data),
                compression,
                uncompressed_page_size,
                descriptor.clone(),
                selected_rows,
            )))
        },
    }
}

pub(super) fn get_page_header(header: &ParquetPageHeader) -> Result<Option<DataPageHeader>> {
    let type_ = header.type_.try_into()?;
    Ok(match type_ {
        PageType::DataPage => {
            let header = header.data_page_header.clone().ok_or_else(|| {
                Error::oos("The page header type is a v1 data page but the v1 header is empty")
            })?;
            let _: Encoding = header.encoding.try_into()?;
            let _: Encoding = header.repetition_level_encoding.try_into()?;
            let _: Encoding = header.definition_level_encoding.try_into()?;

            Some(DataPageHeader::V1(header))
        },
        PageType::DataPageV2 => {
            let header = header.data_page_header_v2.clone().ok_or_else(|| {
                Error::oos("The page header type is a v1 data page but the v1 header is empty")
            })?;
            let _: Encoding = header.encoding.try_into()?;
            Some(DataPageHeader::V2(header))
        },
        _ => None,
    })
}
