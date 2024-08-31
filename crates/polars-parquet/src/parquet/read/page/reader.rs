use std::io::Seek;
use std::sync::OnceLock;

use parquet_format_safe::thrift::protocol::TCompactInputProtocol;
use polars_utils::mmap::{MemReader, MemSlice};

use super::PageIterator;
use crate::parquet::compression::Compression;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetaData, Descriptor};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, DataPageHeader, PageType,
    ParquetPageHeader,
};
use crate::parquet::CowBuffer;

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

/// A fallible [`Iterator`] of [`CompressedDataPage`]. This iterator reads pages back
/// to back until all pages have been consumed.
///
/// The pages from this iterator always have [`None`] [`crate::parquet::page::CompressedDataPage::selected_rows()`] since
/// filter pushdown is not supported without a
/// pre-computed [page index](https://github.com/apache/parquet-format/blob/master/PageIndex.md).
pub struct PageReader {
    // The source
    reader: MemReader,

    compression: Compression,

    // The number of values we have seen so far.
    seen_num_values: i64,

    // The number of total values in this column chunk.
    total_num_values: i64,

    descriptor: Descriptor,

    // The currently allocated buffer.
    pub(crate) scratch: Vec<u8>,

    // Maximum page size (compressed or uncompressed) to limit allocations
    max_page_size: usize,
}

impl PageReader {
    /// Returns a new [`PageReader`].
    ///
    /// It assumes that the reader has been `sought` (`seek`) to the beginning of `column`.
    /// The parameter `max_header_size`
    pub fn new(
        reader: MemReader,
        column: &ColumnChunkMetaData,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        Self::new_with_page_meta(reader, column.into(), scratch, max_page_size)
    }

    /// Create a a new [`PageReader`] with [`PageMetaData`].
    ///
    /// It assumes that the reader has been `sought` (`seek`) to the beginning of `column`.
    pub fn new_with_page_meta(
        reader: MemReader,
        reader_meta: PageMetaData,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        Self {
            reader,
            total_num_values: reader_meta.num_values,
            compression: reader_meta.compression,
            seen_num_values: 0,
            descriptor: reader_meta.descriptor,
            scratch,
            max_page_size,
        }
    }

    /// Returns the reader and this Readers' interval buffer
    pub fn into_inner(self) -> (MemReader, Vec<u8>) {
        (self.reader, self.scratch)
    }

    pub fn total_num_values(&self) -> usize {
        debug_assert!(self.total_num_values >= 0);
        self.total_num_values as usize
    }

    pub fn read_dict(&mut self) -> ParquetResult<Option<CompressedDictPage>> {
        // If there are no pages, we cannot check if the first page is a dictionary page. Just
        // return the fact there is no dictionary page.
        if self.reader.remaining_len() == 0 {
            return Ok(None);
        }

        // a dictionary page exists iff the first data page is not at the start of
        // the column
        let seek_offset = self.reader.position();
        let page_header = read_page_header(&mut self.reader, self.max_page_size)?;
        let page_type = page_header.type_.try_into()?;

        if !matches!(page_type, PageType::DictionaryPage) {
            self.reader
                .seek(std::io::SeekFrom::Start(seek_offset as u64))?;
            return Ok(None);
        }

        let read_size: usize = page_header.compressed_page_size.try_into()?;

        if read_size > self.max_page_size {
            return Err(ParquetError::WouldOverAllocate);
        }

        let buffer = self.reader.read_slice(read_size);

        if buffer.len() != read_size {
            return Err(ParquetError::oos(
                "The page header reported the wrong page size",
            ));
        }

        finish_page(page_header, buffer, self.compression, &self.descriptor).map(|p| {
            if let CompressedPage::Dict(d) = p {
                Some(d)
            } else {
                unreachable!()
            }
        })
    }
}

impl PageIterator for PageReader {
    fn swap_buffer(&mut self, scratch: &mut Vec<u8>) {
        std::mem::swap(&mut self.scratch, scratch)
    }
}

impl Iterator for PageReader {
    type Item = ParquetResult<CompressedPage>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = std::mem::take(&mut self.scratch);
        let maybe_maybe_page = next_page(self).transpose();
        if maybe_maybe_page.is_none() {
            // no page => we take back the buffer
            self.scratch = std::mem::take(&mut buffer);
        }
        maybe_maybe_page
    }
}

/// Reads Page header from Thrift.
pub(super) fn read_page_header(
    reader: &mut MemReader,
    max_size: usize,
) -> ParquetResult<ParquetPageHeader> {
    let mut prot = TCompactInputProtocol::new(reader, max_size);
    let page_header = ParquetPageHeader::read_from_in_protocol(&mut prot)?;
    Ok(page_header)
}

/// This function is lightweight and executes a minimal amount of work so that it is IO bounded.
// Any un-necessary CPU-intensive tasks SHOULD be executed on individual pages.
fn next_page(reader: &mut PageReader) -> ParquetResult<Option<CompressedPage>> {
    if reader.seen_num_values >= reader.total_num_values {
        return Ok(None);
    };
    build_page(reader)
}

pub(super) fn build_page(reader: &mut PageReader) -> ParquetResult<Option<CompressedPage>> {
    let page_header = read_page_header(&mut reader.reader, reader.max_page_size)?;

    reader.seen_num_values += get_page_num_values(&page_header)? as i64;

    let read_size: usize = page_header.compressed_page_size.try_into()?;

    if read_size > reader.max_page_size {
        return Err(ParquetError::WouldOverAllocate);
    }

    let buffer = reader.reader.read_slice(read_size);

    if buffer.len() != read_size {
        return Err(ParquetError::oos(
            "The page header reported the wrong page size",
        ));
    }

    finish_page(page_header, buffer, reader.compression, &reader.descriptor).map(Some)
}

pub(super) fn finish_page(
    page_header: ParquetPageHeader,
    data: MemSlice,
    compression: Compression,
    descriptor: &Descriptor,
) -> ParquetResult<CompressedPage> {
    let type_ = page_header.type_.try_into()?;
    let uncompressed_page_size = page_header.uncompressed_page_size.try_into()?;

    static DO_VERBOSE: OnceLock<bool> = OnceLock::new();
    let do_verbose = *DO_VERBOSE.get_or_init(|| std::env::var("PARQUET_DO_VERBOSE").is_ok());

    match type_ {
        PageType::DictionaryPage => {
            let dict_header = page_header.dictionary_page_header.as_ref().ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a dictionary page but the dictionary header is empty",
                )
            })?;

            if do_verbose {
                println!("DictPage ( )");
            }

            let is_sorted = dict_header.is_sorted.unwrap_or(false);

            // move the buffer to `dict_page`
            let page = CompressedDictPage::new(
                CowBuffer::Borrowed(data),
                compression,
                uncompressed_page_size,
                dict_header.num_values.try_into()?,
                is_sorted,
            );

            Ok(CompressedPage::Dict(page))
        },
        PageType::DataPage => {
            let header = page_header.data_page_header.ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v1 data page but the v1 data header is empty",
                )
            })?;

            if do_verbose {
                println!(
                    "DataPageV1 ( num_values: {}, datatype: {:?}, encoding: {:?} )",
                    header.num_values, descriptor.primitive_type, header.encoding
                );
            }

            Ok(CompressedPage::Data(CompressedDataPage::new_read(
                DataPageHeader::V1(header),
                CowBuffer::Borrowed(data),
                compression,
                uncompressed_page_size,
                descriptor.clone(),
            )))
        },
        PageType::DataPageV2 => {
            let header = page_header.data_page_header_v2.ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v2 data page but the v2 data header is empty",
                )
            })?;

            if do_verbose {
                println!(
                    "DataPageV2 ( num_values: {}, datatype: {:?}, encoding: {:?} )",
                    header.num_values, descriptor.primitive_type, header.encoding
                );
            }

            Ok(CompressedPage::Data(CompressedDataPage::new_read(
                DataPageHeader::V2(header),
                CowBuffer::Borrowed(data),
                compression,
                uncompressed_page_size,
                descriptor.clone(),
            )))
        },
    }
}

pub(super) fn get_page_num_values(header: &ParquetPageHeader) -> ParquetResult<i32> {
    let type_ = header.type_.try_into()?;
    Ok(match type_ {
        PageType::DataPage => {
            header
                .data_page_header
                .as_ref()
                .ok_or_else(|| {
                    ParquetError::oos(
                        "The page header type is a v1 data page but the v1 header is empty",
                    )
                })?
                .num_values
        },
        PageType::DataPageV2 => {
            header
                .data_page_header_v2
                .as_ref()
                .ok_or_else(|| {
                    ParquetError::oos(
                        "The page header type is a v1 data page but the v1 header is empty",
                    )
                })?
                .num_values
        },
        _ => 0,
    })
}
