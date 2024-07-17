use std::io::{Read, Seek};
use std::marker::PhantomData;
use std::sync::Arc;

use parquet_format_safe::thrift::protocol::TCompactInputProtocol;

use super::PageIterator;
use crate::parquet::compression::Compression;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::indexes::Interval;
use crate::parquet::metadata::{ColumnChunkMetaData, Descriptor};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, CowBuffer, DataPageHeader, DictPage, PageType, ParquetPageHeader
};
use crate::parquet::parquet_bridge::Encoding;

pub trait ReadSliced<'a>: Read + Seek {
    fn read_sliced(&mut self, n: usize) -> std::io::Result<CowBuffer<'a>>;
}

impl<'a, 'b, T: ReadSliced<'a>> ReadSliced<'a> for &'b mut T {
    fn read_sliced(&mut self, n: usize) -> std::io::Result<CowBuffer<'a>> {
        T::read_sliced(self, n)
    }
}

impl ReadSliced<'static> for std::fs::File {
    fn read_sliced(&mut self, n: usize) -> std::io::Result<CowBuffer<'static>> {
        let mut buffer = Vec::with_capacity(n);
        self.take(n as u64).read_to_end(&mut buffer)?;
        Ok(CowBuffer::Owned(buffer))
    }
}

impl<'a> ReadSliced<'a> for std::io::Cursor<&'a [u8]> {
    fn read_sliced(&mut self, n: usize) -> std::io::Result<CowBuffer<'a>> {
        self.seek_relative(n as i64)?;
        let seek_position = self.position();
        let slice = *self.get_ref();
        Ok(CowBuffer::Borrowed(
            &slice[seek_position as usize - n..seek_position as usize],
        ))
    }
}

impl ReadSliced<'static> for std::io::Cursor<Vec<u8>> {
    fn read_sliced(&mut self, n: usize) -> std::io::Result<CowBuffer<'static>> {
        self.seek_relative(n as i64)?;
        let seek_position = self.position();
        let slice = self.get_ref();
        Ok(CowBuffer::Owned(
            slice[seek_position as usize - n..seek_position as usize].to_vec(),
        ))
    }
}

impl<'a> ReadSliced<'a> for std::io::Cursor<CowBuffer<'a>> {
    fn read_sliced(&mut self, n: usize) -> std::io::Result<CowBuffer<'a>> {
        self.seek_relative(n as i64)?;
        let seek_position = self.position();
        let slice = self.get_ref();
        Ok(match slice {
            CowBuffer::Borrowed(slice) => {
                CowBuffer::Borrowed(&slice[seek_position as usize - n..seek_position as usize])
            },
            CowBuffer::Owned(_) => {
                CowBuffer::Owned(slice[seek_position as usize - n..seek_position as usize].to_vec())
            },
        })
    }
}

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
pub struct PageReader<'a, R: ReadSliced<'a>> {
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

    dict: Option<DictPage>,

    _pd: PhantomData<&'a ()>,
}

impl<'a, R: ReadSliced<'a>> PageReader<'a, R> {
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
    ) -> ParquetResult<Self> {
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
    ) -> ParquetResult<Self> {
        Self {
            reader,
            total_num_values: reader_meta.num_values,
            compression: reader_meta.compression,
            seen_num_values: 0,
            descriptor: reader_meta.descriptor,
            pages_filter,
            scratch,
            max_page_size,
            dict: None,
            _pd: PhantomData,
        }.with_dict()
    }

    /// Returns the reader and this Readers' interval buffer
    pub fn into_inner(self) -> (R, Vec<u8>) {
        (self.reader, self.scratch)
    }

    pub fn with_dict(mut self) -> ParquetResult<Self> {
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

impl<'a, R: ReadSliced<'a>> PageIterator<'a> for PageReader<'a, R> {
    fn swap_buffer(&mut self, scratch: &mut Vec<u8>) {
        std::mem::swap(&mut self.scratch, scratch)
    }
}

impl<'a, R: ReadSliced<'a>> Iterator for PageReader<'a, R> {
    type Item = ParquetResult<CompressedPage<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = std::mem::take(&mut self.scratch);
        let maybe_maybe_page = next_page(self).transpose();
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
pub(super) fn read_page_header<'a, R: ReadSliced<'a>>(
    reader: &mut R,
    max_size: usize,
) -> ParquetResult<ParquetPageHeader> {
    let mut prot = TCompactInputProtocol::new(reader, max_size);
    let page_header = ParquetPageHeader::read_from_in_protocol(&mut prot)?;
    Ok(page_header)
}

/// This function is lightweight and executes a minimal amount of work so that it is IO bounded.
// Any un-necessary CPU-intensive tasks SHOULD be executed on individual pages.
fn next_page<'a, R: ReadSliced<'a>>(
    reader: &mut PageReader<'a, R>,
) -> ParquetResult<Option<CompressedPage<'a>>> {
    if reader.seen_num_values >= reader.total_num_values {
        return Ok(None);
    };
    build_page(reader).map(Some)
}

pub(super) fn build_page<'a, R: ReadSliced<'a>>(
    reader: &mut PageReader<'a, R>,
) -> ParquetResult<CompressedPage<'a>> {
    let page_header = read_page_header(&mut reader.reader, reader.max_page_size)?;

    reader.seen_num_values += get_page_header(&page_header)?
        .map(|x| x.num_values() as i64)
        .unwrap_or_default();

    let read_size: usize = page_header.compressed_page_size.try_into()?;

    if read_size > reader.max_page_size {
        return Err(ParquetError::WouldOverAllocate);
    }

    let buffer = reader
        .reader
        .read_sliced(read_size)
        .map_err(|_| ParquetError::oos("The page header reported the wrong page size"))?;

    finish_page(
        page_header,
        buffer,
        reader.compression,
        &reader.descriptor,
        None,
    )
}

pub(super) fn finish_page<'a>(
    page_header: ParquetPageHeader,
    data: CowBuffer<'a>,
    compression: Compression,
    descriptor: &Descriptor,
    selected_rows: Option<Vec<Interval>>,
) -> ParquetResult<CompressedPage<'a>> {
    let type_ = page_header.type_.try_into()?;
    let uncompressed_page_size = page_header.uncompressed_page_size.try_into()?;
    match type_ {
        PageType::DictionaryPage => {
            let dict_header = page_header.dictionary_page_header.as_ref().ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a dictionary page but the dictionary header is empty",
                )
            })?;
            let is_sorted = dict_header.is_sorted.unwrap_or(false);

            // move the buffer to `dict_page`
            let page = CompressedDictPage::new(
                data,
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

            Ok(CompressedPage::Data(CompressedDataPage::new_read(
                DataPageHeader::V1(header),
                data,
                compression,
                uncompressed_page_size,
                descriptor.clone(),
                selected_rows,
            )))
        },
        PageType::DataPageV2 => {
            let header = page_header.data_page_header_v2.ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v2 data page but the v2 data header is empty",
                )
            })?;

            Ok(CompressedPage::Data(CompressedDataPage::new_read(
                DataPageHeader::V2(header),
                data,
                compression,
                uncompressed_page_size,
                descriptor.clone(),
                selected_rows,
            )))
        },
    }
}

pub(super) fn get_page_header(header: &ParquetPageHeader) -> ParquetResult<Option<DataPageHeader>> {
    let type_ = header.type_.try_into()?;
    Ok(match type_ {
        PageType::DataPage => {
            let header = header.data_page_header.clone().ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v1 data page but the v1 header is empty",
                )
            })?;
            let _: Encoding = header.encoding.try_into()?;
            let _: Encoding = header.repetition_level_encoding.try_into()?;
            let _: Encoding = header.definition_level_encoding.try_into()?;

            Some(DataPageHeader::V1(header))
        },
        PageType::DataPageV2 => {
            let header = header.data_page_header_v2.clone().ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v1 data page but the v1 header is empty",
                )
            })?;
            let _: Encoding = header.encoding.try_into()?;
            Some(DataPageHeader::V2(header))
        },
        _ => None,
    })
}
