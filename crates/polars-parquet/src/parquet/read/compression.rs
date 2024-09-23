use parquet_format_safe::DataPageHeaderV2;

use super::PageReader;
use crate::parquet::compression::{self, Compression};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{
    CompressedDataPage, CompressedPage, DataPage, DataPageHeader, DictPage, Page,
};
use crate::parquet::CowBuffer;

fn decompress_v1(
    compressed: &[u8],
    compression: Compression,
    buffer: &mut [u8],
) -> ParquetResult<()> {
    compression::decompress(compression, compressed, buffer)
}

fn decompress_v2(
    compressed: &[u8],
    page_header: &DataPageHeaderV2,
    compression: Compression,
    buffer: &mut [u8],
) -> ParquetResult<()> {
    // When processing data page v2, depending on enabled compression for the
    // page, we should account for uncompressed data ('offset') of
    // repetition and definition levels.
    //
    // We always use 0 offset for other pages other than v2, `true` flag means
    // that compression will be applied if decompressor is defined
    let offset = (page_header.definition_levels_byte_length
        + page_header.repetition_levels_byte_length) as usize;
    // When is_compressed flag is missing the page is considered compressed
    let can_decompress = page_header.is_compressed.unwrap_or(true);

    if can_decompress {
        if offset > buffer.len() || offset > compressed.len() {
            return Err(ParquetError::oos(
                "V2 Page Header reported incorrect offset to compressed data",
            ));
        }

        (buffer[..offset]).copy_from_slice(&compressed[..offset]);

        compression::decompress(compression, &compressed[offset..], &mut buffer[offset..])?;
    } else {
        if buffer.len() != compressed.len() {
            return Err(ParquetError::oos(
                "V2 Page Header reported incorrect decompressed size",
            ));
        }
        buffer.copy_from_slice(compressed);
    }
    Ok(())
}

/// Decompresses the page, using `buffer` for decompression.
/// If `page.buffer.len() == 0`, there was no decompression and the buffer was moved.
/// Else, decompression took place.
pub fn decompress(compressed_page: CompressedPage, buffer: &mut Vec<u8>) -> ParquetResult<Page> {
    Ok(match (compressed_page.compression(), compressed_page) {
        (Compression::Uncompressed, CompressedPage::Data(page)) => Page::Data(DataPage::new_read(
            page.header,
            page.buffer,
            page.descriptor,
        )),
        (_, CompressedPage::Data(page)) => {
            // prepare the compression buffer
            let read_size = page.uncompressed_size();

            if read_size > buffer.capacity() {
                // dealloc and ignore region, replacing it by a new region.
                // This won't reallocate - it frees and calls `alloc_zeroed`
                *buffer = vec![0; read_size];
            } else if read_size > buffer.len() {
                // fill what we need with zeros so that we can use them in `Read`.
                // This won't reallocate
                buffer.resize(read_size, 0);
            } else {
                buffer.truncate(read_size);
            }

            match page.header() {
                DataPageHeader::V1(_) => decompress_v1(&page.buffer, page.compression, buffer)?,
                DataPageHeader::V2(header) => {
                    decompress_v2(&page.buffer, header, page.compression, buffer)?
                },
            }
            let buffer = CowBuffer::Owned(std::mem::take(buffer));

            Page::Data(DataPage::new_read(page.header, buffer, page.descriptor))
        },
        (Compression::Uncompressed, CompressedPage::Dict(page)) => Page::Dict(DictPage {
            buffer: page.buffer,
            num_values: page.num_values,
            is_sorted: page.is_sorted,
        }),
        (_, CompressedPage::Dict(page)) => {
            // prepare the compression buffer
            let read_size = page.uncompressed_page_size;

            if read_size > buffer.capacity() {
                // dealloc and ignore region, replacing it by a new region.
                // This won't reallocate - it frees and calls `alloc_zeroed`
                *buffer = vec![0; read_size];
            } else if read_size > buffer.len() {
                // fill what we need with zeros so that we can use them in `Read`.
                // This won't reallocate
                buffer.resize(read_size, 0);
            } else {
                buffer.truncate(read_size);
            }
            decompress_v1(&page.buffer, page.compression(), buffer)?;
            let buffer = CowBuffer::Owned(std::mem::take(buffer));

            Page::Dict(DictPage {
                buffer,
                num_values: page.num_values,
                is_sorted: page.is_sorted,
            })
        },
    })
}

type _Decompressor<I> = streaming_decompression::Decompressor<
    CompressedPage,
    Page,
    fn(CompressedPage, &mut Vec<u8>) -> ParquetResult<Page>,
    ParquetError,
    I,
>;

impl streaming_decompression::Compressed for CompressedPage {
    #[inline]
    fn is_compressed(&self) -> bool {
        self.compression() != Compression::Uncompressed
    }
}

impl streaming_decompression::Decompressed for Page {
    #[inline]
    fn buffer_mut(&mut self) -> &mut Vec<u8> {
        self.buffer_mut()
    }
}

/// A [`FallibleStreamingIterator`] that decompresses [`CompressedPage`] into [`DataPage`].
/// # Implementation
/// This decompressor uses an internal [`Vec<u8>`] to perform decompressions which
/// is reused across pages, so that a single allocation is required.
/// If the pages are not compressed, the internal buffer is not used.
pub struct BasicDecompressor {
    reader: PageReader,
    buffer: Vec<u8>,
}

impl BasicDecompressor {
    /// Create a new [`BasicDecompressor`]
    pub fn new(reader: PageReader, buffer: Vec<u8>) -> Self {
        Self { reader, buffer }
    }

    /// The total number of values is given from the `ColumnChunk` metadata.
    ///
    /// - Nested column: equal to the number of non-null values at the lowest nesting level.
    /// - Unnested column: equal to the number of non-null rows.
    pub fn total_num_values(&self) -> usize {
        self.reader.total_num_values()
    }

    /// Returns its internal buffer, consuming itself.
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }

    pub fn read_dict_page(&mut self) -> ParquetResult<Option<DictPage>> {
        match self.reader.read_dict()? {
            None => Ok(None),
            Some(p) => {
                let num_values = p.num_values;
                let page =
                    decompress(CompressedPage::Dict(p), &mut Vec::with_capacity(num_values))?;

                match page {
                    Page::Dict(d) => Ok(Some(d)),
                    Page::Data(_) => unreachable!(),
                }
            },
        }
    }

    pub fn reuse_page_buffer(&mut self, page: DataPage) {
        let buffer = match page.buffer {
            CowBuffer::Borrowed(_) => return,
            CowBuffer::Owned(vec) => vec,
        };

        if self.buffer.capacity() > buffer.capacity() {
            return;
        };

        self.buffer = buffer;
    }
}

pub struct DataPageItem {
    page: CompressedDataPage,
}

impl DataPageItem {
    pub fn num_values(&self) -> usize {
        self.page.num_values()
    }

    pub fn decompress(self, decompressor: &mut BasicDecompressor) -> ParquetResult<DataPage> {
        let p = decompress(CompressedPage::Data(self.page), &mut decompressor.buffer)?;
        let Page::Data(p) = p else {
            panic!("Decompressing a data page should result in a data page");
        };

        Ok(p)
    }
}

impl Iterator for BasicDecompressor {
    type Item = ParquetResult<DataPageItem>;

    fn next(&mut self) -> Option<Self::Item> {
        let page = match self.reader.next() {
            None => return None,
            Some(Err(e)) => return Some(Err(e)),
            Some(Ok(p)) => p,
        };

        let CompressedPage::Data(page) = page else {
            return Some(Err(ParquetError::oos(
                "Found dictionary page beyond the first page of a column chunk",
            )));
        };

        Some(Ok(DataPageItem { page }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.reader.size_hint()
    }
}
