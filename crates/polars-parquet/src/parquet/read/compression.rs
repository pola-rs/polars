use parquet_format_safe::DataPageHeaderV2;

use crate::parquet::compression::{self, Compression};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{CompressedPage, DataPage, DataPageHeader, DictPage, Page};
use crate::parquet::{CowBuffer, FallibleStreamingIterator};

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

/// decompresses a [`CompressedDataPage`] into `buffer`.
/// If the page is un-compressed, `buffer` is swapped instead.
/// Returns whether the page was decompressed.
pub fn decompress_buffer(
    compressed_page: &mut CompressedPage,
    buffer: &mut Vec<u8>,
) -> ParquetResult<bool> {
    if compressed_page.compression() != Compression::Uncompressed {
        // prepare the compression buffer
        let read_size = compressed_page.uncompressed_size();

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
        match compressed_page {
            CompressedPage::Data(compressed_page) => match compressed_page.header() {
                DataPageHeader::V1(_) => {
                    decompress_v1(&compressed_page.buffer, compressed_page.compression, buffer)?
                },
                DataPageHeader::V2(header) => decompress_v2(
                    &compressed_page.buffer,
                    header,
                    compressed_page.compression,
                    buffer,
                )?,
            },
            CompressedPage::Dict(page) => decompress_v1(&page.buffer, page.compression(), buffer)?,
        }
        Ok(true)
    } else {
        // page.buffer is already decompressed => swap it with `buffer`, making `page.buffer` the
        // decompression buffer and `buffer` the decompressed buffer
        std::mem::swap(&mut compressed_page.buffer().to_vec(), buffer);
        Ok(false)
    }
}

fn create_page(compressed_page: CompressedPage, buffer: Vec<u8>) -> Page {
    match compressed_page {
        CompressedPage::Data(page) => Page::Data(DataPage::new_read(
            page.header,
            CowBuffer::Owned(buffer),
            page.descriptor,
            page.selected_rows,
        )),
        CompressedPage::Dict(page) => Page::Dict(DictPage {
            buffer: CowBuffer::Owned(buffer),
            num_values: page.num_values,
            is_sorted: page.is_sorted,
        }),
    }
}

/// Decompresses the page, using `buffer` for decompression.
/// If `page.buffer.len() == 0`, there was no decompression and the buffer was moved.
/// Else, decompression took place.
pub fn decompress(
    mut compressed_page: CompressedPage,
    buffer: &mut Vec<u8>,
) -> ParquetResult<Page> {
    decompress_buffer(&mut compressed_page, buffer)?;
    Ok(create_page(compressed_page, std::mem::take(buffer)))
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
pub struct BasicDecompressor<I: Iterator<Item = ParquetResult<CompressedPage>>> {
    iter: _Decompressor<I>,
    peeked: Option<Page>,
}

impl<I> BasicDecompressor<I>
where
    I: Iterator<Item = ParquetResult<CompressedPage>>,
{
    /// Returns a new [`BasicDecompressor`].
    pub fn new(iter: I, buffer: Vec<u8>) -> Self {
        Self {
            iter: _Decompressor::new(iter, buffer, decompress),
            peeked: None,
        }
    }

    /// Returns its internal buffer, consuming itself.
    pub fn into_inner(self) -> Vec<u8> {
        self.iter.into_inner()
    }

    pub fn read_dict_page(&mut self) -> ParquetResult<Option<DictPage>> {
        match self.iter.next()? {
            Some(Page::Data(page)) => {
                self.peeked = Some(Page::Data(page.clone()));
                Ok(None)
            },
            Some(Page::Dict(page)) => Ok(Some(page.clone())),
            None => Ok(None),
        }
    }
}

impl<I> FallibleStreamingIterator for BasicDecompressor<I>
where
    I: Iterator<Item = ParquetResult<CompressedPage>>,
{
    type Item = Page;
    type Error = ParquetError;

    fn advance(&mut self) -> ParquetResult<()> {
        if self.peeked.take().is_some() {
            return Ok(());
        }

        self.iter.advance()
    }

    fn get(&self) -> Option<&Self::Item> {
        if let Some(peeked) = self.peeked.as_ref() {
            return Some(peeked);
        }

        self.iter.get()
    }
}
