use parquet_format_safe::DataPageHeaderV2;
use streaming_decompression;

use crate::compression::{self, Compression};
use crate::error::{Error, Result};
use crate::page::{CompressedPage, DataPage, DataPageHeader, DictPage, Page};
use crate::FallibleStreamingIterator;

use super::page::PageIterator;

fn decompress_v1(compressed: &[u8], compression: Compression, buffer: &mut [u8]) -> Result<()> {
    compression::decompress(compression, compressed, buffer)
}

fn decompress_v2(
    compressed: &[u8],
    page_header: &DataPageHeaderV2,
    compression: Compression,
    buffer: &mut [u8],
) -> Result<()> {
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
            return Err(Error::OutOfSpec(
                "V2 Page Header reported incorrect offset to compressed data".to_string(),
            ));
        }

        (buffer[..offset]).copy_from_slice(&compressed[..offset]);

        compression::decompress(compression, &compressed[offset..], &mut buffer[offset..])?;
    } else {
        if buffer.len() != compressed.len() {
            return Err(Error::OutOfSpec(
                "V2 Page Header reported incorrect decompressed size".to_string(),
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
) -> Result<bool> {
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
                }
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
        std::mem::swap(compressed_page.buffer(), buffer);
        Ok(false)
    }
}

fn create_page(compressed_page: CompressedPage, buffer: Vec<u8>) -> Page {
    match compressed_page {
        CompressedPage::Data(page) => Page::Data(DataPage::new_read(
            page.header,
            buffer,
            page.descriptor,
            page.selected_rows,
        )),
        CompressedPage::Dict(page) => Page::Dict(DictPage {
            buffer,
            num_values: page.num_values,
            is_sorted: page.is_sorted,
        }),
    }
}

/// Decompresses the page, using `buffer` for decompression.
/// If `page.buffer.len() == 0`, there was no decompression and the buffer was moved.
/// Else, decompression took place.
pub fn decompress(mut compressed_page: CompressedPage, buffer: &mut Vec<u8>) -> Result<Page> {
    decompress_buffer(&mut compressed_page, buffer)?;
    Ok(create_page(compressed_page, std::mem::take(buffer)))
}

fn decompress_reuse<P: PageIterator>(
    mut compressed_page: CompressedPage,
    iterator: &mut P,
    buffer: &mut Vec<u8>,
) -> Result<(Page, bool)> {
    let was_decompressed = decompress_buffer(&mut compressed_page, buffer)?;

    if was_decompressed {
        iterator.swap_buffer(compressed_page.buffer())
    };

    let new_page = create_page(compressed_page, std::mem::take(buffer));

    Ok((new_page, was_decompressed))
}

/// Decompressor that allows re-using the page buffer of [`PageIterator`].
/// # Implementation
/// The implementation depends on whether a page is compressed or not.
/// > `PageReader(a)`, `CompressedPage(b)`, `Decompressor(c)`, `DecompressedPage(d)`
/// ### un-compressed pages:
/// > page iter: `a` is swapped with `b`
/// > decompress iter: `b` is swapped with `d`, `b` is swapped with `a`
/// therefore:
/// * `PageReader` has its buffer back
/// * `Decompressor`'s buffer is un-used
/// * `DecompressedPage` has the same data as `CompressedPage` had
/// ### compressed pages:
/// > page iter: `a` is swapped with `b`
/// > decompress iter:
/// > * `b` is decompressed into `c`
/// > * `b` is swapped with `a`
/// > * `c` is moved to `d`
/// > * (next iteration): `d` is moved to `c`
/// therefore, while the page is available:
/// * `PageReader` has its buffer back
/// * `Decompressor`'s buffer empty
/// * `DecompressedPage` has the decompressed buffer
/// after the page is used:
/// * `PageReader` has its buffer back
/// * `Decompressor` has its buffer back
/// * `DecompressedPage` has an empty buffer
pub struct Decompressor<P: PageIterator> {
    iter: P,
    buffer: Vec<u8>,
    current: Option<Page>,
    was_decompressed: bool,
}

impl<P: PageIterator> Decompressor<P> {
    /// Creates a new [`Decompressor`].
    pub fn new(iter: P, buffer: Vec<u8>) -> Self {
        Self {
            iter,
            buffer,
            current: None,
            was_decompressed: false,
        }
    }

    /// Returns two buffers: the first buffer corresponds to the page buffer,
    /// the second to the decompression buffer.
    pub fn into_buffers(mut self) -> (Vec<u8>, Vec<u8>) {
        let mut page_buffer = vec![];
        self.iter.swap_buffer(&mut page_buffer);
        (page_buffer, self.buffer)
    }
}

impl<P: PageIterator> FallibleStreamingIterator for Decompressor<P> {
    type Item = Page;
    type Error = Error;

    fn advance(&mut self) -> Result<()> {
        if let Some(page) = self.current.as_mut() {
            if self.was_decompressed {
                self.buffer = std::mem::take(page.buffer());
            } else {
                self.iter.swap_buffer(page.buffer());
            }
        }

        let next = self
            .iter
            .next()
            .map(|x| {
                x.and_then(|x| {
                    let (page, was_decompressed) =
                        decompress_reuse(x, &mut self.iter, &mut self.buffer)?;
                    self.was_decompressed = was_decompressed;
                    Ok(page)
                })
            })
            .transpose()?;
        self.current = next;
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }
}

type _Decompressor<I> = streaming_decompression::Decompressor<
    CompressedPage,
    Page,
    fn(CompressedPage, &mut Vec<u8>) -> Result<Page>,
    Error,
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
        self.buffer()
    }
}

/// A [`FallibleStreamingIterator`] that decompresses [`CompressedPage`] into [`DataPage`].
/// # Implementation
/// This decompressor uses an internal [`Vec<u8>`] to perform decompressions which
/// is re-used across pages, so that a single allocation is required.
/// If the pages are not compressed, the internal buffer is not used.
pub struct BasicDecompressor<I: Iterator<Item = Result<CompressedPage>>> {
    iter: _Decompressor<I>,
}

impl<I> BasicDecompressor<I>
where
    I: Iterator<Item = Result<CompressedPage>>,
{
    /// Returns a new [`BasicDecompressor`].
    pub fn new(iter: I, buffer: Vec<u8>) -> Self {
        Self {
            iter: _Decompressor::new(iter, buffer, decompress),
        }
    }

    /// Returns its internal buffer, consuming itself.
    pub fn into_inner(self) -> Vec<u8> {
        self.iter.into_inner()
    }
}

impl<I> FallibleStreamingIterator for BasicDecompressor<I>
where
    I: Iterator<Item = Result<CompressedPage>>,
{
    type Item = Page;
    type Error = Error;

    fn advance(&mut self) -> Result<()> {
        self.iter.advance()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.iter.get()
    }
}
