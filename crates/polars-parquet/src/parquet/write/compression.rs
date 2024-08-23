use crate::parquet::compression::CompressionOptions;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, DataPage, DataPageHeader, DictPage,
    Page,
};
use crate::parquet::{compression, CowBuffer, FallibleStreamingIterator};

/// Compresses a [`DataPage`] into a [`CompressedDataPage`].
fn compress_data(
    page: DataPage,
    mut compressed_buffer: Vec<u8>,
    compression: CompressionOptions,
) -> ParquetResult<CompressedDataPage> {
    let DataPage {
        mut buffer,
        header,
        descriptor,
        num_rows,
    } = page;
    let uncompressed_page_size = buffer.len();
    let num_rows = num_rows.expect("We should have num_rows when we are writing");
    if compression != CompressionOptions::Uncompressed {
        match &header {
            DataPageHeader::V1(_) => {
                compression::compress(compression, &buffer, &mut compressed_buffer)?;
            },
            DataPageHeader::V2(header) => {
                let levels_byte_length = (header.repetition_levels_byte_length
                    + header.definition_levels_byte_length)
                    as usize;
                compressed_buffer.extend_from_slice(&buffer[..levels_byte_length]);
                compression::compress(
                    compression,
                    &buffer[levels_byte_length..],
                    &mut compressed_buffer,
                )?;
            },
        };
    } else {
        std::mem::swap(buffer.to_mut(), &mut compressed_buffer);
    }

    Ok(CompressedDataPage::new(
        header,
        CowBuffer::Owned(compressed_buffer),
        compression.into(),
        uncompressed_page_size,
        descriptor,
        num_rows,
    ))
}

fn compress_dict(
    page: DictPage,
    mut compressed_buffer: Vec<u8>,
    compression: CompressionOptions,
) -> ParquetResult<CompressedDictPage> {
    let DictPage {
        buffer,
        num_values,
        is_sorted,
    } = page;

    let uncompressed_page_size = buffer.len();
    let compressed_buffer = if compression != CompressionOptions::Uncompressed {
        compression::compress(compression, &buffer, &mut compressed_buffer)?;
        CowBuffer::Owned(compressed_buffer)
    } else {
        buffer
    };

    Ok(CompressedDictPage::new(
        compressed_buffer,
        compression.into(),
        uncompressed_page_size,
        num_values,
        is_sorted,
    ))
}

/// Compresses an [`EncodedPage`] into a [`CompressedPage`] using `compressed_buffer` as the
/// intermediary buffer.
///
/// `compressed_buffer` is taken by value because it becomes owned by [`CompressedPage`]
///
/// # Errors
/// Errors if the compressor fails
pub fn compress(
    page: Page,
    compressed_buffer: Vec<u8>,
    compression: CompressionOptions,
) -> ParquetResult<CompressedPage> {
    match page {
        Page::Data(page) => {
            compress_data(page, compressed_buffer, compression).map(CompressedPage::Data)
        },
        Page::Dict(page) => {
            compress_dict(page, compressed_buffer, compression).map(CompressedPage::Dict)
        },
    }
}

/// A [`FallibleStreamingIterator`] that consumes [`Page`] and yields [`CompressedPage`]
/// holding a reusable buffer ([`Vec<u8>`]) for compression.
pub struct Compressor<I: Iterator<Item = ParquetResult<Page>>> {
    iter: I,
    compression: CompressionOptions,
    buffer: Vec<u8>,
    current: Option<CompressedPage>,
}

impl<I: Iterator<Item = ParquetResult<Page>>> Compressor<I> {
    /// Creates a new [`Compressor`]
    pub fn new(iter: I, compression: CompressionOptions, buffer: Vec<u8>) -> Self {
        Self {
            iter,
            compression,
            buffer,
            current: None,
        }
    }

    /// Creates a new [`Compressor`] (same as `new`)
    pub fn new_from_vec(iter: I, compression: CompressionOptions, buffer: Vec<u8>) -> Self {
        Self::new(iter, compression, buffer)
    }

    /// Deconstructs itself into its iterator and scratch buffer.
    pub fn into_inner(mut self) -> (I, Vec<u8>) {
        let mut buffer = if let Some(page) = self.current.as_mut() {
            std::mem::take(page.buffer_mut())
        } else {
            std::mem::take(&mut self.buffer)
        };
        buffer.clear();
        (self.iter, buffer)
    }
}

impl<I: Iterator<Item = ParquetResult<Page>>> FallibleStreamingIterator for Compressor<I> {
    type Item = CompressedPage;
    type Error = ParquetError;

    fn advance(&mut self) -> std::result::Result<(), Self::Error> {
        let mut compressed_buffer = if let Some(page) = self.current.as_mut() {
            std::mem::take(page.buffer_mut())
        } else {
            std::mem::take(&mut self.buffer)
        };
        compressed_buffer.clear();

        let next = self
            .iter
            .next()
            .map(|x| x.and_then(|page| compress(page, compressed_buffer, self.compression)))
            .transpose()?;
        self.current = next;
        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }
}
