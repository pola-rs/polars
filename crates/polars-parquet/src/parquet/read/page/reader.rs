use std::io::{Cursor, Seek};
use std::sync::OnceLock;

use polars_buffer::Buffer;
use polars_parquet_format::thrift::protocol::TCompactInputProtocol;

use super::PageIterator;
use crate::parquet::CowBuffer;
use crate::parquet::compression::Compression;
use crate::parquet::encryption::decrypt::{CryptoContext, read_and_decrypt};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetadata, Descriptor};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, DataPageHeader, PageType,
    ParquetPageHeader,
};
use crate::write::Encoding;

/// This meta is a small part of [`ColumnChunkMetadata`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageMetaData {
    /// The start offset of this column chunk in file.
    pub column_start: u64,
    /// The dictionary page offset, if the column chunk has a dictionary page.
    pub dictionary_page_offset: Option<u64>,
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
        dictionary_page_offset: Option<u64>,
        num_values: i64,
        compression: Compression,
        descriptor: Descriptor,
    ) -> Self {
        Self {
            column_start,
            dictionary_page_offset,
            num_values,
            compression,
            descriptor,
        }
    }
}

impl From<&ColumnChunkMetadata> for PageMetaData {
    fn from(column: &ColumnChunkMetadata) -> Self {
        Self {
            column_start: column.byte_range().start,
            dictionary_page_offset: column.dictionary_page_offset().map(|x| x as u64),
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
    reader: Cursor<Buffer<u8>>,

    compression: Compression,
    dictionary_page_offset: Option<u64>,

    // The number of values we have seen so far.
    seen_num_values: i64,

    // The number of total values in this column chunk.
    total_num_values: i64,

    descriptor: Descriptor,

    // The currently allocated buffer.
    pub(crate) scratch: Vec<u8>,

    // Maximum page size (compressed or uncompressed) to limit allocations
    max_page_size: usize,

    crypto_context: Option<CryptoContext>,
    page_ordinal: usize,
}

impl PageReader {
    /// Returns a new [`PageReader`].
    ///
    /// It assumes that the reader has been `sought` (`seek`) to the beginning of `column`.
    /// The parameter `max_header_size`
    pub fn new(
        reader: Cursor<Buffer<u8>>,
        column: &ColumnChunkMetadata,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        Self::new_with_page_meta(reader, column.into(), scratch, max_page_size)
            .with_crypto_context(column.crypto_context().cloned())
    }

    /// Create a new [`PageReader`] with [`PageMetaData`].
    ///
    /// It assumes that the reader has been `sought` (`seek`) to the beginning of `column`.
    pub fn new_with_page_meta(
        reader: Cursor<Buffer<u8>>,
        reader_meta: PageMetaData,
        scratch: Vec<u8>,
        max_page_size: usize,
    ) -> Self {
        let dictionary_page_offset = reader_meta.dictionary_page_offset.and_then(|offset| {
            offset
                .checked_sub(reader_meta.column_start)
                .and_then(|relative_offset| reader.position().checked_add(relative_offset))
        });

        Self {
            reader,
            total_num_values: reader_meta.num_values,
            compression: reader_meta.compression,
            dictionary_page_offset,
            seen_num_values: 0,
            descriptor: reader_meta.descriptor,
            scratch,
            max_page_size,
            crypto_context: None,
            page_ordinal: 0,
        }
    }

    pub(crate) fn with_crypto_context(mut self, crypto_context: Option<CryptoContext>) -> Self {
        self.crypto_context = crypto_context;
        self
    }

    /// Returns the reader and this Readers' interval buffer
    pub fn into_inner(self) -> (Cursor<Buffer<u8>>, Vec<u8>) {
        (self.reader, self.scratch)
    }

    pub fn total_num_values(&self) -> usize {
        debug_assert!(self.total_num_values >= 0);
        self.total_num_values as usize
    }

    pub fn read_dict(&mut self) -> ParquetResult<Option<CompressedDictPage>> {
        // If there are no pages, we cannot check if the first page is a dictionary page. Just
        // return the fact there is no dictionary page.
        if self.reader.position() == self.reader.get_ref().len() as u64 {
            return Ok(None);
        }

        if self.crypto_context.is_some() {
            let Some(dictionary_page_offset) = self.dictionary_page_offset else {
                return Ok(None);
            };

            if self.reader.position() != dictionary_page_offset {
                return Ok(None);
            }
        }

        // a dictionary page exists iff the first data page is not at the start of
        // the column
        let seek_offset = self.reader.position();
        let page_header = read_page_header_with_crypto(
            &mut self.reader,
            self.max_page_size,
            self.crypto_context.as_ref(),
            self.page_ordinal,
            true,
        )?;
        let page_type = page_header.type_.try_into()?;

        if !matches!(page_type, PageType::DictionaryPage) {
            self.reader.seek(std::io::SeekFrom::Start(seek_offset))?;
            return Ok(None);
        }

        let read_size: usize = page_header.compressed_page_size.try_into()?;

        if read_size > self.max_page_size {
            return Err(ParquetError::WouldOverAllocate);
        }

        // Read read_size into new buffer and advance reader.
        let orig_buf = self.reader.get_ref();
        let pos = self.reader.position() as usize;
        let new_pos = (pos + read_size).min(orig_buf.len());
        let buffer = orig_buf.clone().sliced(pos..new_pos);
        self.reader.set_position(new_pos as u64);

        if buffer.len() != read_size {
            return Err(ParquetError::oos(
                "The page header reported the wrong page size",
            ));
        }

        let buffer = decrypt_page_data(
            buffer,
            self.crypto_context.as_ref(),
            self.page_ordinal,
            true,
        )?;

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
    reader: &mut Cursor<Buffer<u8>>,
    max_size: usize,
) -> ParquetResult<ParquetPageHeader> {
    let mut prot = TCompactInputProtocol::new(reader, max_size);
    let page_header = ParquetPageHeader::read_from_in_protocol(&mut prot)?;
    Ok(page_header)
}

fn read_page_header_with_crypto(
    reader: &mut Cursor<Buffer<u8>>,
    max_size: usize,
    crypto_context: Option<&CryptoContext>,
    page_ordinal: usize,
    dictionary_page: bool,
) -> ParquetResult<ParquetPageHeader> {
    let Some(crypto_context) = crypto_context else {
        return read_page_header(reader, max_size);
    };

    let page_crypto_context = if dictionary_page {
        crypto_context.for_dictionary_page()
    } else {
        crypto_context.with_page_ordinal(page_ordinal)
    };
    let aad = page_crypto_context.create_page_header_aad()?;
    let max_ciphertext_len = max_size
        .saturating_add(crate::parquet::encryption::ciphers::NONCE_LEN)
        .saturating_add(crate::parquet::encryption::ciphers::TAG_LEN)
        .min(
            reader
                .get_ref()
                .len()
                .saturating_sub(reader.position() as usize)
                .saturating_sub(crate::parquet::encryption::ciphers::SIZE_LEN),
        );
    let decrypted_header = read_and_decrypt(
        page_crypto_context.metadata_decryptor(),
        reader,
        &aad,
        max_ciphertext_len,
    )
    .map_err(|_| ParquetError::oos("failed to decrypt parquet page header"))?;

    let mut header_reader = Cursor::new(Buffer::from_vec(decrypted_header));
    read_page_header(&mut header_reader, max_size)
}

pub(super) fn decrypt_page_data(
    buffer: Buffer<u8>,
    crypto_context: Option<&CryptoContext>,
    page_ordinal: usize,
    dictionary_page: bool,
) -> ParquetResult<Buffer<u8>> {
    let Some(crypto_context) = crypto_context else {
        return Ok(buffer);
    };

    let page_crypto_context = if dictionary_page {
        crypto_context.for_dictionary_page()
    } else {
        crypto_context.with_page_ordinal(page_ordinal)
    };
    let aad = page_crypto_context.create_page_aad()?;
    let decrypted = page_crypto_context
        .data_decryptor()
        .decrypt(&buffer, &aad)
        .map_err(|_| ParquetError::oos("failed to decrypt parquet page data"))?;
    Ok(Buffer::from_vec(decrypted))
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
    let page_ordinal = reader.page_ordinal;
    let page_header = read_page_header_with_crypto(
        &mut reader.reader,
        reader.max_page_size,
        reader.crypto_context.as_ref(),
        page_ordinal,
        false,
    )?;

    reader.seen_num_values += get_page_num_values(&page_header)? as i64;
    reader.page_ordinal += 1;

    let read_size: usize = page_header.compressed_page_size.try_into()?;

    if read_size > reader.max_page_size {
        return Err(ParquetError::WouldOverAllocate);
    }

    // Read read_size into new buffer and advance reader.
    let orig_buf = reader.reader.get_ref();
    let pos = reader.reader.position() as usize;
    let new_pos = (pos + read_size).min(orig_buf.len());
    let buffer = orig_buf.clone().sliced(pos..new_pos);
    reader.reader.set_position(new_pos as u64);

    if buffer.len() != read_size {
        return Err(ParquetError::oos(
            "The page header reported the wrong page size",
        ));
    }

    let buffer = decrypt_page_data(buffer, reader.crypto_context.as_ref(), page_ordinal, false)?;

    finish_page(page_header, buffer, reader.compression, &reader.descriptor).map(Some)
}

pub(super) fn finish_page(
    page_header: ParquetPageHeader,
    data: Buffer<u8>,
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
                eprintln!(
                    "Parquet DictPage ( num_values: {}, datatype: {:?} )",
                    dict_header.num_values, descriptor.primitive_type
                );
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
                eprintln!(
                    "Parquet DataPageV1 ( num_values: {}, datatype: {:?}, encoding: {:?} )",
                    header.num_values,
                    descriptor.primitive_type,
                    Encoding::try_from(header.encoding).ok()
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
                    "Parquet DataPageV2 ( num_values: {}, datatype: {:?}, encoding: {:?} )",
                    header.num_values,
                    descriptor.primitive_type,
                    Encoding::try_from(header.encoding).ok()
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
