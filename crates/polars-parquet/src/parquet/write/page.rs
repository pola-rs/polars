use std::io::Write;
use std::sync::Arc;

#[cfg(feature = "async")]
use futures::{AsyncWrite, AsyncWriteExt};
use polars_parquet_format::thrift::protocol::TCompactOutputProtocol;
#[cfg(feature = "async")]
use polars_parquet_format::thrift::protocol::TCompactOutputStreamProtocol;
use polars_parquet_format::{DictionaryPageHeader, Encoding, PageType};

use crate::parquet::compression::Compression;
use crate::parquet::encryption::ciphers::BlockEncryptor;
use crate::parquet::encryption::encrypt::{
    FileEncryptor, encrypt_bytes, write_encrypted_thrift_object,
};
use crate::parquet::encryption::modules::{ModuleType, create_module_aad};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, DataPageHeader, ParquetPageHeader,
};
use crate::parquet::statistics::Statistics;

pub(crate) fn is_data_page(page: &PageWriteSpec) -> bool {
    page.header.type_ == PageType::DATA_PAGE || page.header.type_ == PageType::DATA_PAGE_V2
}

pub(crate) fn is_dict_page(page: &PageWriteSpec) -> bool {
    page.header.type_ == PageType::DICTIONARY_PAGE
}

#[derive(Debug)]
pub(crate) struct PageEncryptor {
    file_encryptor: Arc<FileEncryptor>,
    block_encryptor: Box<dyn BlockEncryptor>,
    row_group_index: usize,
    column_index: usize,
    page_index: usize,
}

impl PageEncryptor {
    pub(crate) fn create_if_column_encrypted(
        file_encryptor: Option<&Arc<FileEncryptor>>,
        row_group_index: usize,
        column_index: usize,
        column_path: &str,
    ) -> ParquetResult<Option<Self>> {
        let Some(file_encryptor) = file_encryptor else {
            return Ok(None);
        };
        if !file_encryptor.is_column_encrypted(column_path) {
            return Ok(None);
        }
        let block_encryptor = file_encryptor.get_column_encryptor(column_path)?;
        Ok(Some(Self {
            file_encryptor: Arc::clone(file_encryptor),
            block_encryptor,
            row_group_index,
            column_index,
            page_index: 0,
        }))
    }

    fn encrypt_page(&mut self, page: &CompressedPage) -> ParquetResult<Vec<u8>> {
        let module_type = match page {
            CompressedPage::Data(_) => ModuleType::DataPage,
            CompressedPage::Dict(_) => ModuleType::DictionaryPage,
        };
        let aad = create_module_aad(
            self.file_encryptor.file_aad(),
            module_type,
            self.row_group_index,
            self.column_index,
            Some(self.page_index),
        )?;
        let data = match page {
            CompressedPage::Data(page) => &page.buffer[..],
            CompressedPage::Dict(page) => &page.buffer[..],
        };
        encrypt_bytes(data, &mut *self.block_encryptor, &aad)
    }

    fn write_page_header<W: Write>(
        &mut self,
        writer: &mut W,
        header: &ParquetPageHeader,
    ) -> ParquetResult<u64> {
        let module_type = match header.type_ {
            PageType::DATA_PAGE | PageType::DATA_PAGE_V2 => ModuleType::DataPageHeader,
            PageType::DICTIONARY_PAGE => ModuleType::DictionaryPageHeader,
            _ => {
                return Err(ParquetError::not_supported(format!(
                    "page type {:?} cannot be encrypted",
                    header.type_
                )));
            },
        };
        let aad = create_module_aad(
            self.file_encryptor.file_aad(),
            module_type,
            self.row_group_index,
            self.column_index,
            Some(self.page_index),
        )?;
        write_encrypted_thrift_object(writer, &mut *self.block_encryptor, &aad, |protocol| {
            header.write_to_out_protocol(protocol)
        })
    }

    fn advance_page(&mut self, header: &ParquetPageHeader) {
        if matches!(header.type_, PageType::DATA_PAGE | PageType::DATA_PAGE_V2) {
            self.page_index += 1;
        }
    }
}

fn maybe_bytes(uncompressed: usize, compressed: usize) -> ParquetResult<(i32, i32)> {
    let uncompressed_page_size: i32 = uncompressed.try_into().map_err(|_| {
        ParquetError::oos(format!(
            "A page can only contain i32::MAX uncompressed bytes. This one contains {uncompressed}"
        ))
    })?;

    let compressed_page_size: i32 = compressed.try_into().map_err(|_| {
        ParquetError::oos(format!(
            "A page can only contain i32::MAX compressed bytes. This one contains {compressed}"
        ))
    })?;

    Ok((uncompressed_page_size, compressed_page_size))
}

/// Contains page write metrics.
pub struct PageWriteSpec {
    pub header: ParquetPageHeader,
    #[allow(dead_code)]
    pub num_values: usize,
    /// The number of actual rows. For non-nested values, this is equal to the number of values.
    pub num_rows: usize,
    pub header_size: u64,
    pub offset: u64,
    pub bytes_written: u64,
    pub compression: Compression,
    pub statistics: Option<Statistics>,
}

pub fn write_page<W: Write>(
    writer: &mut W,
    offset: u64,
    compressed_page: &CompressedPage,
    mut page_encryptor: Option<&mut PageEncryptor>,
) -> ParquetResult<PageWriteSpec> {
    let num_values = compressed_page.num_values();
    let num_rows = compressed_page
        .num_rows()
        .expect("We should have num_rows when we are writing");

    let encrypted_buffer = match page_encryptor.as_deref_mut() {
        Some(page_encryptor) => Some(page_encryptor.encrypt_page(compressed_page)?),
        None => None,
    };

    let header = match &compressed_page {
        CompressedPage::Data(compressed_page) => assemble_data_page_header(
            compressed_page,
            encrypted_buffer.as_ref().map(|buffer| buffer.len()),
        ),
        CompressedPage::Dict(compressed_page) => assemble_dict_page_header(
            compressed_page,
            encrypted_buffer.as_ref().map(|buffer| buffer.len()),
        ),
    }?;

    let header_size = match page_encryptor.as_deref_mut() {
        Some(page_encryptor) => page_encryptor.write_page_header(writer, &header)?,
        None => write_page_header(writer, &header)?,
    };
    let mut bytes_written = header_size;

    bytes_written += if let Some(buffer) = encrypted_buffer {
        writer.write_all(&buffer)?;
        buffer.len() as u64
    } else {
        match &compressed_page {
            CompressedPage::Data(compressed_page) => {
                writer.write_all(&compressed_page.buffer)?;
                compressed_page.buffer.len() as u64
            },
            CompressedPage::Dict(compressed_page) => {
                writer.write_all(&compressed_page.buffer)?;
                compressed_page.buffer.len() as u64
            },
        }
    };

    if let Some(page_encryptor) = page_encryptor {
        page_encryptor.advance_page(&header);
    }

    let statistics = match &compressed_page {
        CompressedPage::Data(compressed_page) => compressed_page.statistics().transpose()?,
        CompressedPage::Dict(_) => None,
    };

    Ok(PageWriteSpec {
        header,
        header_size,
        offset,
        bytes_written,
        compression: compressed_page.compression(),
        statistics,
        num_values,
        num_rows,
    })
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_page_async<W: AsyncWrite + Unpin + Send>(
    writer: &mut W,
    offset: u64,
    compressed_page: &CompressedPage,
) -> ParquetResult<PageWriteSpec> {
    let num_values = compressed_page.num_values();
    let num_rows = compressed_page
        .num_rows()
        .expect("We should have the num_rows when we are writing");

    let header = match &compressed_page {
        CompressedPage::Data(compressed_page) => assemble_data_page_header(compressed_page, None),
        CompressedPage::Dict(compressed_page) => assemble_dict_page_header(compressed_page, None),
    }?;

    let header_size = write_page_header_async(writer, &header).await?;
    let mut bytes_written = header_size as u64;

    bytes_written += match &compressed_page {
        CompressedPage::Data(compressed_page) => {
            writer.write_all(&compressed_page.buffer).await?;
            compressed_page.buffer.len() as u64
        },
        CompressedPage::Dict(compressed_page) => {
            writer.write_all(&compressed_page.buffer).await?;
            compressed_page.buffer.len() as u64
        },
    };

    let statistics = match &compressed_page {
        CompressedPage::Data(compressed_page) => compressed_page.statistics().transpose()?,
        CompressedPage::Dict(_) => None,
    };

    Ok(PageWriteSpec {
        header,
        header_size,
        offset,
        bytes_written,
        compression: compressed_page.compression(),
        statistics,
        num_rows,
        num_values,
    })
}

fn assemble_data_page_header(
    page: &CompressedDataPage,
    encrypted_size: Option<usize>,
) -> ParquetResult<ParquetPageHeader> {
    let (uncompressed_page_size, compressed_page_size) = maybe_bytes(
        page.uncompressed_size(),
        encrypted_size.unwrap_or(page.compressed_size()),
    )?;

    let mut page_header = ParquetPageHeader {
        type_: match page.header() {
            DataPageHeader::V1(_) => PageType::DATA_PAGE,
            DataPageHeader::V2(_) => PageType::DATA_PAGE_V2,
        },
        uncompressed_page_size,
        compressed_page_size,
        crc: None,
        data_page_header: None,
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    };

    match page.header() {
        DataPageHeader::V1(header) => {
            page_header.data_page_header = Some(header.clone());
        },
        DataPageHeader::V2(header) => {
            page_header.data_page_header_v2 = Some(header.clone());
        },
    }
    Ok(page_header)
}

fn assemble_dict_page_header(
    page: &CompressedDictPage,
    encrypted_size: Option<usize>,
) -> ParquetResult<ParquetPageHeader> {
    let (uncompressed_page_size, compressed_page_size) = maybe_bytes(
        page.uncompressed_page_size,
        encrypted_size.unwrap_or(page.buffer.len()),
    )?;

    let num_values: i32 = page.num_values.try_into().map_err(|_| {
        ParquetError::oos(format!(
            "A dictionary page can only contain i32::MAX items. This one contains {}",
            page.num_values
        ))
    })?;

    Ok(ParquetPageHeader {
        type_: PageType::DICTIONARY_PAGE,
        uncompressed_page_size,
        compressed_page_size,
        crc: None,
        data_page_header: None,
        index_page_header: None,
        dictionary_page_header: Some(DictionaryPageHeader {
            num_values,
            encoding: Encoding::PLAIN,
            is_sorted: None,
        }),
        data_page_header_v2: None,
    })
}

/// writes the page header into `writer`, returning the number of bytes used in the process.
fn write_page_header<W: Write>(
    mut writer: &mut W,
    header: &ParquetPageHeader,
) -> ParquetResult<u64> {
    let mut protocol = TCompactOutputProtocol::new(&mut writer);
    Ok(header.write_to_out_protocol(&mut protocol)? as u64)
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
/// writes the page header into `writer`, returning the number of bytes used in the process.
async fn write_page_header_async<W: AsyncWrite + Unpin + Send>(
    mut writer: &mut W,
    header: &ParquetPageHeader,
) -> ParquetResult<u64> {
    let mut protocol = TCompactOutputStreamProtocol::new(&mut writer);
    Ok(header.write_to_out_stream_protocol(&mut protocol).await? as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::CowBuffer;

    #[test]
    fn dict_too_large() {
        let page = CompressedDictPage::new(
            CowBuffer::Owned(vec![]),
            Compression::Uncompressed,
            i32::MAX as usize + 1,
            100,
            false,
        );
        assert!(assemble_dict_page_header(&page, None).is_err());
    }

    #[test]
    fn dict_too_many_values() {
        let page = CompressedDictPage::new(
            CowBuffer::Owned(vec![]),
            Compression::Uncompressed,
            0,
            i32::MAX as usize + 1,
            false,
        );
        assert!(assemble_dict_page_header(&page, None).is_err());
    }
}
