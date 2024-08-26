use std::io::Write;

#[cfg(feature = "async")]
use futures::{AsyncWrite, AsyncWriteExt};
use parquet_format_safe::thrift::protocol::TCompactOutputProtocol;
#[cfg(feature = "async")]
use parquet_format_safe::thrift::protocol::TCompactOutputStreamProtocol;
use parquet_format_safe::{DictionaryPageHeader, Encoding, PageType};

use crate::parquet::compression::Compression;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{
    CompressedDataPage, CompressedDictPage, CompressedPage, DataPageHeader, ParquetPageHeader,
};
use crate::parquet::statistics::Statistics;

pub(crate) fn is_data_page(page: &PageWriteSpec) -> bool {
    page.header.type_ == PageType::DATA_PAGE || page.header.type_ == PageType::DATA_PAGE_V2
}

fn maybe_bytes(uncompressed: usize, compressed: usize) -> ParquetResult<(i32, i32)> {
    let uncompressed_page_size: i32 = uncompressed.try_into().map_err(|_| {
        ParquetError::oos(format!(
            "A page can only contain i32::MAX uncompressed bytes. This one contains {}",
            uncompressed
        ))
    })?;

    let compressed_page_size: i32 = compressed.try_into().map_err(|_| {
        ParquetError::oos(format!(
            "A page can only contain i32::MAX compressed bytes. This one contains {}",
            compressed
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
) -> ParquetResult<PageWriteSpec> {
    let num_values = compressed_page.num_values();
    let num_rows = compressed_page
        .num_rows()
        .expect("We should have num_rows when we are writing");

    let header = match &compressed_page {
        CompressedPage::Data(compressed_page) => assemble_data_page_header(compressed_page),
        CompressedPage::Dict(compressed_page) => assemble_dict_page_header(compressed_page),
    }?;

    let header_size = write_page_header(writer, &header)?;
    let mut bytes_written = header_size;

    bytes_written += match &compressed_page {
        CompressedPage::Data(compressed_page) => {
            writer.write_all(&compressed_page.buffer)?;
            compressed_page.buffer.len() as u64
        },
        CompressedPage::Dict(compressed_page) => {
            writer.write_all(&compressed_page.buffer)?;
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
        CompressedPage::Data(compressed_page) => assemble_data_page_header(compressed_page),
        CompressedPage::Dict(compressed_page) => assemble_dict_page_header(compressed_page),
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

fn assemble_data_page_header(page: &CompressedDataPage) -> ParquetResult<ParquetPageHeader> {
    let (uncompressed_page_size, compressed_page_size) =
        maybe_bytes(page.uncompressed_size(), page.compressed_size())?;

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

fn assemble_dict_page_header(page: &CompressedDictPage) -> ParquetResult<ParquetPageHeader> {
    let (uncompressed_page_size, compressed_page_size) =
        maybe_bytes(page.uncompressed_page_size, page.buffer.len())?;

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
        assert!(assemble_dict_page_header(&page).is_err());
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
        assert!(assemble_dict_page_header(&page).is_err());
    }
}
