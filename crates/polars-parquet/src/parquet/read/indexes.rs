use std::io::{Cursor, Read, Seek, SeekFrom};

use polars_buffer::Buffer;
use polars_parquet_format::thrift::protocol::TCompactInputProtocol;
use polars_parquet_format::{ColumnIndex, OffsetIndex};

use crate::parquet::encryption::decrypt::CryptoContext;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::ColumnChunkMetadata;

const MAX_PAGE_INDEX_ALLOCATION: usize = 256 * 1024 * 1024;

fn read_index_bytes<R: Read + Seek>(
    reader: &mut R,
    offset: Option<i64>,
    length: Option<i32>,
    crypto_context: Option<&CryptoContext>,
    create_aad: impl FnOnce(&CryptoContext) -> ParquetResult<Vec<u8>>,
) -> ParquetResult<Option<Vec<u8>>> {
    let (Some(offset), Some(length)) = (offset, length) else {
        return Ok(None);
    };
    let offset: u64 = offset
        .try_into()
        .map_err(|_| ParquetError::oos("page index offset is negative"))?;
    let length: usize = length
        .try_into()
        .map_err(|_| ParquetError::oos("page index length is negative"))?;
    if length > MAX_PAGE_INDEX_ALLOCATION {
        return Err(ParquetError::WouldOverAllocate);
    }
    let end = reader.seek(SeekFrom::End(0))?;
    let requested_end = offset
        .checked_add(length as u64)
        .ok_or_else(|| ParquetError::oos("page index range overflows file offset"))?;
    if requested_end > end {
        return Err(ParquetError::oos("page index range exceeds file size"));
    }
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(length)?;
    buffer.resize(length, 0);
    reader.seek(SeekFrom::Start(offset))?;
    reader.read_exact(&mut buffer)?;

    if let Some(crypto_context) = crypto_context {
        let aad = create_aad(crypto_context)?;
        buffer = crypto_context
            .metadata_decryptor()
            .decrypt(&buffer, &aad)
            .map_err(|_| ParquetError::oos("failed to decrypt parquet page index"))?;
    }
    Ok(Some(buffer))
}

/// Read and, when necessary, decrypt a column index.
pub fn read_column_index<R: Read + Seek>(
    reader: &mut R,
    column: &ColumnChunkMetadata,
) -> ParquetResult<Option<ColumnIndex>> {
    let Some(bytes) = read_index_bytes(
        reader,
        column.column_index_offset(),
        column.column_index_length(),
        column.crypto_context(),
        CryptoContext::create_column_index_aad,
    )?
    else {
        return Ok(None);
    };
    let mut cursor = Cursor::new(Buffer::from_vec(bytes));
    let mut protocol = TCompactInputProtocol::new(&mut cursor, MAX_PAGE_INDEX_ALLOCATION);
    Ok(Some(ColumnIndex::read_from_in_protocol(&mut protocol)?))
}

/// Read and, when necessary, decrypt an offset index.
pub fn read_offset_index<R: Read + Seek>(
    reader: &mut R,
    column: &ColumnChunkMetadata,
) -> ParquetResult<Option<OffsetIndex>> {
    let Some(bytes) = read_index_bytes(
        reader,
        column.offset_index_offset(),
        column.offset_index_length(),
        column.crypto_context(),
        CryptoContext::create_offset_index_aad,
    )?
    else {
        return Ok(None);
    };
    let mut cursor = Cursor::new(Buffer::from_vec(bytes));
    let mut protocol = TCompactInputProtocol::new(&mut cursor, MAX_PAGE_INDEX_ALLOCATION);
    Ok(Some(OffsetIndex::read_from_in_protocol(&mut protocol)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_index_length_is_checked_before_allocation() {
        let mut reader = Cursor::new([0u8; 1]);
        let result = read_index_bytes(
            &mut reader,
            Some(0),
            Some((MAX_PAGE_INDEX_ALLOCATION + 1) as i32),
            None,
            |_| unreachable!(),
        );
        assert!(matches!(result, Err(ParquetError::WouldOverAllocate)));
    }
}
