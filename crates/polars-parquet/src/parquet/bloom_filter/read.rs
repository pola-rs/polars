use std::io::{Read, Seek, SeekFrom};

use polars_parquet_format::thrift::protocol::TCompactInputProtocol;
use polars_parquet_format::{
    BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHeader, SplitBlockAlgorithm,
    Uncompressed,
};

use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::ColumnChunkMetadata;

/// Returns the bitset length if the header is supported, otherwise clears `bitset` and returns `None`.
fn supported_bitset_length(
    header: &BloomFilterHeader,
    bitset: &mut Vec<u8>,
) -> ParquetResult<Option<usize>> {
    if header.algorithm != BloomFilterAlgorithm::BLOCK(SplitBlockAlgorithm {})
        || header.compression != BloomFilterCompression::UNCOMPRESSED(Uncompressed {})
    {
        bitset.clear();
        return Ok(None);
    }
    Ok(Some(header.num_bytes.try_into()?))
}

fn prepare_bitset(bitset: &mut Vec<u8>, length: usize) -> ParquetResult<()> {
    bitset.clear();
    bitset.try_reserve(length)?;
    Ok(())
}

/// Reads the bloom filter associated to [`ColumnChunkMetadata`] into `bitset`.
/// Results in an empty `bitset` if there is no associated bloom filter or the algorithm is not supported.
/// # Error
/// Errors if the column contains no metadata or the filter can't be read or deserialized.
pub fn read<R: Read + Seek>(
    column_metadata: &ColumnChunkMetadata,
    mut reader: &mut R,
    bitset: &mut Vec<u8>,
) -> ParquetResult<()> {
    let offset = if let Some(offset) = column_metadata.bloom_filter_offset() {
        offset as u64
    } else {
        bitset.clear();
        return Ok(());
    };
    reader.seek(SeekFrom::Start(offset))?;

    // deserialize header
    let mut prot = TCompactInputProtocol::new(&mut reader, usize::MAX); // max is ok since `BloomFilterHeader` never allocates
    let header = BloomFilterHeader::read_from_in_protocol(&mut prot)?;
    let Some(length) = supported_bitset_length(&header, bitset)? else {
        return Ok(());
    };

    prepare_bitset(bitset, length)?;
    reader.by_ref().take(length as u64).read_to_end(bitset)?;

    Ok(())
}

/// Parse a bloom filter from an in-memory slice (header + bitset bytes).
pub fn read_from_bytes(bytes: &[u8], bitset: &mut Vec<u8>) -> ParquetResult<()> {
    let mut reader = std::io::Cursor::new(bytes);
    let mut prot = TCompactInputProtocol::new(&mut reader, usize::MAX);
    let header = BloomFilterHeader::read_from_in_protocol(&mut prot)?;
    let Some(length) = supported_bitset_length(&header, bitset)? else {
        return Ok(());
    };

    let pos = reader.position() as usize;
    let available = bytes.len().saturating_sub(pos);
    if available < length {
        bitset.clear();
        return Ok(());
    }

    prepare_bitset(bitset, length)?;
    bitset.extend_from_slice(&bytes[pos..pos + length]);
    Ok(())
}
