use std::io::{Read, Seek, SeekFrom};

use parquet_format_safe::thrift::protocol::TCompactInputProtocol;
use parquet_format_safe::{
    BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHeader, SplitBlockAlgorithm,
    Uncompressed,
};

use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::ColumnChunkMetaData;

/// Reads the bloom filter associated to [`ColumnChunkMetaData`] into `bitset`.
/// Results in an empty `bitset` if there is no associated bloom filter or the algorithm is not supported.
/// # Error
/// Errors if the column contains no metadata or the filter can't be read or deserialized.
pub fn read<R: Read + Seek>(
    column_metadata: &ColumnChunkMetaData,
    mut reader: &mut R,
    bitset: &mut Vec<u8>,
) -> ParquetResult<()> {
    let offset = column_metadata.metadata().bloom_filter_offset;

    let offset = if let Some(offset) = offset {
        offset as u64
    } else {
        bitset.clear();
        return Ok(());
    };
    reader.seek(SeekFrom::Start(offset))?;

    // deserialize header
    let mut prot = TCompactInputProtocol::new(&mut reader, usize::MAX); // max is ok since `BloomFilterHeader` never allocates
    let header = BloomFilterHeader::read_from_in_protocol(&mut prot)?;

    if header.algorithm != BloomFilterAlgorithm::BLOCK(SplitBlockAlgorithm {}) {
        bitset.clear();
        return Ok(());
    }
    if header.compression != BloomFilterCompression::UNCOMPRESSED(Uncompressed {}) {
        bitset.clear();
        return Ok(());
    }

    let length: usize = header.num_bytes.try_into()?;

    bitset.clear();
    bitset.try_reserve(length)?;
    reader.by_ref().take(length as u64).read_to_end(bitset)?;

    Ok(())
}
