use crate::arrow::read::expr::ParquetScalar;
use crate::parquet::bloom_filter::hash::{hash_byte, hash_native};
use crate::parquet::bloom_filter::is_maybe_in_bitset;
use crate::parquet::bloom_filter::read::{BloomFilterLayout, read_from_bytes};
use crate::parquet::bloom_filter::split_block::{
    BLOCK_SIZE, hash_to_block_index, is_maybe_in_block,
};
use crate::parquet::error::ParquetResult;

/// Hash a value for split-block bloom filter membership tests.
///
/// Narrow integer logical types (`Int8`/`Int16`/`UInt8`/`UInt16`) are widened to physical `INT32`
/// before hashing, matching Parquet writers.
pub fn hash_parquet_scalar(scalar: &ParquetScalar) -> Option<u64> {
    use ParquetScalar as S;
    match scalar {
        S::Null => None,
        S::Boolean(v) => Some(hash_byte([*v as u8])),
        // The following four columns are stored as physical INT32 in Parquet.
        S::Int8(v) => Some(hash_native(i32::from(*v))),
        S::UInt8(v) => Some(hash_native(i32::from(*v))),
        S::Int16(v) => Some(hash_native(i32::from(*v))),
        S::UInt16(v) => Some(hash_native(i32::from(*v))),
        S::Int32(v) => Some(hash_native(*v)),
        S::Int64(v) => Some(hash_native(*v)),
        // UInt32 columns are stored as physical INT32.
        S::UInt32(v) => Some(hash_byte(v.to_le_bytes())),
        // UInt64 columns are stored as physical INT64.
        S::UInt64(v) => Some(hash_byte(v.to_le_bytes())),
        S::Float32(v) => Some(hash_native(*v)),
        S::Float64(v) => Some(hash_native(*v)),
        // String columns are stored as physical BYTE_ARRAY.
        S::String(v) => Some(hash_byte(v.as_bytes())),
        S::Binary(v) | S::FixedSizeBinary(v) => Some(hash_byte(v)),
    }
}

/// Returns whether any precomputed bloom hash might be present in a bloom filter.
///
/// `true` means a value **may** be present (or the probe is inconclusive); `false` means all
/// hashes are **definitely not** present. Bloom filters only support safe skipping on `false`.
pub fn might_contain_any_hashes(
    bytes: &[u8],
    hashes: &[u64],
    bitset: &mut Vec<u8>,
) -> ParquetResult<bool> {
    read_from_bytes(bytes, bitset)?;
    Ok(if bitset.is_empty() {
        true
    } else {
        hashes.iter().any(|&hash| is_maybe_in_bitset(bitset, hash))
    })
}

/// Sorted unique block indices touched by `hashes`.
pub fn unique_block_indices(hashes: &[u64], bitset_num_bytes: usize) -> Vec<usize> {
    if bitset_num_bytes == 0 {
        return vec![];
    }
    let mut indices: Vec<usize> = hashes
        .iter()
        .map(|&hash| hash_to_block_index(hash, bitset_num_bytes))
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Whether block-wise I/O is cheaper than reading the full serialized filter slice.
pub fn prefer_block_reads(
    unique_blocks: usize,
    layout: &BloomFilterLayout,
    bloom_slice_len: usize,
) -> bool {
    if layout.bitset_num_bytes == 0 || unique_blocks == 0 {
        return false;
    }
    layout.header_len + unique_blocks * BLOCK_SIZE < bloom_slice_len
}

/// Probe precomputed hashes against individually loaded blocks.
///
/// Missing or short blocks are treated as inconclusive (`true`).
pub fn any_hashes_might_be_in_blocks<'a>(
    hashes: &[u64],
    bitset_num_bytes: usize,
    block: impl Fn(usize) -> Option<&'a [u8]>,
) -> bool {
    if bitset_num_bytes == 0 {
        return true;
    }
    hashes.iter().any(|&hash| {
        let idx = hash_to_block_index(hash, bitset_num_bytes);
        match block(idx) {
            Some(bytes) if bytes.len() == BLOCK_SIZE => is_maybe_in_block(bytes, hash),
            _ => true,
        }
    })
}
