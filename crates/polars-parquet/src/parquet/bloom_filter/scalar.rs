use xxhash_rust::xxh64::xxh64;

use crate::arrow::read::expr::ParquetScalar;
use crate::parquet::bloom_filter::is_in_set;
use crate::parquet::bloom_filter::read::read_from_bytes;
use crate::parquet::error::ParquetResult;

const SEED: u64 = 0;

fn hash_bytes(bytes: &[u8]) -> u64 {
    xxh64(bytes, SEED)
}

/// Hash a value for split-block bloom filter membership tests.
pub fn hash_parquet_scalar(scalar: &ParquetScalar) -> Option<u64> {
    use ParquetScalar as S;
    match scalar {
        S::Null => None,
        S::Boolean(v) => Some(hash_bytes(&[*v as u8])),
        S::Int8(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::Int16(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::Int32(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::Int64(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::UInt8(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::UInt16(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::UInt32(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::UInt64(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::Float32(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::Float64(v) => Some(hash_bytes(&v.to_le_bytes())),
        S::String(v) => Some(hash_bytes(v.as_bytes())),
        S::Binary(v) | S::FixedSizeBinary(v) => Some(hash_bytes(v)),
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
    Ok(any_hash_might_be_in_prepared_bitset(hashes, bitset))
}

fn any_hash_might_be_in_prepared_bitset(hashes: &[u64], bitset: &[u8]) -> bool {
    if bitset.is_empty() {
        return true;
    }
    hashes.iter().any(|&hash| is_in_set(bitset, hash))
}
