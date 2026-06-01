use xxhash_rust::xxh64::xxh64;

use crate::arrow::read::expr::ParquetScalar;
use crate::parquet::bloom_filter::{is_in_set, read::read_from_bytes};
use crate::parquet::error::ParquetResult;

const SEED: u64 = 0;

fn hash_bytes(bytes: &[u8]) -> u64 {
    xxh64(bytes, SEED)
}

fn hash_parquet_scalar(scalar: &ParquetScalar) -> Option<u64> {
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

/// Returns whether `scalar` might be present in a bloom filter given its serialized bytes.
///
/// `true` means the value **may** be present (or the probe is inconclusive); `false` means it is
/// **definitely not** present. Bloom filters only support safe skipping on `false`.
pub fn might_contain_scalar_bytes(bytes: &[u8], scalar: &ParquetScalar, bitset: &mut Vec<u8>) -> ParquetResult<bool> {
    read_from_bytes(bytes, bitset)?;
    Ok(scalar_might_be_in_prepared_bitset(scalar, bitset))
}

/// Like [`might_contain_scalar_bytes`], but parses `bytes` once and probes every scalar.
pub fn might_contain_any_scalar_bytes(
    bytes: &[u8],
    scalars: &[ParquetScalar],
    bitset: &mut Vec<u8>,
) -> ParquetResult<bool> {
    read_from_bytes(bytes, bitset)?;
    Ok(scalars
        .iter()
        .any(|scalar| scalar_might_be_in_prepared_bitset(scalar, bitset)))
}

fn scalar_might_be_in_prepared_bitset(scalar: &ParquetScalar, bitset: &[u8]) -> bool {
    let Some(hash) = hash_parquet_scalar(scalar) else {
        return true;
    };
    if bitset.is_empty() {
        return true;
    }
    is_in_set(bitset, hash)
}
