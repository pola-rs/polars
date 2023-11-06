use xxhash_rust::xxh64::xxh64;

use crate::parquet::types::NativeType;

const SEED: u64 = 0;

/// (xxh64) hash of a [`NativeType`].
#[inline]
pub fn hash_native<T: NativeType>(value: T) -> u64 {
    xxh64(value.to_le_bytes().as_ref(), SEED)
}

/// (xxh64) hash of a sequence of bytes (e.g. ByteArray).
#[inline]
pub fn hash_byte<A: AsRef<[u8]>>(value: A) -> u64 {
    xxh64(value.as_ref(), SEED)
}
