pub mod bitpacked;
pub mod byte_stream_split;
pub mod delta_bitpacked;
pub mod delta_byte_array;
pub mod delta_length_byte_array;
pub mod hybrid_rle;
pub mod plain_byte_array;
pub mod uleb128;
pub mod zigzag_leb128;

pub use crate::parquet::parquet_bridge::Encoding;

/// # Panics
/// This function panics iff `values.len() < 4`.
#[inline]
pub fn get_length(values: &[u8]) -> Option<usize> {
    values
        .get(0..4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()) as usize)
}

/// Returns the ceil of value / 8
#[inline]
pub fn ceil8(value: usize) -> usize {
    value / 8 + ((value % 8 != 0) as usize)
}
