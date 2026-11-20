//! Malformed page data gives an error rather than a panic.

use polars_parquet::parquet::encoding::{bitpacked, byte_stream_split, delta_byte_array};

#[test]
fn test_delta_header_rejects_invalid_block_layout() {
    // The prefix lengths' DELTA_BINARY_PACKED header:
    // <block size> <miniblocks per block> <total count> <first value>
    let headers: [&[u8]; 5] = [
        &[0x0a],                         // block size 10
        &[0x00, 0x04, 0x02, 0x00],       // block size 0
        &[0x80, 0x01, 0x00],             // no miniblocks
        &[0x80, 0x01, 0x03, 0x02, 0x00], // 128 values over 3 miniblocks
        &[0x80, 0x01, 0x08, 0x02, 0x00], // 16 values per miniblock
    ];
    for header in headers {
        assert!(
            delta_byte_array::Decoder::try_new(header).is_err(),
            "{header:?}"
        );
    }
}

#[test]
fn test_bitpacked_rejects_more_bits_than_the_type() {
    assert!(bitpacked::Decoder::<u32>::try_new(&[0; 7], 49, 1).is_err());
    assert!(bitpacked::Decoder::<u64>::try_new(&[0; 9], 65, 1).is_err());
    assert!(bitpacked::Decoder::<u32>::try_new_allow_zero(&[0; 5], 33, 1).is_err());
    assert!(bitpacked::Decoder::<u32>::try_new(&[0; 4], 32, 1).is_ok());
}

#[test]
fn test_byte_stream_split_rejects_zero_element_size() {
    assert!(byte_stream_split::Decoder::try_new(&[], 0).is_err());
}
