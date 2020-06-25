// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{cmp, mem};

use super::rle::{RleDecoder, RleEncoder};

use crate::basic::Encoding;
use crate::data_type::AsBytes;
use crate::errors::{ParquetError, Result};
use crate::util::{
    bit_util::{ceil, log2, BitReader, BitWriter},
    memory::ByteBufferPtr,
};

/// Computes max buffer size for level encoder/decoder based on encoding, max
/// repetition/definition level and number of total buffered values (includes null
/// values).
#[inline]
pub fn max_buffer_size(
    encoding: Encoding,
    max_level: i16,
    num_buffered_values: usize,
) -> usize {
    let bit_width = log2(max_level as u64 + 1) as u8;
    match encoding {
        Encoding::RLE => {
            RleEncoder::max_buffer_size(bit_width, num_buffered_values)
                + RleEncoder::min_buffer_size(bit_width)
        }
        Encoding::BIT_PACKED => {
            ceil((num_buffered_values * bit_width as usize) as i64, 8) as usize
        }
        _ => panic!("Unsupported encoding type {}", encoding),
    }
}

/// Encoder for definition/repetition levels.
/// Currently only supports RLE and BIT_PACKED (dev/null) encoding, including v2.
pub enum LevelEncoder {
    RLE(RleEncoder),
    RLE_V2(RleEncoder),
    BIT_PACKED(u8, BitWriter),
}

impl LevelEncoder {
    /// Creates new level encoder based on encoding, max level and underlying byte buffer.
    /// For bit packed encoding it is assumed that buffer is already allocated with
    /// `levels::max_buffer_size` method.
    ///
    /// Used to encode levels for Data Page v1.
    ///
    /// Panics, if encoding is not supported.
    pub fn v1(encoding: Encoding, max_level: i16, byte_buffer: Vec<u8>) -> Self {
        let bit_width = log2(max_level as u64 + 1) as u8;
        match encoding {
            Encoding::RLE => LevelEncoder::RLE(RleEncoder::new_from_buf(
                bit_width,
                byte_buffer,
                mem::size_of::<i32>(),
            )),
            Encoding::BIT_PACKED => {
                // Here we set full byte buffer without adjusting for num_buffered_values,
                // because byte buffer will already be allocated with size from
                // `max_buffer_size()` method.
                LevelEncoder::BIT_PACKED(
                    bit_width,
                    BitWriter::new_from_buf(byte_buffer, 0),
                )
            }
            _ => panic!("Unsupported encoding type {}", encoding),
        }
    }

    /// Creates new level encoder based on RLE encoding. Used to encode Data Page v2
    /// repetition and definition levels.
    pub fn v2(max_level: i16, byte_buffer: Vec<u8>) -> Self {
        let bit_width = log2(max_level as u64 + 1) as u8;
        LevelEncoder::RLE_V2(RleEncoder::new_from_buf(bit_width, byte_buffer, 0))
    }

    /// Put/encode levels vector into this level encoder.
    /// Returns number of encoded values that are less than or equal to length of the
    /// input buffer.
    ///
    /// RLE and BIT_PACKED level encoders return Err() when internal buffer overflows or
    /// flush fails.
    #[inline]
    pub fn put(&mut self, buffer: &[i16]) -> Result<usize> {
        let mut num_encoded = 0;
        match *self {
            LevelEncoder::RLE(ref mut encoder)
            | LevelEncoder::RLE_V2(ref mut encoder) => {
                for value in buffer {
                    if !encoder.put(*value as u64)? {
                        return Err(general_err!("RLE buffer is full"));
                    }
                    num_encoded += 1;
                }
                encoder.flush()?;
            }
            LevelEncoder::BIT_PACKED(bit_width, ref mut encoder) => {
                for value in buffer {
                    if !encoder.put_value(*value as u64, bit_width as usize) {
                        return Err(general_err!("Not enough bytes left"));
                    }
                    num_encoded += 1;
                }
                encoder.flush();
            }
        }
        Ok(num_encoded)
    }

    /// Finalizes level encoder, flush all intermediate buffers and return resulting
    /// encoded buffer. Returned buffer is already truncated to encoded bytes only.
    #[inline]
    pub fn consume(self) -> Result<Vec<u8>> {
        match self {
            LevelEncoder::RLE(encoder) => {
                let mut encoded_data = encoder.consume()?;
                // Account for the buffer offset
                let encoded_len = encoded_data.len() - mem::size_of::<i32>();
                let len = (encoded_len as i32).to_le();
                let len_bytes = len.as_bytes();
                encoded_data[0..len_bytes.len()].copy_from_slice(len_bytes);
                Ok(encoded_data)
            }
            LevelEncoder::RLE_V2(encoder) => encoder.consume(),
            LevelEncoder::BIT_PACKED(_, encoder) => Ok(encoder.consume()),
        }
    }
}

/// Decoder for definition/repetition levels.
/// Currently only supports RLE and BIT_PACKED encoding for Data Page v1 and
/// RLE for Data Page v2.
pub enum LevelDecoder {
    RLE(Option<usize>, RleDecoder),
    RLE_V2(Option<usize>, RleDecoder),
    BIT_PACKED(Option<usize>, u8, BitReader),
}

impl LevelDecoder {
    /// Creates new level decoder based on encoding and max definition/repetition level.
    /// This method only initializes level decoder, `set_data` method must be called
    /// before reading any value.
    ///
    /// Used to encode levels for Data Page v1.
    ///
    /// Panics if encoding is not supported
    pub fn v1(encoding: Encoding, max_level: i16) -> Self {
        let bit_width = log2(max_level as u64 + 1) as u8;
        match encoding {
            Encoding::RLE => LevelDecoder::RLE(None, RleDecoder::new(bit_width)),
            Encoding::BIT_PACKED => {
                LevelDecoder::BIT_PACKED(None, bit_width, BitReader::from(Vec::new()))
            }
            _ => panic!("Unsupported encoding type {}", encoding),
        }
    }

    /// Creates new level decoder based on RLE encoding.
    /// Used to decode Data Page v2 repetition and definition levels.
    ///
    /// To set data for this decoder, use `set_data_range` method.
    pub fn v2(max_level: i16) -> Self {
        let bit_width = log2(max_level as u64 + 1) as u8;
        LevelDecoder::RLE_V2(None, RleDecoder::new(bit_width))
    }

    /// Sets data for this level decoder, and returns total number of bytes set.
    /// This is used for Data Page v1 levels.
    ///
    /// `data` is encoded data as byte buffer, `num_buffered_values` represents total
    /// number of values that is expected.
    ///
    /// Both RLE and BIT_PACKED level decoders set `num_buffered_values` as total number
    /// of values that they can return and track num values.
    #[inline]
    pub fn set_data(&mut self, num_buffered_values: usize, data: ByteBufferPtr) -> usize {
        match *self {
            LevelDecoder::RLE(ref mut num_values, ref mut decoder) => {
                *num_values = Some(num_buffered_values);
                let i32_size = mem::size_of::<i32>();
                let data_size = read_num_bytes!(i32, i32_size, data.as_ref()) as usize;
                decoder.set_data(data.range(i32_size, data_size));
                i32_size + data_size
            }
            LevelDecoder::BIT_PACKED(ref mut num_values, bit_width, ref mut decoder) => {
                *num_values = Some(num_buffered_values);
                // Set appropriate number of bytes: if max size is larger than buffer -
                // set full buffer
                let num_bytes =
                    ceil((num_buffered_values * bit_width as usize) as i64, 8);
                let data_size = cmp::min(num_bytes as usize, data.len());
                decoder.reset(data.range(data.start(), data_size));
                data_size
            }
            _ => panic!(),
        }
    }

    /// Sets byte array explicitly when start position `start` and length `len` are known
    /// in advance. Only supported by RLE level decoder and used for Data Page v2 levels.
    /// Returns number of total bytes set for this decoder (len).
    #[inline]
    pub fn set_data_range(
        &mut self,
        num_buffered_values: usize,
        data: &ByteBufferPtr,
        start: usize,
        len: usize,
    ) -> usize {
        match *self {
            LevelDecoder::RLE_V2(ref mut num_values, ref mut decoder) => {
                decoder.set_data(data.range(start, len));
                *num_values = Some(num_buffered_values);
                len
            }
            _ => panic!(
                "set_data_range() method is only supported by RLE v2 encoding type"
            ),
        }
    }

    /// Returns true if data is set for decoder, false otherwise.
    #[inline]
    pub fn is_data_set(&self) -> bool {
        match self {
            LevelDecoder::RLE(ref num_values, _) => num_values.is_some(),
            LevelDecoder::RLE_V2(ref num_values, _) => num_values.is_some(),
            LevelDecoder::BIT_PACKED(ref num_values, ..) => num_values.is_some(),
        }
    }

    /// Decodes values and puts them into `buffer`.
    /// Returns number of values that were successfully decoded (less than or equal to
    /// buffer length).
    #[inline]
    pub fn get(&mut self, buffer: &mut [i16]) -> Result<usize> {
        assert!(self.is_data_set(), "No data set for decoding");
        match *self {
            LevelDecoder::RLE(ref mut num_values, ref mut decoder)
            | LevelDecoder::RLE_V2(ref mut num_values, ref mut decoder) => {
                // Max length we can read
                let len = cmp::min(num_values.unwrap(), buffer.len());
                let values_read = decoder.get_batch::<i16>(&mut buffer[0..len])?;
                *num_values = num_values.map(|len| len - values_read);
                Ok(values_read)
            }
            LevelDecoder::BIT_PACKED(ref mut num_values, bit_width, ref mut decoder) => {
                // When extracting values from bit reader, it might return more values
                // than left because of padding to a full byte, we use
                // num_values to track precise number of values.
                let len = cmp::min(num_values.unwrap(), buffer.len());
                let values_read =
                    decoder.get_batch::<i16>(&mut buffer[..len], bit_width as usize);
                *num_values = num_values.map(|len| len - values_read);
                Ok(values_read)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::util::test_common::random_numbers_range;

    fn test_internal_roundtrip(enc: Encoding, levels: &[i16], max_level: i16, v2: bool) {
        let size = max_buffer_size(enc, max_level, levels.len());
        let mut encoder = if v2 {
            LevelEncoder::v2(max_level, vec![0; size])
        } else {
            LevelEncoder::v1(enc, max_level, vec![0; size])
        };
        encoder.put(&levels).expect("put() should be OK");
        let encoded_levels = encoder.consume().expect("consume() should be OK");

        let byte_buf = ByteBufferPtr::new(encoded_levels);
        let mut decoder;
        if v2 {
            decoder = LevelDecoder::v2(max_level);
            decoder.set_data_range(levels.len(), &byte_buf, 0, byte_buf.len());
        } else {
            decoder = LevelDecoder::v1(enc, max_level);
            decoder.set_data(levels.len(), byte_buf);
        };

        let mut buffer = vec![0; levels.len()];
        let num_decoded = decoder.get(&mut buffer).expect("get() should be OK");
        assert_eq!(num_decoded, levels.len());
        assert_eq!(buffer, levels);
    }

    // Performs incremental read until all bytes are read
    fn test_internal_roundtrip_incremental(
        enc: Encoding,
        levels: &[i16],
        max_level: i16,
        v2: bool,
    ) {
        let size = max_buffer_size(enc, max_level, levels.len());
        let mut encoder = if v2 {
            LevelEncoder::v2(max_level, vec![0; size])
        } else {
            LevelEncoder::v1(enc, max_level, vec![0; size])
        };
        encoder.put(&levels).expect("put() should be OK");
        let encoded_levels = encoder.consume().expect("consume() should be OK");

        let byte_buf = ByteBufferPtr::new(encoded_levels);
        let mut decoder;
        if v2 {
            decoder = LevelDecoder::v2(max_level);
            decoder.set_data_range(levels.len(), &byte_buf, 0, byte_buf.len());
        } else {
            decoder = LevelDecoder::v1(enc, max_level);
            decoder.set_data(levels.len(), byte_buf);
        }

        let mut buffer = vec![0; levels.len() * 2];
        let mut total_decoded = 0;
        let mut safe_stop = levels.len() * 2; // still terminate in case of issues in the code
        while safe_stop > 0 {
            safe_stop -= 1;
            let num_decoded = decoder
                .get(&mut buffer[total_decoded..total_decoded + 1])
                .expect("get() should be OK");
            if num_decoded == 0 {
                break;
            }
            total_decoded += num_decoded;
        }
        assert!(
            safe_stop > 0,
            "Failed to read values incrementally, reached safe stop"
        );
        assert_eq!(total_decoded, levels.len());
        assert_eq!(&buffer[0..levels.len()], levels);
    }

    // Tests encoding/decoding of values when output buffer is larger than number of
    // encoded values
    fn test_internal_roundtrip_underflow(
        enc: Encoding,
        levels: &[i16],
        max_level: i16,
        v2: bool,
    ) {
        let size = max_buffer_size(enc, max_level, levels.len());
        let mut encoder = if v2 {
            LevelEncoder::v2(max_level, vec![0; size])
        } else {
            LevelEncoder::v1(enc, max_level, vec![0; size])
        };
        // Encode only one value
        let num_encoded = encoder.put(&levels[0..1]).expect("put() should be OK");
        let encoded_levels = encoder.consume().expect("consume() should be OK");
        assert_eq!(num_encoded, 1);

        let byte_buf = ByteBufferPtr::new(encoded_levels);
        let mut decoder;
        // Set one encoded value as `num_buffered_values`
        if v2 {
            decoder = LevelDecoder::v2(max_level);
            decoder.set_data_range(1, &byte_buf, 0, byte_buf.len());
        } else {
            decoder = LevelDecoder::v1(enc, max_level);
            decoder.set_data(1, byte_buf);
        }

        let mut buffer = vec![0; levels.len()];
        let num_decoded = decoder.get(&mut buffer).expect("get() should be OK");
        assert_eq!(num_decoded, num_encoded);
        assert_eq!(buffer[0..num_decoded], levels[0..num_decoded]);
    }

    // Tests when encoded values are larger than encoder's buffer
    fn test_internal_roundtrip_overflow(
        enc: Encoding,
        levels: &[i16],
        max_level: i16,
        v2: bool,
    ) {
        let size = max_buffer_size(enc, max_level, levels.len());
        let mut encoder = if v2 {
            LevelEncoder::v2(max_level, vec![0; size])
        } else {
            LevelEncoder::v1(enc, max_level, vec![0; size])
        };
        let mut found_err = false;
        // Insert a large number of values, so we run out of space
        for _ in 0..100 {
            match encoder.put(&levels) {
                Err(err) => {
                    assert!(format!("{}", err).contains("Not enough bytes left"));
                    found_err = true;
                    break;
                }
                Ok(_) => {}
            }
        }
        if !found_err {
            panic!("Failed test: no buffer overflow");
        }
    }

    #[test]
    fn test_roundtrip_one() {
        let levels = vec![0, 1, 1, 1, 1, 0, 0, 0, 0, 1];
        let max_level = 1;
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip(Encoding::BIT_PACKED, &levels, max_level, false);
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_roundtrip() {
        let levels = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let max_level = 10;
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip(Encoding::BIT_PACKED, &levels, max_level, false);
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_roundtrip_incremental() {
        let levels = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let max_level = 10;
        test_internal_roundtrip_incremental(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip_incremental(
            Encoding::BIT_PACKED,
            &levels,
            max_level,
            false,
        );
        test_internal_roundtrip_incremental(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_roundtrip_all_zeros() {
        let levels = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let max_level = 1;
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip(Encoding::BIT_PACKED, &levels, max_level, false);
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_roundtrip_random() {
        // This test is mainly for bit packed level encoder/decoder
        let mut levels = Vec::new();
        let max_level = 5;
        random_numbers_range::<i16>(120, 0, max_level, &mut levels);
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip(Encoding::BIT_PACKED, &levels, max_level, false);
        test_internal_roundtrip(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_roundtrip_underflow() {
        let levels = vec![1, 1, 2, 3, 2, 1, 1, 2, 3, 1];
        let max_level = 3;
        test_internal_roundtrip_underflow(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip_underflow(
            Encoding::BIT_PACKED,
            &levels,
            max_level,
            false,
        );
        test_internal_roundtrip_underflow(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_roundtrip_overflow() {
        let levels = vec![1, 1, 2, 3, 2, 1, 1, 2, 3, 1];
        let max_level = 3;
        test_internal_roundtrip_overflow(Encoding::RLE, &levels, max_level, false);
        test_internal_roundtrip_overflow(Encoding::BIT_PACKED, &levels, max_level, false);
        test_internal_roundtrip_overflow(Encoding::RLE, &levels, max_level, true);
    }

    #[test]
    fn test_rle_decoder_set_data_range() {
        // Buffer containing both repetition and definition levels
        let buffer = ByteBufferPtr::new(vec![5, 198, 2, 5, 42, 168, 10, 0, 2, 3, 36, 73]);

        let max_rep_level = 1;
        let mut decoder = LevelDecoder::v2(max_rep_level);
        assert_eq!(decoder.set_data_range(10, &buffer, 0, 3), 3);
        let mut result = vec![0; 10];
        let num_decoded = decoder.get(&mut result).expect("get() should be OK");
        assert_eq!(num_decoded, 10);
        assert_eq!(result, vec![0, 1, 1, 0, 0, 0, 1, 1, 0, 1]);

        let max_def_level = 2;
        let mut decoder = LevelDecoder::v2(max_def_level);
        assert_eq!(decoder.set_data_range(10, &buffer, 3, 5), 5);
        let mut result = vec![0; 10];
        let num_decoded = decoder.get(&mut result).expect("get() should be OK");
        assert_eq!(num_decoded, 10);
        assert_eq!(result, vec![2, 2, 2, 0, 0, 2, 2, 2, 2, 2]);
    }

    #[test]
    #[should_panic(
        expected = "set_data_range() method is only supported by RLE v2 encoding type"
    )]
    fn test_bit_packed_decoder_set_data_range() {
        // Buffer containing both repetition and definition levels
        let buffer = ByteBufferPtr::new(vec![1, 2, 3, 4, 5]);
        let max_level = 1;
        let mut decoder = LevelDecoder::v1(Encoding::BIT_PACKED, max_level);
        decoder.set_data_range(10, &buffer, 0, 3);
    }

    #[test]
    fn test_bit_packed_decoder_set_data() {
        // Test the maximum size that is assigned based on number of values and buffer
        // length
        let buffer = ByteBufferPtr::new(vec![1, 2, 3, 4, 5]);
        let max_level = 1;
        let mut decoder = LevelDecoder::v1(Encoding::BIT_PACKED, max_level);
        // This should reset to entire buffer
        assert_eq!(decoder.set_data(1024, buffer.all()), buffer.len());
        // This should set smallest num bytes
        assert_eq!(decoder.set_data(3, buffer.all()), 1);
    }

    #[test]
    #[should_panic(expected = "No data set for decoding")]
    fn test_rle_level_decoder_get_no_set_data() {
        // `get()` normally panics because bit_reader is not set for RLE decoding
        // we have explicit check now in set_data
        let max_rep_level = 2;
        let mut decoder = LevelDecoder::v1(Encoding::RLE, max_rep_level);
        let mut buffer = vec![0; 16];
        decoder.get(&mut buffer).unwrap();
    }

    #[test]
    #[should_panic(expected = "No data set for decoding")]
    fn test_bit_packed_level_decoder_get_no_set_data() {
        let max_rep_level = 2;
        let mut decoder = LevelDecoder::v1(Encoding::BIT_PACKED, max_rep_level);
        let mut buffer = vec![0; 16];
        decoder.get(&mut buffer).unwrap();
    }
}
