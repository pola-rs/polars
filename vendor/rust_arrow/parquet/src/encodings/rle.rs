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

use std::{cmp, mem::size_of};

use crate::errors::{ParquetError, Result};
use crate::util::{
    bit_util::{self, from_ne_slice, BitReader, BitWriter, FromBytes},
    memory::ByteBufferPtr,
};

/// Rle/Bit-Packing Hybrid Encoding
/// The grammar for this encoding looks like the following (copied verbatim
/// from https://github.com/Parquet/parquet-format/blob/master/Encodings.md):
///
/// rle-bit-packed-hybrid: <length> <encoded-data>
/// length := length of the <encoded-data> in bytes stored as 4 bytes little endian
/// encoded-data := <run>*
/// run := <bit-packed-run> | <rle-run>
/// bit-packed-run := <bit-packed-header> <bit-packed-values>
/// bit-packed-header := varint-encode(<bit-pack-count> << 1 | 1)
/// we always bit-pack a multiple of 8 values at a time, so we only store the number of
/// values / 8
/// bit-pack-count := (number of values in this run) / 8
/// bit-packed-values := *see 1 below*
/// rle-run := <rle-header> <repeated-value>
/// rle-header := varint-encode( (number of times repeated) << 1)
/// repeated-value := value that is repeated, using a fixed-width of
/// round-up-to-next-byte(bit-width)

/// Maximum groups per bit-packed run. Current value is 64.
const MAX_GROUPS_PER_BIT_PACKED_RUN: usize = 1 << 6;
const MAX_VALUES_PER_BIT_PACKED_RUN: usize = MAX_GROUPS_PER_BIT_PACKED_RUN * 8;
const MAX_WRITER_BUF_SIZE: usize = 1 << 10;

/// A RLE/Bit-Packing hybrid encoder.
// TODO: tracking memory usage
pub struct RleEncoder {
    // Number of bits needed to encode the value. Must be in the range of [0, 64].
    bit_width: u8,

    // Underlying writer which holds an internal buffer.
    bit_writer: BitWriter,

    // If this is true, the buffer is full and subsequent `put()` calls will fail.
    buffer_full: bool,

    // The maximum byte size a single run can take.
    max_run_byte_size: usize,

    // Buffered values for bit-packed runs.
    buffered_values: [u64; 8],

    // Number of current buffered values. Must be less than 8.
    num_buffered_values: usize,

    // The current (also last) value that was written and the count of how many
    // times in a row that value has been seen.
    current_value: u64,

    // The number of repetitions for `current_value`. If this gets too high we'd
    // switch to use RLE encoding.
    repeat_count: usize,

    // Number of bit-packed values in the current run. This doesn't include values
    // in `buffered_values`.
    bit_packed_count: usize,

    // The position of the indicator byte in the `bit_writer`.
    indicator_byte_pos: i64,
}

impl RleEncoder {
    pub fn new(bit_width: u8, buffer_len: usize) -> Self {
        let buffer = vec![0; buffer_len];
        RleEncoder::new_from_buf(bit_width, buffer, 0)
    }

    /// Initialize the encoder from existing `buffer` and the starting offset `start`.
    pub fn new_from_buf(bit_width: u8, buffer: Vec<u8>, start: usize) -> Self {
        assert!(bit_width <= 64, "bit_width ({}) out of range.", bit_width);
        let max_run_byte_size = RleEncoder::min_buffer_size(bit_width);
        assert!(
            buffer.len() >= max_run_byte_size,
            "buffer length {} must be greater than {}",
            buffer.len(),
            max_run_byte_size
        );
        let bit_writer = BitWriter::new_from_buf(buffer, start);
        RleEncoder {
            bit_width,
            bit_writer,
            buffer_full: false,
            max_run_byte_size,
            buffered_values: [0; 8],
            num_buffered_values: 0,
            current_value: 0,
            repeat_count: 0,
            bit_packed_count: 0,
            indicator_byte_pos: -1,
        }
    }

    /// Returns the minimum buffer size needed to use the encoder for `bit_width`.
    /// This is the maximum length of a single run for `bit_width`.
    pub fn min_buffer_size(bit_width: u8) -> usize {
        let max_bit_packed_run_size = 1 + bit_util::ceil(
            (MAX_VALUES_PER_BIT_PACKED_RUN * bit_width as usize) as i64,
            8,
        );
        let max_rle_run_size =
            bit_util::MAX_VLQ_BYTE_LEN + bit_util::ceil(bit_width as i64, 8) as usize;
        std::cmp::max(max_bit_packed_run_size as usize, max_rle_run_size)
    }

    /// Returns the maximum buffer size takes to encode `num_values` values with
    /// `bit_width`.
    pub fn max_buffer_size(bit_width: u8, num_values: usize) -> usize {
        // First the maximum size for bit-packed run
        let bytes_per_run = bit_width;
        let num_runs = bit_util::ceil(num_values as i64, 8) as usize;
        let bit_packed_max_size = num_runs + num_runs * bytes_per_run as usize;

        // Second the maximum size for RLE run
        let min_rle_run_size = 1 + bit_util::ceil(bit_width as i64, 8) as usize;
        let rle_max_size =
            bit_util::ceil(num_values as i64, 8) as usize * min_rle_run_size;
        std::cmp::max(bit_packed_max_size, rle_max_size) as usize
    }

    /// Encodes `value`, which must be representable with `bit_width` bits.
    /// Returns true if the value fits in buffer, false if it doesn't, or
    /// error if something is wrong.
    #[inline]
    pub fn put(&mut self, value: u64) -> Result<bool> {
        // This function buffers 8 values at a time. After seeing 8 values, it
        // decides whether the current run should be encoded in bit-packed or RLE.
        if self.buffer_full {
            // The value cannot fit in the current buffer.
            return Ok(false);
        }
        if self.current_value == value {
            self.repeat_count += 1;
            if self.repeat_count > 8 {
                // A continuation of last value. No need to buffer.
                return Ok(true);
            }
        } else {
            if self.repeat_count >= 8 {
                // The current RLE run has ended and we've gathered enough. Flush first.
                assert_eq!(self.bit_packed_count, 0);
                self.flush_rle_run()?;
            }
            self.repeat_count = 1;
            self.current_value = value;
        }

        self.buffered_values[self.num_buffered_values] = value;
        self.num_buffered_values += 1;
        if self.num_buffered_values == 8 {
            // Buffered values are full. Flush them.
            assert_eq!(self.bit_packed_count % 8, 0);
            self.flush_buffered_values()?;
        }

        Ok(true)
    }

    #[inline]
    pub fn buffer(&self) -> &[u8] {
        self.bit_writer.buffer()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bit_writer.bytes_written()
    }

    #[inline]
    pub fn consume(mut self) -> Result<Vec<u8>> {
        self.flush()?;
        Ok(self.bit_writer.consume())
    }

    /// Borrow equivalent of the `consume` method.
    /// Call `clear()` after invoking this method.
    #[inline]
    pub fn flush_buffer(&mut self) -> Result<&[u8]> {
        self.flush()?;
        Ok(self.bit_writer.flush_buffer())
    }

    /// Clears the internal state so this encoder can be reused (e.g., after becoming
    /// full).
    #[inline]
    pub fn clear(&mut self) {
        self.bit_writer.clear();
        self.buffer_full = false;
        self.num_buffered_values = 0;
        self.current_value = 0;
        self.repeat_count = 0;
        self.bit_packed_count = 0;
        self.indicator_byte_pos = -1;
    }

    /// Flushes all remaining values and return the final byte buffer maintained by the
    /// internal writer.
    #[inline]
    pub fn flush(&mut self) -> Result<()> {
        if self.bit_packed_count > 0
            || self.repeat_count > 0
            || self.num_buffered_values > 0
        {
            let all_repeat = self.bit_packed_count == 0
                && (self.repeat_count == self.num_buffered_values
                    || self.num_buffered_values == 0);
            if self.repeat_count > 0 && all_repeat {
                self.flush_rle_run()?;
            } else {
                // Buffer the last group of bit-packed values to 8 by padding with 0s.
                if self.num_buffered_values > 0 {
                    while self.num_buffered_values < 8 {
                        self.buffered_values[self.num_buffered_values] = 0;
                        self.num_buffered_values += 1;
                    }
                }
                self.bit_packed_count += self.num_buffered_values;
                self.flush_bit_packed_run(true)?;
                self.repeat_count = 0;
            }
        }
        Ok(())
    }

    #[inline]
    fn flush_rle_run(&mut self) -> Result<()> {
        assert!(self.repeat_count > 0);
        let indicator_value = self.repeat_count << 1 | 0;
        let mut result = self.bit_writer.put_vlq_int(indicator_value as u64);
        result &= self.bit_writer.put_aligned(
            self.current_value,
            bit_util::ceil(self.bit_width as i64, 8) as usize,
        );
        if !result {
            return Err(general_err!("Failed to write RLE run"));
        }
        self.num_buffered_values = 0;
        self.repeat_count = 0;
        Ok(())
    }

    #[inline]
    fn flush_bit_packed_run(&mut self, update_indicator_byte: bool) -> Result<()> {
        if self.indicator_byte_pos < 0 {
            self.indicator_byte_pos = self.bit_writer.skip(1)? as i64;
        }

        // Write all buffered values as bit-packed literals
        for i in 0..self.num_buffered_values {
            let _ = self
                .bit_writer
                .put_value(self.buffered_values[i], self.bit_width as usize);
        }
        self.num_buffered_values = 0;
        if update_indicator_byte {
            // Write the indicator byte to the reserved position in `bit_writer`
            let num_groups = self.bit_packed_count / 8;
            let indicator_byte = ((num_groups << 1) | 1) as u8;
            if !self.bit_writer.put_aligned_offset(
                indicator_byte,
                1,
                self.indicator_byte_pos as usize,
            ) {
                return Err(general_err!("Not enough space to write indicator byte"));
            }
            self.indicator_byte_pos = -1;
            self.bit_packed_count = 0;
        }
        Ok(())
    }

    #[inline]
    fn flush_buffered_values(&mut self) -> Result<()> {
        if self.repeat_count >= 8 {
            self.num_buffered_values = 0;
            if self.bit_packed_count > 0 {
                // In this case we choose RLE encoding. Flush the current buffered values
                // as bit-packed encoding.
                assert_eq!(self.bit_packed_count % 8, 0);
                self.flush_bit_packed_run(true)?
            }
            return Ok(());
        }

        self.bit_packed_count += self.num_buffered_values;
        let num_groups = self.bit_packed_count / 8;
        if num_groups + 1 >= MAX_GROUPS_PER_BIT_PACKED_RUN {
            // We've reached the maximum value that can be hold in a single bit-packed
            // run.
            assert!(self.indicator_byte_pos >= 0);
            self.flush_bit_packed_run(true)?;
        } else {
            self.flush_bit_packed_run(false)?;
        }
        self.repeat_count = 0;
        Ok(())
    }
}

/// A RLE/Bit-Packing hybrid decoder.
pub struct RleDecoder {
    // Number of bits used to encode the value. Must be between [0, 64].
    bit_width: u8,

    // Bit reader loaded with input buffer.
    bit_reader: Option<BitReader>,

    // Buffer used when `bit_reader` is not `None`, for batch reading.
    index_buf: Option<[i32; 1024]>,

    // The remaining number of values in RLE for this run
    rle_left: u32,

    // The remaining number of values in Bit-Packing for this run
    bit_packed_left: u32,

    // The current value for the case of RLE mode
    current_value: Option<u64>,
}

impl RleDecoder {
    pub fn new(bit_width: u8) -> Self {
        RleDecoder {
            bit_width,
            rle_left: 0,
            bit_packed_left: 0,
            bit_reader: None,
            index_buf: None,
            current_value: None,
        }
    }

    pub fn set_data(&mut self, data: ByteBufferPtr) {
        if let Some(ref mut bit_reader) = self.bit_reader {
            bit_reader.reset(data);
        } else {
            self.bit_reader = Some(BitReader::new(data));
            self.index_buf = Some([0; 1024]);
        }

        let _ = self.reload();
    }

    #[inline]
    pub fn get<T: FromBytes>(&mut self) -> Result<Option<T>> {
        assert!(size_of::<T>() <= 8);

        while self.rle_left <= 0 && self.bit_packed_left <= 0 {
            if !self.reload() {
                return Ok(None);
            }
        }

        let value = if self.rle_left > 0 {
            let rle_value = from_ne_slice(
                &self
                    .current_value
                    .as_mut()
                    .expect("current_value should be Some")
                    .to_ne_bytes(),
            );
            self.rle_left -= 1;
            rle_value
        } else {
            // self.bit_packed_left > 0
            let bit_reader = self.bit_reader.as_mut().expect("bit_reader should be Some");
            let bit_packed_value = bit_reader
                .get_value(self.bit_width as usize)
                .ok_or(eof_err!("Not enough data for 'bit_packed_value'"))?;
            self.bit_packed_left -= 1;
            bit_packed_value
        };

        Ok(Some(value))
    }

    #[inline]
    pub fn get_batch<T: FromBytes>(&mut self, buffer: &mut [T]) -> Result<usize> {
        assert!(self.bit_reader.is_some());
        assert!(size_of::<T>() <= 8);

        let mut values_read = 0;
        while values_read < buffer.len() {
            if self.rle_left > 0 {
                assert!(self.current_value.is_some());
                let num_values =
                    cmp::min(buffer.len() - values_read, self.rle_left as usize);
                for i in 0..num_values {
                    let repeated_value = from_ne_slice(
                        &self.current_value.as_mut().unwrap().to_ne_bytes(),
                    );
                    buffer[values_read + i] = repeated_value;
                }
                self.rle_left -= num_values as u32;
                values_read += num_values;
            } else if self.bit_packed_left > 0 {
                assert!(self.bit_reader.is_some());
                let mut num_values =
                    cmp::min(buffer.len() - values_read, self.bit_packed_left as usize);
                if let Some(ref mut bit_reader) = self.bit_reader {
                    num_values = bit_reader.get_batch::<T>(
                        &mut buffer[values_read..values_read + num_values],
                        self.bit_width as usize,
                    );
                    self.bit_packed_left -= num_values as u32;
                    values_read += num_values;
                }
            } else {
                if !self.reload() {
                    break;
                }
            }
        }

        Ok(values_read)
    }

    #[inline]
    pub fn get_batch_with_dict<T>(
        &mut self,
        dict: &[T],
        buffer: &mut [T],
        max_values: usize,
    ) -> Result<usize>
    where
        T: Default + Clone,
    {
        assert!(buffer.len() >= max_values);

        let mut values_read = 0;
        while values_read < max_values {
            if self.rle_left > 0 {
                assert!(self.current_value.is_some());
                let num_values =
                    cmp::min(max_values - values_read, self.rle_left as usize);
                let dict_idx = self.current_value.unwrap() as usize;
                for i in 0..num_values {
                    buffer[values_read + i] = dict[dict_idx].clone();
                }
                self.rle_left -= num_values as u32;
                values_read += num_values;
            } else if self.bit_packed_left > 0 {
                assert!(self.bit_reader.is_some());
                let mut num_values =
                    cmp::min(max_values - values_read, self.bit_packed_left as usize);
                if let Some(ref mut bit_reader) = self.bit_reader {
                    let mut index_buf = self.index_buf.unwrap();
                    num_values = cmp::min(num_values, index_buf.len());
                    loop {
                        num_values = bit_reader.get_batch::<i32>(
                            &mut index_buf[..num_values],
                            self.bit_width as usize,
                        );
                        for i in 0..num_values {
                            buffer[values_read + i] = dict[index_buf[i] as usize].clone();
                        }
                        self.bit_packed_left -= num_values as u32;
                        values_read += num_values;
                        if num_values < index_buf.len() {
                            break;
                        }
                    }
                }
            } else {
                if !self.reload() {
                    break;
                }
            }
        }

        Ok(values_read)
    }

    #[inline]
    fn reload(&mut self) -> bool {
        assert!(self.bit_reader.is_some());
        if let Some(ref mut bit_reader) = self.bit_reader {
            if let Some(indicator_value) = bit_reader.get_vlq_int() {
                if indicator_value & 1 == 1 {
                    self.bit_packed_left = ((indicator_value >> 1) * 8) as u32;
                } else {
                    self.rle_left = (indicator_value >> 1) as u32;
                    let value_width = bit_util::ceil(self.bit_width as i64, 8);
                    self.current_value =
                        bit_reader.get_aligned::<u64>(value_width as usize);
                    assert!(self.current_value.is_some());
                }
                return true;
            } else {
                return false;
            }
        }
        return false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{self, distributions::Standard, thread_rng, Rng, SeedableRng};

    use crate::util::memory::ByteBufferPtr;

    const MAX_WIDTH: usize = 32;

    #[test]
    fn test_rle_decode_int32() {
        // Test data: 0-7 with bit width 3
        // 00000011 10001000 11000110 11111010
        let data = ByteBufferPtr::new(vec![0x03, 0x88, 0xC6, 0xFA]);
        let mut decoder: RleDecoder = RleDecoder::new(3);
        decoder.set_data(data);
        let mut buffer = vec![0; 8];
        let expected = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let result = decoder.get_batch::<i32>(&mut buffer);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_rle_consume_flush_buffer() {
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3];
        let mut encoder1 = RleEncoder::new(3, 256);
        let mut encoder2 = RleEncoder::new(3, 256);
        for value in data {
            encoder1.put(value as u64).unwrap();
            encoder2.put(value as u64).unwrap();
        }
        let res1 = encoder1.flush_buffer().unwrap();
        let res2 = encoder2.consume().unwrap();
        assert_eq!(res1, &res2[..]);
    }

    #[test]
    fn test_rle_decode_bool() {
        // RLE test data: 50 1s followed by 50 0s
        // 01100100 00000001 01100100 00000000
        let data1 = ByteBufferPtr::new(vec![0x64, 0x01, 0x64, 0x00]);

        // Bit-packing test data: alternating 1s and 0s, 100 total
        // 100 / 8 = 13 groups
        // 00011011 10101010 ... 00001010
        let data2 = ByteBufferPtr::new(vec![
            0x1B, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
            0x0A,
        ]);

        let mut decoder: RleDecoder = RleDecoder::new(1);
        decoder.set_data(data1);
        let mut buffer = vec![false; 100];
        let mut expected = vec![];
        for i in 0..100 {
            if i < 50 {
                expected.push(true);
            } else {
                expected.push(false);
            }
        }
        let result = decoder.get_batch::<bool>(&mut buffer);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);

        decoder.set_data(data2);
        let mut buffer = vec![false; 100];
        let mut expected = vec![];
        for i in 0..100 {
            if i % 2 == 0 {
                expected.push(false);
            } else {
                expected.push(true);
            }
        }
        let result = decoder.get_batch::<bool>(&mut buffer);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_rle_decode_with_dict_int32() {
        // Test RLE encoding: 3 0s followed by 4 1s followed by 5 2s
        // 00000110 00000000 00001000 00000001 00001010 00000010
        let dict = vec![10, 20, 30];
        let data = ByteBufferPtr::new(vec![0x06, 0x00, 0x08, 0x01, 0x0A, 0x02]);
        let mut decoder: RleDecoder = RleDecoder::new(3);
        decoder.set_data(data);
        let mut buffer = vec![0; 12];
        let expected = vec![10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 30];
        let result = decoder.get_batch_with_dict::<i32>(&dict, &mut buffer, 12);
        assert!(result.is_ok());
        assert_eq!(buffer, expected);

        // Test bit-pack encoding: 345345345455 (2 groups: 8 and 4)
        // 011 100 101 011 100 101 011 100 101 100 101 101
        // 00000011 01100011 11000111 10001110 00000011 01100101 00001011
        let dict = vec!["aaa", "bbb", "ccc", "ddd", "eee", "fff"];
        let data = ByteBufferPtr::new(vec![0x03, 0x63, 0xC7, 0x8E, 0x03, 0x65, 0x0B]);
        let mut decoder: RleDecoder = RleDecoder::new(3);
        decoder.set_data(data);
        let mut buffer = vec![""; 12];
        let expected = vec![
            "ddd", "eee", "fff", "ddd", "eee", "fff", "ddd", "eee", "fff", "eee", "fff",
            "fff",
        ];
        let result = decoder.get_batch_with_dict::<&str>(
            dict.as_slice(),
            buffer.as_mut_slice(),
            12,
        );
        assert!(result.is_ok());
        assert_eq!(buffer, expected);
    }

    fn validate_rle(
        values: &[i64],
        bit_width: u8,
        expected_encoding: Option<&[u8]>,
        expected_len: i32,
    ) {
        let buffer_len = 64 * 1024;
        let mut encoder = RleEncoder::new(bit_width, buffer_len);
        for v in values {
            let result = encoder.put(*v as u64);
            assert!(result.is_ok());
        }
        let buffer = ByteBufferPtr::new(encoder.consume().expect("Expect consume() OK"));
        if expected_len != -1 {
            assert_eq!(buffer.len(), expected_len as usize);
        }
        match expected_encoding {
            Some(b) => assert_eq!(buffer.as_ref(), b),
            _ => (),
        }

        // Verify read
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer.all());
        for v in values {
            let val: i64 = decoder
                .get()
                .expect("get() should be OK")
                .expect("get() should return more value");
            assert_eq!(val, *v);
        }

        // Verify batch read
        decoder.set_data(buffer);
        let mut values_read: Vec<i64> = vec![0; values.len()];
        decoder
            .get_batch(&mut values_read[..])
            .expect("get_batch() should be OK");
        assert_eq!(&values_read[..], values);
    }

    #[test]
    fn test_rle_specific_sequences() {
        let mut expected_buffer = Vec::new();
        let mut values = Vec::new();
        for _ in 0..50 {
            values.push(0);
        }
        for _ in 0..50 {
            values.push(1);
        }
        expected_buffer.push(50 << 1);
        expected_buffer.push(0);
        expected_buffer.push(50 << 1);
        expected_buffer.push(1);

        for width in 1..9 {
            validate_rle(&values[..], width, Some(&expected_buffer[..]), 4);
        }
        for width in 9..MAX_WIDTH + 1 {
            validate_rle(
                &values[..],
                width as u8,
                None,
                2 * (1 + bit_util::ceil(width as i64, 8) as i32),
            );
        }

        // Test 100 0's and 1's alternating
        values.clear();
        expected_buffer.clear();
        for i in 0..101 {
            values.push(i % 2);
        }
        let num_groups = bit_util::ceil(100, 8) as u8;
        expected_buffer.push(((num_groups << 1) as u8) | 1);
        for _ in 1..(100 / 8) + 1 {
            expected_buffer.push(0b10101010);
        }
        // For the last 4 0 and 1's, padded with 0.
        expected_buffer.push(0b00001010);
        validate_rle(
            &values,
            1,
            Some(&expected_buffer[..]),
            1 + num_groups as i32,
        );
        for width in 2..MAX_WIDTH + 1 {
            let num_values = bit_util::ceil(100, 8) * 8;
            validate_rle(
                &values,
                width as u8,
                None,
                1 + bit_util::ceil(width as i64 * num_values, 8) as i32,
            );
        }
    }

    // `validate_rle` on `num_vals` with width `bit_width`. If `value` is -1, that value
    // is used, otherwise alternating values are used.
    fn test_rle_values(bit_width: usize, num_vals: usize, value: i32) {
        let mod_val = if bit_width == 64 {
            1
        } else {
            1u64 << bit_width
        };
        let mut values: Vec<i64> = vec![];
        for v in 0..num_vals {
            let val = if value == -1 {
                v as i64 % mod_val as i64
            } else {
                value as i64
            };
            values.push(val);
        }
        validate_rle(&values, bit_width as u8, None, -1);
    }

    #[test]
    fn test_values() {
        for width in 1..MAX_WIDTH + 1 {
            test_rle_values(width, 1, -1);
            test_rle_values(width, 1024, -1);
            test_rle_values(width, 1024, 0);
            test_rle_values(width, 1024, 1);
        }
    }

    #[test]
    fn test_rle_specific_roundtrip() {
        let bit_width = 1;
        let buffer_len = RleEncoder::min_buffer_size(bit_width);
        let values: Vec<i16> = vec![0, 1, 1, 1, 1, 0, 0, 0, 0, 1];
        let mut encoder = RleEncoder::new(bit_width, buffer_len);
        for v in &values {
            assert!(encoder.put(*v as u64).expect("put() should be OK"));
        }
        let buffer = encoder.consume().expect("consume() should be OK");
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(ByteBufferPtr::new(buffer));
        let mut actual_values: Vec<i16> = vec![0; values.len()];
        decoder
            .get_batch(&mut actual_values)
            .expect("get_batch() should be OK");
        assert_eq!(actual_values, values);
    }

    fn test_round_trip(values: &[i32], bit_width: u8) {
        let buffer_len = 64 * 1024;
        let mut encoder = RleEncoder::new(bit_width, buffer_len);
        for v in values {
            let result = encoder.put(*v as u64).expect("put() should be OK");
            assert!(result, "put() should not return false");
        }

        let buffer =
            ByteBufferPtr::new(encoder.consume().expect("consume() should be OK"));

        // Verify read
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer.all());
        for v in values {
            let val = decoder
                .get::<i32>()
                .expect("get() should be OK")
                .expect("get() should return value");
            assert_eq!(val, *v);
        }

        // Verify batch read
        let mut decoder = RleDecoder::new(bit_width);
        decoder.set_data(buffer);
        let mut values_read: Vec<i32> = vec![0; values.len()];
        decoder
            .get_batch(&mut values_read[..])
            .expect("get_batch() should be OK");
        assert_eq!(&values_read[..], values);
    }

    #[test]
    fn test_random() {
        let seed_len = 32;
        let niters = 50;
        let ngroups = 1000;
        let max_group_size = 15;
        let mut values = vec![];

        for _ in 0..niters {
            values.clear();
            let mut rng = thread_rng();
            let seed_vec: Vec<u8> =
                rng.sample_iter::<u8, _>(&Standard).take(seed_len).collect();
            let mut seed = [0u8; 32];
            seed.copy_from_slice(&seed_vec[0..seed_len]);
            let mut gen = rand::rngs::StdRng::from_seed(seed);

            let mut parity = false;
            for _ in 0..ngroups {
                let mut group_size = gen.gen_range(1, 20);
                if group_size > max_group_size {
                    group_size = 1;
                }
                for _ in 0..group_size {
                    values.push(parity as i32);
                }
                parity = !parity;
            }
            let bit_width = bit_util::num_required_bits(values.len() as u64);
            assert!(bit_width < 64);
            test_round_trip(&values[..], bit_width as u8);
        }
    }
}
