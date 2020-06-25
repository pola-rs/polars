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

//! Utils for working with bits

#[cfg(feature = "simd")]
use packed_simd::u8x64;

static BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

static POPCOUNT_TABLE: [u8; 256] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
    3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4,
    3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4,
    3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7,
    6, 7, 7, 8,
];

/// Returns the nearest number that is `>=` than `num` and is a multiple of 64
#[inline]
pub fn round_upto_multiple_of_64(num: usize) -> usize {
    round_upto_power_of_2(num, 64)
}

/// Returns the nearest multiple of `factor` that is `>=` than `num`. Here `factor` must
/// be a power of 2.
fn round_upto_power_of_2(num: usize, factor: usize) -> usize {
    debug_assert!(factor > 0 && (factor & (factor - 1)) == 0);
    (num + (factor - 1)) & !(factor - 1)
}

/// Returns whether bit at position `i` in `data` is set or not
#[inline]
pub fn get_bit(data: &[u8], i: usize) -> bool {
    (data[i >> 3] & BIT_MASK[i & 7]) != 0
}

/// Returns whether bit at position `i` in `data` is set or not.
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that `i` is within bounds.
#[inline]
pub unsafe fn get_bit_raw(data: *const u8, i: usize) -> bool {
    (*data.add(i >> 3) & BIT_MASK[i & 7]) != 0
}

/// Sets bit at position `i` for `data`
#[inline]
pub fn set_bit(data: &mut [u8], i: usize) {
    data[i >> 3] |= BIT_MASK[i & 7]
}

/// Sets bit at position `i` for `data`
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that `i` is within bounds.
#[inline]
pub unsafe fn set_bit_raw(data: *mut u8, i: usize) {
    *data.add(i >> 3) |= BIT_MASK[i & 7]
}

/// Sets bits in the non-inclusive range `start..end` for `data`
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that both `start` and `end` are within bounds.
#[inline]
pub unsafe fn set_bits_raw(data: *mut u8, start: usize, end: usize) {
    let start_byte = (start >> 3) as isize;
    let end_byte = (end >> 3) as isize;

    let start_offset = (start & 7) as u8;
    let end_offset = (end & 7) as u8;

    // All set apart from lowest `start_offset` bits
    let start_mask = !((1 << start_offset) - 1);
    // All clear apart from lowest `end_offset` bits
    let end_mask = (1 << end_offset) - 1;

    if start_byte == end_byte {
        *data.offset(start_byte) |= start_mask & end_mask;
    } else {
        *data.offset(start_byte) |= start_mask;
        for i in (start_byte + 1)..end_byte {
            *data.offset(i) = 0xFF;
        }
        *data.offset(end_byte) |= end_mask;
    }
}

/// Returns the number of 1-bits in `data`
#[inline]
pub fn count_set_bits(data: &[u8]) -> usize {
    let mut count: usize = 0;
    for u in data {
        count += POPCOUNT_TABLE[*u as usize] as usize;
    }
    count
}

/// Returns the number of 1-bits in `data`, starting from `offset` with `length` bits
/// inspected. Note that both `offset` and `length` are measured in bits.
#[inline]
pub fn count_set_bits_offset(data: &[u8], offset: usize, length: usize) -> usize {
    let bit_end = offset + length;
    assert!(bit_end <= (data.len() << 3));

    let byte_start = std::cmp::min(round_upto_power_of_2(offset, 8), bit_end);
    let num_bytes = (bit_end - byte_start) >> 3;

    let mut result = 0;

    for i in offset..byte_start {
        if get_bit(data, i) {
            result += 1;
        }
    }
    for i in 0..num_bytes {
        result += POPCOUNT_TABLE[data[(byte_start >> 3) + i] as usize] as usize;
    }
    for i in (byte_start + (num_bytes << 3))..bit_end {
        if get_bit(data, i) {
            result += 1;
        }
    }

    result
}

/// Returns the ceil of `value`/`divisor`
#[inline]
pub fn ceil(value: usize, divisor: usize) -> usize {
    let (quot, rem) = (value / divisor, value % divisor);
    if rem > 0 && divisor > 0 {
        quot + 1
    } else {
        quot
    }
}

/// Performs SIMD bitwise binary operations.
///
/// Note that each slice should be 64 bytes and it is the callers responsibility to ensure
/// that this is the case.  If passed slices larger than 64 bytes the operation will only
/// be performed on the first 64 bytes.  Slices less than 64 bytes will panic.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
pub unsafe fn bitwise_bin_op_simd<F>(left: &[u8], right: &[u8], result: &mut [u8], op: F)
where
    F: Fn(u8x64, u8x64) -> u8x64,
{
    let left_simd = u8x64::from_slice_unaligned_unchecked(left);
    let right_simd = u8x64::from_slice_unaligned_unchecked(right);
    let simd_result = op(left_simd, right_simd);
    simd_result.write_to_slice_unaligned_unchecked(result);
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_round_upto_multiple_of_64() {
        assert_eq!(0, round_upto_multiple_of_64(0));
        assert_eq!(64, round_upto_multiple_of_64(1));
        assert_eq!(64, round_upto_multiple_of_64(63));
        assert_eq!(64, round_upto_multiple_of_64(64));
        assert_eq!(128, round_upto_multiple_of_64(65));
        assert_eq!(192, round_upto_multiple_of_64(129));
    }

    #[test]
    fn test_get_bit() {
        // 00001101
        assert_eq!(true, get_bit(&[0b00001101], 0));
        assert_eq!(false, get_bit(&[0b00001101], 1));
        assert_eq!(true, get_bit(&[0b00001101], 2));
        assert_eq!(true, get_bit(&[0b00001101], 3));

        // 01001001 01010010
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 0));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 1));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 2));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 3));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 4));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 5));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 6));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 7));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 8));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 9));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 10));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 11));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 12));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 13));
        assert_eq!(true, get_bit(&[0b01001001, 0b01010010], 14));
        assert_eq!(false, get_bit(&[0b01001001, 0b01010010], 15));
    }

    #[test]
    fn test_get_bit_raw() {
        const NUM_BYTE: usize = 10;
        let mut buf = vec![0; NUM_BYTE];
        let mut expected = vec![];
        let mut rng = thread_rng();
        for i in 0..8 * NUM_BYTE {
            let b = rng.gen_bool(0.5);
            expected.push(b);
            if b {
                set_bit(&mut buf[..], i)
            }
        }

        let raw_ptr = buf.as_ptr();
        for (i, b) in expected.iter().enumerate() {
            unsafe {
                assert_eq!(*b, get_bit_raw(raw_ptr, i));
            }
        }
    }

    #[test]
    fn test_set_bit() {
        let mut b = [0b00000000];
        set_bit(&mut b, 0);
        assert_eq!([0b00000001], b);
        set_bit(&mut b, 2);
        assert_eq!([0b00000101], b);
        set_bit(&mut b, 5);
        assert_eq!([0b00100101], b);
    }

    #[test]
    fn test_set_bit_raw() {
        const NUM_BYTE: usize = 10;
        let mut buf = vec![0; NUM_BYTE];
        let mut expected = vec![];
        let mut rng = thread_rng();
        for i in 0..8 * NUM_BYTE {
            let b = rng.gen_bool(0.5);
            expected.push(b);
            if b {
                unsafe {
                    set_bit_raw(buf.as_mut_ptr(), i);
                }
            }
        }

        let raw_ptr = buf.as_ptr();
        for (i, b) in expected.iter().enumerate() {
            unsafe {
                assert_eq!(*b, get_bit_raw(raw_ptr, i));
            }
        }
    }

    #[test]
    fn test_set_bits_raw() {
        const NUM_BYTE: usize = 64;
        const NUM_BLOCKS: usize = 12;
        const MAX_BLOCK_SIZE: usize = 32;
        let mut buf = vec![0; NUM_BYTE];

        let mut expected = Vec::with_capacity(NUM_BYTE * 8);
        expected.resize(NUM_BYTE * 8, false);

        let mut rng = thread_rng();

        for _ in 0..NUM_BLOCKS {
            let start = rng.gen_range(0, NUM_BYTE * 8 - MAX_BLOCK_SIZE);
            let end = start + rng.gen_range(1, MAX_BLOCK_SIZE);
            unsafe {
                set_bits_raw(buf.as_mut_ptr(), start, end);
            }
            for i in start..end {
                expected[i] = true;
            }
        }

        let raw_ptr = buf.as_ptr();
        for (i, b) in expected.iter().enumerate() {
            unsafe {
                assert_eq!(*b, get_bit_raw(raw_ptr, i));
            }
        }
    }

    #[test]
    fn test_get_set_bit_roundtrip() {
        const NUM_BYTES: usize = 10;
        const NUM_SETS: usize = 10;

        let mut buffer: [u8; NUM_BYTES * 8] = [0; NUM_BYTES * 8];
        let mut v = HashSet::new();
        let mut rng = thread_rng();
        for _ in 0..NUM_SETS {
            let offset = rng.gen_range(0, 8 * NUM_BYTES);
            v.insert(offset);
            set_bit(&mut buffer[..], offset);
        }
        for i in 0..NUM_BYTES * 8 {
            assert_eq!(v.contains(&i), get_bit(&buffer[..], i));
        }
    }

    #[test]
    fn test_count_bits_slice() {
        assert_eq!(0, count_set_bits(&[0b00000000]));
        assert_eq!(8, count_set_bits(&[0b11111111]));
        assert_eq!(3, count_set_bits(&[0b00001101]));
        assert_eq!(6, count_set_bits(&[0b01001001, 0b01010010]));
    }

    #[test]
    fn test_count_bits_offset_slice() {
        assert_eq!(8, count_set_bits_offset(&[0b11111111], 0, 8));
        assert_eq!(3, count_set_bits_offset(&[0b11111111], 0, 3));
        assert_eq!(5, count_set_bits_offset(&[0b11111111], 3, 5));
        assert_eq!(1, count_set_bits_offset(&[0b11111111], 3, 1));
        assert_eq!(0, count_set_bits_offset(&[0b11111111], 8, 0));
        assert_eq!(2, count_set_bits_offset(&[0b01010101], 0, 3));
        assert_eq!(16, count_set_bits_offset(&[0b11111111, 0b11111111], 0, 16));
        assert_eq!(10, count_set_bits_offset(&[0b11111111, 0b11111111], 0, 10));
        assert_eq!(10, count_set_bits_offset(&[0b11111111, 0b11111111], 3, 10));
        assert_eq!(8, count_set_bits_offset(&[0b11111111, 0b11111111], 8, 8));
        assert_eq!(5, count_set_bits_offset(&[0b11111111, 0b11111111], 11, 5));
        assert_eq!(0, count_set_bits_offset(&[0b11111111, 0b11111111], 16, 0));
        assert_eq!(2, count_set_bits_offset(&[0b01101101, 0b10101010], 7, 5));
        assert_eq!(4, count_set_bits_offset(&[0b01101101, 0b10101010], 7, 9));
    }

    #[test]
    fn test_ceil() {
        assert_eq!(ceil(0, 1), 0);
        assert_eq!(ceil(1, 1), 1);
        assert_eq!(ceil(1, 2), 1);
        assert_eq!(ceil(1, 8), 1);
        assert_eq!(ceil(7, 8), 1);
        assert_eq!(ceil(8, 8), 1);
        assert_eq!(ceil(9, 8), 2);
        assert_eq!(ceil(9, 9), 1);
        assert_eq!(ceil(10000000000, 10), 1000000000);
        assert_eq!(ceil(10, 10000000000), 1);
        assert_eq!(ceil(10000000000, 1000000000), 10);
    }

    #[test]
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    fn test_bitwise_and_simd() {
        let buf1 = [0b00110011u8; 64];
        let buf2 = [0b11110000u8; 64];
        let mut buf3 = [0b00000000; 64];
        unsafe { bitwise_bin_op_simd(&buf1, &buf2, &mut buf3, |a, b| a & b) };
        for i in buf3.iter() {
            assert_eq!(&0b00110000u8, i);
        }
    }

    #[test]
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    fn test_bitwise_or_simd() {
        let buf1 = [0b00110011u8; 64];
        let buf2 = [0b11110000u8; 64];
        let mut buf3 = [0b00000000; 64];
        unsafe { bitwise_bin_op_simd(&buf1, &buf2, &mut buf3, |a, b| a | b) };
        for i in buf3.iter() {
            assert_eq!(&0b11110011u8, i);
        }
    }
}
