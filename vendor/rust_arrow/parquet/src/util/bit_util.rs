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

use crate::data_type::AsBytes;
use crate::errors::{ParquetError, Result};
use crate::util::{bit_packing::unpack32, memory::ByteBufferPtr};

pub fn from_ne_slice<T: FromBytes>(bs: &[u8]) -> T {
    let mut b = T::Buffer::default();
    {
        let b = b.as_mut();
        let bs = &bs[..b.len()];
        b.copy_from_slice(bs);
    }
    T::from_ne_bytes(b)
}

pub trait FromBytes: Sized {
    type Buffer: AsMut<[u8]> + Default;
    fn from_le_bytes(bs: Self::Buffer) -> Self;
    fn from_be_bytes(bs: Self::Buffer) -> Self;
    fn from_ne_bytes(bs: Self::Buffer) -> Self;
}

macro_rules! from_le_bytes {
    ($($ty: ty),*) => {
        $(
        impl FromBytes for $ty {
            type Buffer = [u8; size_of::<Self>()];
            fn from_le_bytes(bs: Self::Buffer) -> Self {
                <$ty>::from_le_bytes(bs)
            }
            fn from_be_bytes(bs: Self::Buffer) -> Self {
                <$ty>::from_be_bytes(bs)
            }
            fn from_ne_bytes(bs: Self::Buffer) -> Self {
                <$ty>::from_ne_bytes(bs)
            }
        }
        )*
    };
}

impl FromBytes for bool {
    type Buffer = [u8; 1];
    fn from_le_bytes(bs: Self::Buffer) -> Self {
        Self::from_ne_bytes(bs)
    }
    fn from_be_bytes(bs: Self::Buffer) -> Self {
        Self::from_ne_bytes(bs)
    }
    fn from_ne_bytes(bs: Self::Buffer) -> Self {
        match bs[0] {
            0 => false,
            1 => true,
            _ => panic!("Invalid byte when reading bool"),
        }
    }
}

from_le_bytes! { u8, u16, u32, u64, i8, i16, i32, i64, f32, f64 }

/// Reads `$size` of bytes from `$src`, and reinterprets them as type `$ty`, in
/// little-endian order. `$ty` must implement the `Default` trait. Otherwise this won't
/// compile.
/// This is copied and modified from byteorder crate.
macro_rules! read_num_bytes {
    ($ty:ty, $size:expr, $src:expr) => {{
        assert!($size <= $src.len());
        let mut buffer = <$ty as $crate::util::bit_util::FromBytes>::Buffer::default();
        buffer.as_mut()[..$size].copy_from_slice(&$src[..$size]);
        <$ty>::from_ne_bytes(buffer)
    }};
}

/// Converts value `val` of type `T` to a byte vector, by reading `num_bytes` from `val`.
/// NOTE: if `val` is less than the size of `T` then it can be truncated.
#[inline]
pub fn convert_to_bytes<T>(val: &T, num_bytes: usize) -> Vec<u8>
where
    T: ?Sized + AsBytes,
{
    let mut bytes: Vec<u8> = vec![0; num_bytes];
    memcpy_value(val.as_bytes(), num_bytes, &mut bytes);
    bytes
}

#[inline]
pub fn memcpy(source: &[u8], target: &mut [u8]) {
    assert!(target.len() >= source.len());
    target[..source.len()].copy_from_slice(source)
}

#[inline]
pub fn memcpy_value<T>(source: &T, num_bytes: usize, target: &mut [u8])
where
    T: ?Sized + AsBytes,
{
    assert!(
        target.len() >= num_bytes,
        "Not enough space. Only had {} bytes but need to put {} bytes",
        target.len(),
        num_bytes
    );
    memcpy(&source.as_bytes()[..num_bytes], target)
}

/// Returns the ceil of value/divisor
#[inline]
pub fn ceil(value: i64, divisor: i64) -> i64 {
    let mut result = value / divisor;
    if value % divisor != 0 {
        result += 1
    };
    result
}

/// Returns ceil(log2(x))
#[inline]
pub fn log2(mut x: u64) -> i32 {
    if x == 1 {
        return 0;
    }
    x -= 1;
    let mut result = 0;
    while x > 0 {
        x >>= 1;
        result += 1;
    }
    result
}

/// Returns the `num_bits` least-significant bits of `v`
#[inline]
pub fn trailing_bits(v: u64, num_bits: usize) -> u64 {
    if num_bits == 0 {
        return 0;
    }
    if num_bits >= 64 {
        return v;
    }
    let n = 64 - num_bits;
    (v << n) >> n
}

#[inline]
pub fn set_array_bit(bits: &mut [u8], i: usize) {
    bits[i / 8] |= 1 << (i % 8);
}

#[inline]
pub fn unset_array_bit(bits: &mut [u8], i: usize) {
    bits[i / 8] &= !(1 << (i % 8));
}

/// Returns the minimum number of bits needed to represent the value 'x'
#[inline]
pub fn num_required_bits(x: u64) -> usize {
    for i in (0..64).rev() {
        if x & (1u64 << i) != 0 {
            return i + 1;
        }
    }
    0
}

static BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Returns whether bit at position `i` in `data` is set or not
#[inline]
pub fn get_bit(data: &[u8], i: usize) -> bool {
    (data[i >> 3] & BIT_MASK[i & 7]) != 0
}

/// Utility class for writing bit/byte streams. This class can write data in either
/// bit packed or byte aligned fashion.
pub struct BitWriter {
    buffer: Vec<u8>,
    max_bytes: usize,
    buffered_values: u64,
    byte_offset: usize,
    bit_offset: usize,
    start: usize,
}

impl BitWriter {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            buffer: vec![0; max_bytes],
            max_bytes,
            buffered_values: 0,
            byte_offset: 0,
            bit_offset: 0,
            start: 0,
        }
    }

    /// Initializes the writer from the existing buffer `buffer` and starting
    /// offset `start`.
    pub fn new_from_buf(buffer: Vec<u8>, start: usize) -> Self {
        assert!(start < buffer.len());
        let len = buffer.len();
        Self {
            buffer,
            max_bytes: len,
            buffered_values: 0,
            byte_offset: start,
            bit_offset: 0,
            start,
        }
    }

    /// Consumes and returns the current buffer.
    #[inline]
    pub fn consume(mut self) -> Vec<u8> {
        self.flush();
        self.buffer.truncate(self.byte_offset);
        self.buffer
    }

    /// Flushes the internal buffered bits and returns the buffer's content.
    /// This is a borrow equivalent of `consume` method.
    #[inline]
    pub fn flush_buffer(&mut self) -> &[u8] {
        self.flush();
        &self.buffer()[0..self.byte_offset]
    }

    /// Clears the internal state so the buffer can be reused.
    #[inline]
    pub fn clear(&mut self) {
        self.buffered_values = 0;
        self.byte_offset = self.start;
        self.bit_offset = 0;
    }

    /// Flushes the internal buffered bits and the align the buffer to the next byte.
    #[inline]
    pub fn flush(&mut self) {
        let num_bytes = ceil(self.bit_offset as i64, 8) as usize;
        assert!(self.byte_offset + num_bytes <= self.max_bytes);
        memcpy_value(
            &self.buffered_values,
            num_bytes,
            &mut self.buffer[self.byte_offset..],
        );
        self.buffered_values = 0;
        self.bit_offset = 0;
        self.byte_offset += num_bytes;
    }

    /// Advances the current offset by skipping `num_bytes`, flushing the internal bit
    /// buffer first.
    /// This is useful when you want to jump over `num_bytes` bytes and come back later
    /// to fill these bytes.
    ///
    /// Returns error if `num_bytes` is beyond the boundary of the internal buffer.
    /// Otherwise, returns the old offset.
    #[inline]
    pub fn skip(&mut self, num_bytes: usize) -> Result<usize> {
        self.flush();
        assert!(self.byte_offset <= self.max_bytes);
        if self.byte_offset + num_bytes > self.max_bytes {
            return Err(general_err!(
                "Not enough bytes left in BitWriter. Need {} but only have {}",
                self.byte_offset + num_bytes,
                self.max_bytes
            ));
        }
        let result = self.byte_offset;
        self.byte_offset += num_bytes;
        Ok(result)
    }

    /// Returns a slice containing the next `num_bytes` bytes starting from the current
    /// offset, and advances the underlying buffer by `num_bytes`.
    /// This is useful when you want to jump over `num_bytes` bytes and come back later
    /// to fill these bytes.
    #[inline]
    pub fn get_next_byte_ptr(&mut self, num_bytes: usize) -> Result<&mut [u8]> {
        let offset = self.skip(num_bytes)?;
        Ok(&mut self.buffer[offset..offset + num_bytes])
    }

    #[inline]
    pub fn bytes_written(&self) -> usize {
        self.byte_offset - self.start + ceil(self.bit_offset as i64, 8) as usize
    }

    #[inline]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer[self.start..]
    }

    #[inline]
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Returns the internal buffer length. This is the maximum number of bytes that this
    /// writer can write. User needs to call `consume` to consume the current buffer
    /// before more data can be written.
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.max_bytes
    }

    pub fn write_at(&mut self, offset: usize, value: u8) {
        self.buffer[offset] = value;
    }

    /// Writes the `num_bits` LSB of value `v` to the internal buffer of this writer.
    /// The `num_bits` must not be greater than 64. This is bit packed.
    ///
    /// Returns false if there's not enough room left. True otherwise.
    #[inline]
    pub fn put_value(&mut self, v: u64, num_bits: usize) -> bool {
        assert!(num_bits <= 64);
        assert_eq!(v.checked_shr(num_bits as u32).unwrap_or(0), 0); // covers case v >> 64

        if self.byte_offset * 8 + self.bit_offset + num_bits > self.max_bytes as usize * 8
        {
            return false;
        }

        self.buffered_values |= v << self.bit_offset;
        self.bit_offset += num_bits;
        if self.bit_offset >= 64 {
            memcpy_value(
                &self.buffered_values,
                8,
                &mut self.buffer[self.byte_offset..],
            );
            self.byte_offset += 8;
            self.bit_offset -= 64;
            self.buffered_values = 0;
            // Perform checked right shift: v >> offset, where offset < 64, otherwise we
            // shift all bits
            self.buffered_values = v
                .checked_shr((num_bits - self.bit_offset) as u32)
                .unwrap_or(0);
        }
        assert!(self.bit_offset < 64);
        true
    }

    /// Writes `val` of `num_bytes` bytes to the next aligned byte. If size of `T` is
    /// larger than `num_bytes`, extra higher ordered bytes will be ignored.
    ///
    /// Returns false if there's not enough room left. True otherwise.
    #[inline]
    pub fn put_aligned<T: AsBytes>(&mut self, val: T, num_bytes: usize) -> bool {
        let result = self.get_next_byte_ptr(num_bytes);
        if result.is_err() {
            // TODO: should we return `Result` for this func?
            return false;
        }
        let mut ptr = result.unwrap();
        memcpy_value(&val, num_bytes, &mut ptr);
        true
    }

    /// Writes `val` of `num_bytes` bytes at the designated `offset`. The `offset` is the
    /// offset starting from the beginning of the internal buffer that this writer
    /// maintains. Note that this will overwrite any existing data between `offset` and
    /// `offset + num_bytes`. Also that if size of `T` is larger than `num_bytes`, extra
    /// higher ordered bytes will be ignored.
    ///
    /// Returns false if there's not enough room left, or the `pos` is not valid.
    /// True otherwise.
    #[inline]
    pub fn put_aligned_offset<T: AsBytes>(
        &mut self,
        val: T,
        num_bytes: usize,
        offset: usize,
    ) -> bool {
        if num_bytes + offset > self.max_bytes {
            return false;
        }
        memcpy_value(
            &val,
            num_bytes,
            &mut self.buffer[offset..offset + num_bytes],
        );
        true
    }

    /// Writes a VLQ encoded integer `v` to this buffer. The value is byte aligned.
    ///
    /// Returns false if there's not enough room left. True otherwise.
    #[inline]
    pub fn put_vlq_int(&mut self, mut v: u64) -> bool {
        let mut result = true;
        while v & 0xFFFFFFFFFFFFFF80 != 0 {
            result &= self.put_aligned::<u8>(((v & 0x7F) | 0x80) as u8, 1);
            v >>= 7;
        }
        result &= self.put_aligned::<u8>((v & 0x7F) as u8, 1);
        result
    }

    /// Writes a zigzag-VLQ encoded (in little endian order) int `v` to this buffer.
    /// Zigzag-VLQ is a variant of VLQ encoding where negative and positive
    /// numbers are encoded in a zigzag fashion.
    /// See: https://developers.google.com/protocol-buffers/docs/encoding
    ///
    /// Returns false if there's not enough room left. True otherwise.
    #[inline]
    pub fn put_zigzag_vlq_int(&mut self, v: i64) -> bool {
        let u: u64 = ((v << 1) ^ (v >> 63)) as u64;
        self.put_vlq_int(u)
    }
}

/// Maximum byte length for a VLQ encoded integer
/// MAX_VLQ_BYTE_LEN = 5 for i32, and MAX_VLQ_BYTE_LEN = 10 for i64
pub const MAX_VLQ_BYTE_LEN: usize = 10;

pub struct BitReader {
    // The byte buffer to read from, passed in by client
    buffer: ByteBufferPtr,

    // Bytes are memcpy'd from `buffer` and values are read from this variable.
    // This is faster than reading values byte by byte directly from `buffer`
    buffered_values: u64,

    //
    // End                                         Start
    // |............|B|B|B|B|B|B|B|B|..............|
    //                   ^          ^
    //                 bit_offset   byte_offset
    //
    // Current byte offset in `buffer`
    byte_offset: usize,

    // Current bit offset in `buffered_values`
    bit_offset: usize,

    // Total number of bytes in `buffer`
    total_bytes: usize,
}

/// Utility class to read bit/byte stream. This class can read bits or bytes that are
/// either byte aligned or not.
impl BitReader {
    pub fn new(buffer: ByteBufferPtr) -> Self {
        let total_bytes = buffer.len();
        let num_bytes = cmp::min(8, total_bytes);
        let buffered_values = read_num_bytes!(u64, num_bytes, buffer.as_ref());
        BitReader {
            buffer,
            buffered_values,
            byte_offset: 0,
            bit_offset: 0,
            total_bytes,
        }
    }

    #[inline]
    pub fn reset(&mut self, buffer: ByteBufferPtr) {
        self.buffer = buffer;
        self.total_bytes = self.buffer.len();
        let num_bytes = cmp::min(8, self.total_bytes);
        self.buffered_values = read_num_bytes!(u64, num_bytes, self.buffer.as_ref());
        self.byte_offset = 0;
        self.bit_offset = 0;
    }

    /// Gets the current byte offset
    #[inline]
    pub fn get_byte_offset(&self) -> usize {
        self.byte_offset + ceil(self.bit_offset as i64, 8) as usize
    }

    /// Reads a value of type `T` and of size `num_bits`.
    ///
    /// Returns `None` if there's not enough data available. `Some` otherwise.
    #[inline]
    pub fn get_value<T: FromBytes>(&mut self, num_bits: usize) -> Option<T> {
        assert!(num_bits <= 64);
        assert!(num_bits <= size_of::<T>() * 8);

        if self.byte_offset * 8 + self.bit_offset + num_bits > self.total_bytes * 8 {
            return None;
        }

        let mut v = trailing_bits(self.buffered_values, self.bit_offset + num_bits)
            >> self.bit_offset;
        self.bit_offset += num_bits;

        if self.bit_offset >= 64 {
            self.byte_offset += 8;
            self.bit_offset -= 64;

            self.reload_buffer_values();
            v |= trailing_bits(self.buffered_values, self.bit_offset)
                .wrapping_shl((num_bits - self.bit_offset) as u32);
        }

        // TODO: better to avoid copying here
        Some(from_ne_slice(v.as_bytes()))
    }

    #[inline]
    pub fn get_batch<T: FromBytes>(&mut self, batch: &mut [T], num_bits: usize) -> usize {
        assert!(num_bits <= 32);
        assert!(num_bits <= size_of::<T>() * 8);

        let mut values_to_read = batch.len();
        let needed_bits = num_bits * values_to_read;
        let remaining_bits = (self.total_bytes - self.byte_offset) * 8 - self.bit_offset;
        if remaining_bits < needed_bits {
            values_to_read = remaining_bits / num_bits;
        }

        let mut i = 0;

        // First align bit offset to byte offset
        if self.bit_offset != 0 {
            while i < values_to_read && self.bit_offset != 0 {
                batch[i] = self
                    .get_value(num_bits)
                    .expect("expected to have more data");
                i += 1;
            }
        }

        unsafe {
            let in_buf = &self.buffer.data()[self.byte_offset..];
            let mut in_ptr = in_buf as *const [u8] as *const u8 as *const u32;
            // FIXME assert!(memory::is_ptr_aligned(in_ptr));
            if size_of::<T>() == 4 {
                while values_to_read - i >= 32 {
                    let out_ptr = &mut batch[i..] as *mut [T] as *mut T as *mut u32;
                    in_ptr = unpack32(in_ptr, out_ptr, num_bits);
                    self.byte_offset += 4 * num_bits;
                    i += 32;
                }
            } else {
                let mut out_buf = [0u32; 32];
                let out_ptr = &mut out_buf as &mut [u32] as *mut [u32] as *mut u32;
                while values_to_read - i >= 32 {
                    in_ptr = unpack32(in_ptr, out_ptr, num_bits);
                    self.byte_offset += 4 * num_bits;
                    for n in 0..32 {
                        // We need to copy from smaller size to bigger size to avoid
                        // overwriting other memory regions.
                        if size_of::<T>() > size_of::<u32>() {
                            std::ptr::copy_nonoverlapping(
                                out_buf[n..].as_ptr() as *const u32,
                                &mut batch[i] as *mut T as *mut u32,
                                1,
                            );
                        } else {
                            std::ptr::copy_nonoverlapping(
                                out_buf[n..].as_ptr() as *const T,
                                &mut batch[i] as *mut T,
                                1,
                            );
                        }
                        i += 1;
                    }
                }
            }
        }

        assert!(values_to_read - i < 32);

        self.reload_buffer_values();
        while i < values_to_read {
            batch[i] = self
                .get_value(num_bits)
                .expect("expected to have more data");
            i += 1;
        }

        values_to_read
    }

    /// Reads a `num_bytes`-sized value from this buffer and return it.
    /// `T` needs to be a little-endian native type. The value is assumed to be byte
    /// aligned so the bit reader will be advanced to the start of the next byte before
    /// reading the value.

    /// Returns `Some` if there's enough bytes left to form a value of `T`.
    /// Otherwise `None`.
    #[inline]
    pub fn get_aligned<T: FromBytes>(&mut self, num_bytes: usize) -> Option<T> {
        let bytes_read = ceil(self.bit_offset as i64, 8) as usize;
        if self.byte_offset + bytes_read + num_bytes > self.total_bytes {
            return None;
        }

        // Advance byte_offset to next unread byte and read num_bytes
        self.byte_offset += bytes_read;
        let v = read_num_bytes!(
            T,
            num_bytes,
            self.buffer.start_from(self.byte_offset).as_ref()
        );
        self.byte_offset += num_bytes;

        // Reset buffered_values
        self.bit_offset = 0;
        self.reload_buffer_values();
        Some(v)
    }

    /// Reads a VLQ encoded (in little endian order) int from the stream.
    /// The encoded int must start at the beginning of a byte.
    ///
    /// Returns `None` if there's not enough bytes in the stream. `Some` otherwise.
    #[inline]
    pub fn get_vlq_int(&mut self) -> Option<i64> {
        let mut shift = 0;
        let mut v: i64 = 0;
        while let Some(byte) = self.get_aligned::<u8>(1) {
            v |= ((byte & 0x7F) as i64) << shift;
            shift += 7;
            assert!(
                shift <= MAX_VLQ_BYTE_LEN * 7,
                "Num of bytes exceed MAX_VLQ_BYTE_LEN ({})",
                MAX_VLQ_BYTE_LEN
            );
            if byte & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    /// Reads a zigzag-VLQ encoded (in little endian order) int from the stream
    /// Zigzag-VLQ is a variant of VLQ encoding where negative and positive numbers are
    /// encoded in a zigzag fashion.
    /// See: https://developers.google.com/protocol-buffers/docs/encoding
    ///
    /// Note: the encoded int must start at the beginning of a byte.
    ///
    /// Returns `None` if the number of bytes there's not enough bytes in the stream.
    /// `Some` otherwise.
    #[inline]
    pub fn get_zigzag_vlq_int(&mut self) -> Option<i64> {
        self.get_vlq_int().map(|v| {
            let u = v as u64;
            (u >> 1) as i64 ^ -((u & 1) as i64)
        })
    }

    #[inline]
    fn reload_buffer_values(&mut self) {
        let bytes_to_read = cmp::min(self.total_bytes - self.byte_offset, 8);
        self.buffered_values = read_num_bytes!(
            u64,
            bytes_to_read,
            self.buffer.start_from(self.byte_offset).as_ref()
        );
    }
}

impl From<Vec<u8>> for BitReader {
    #[inline]
    fn from(buffer: Vec<u8>) -> Self {
        BitReader::new(ByteBufferPtr::new(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_common::*;
    use super::*;

    use rand::distributions::{Distribution, Standard};
    use std::fmt::Debug;

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
    fn test_bit_reader_get_byte_offset() {
        let buffer = vec![255; 10];
        let mut bit_reader = BitReader::from(buffer);
        assert_eq!(bit_reader.get_byte_offset(), 0); // offset (0 bytes, 0 bits)
        bit_reader.get_value::<i32>(6);
        assert_eq!(bit_reader.get_byte_offset(), 1); // offset (0 bytes, 6 bits)
        bit_reader.get_value::<i32>(10);
        assert_eq!(bit_reader.get_byte_offset(), 2); // offset (0 bytes, 16 bits)
        bit_reader.get_value::<i32>(20);
        assert_eq!(bit_reader.get_byte_offset(), 5); // offset (0 bytes, 36 bits)
        bit_reader.get_value::<i32>(30);
        assert_eq!(bit_reader.get_byte_offset(), 9); // offset (8 bytes, 2 bits)
    }

    #[test]
    fn test_bit_reader_get_value() {
        let buffer = vec![255, 0];
        let mut bit_reader = BitReader::from(buffer);
        assert_eq!(bit_reader.get_value::<i32>(1), Some(1));
        assert_eq!(bit_reader.get_value::<i32>(2), Some(3));
        assert_eq!(bit_reader.get_value::<i32>(3), Some(7));
        assert_eq!(bit_reader.get_value::<i32>(4), Some(3));
    }

    #[test]
    fn test_bit_reader_get_value_boundary() {
        let buffer = vec![10, 0, 0, 0, 20, 0, 30, 0, 0, 0, 40, 0];
        let mut bit_reader = BitReader::from(buffer);
        assert_eq!(bit_reader.get_value::<i64>(32), Some(10));
        assert_eq!(bit_reader.get_value::<i64>(16), Some(20));
        assert_eq!(bit_reader.get_value::<i64>(32), Some(30));
        assert_eq!(bit_reader.get_value::<i64>(16), Some(40));
    }

    #[test]
    fn test_bit_reader_get_aligned() {
        // 01110101 11001011
        let buffer = ByteBufferPtr::new(vec![0x75, 0xCB]);
        let mut bit_reader = BitReader::new(buffer.all());
        assert_eq!(bit_reader.get_value::<i32>(3), Some(5));
        assert_eq!(bit_reader.get_aligned::<i32>(1), Some(203));
        assert_eq!(bit_reader.get_value::<i32>(1), None);
        bit_reader.reset(buffer.all());
        assert_eq!(bit_reader.get_aligned::<i32>(3), None);
    }

    #[test]
    fn test_bit_reader_get_vlq_int() {
        // 10001001 00000001 11110010 10110101 00000110
        let buffer: Vec<u8> = vec![0x89, 0x01, 0xF2, 0xB5, 0x06];
        let mut bit_reader = BitReader::from(buffer);
        assert_eq!(bit_reader.get_vlq_int(), Some(137));
        assert_eq!(bit_reader.get_vlq_int(), Some(105202));
    }

    #[test]
    fn test_bit_reader_get_zigzag_vlq_int() {
        let buffer: Vec<u8> = vec![0, 1, 2, 3];
        let mut bit_reader = BitReader::from(buffer);
        assert_eq!(bit_reader.get_zigzag_vlq_int(), Some(0));
        assert_eq!(bit_reader.get_zigzag_vlq_int(), Some(-1));
        assert_eq!(bit_reader.get_zigzag_vlq_int(), Some(1));
        assert_eq!(bit_reader.get_zigzag_vlq_int(), Some(-2));
    }

    #[test]
    fn test_set_array_bit() {
        let mut buffer = vec![0, 0, 0];
        set_array_bit(&mut buffer[..], 1);
        assert_eq!(buffer, vec![2, 0, 0]);
        set_array_bit(&mut buffer[..], 4);
        assert_eq!(buffer, vec![18, 0, 0]);
        unset_array_bit(&mut buffer[..], 1);
        assert_eq!(buffer, vec![16, 0, 0]);
        set_array_bit(&mut buffer[..], 10);
        assert_eq!(buffer, vec![16, 4, 0]);
        set_array_bit(&mut buffer[..], 10);
        assert_eq!(buffer, vec![16, 4, 0]);
        set_array_bit(&mut buffer[..], 11);
        assert_eq!(buffer, vec![16, 12, 0]);
        unset_array_bit(&mut buffer[..], 10);
        assert_eq!(buffer, vec![16, 8, 0]);
    }

    #[test]
    fn test_num_required_bits() {
        assert_eq!(num_required_bits(0), 0);
        assert_eq!(num_required_bits(1), 1);
        assert_eq!(num_required_bits(2), 2);
        assert_eq!(num_required_bits(4), 3);
        assert_eq!(num_required_bits(8), 4);
        assert_eq!(num_required_bits(10), 4);
        assert_eq!(num_required_bits(12), 4);
        assert_eq!(num_required_bits(16), 5);
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
    fn test_log2() {
        assert_eq!(log2(1), 0);
        assert_eq!(log2(2), 1);
        assert_eq!(log2(3), 2);
        assert_eq!(log2(4), 2);
        assert_eq!(log2(5), 3);
        assert_eq!(log2(5), 3);
        assert_eq!(log2(6), 3);
        assert_eq!(log2(7), 3);
        assert_eq!(log2(8), 3);
        assert_eq!(log2(9), 4);
    }

    #[test]
    fn test_skip() {
        let mut writer = BitWriter::new(5);
        let old_offset = writer.skip(1).expect("skip() should return OK");
        writer.put_aligned(42, 4);
        writer.put_aligned_offset(0x10, 1, old_offset);
        let result = writer.consume();
        assert_eq!(result.as_ref(), [0x10, 42, 0, 0, 0]);

        writer = BitWriter::new(4);
        let result = writer.skip(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_next_byte_ptr() {
        let mut writer = BitWriter::new(5);
        {
            let first_byte = writer
                .get_next_byte_ptr(1)
                .expect("get_next_byte_ptr() should return OK");
            first_byte[0] = 0x10;
        }
        writer.put_aligned(42, 4);
        let result = writer.consume();
        assert_eq!(result.as_ref(), [0x10, 42, 0, 0, 0]);
    }

    #[test]
    fn test_consume_flush_buffer() {
        let mut writer1 = BitWriter::new(3);
        let mut writer2 = BitWriter::new(3);
        for i in 1..10 {
            writer1.put_value(i, 4);
            writer2.put_value(i, 4);
        }
        let res1 = writer1.flush_buffer();
        let res2 = writer2.consume();
        assert_eq!(res1, &res2[..]);
    }

    #[test]
    fn test_put_get_bool() {
        let len = 8;
        let mut writer = BitWriter::new(len);

        for i in 0..8 {
            let result = writer.put_value(i % 2, 1);
            assert!(result);
        }

        writer.flush();
        {
            let buffer = writer.buffer();
            assert_eq!(buffer[0], 0b10101010);
        }

        // Write 00110011
        for i in 0..8 {
            let result = match i {
                0 | 1 | 4 | 5 => writer.put_value(false as u64, 1),
                _ => writer.put_value(true as u64, 1),
            };
            assert!(result);
        }
        writer.flush();
        {
            let buffer = writer.buffer();
            assert_eq!(buffer[0], 0b10101010);
            assert_eq!(buffer[1], 0b11001100);
        }

        let mut reader = BitReader::from(writer.consume());

        for i in 0..8 {
            let val = reader
                .get_value::<u8>(1)
                .expect("get_value() should return OK");
            assert_eq!(val, i % 2);
        }

        for i in 0..8 {
            let val = reader
                .get_value::<bool>(1)
                .expect("get_value() should return OK");
            match i {
                0 | 1 | 4 | 5 => assert_eq!(val, false),
                _ => assert_eq!(val, true),
            }
        }
    }

    #[test]
    fn test_put_value_roundtrip() {
        test_put_value_rand_numbers(32, 2);
        test_put_value_rand_numbers(32, 3);
        test_put_value_rand_numbers(32, 4);
        test_put_value_rand_numbers(32, 5);
        test_put_value_rand_numbers(32, 6);
        test_put_value_rand_numbers(32, 7);
        test_put_value_rand_numbers(32, 8);
        test_put_value_rand_numbers(64, 16);
        test_put_value_rand_numbers(64, 24);
        test_put_value_rand_numbers(64, 32);
    }

    fn test_put_value_rand_numbers(total: usize, num_bits: usize) {
        assert!(num_bits < 64);
        let num_bytes = ceil(num_bits as i64, 8);
        let mut writer = BitWriter::new(num_bytes as usize * total);
        let values: Vec<u64> = random_numbers::<u64>(total)
            .iter()
            .map(|v| v & ((1 << num_bits) - 1))
            .collect();
        for i in 0..total {
            assert!(
                writer.put_value(values[i] as u64, num_bits),
                "[{}]: put_value() failed",
                i
            );
        }

        let mut reader = BitReader::from(writer.consume());
        for i in 0..total {
            let v = reader
                .get_value::<u64>(num_bits)
                .expect("get_value() should return OK");
            assert_eq!(
                v, values[i],
                "[{}]: expected {} but got {}",
                i, values[i], v
            );
        }
    }

    #[test]
    fn test_get_batch() {
        const SIZE: &[usize] = &[1, 31, 32, 33, 128, 129];
        for s in SIZE {
            for i in 0..33 {
                match i {
                    0..=8 => test_get_batch_helper::<u8>(*s, i),
                    9..=16 => test_get_batch_helper::<u16>(*s, i),
                    _ => test_get_batch_helper::<u32>(*s, i),
                }
            }
        }
    }

    fn test_get_batch_helper<T>(total: usize, num_bits: usize)
    where
        T: FromBytes + Default + Clone + Debug + Eq,
    {
        assert!(num_bits <= 32);
        let num_bytes = ceil(num_bits as i64, 8);
        let mut writer = BitWriter::new(num_bytes as usize * total);

        let values: Vec<u32> = random_numbers::<u32>(total)
            .iter()
            .map(|v| v & ((1u64 << num_bits) - 1) as u32)
            .collect();

        // Generic values used to check against actual values read from `get_batch`.
        let expected_values: Vec<T> =
            values.iter().map(|v| from_ne_slice(v.as_bytes())).collect();

        for i in 0..total {
            assert!(writer.put_value(values[i] as u64, num_bits));
        }

        let buf = writer.consume();
        let mut reader = BitReader::from(buf);
        let mut batch = vec![T::default(); values.len()];
        let values_read = reader.get_batch::<T>(&mut batch, num_bits);
        assert_eq!(values_read, values.len());
        for i in 0..batch.len() {
            assert_eq!(
                batch[i], expected_values[i],
                "num_bits = {}, index = {}",
                num_bits, i
            );
        }
    }

    #[test]
    fn test_put_aligned_roundtrip() {
        test_put_aligned_rand_numbers::<u8>(4, 3);
        test_put_aligned_rand_numbers::<u8>(16, 5);
        test_put_aligned_rand_numbers::<i16>(32, 7);
        test_put_aligned_rand_numbers::<i16>(32, 9);
        test_put_aligned_rand_numbers::<i32>(32, 11);
        test_put_aligned_rand_numbers::<i32>(32, 13);
        test_put_aligned_rand_numbers::<i64>(32, 17);
        test_put_aligned_rand_numbers::<i64>(32, 23);
    }

    fn test_put_aligned_rand_numbers<T>(total: usize, num_bits: usize)
    where
        T: Copy + FromBytes + AsBytes + Debug + PartialEq,
        Standard: Distribution<T>,
    {
        assert!(num_bits <= 32);
        assert!(total % 2 == 0);

        let aligned_value_byte_width = std::mem::size_of::<T>();
        let value_byte_width = ceil(num_bits as i64, 8) as usize;
        let mut writer =
            BitWriter::new((total / 2) * (aligned_value_byte_width + value_byte_width));
        let values: Vec<u32> = random_numbers::<u32>(total / 2)
            .iter()
            .map(|v| v & ((1 << num_bits) - 1))
            .collect();
        let aligned_values = random_numbers::<T>(total / 2);

        for i in 0..total {
            let j = i / 2;
            if i % 2 == 0 {
                assert!(
                    writer.put_value(values[j] as u64, num_bits),
                    "[{}]: put_value() failed",
                    i
                );
            } else {
                assert!(
                    writer.put_aligned::<T>(aligned_values[j], aligned_value_byte_width),
                    "[{}]: put_aligned() failed",
                    i
                );
            }
        }

        let mut reader = BitReader::from(writer.consume());
        for i in 0..total {
            let j = i / 2;
            if i % 2 == 0 {
                let v = reader
                    .get_value::<u64>(num_bits)
                    .expect("get_value() should return OK");
                assert_eq!(
                    v, values[j] as u64,
                    "[{}]: expected {} but got {}",
                    i, values[j], v
                );
            } else {
                let v = reader
                    .get_aligned::<T>(aligned_value_byte_width)
                    .expect("get_aligned() should return OK");
                assert_eq!(
                    v, aligned_values[j],
                    "[{}]: expected {:?} but got {:?}",
                    i, aligned_values[j], v
                );
            }
        }
    }

    #[test]
    fn test_put_vlq_int() {
        let total = 64;
        let mut writer = BitWriter::new(total * 32);
        let values = random_numbers::<u32>(total);
        for i in 0..total {
            assert!(
                writer.put_vlq_int(values[i] as u64),
                "[{}]; put_vlq_int() failed",
                i
            );
        }

        let mut reader = BitReader::from(writer.consume());
        for i in 0..total {
            let v = reader
                .get_vlq_int()
                .expect("get_vlq_int() should return OK");
            assert_eq!(
                v as u32, values[i],
                "[{}]: expected {} but got {}",
                i, values[i], v
            );
        }
    }

    #[test]
    fn test_put_zigzag_vlq_int() {
        let total = 64;
        let mut writer = BitWriter::new(total * 32);
        let values = random_numbers::<i32>(total);
        for i in 0..total {
            assert!(
                writer.put_zigzag_vlq_int(values[i] as i64),
                "[{}]; put_zigzag_vlq_int() failed",
                i
            );
        }

        let mut reader = BitReader::from(writer.consume());
        for i in 0..total {
            let v = reader
                .get_zigzag_vlq_int()
                .expect("get_zigzag_vlq_int() should return OK");
            assert_eq!(
                v as i32, values[i],
                "[{}]: expected {} but got {}",
                i, values[i], v
            );
        }
    }
}
