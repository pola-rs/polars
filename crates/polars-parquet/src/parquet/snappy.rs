//! Snappy decompression into uninitialized memory.
//!
//! This is the decompressor of the `snap` crate (version 1.1.2), changed to write into
//! `[MaybeUninit<u8>]` as proposed in <https://github.com/BurntSushi/rust-snappy/pull/65>. Once
//! `snap` has such an API, this module can be removed.
//!
//! Copyright 2011, The Snappy-Rust Authors. All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without modification, are permitted
//! provided that the following conditions are met:
//!
//! * Redistributions of source code must retain the above copyright notice, this list of
//!   conditions and the following disclaimer.
//! * Redistributions in binary form must reproduce the above copyright notice, this list of
//!   conditions and the following disclaimer in the documentation and/or other materials provided
//!   with the distribution.
//! * Neither the name of the copyright holder nor the names of its contributors may be used to
//!   endorse or promote products derived from this software without specific prior written
//!   permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
//! IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
//! FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//! CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//! THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
//! OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//! POSSIBILITY OF SUCH DAMAGE.

use std::mem::MaybeUninit;
use std::ptr;

use crate::parquet::error::{ParquetError, ParquetResult};

/// The largest decompressed size that Snappy supports.
const MAX_DECOMPRESSED_LEN: u64 = u32::MAX as u64;

/// Maps the size of an integer in bytes to the mask of its bits in a 32 bit integer.
const WORD_MASK: [usize; 5] = [0, 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF];

/// For each tag byte, the attributes derived from it, with bits `xxaa abbb xxcc cccc`:
///
/// * `a`: the number of bytes after the tag byte that encode the offset of a copy (1 for copy 1,
///   2 for copy 2, 4 for copy 4). For literals longer than 60 bytes, the number of bytes that
///   encode the length.
/// * `b`: the three high bits of the offset of a copy 1 (0 otherwise). They line up with the bits
///   of the total offset, which saves a shift.
/// * `c`: the length of the copy, or of the literal if it is at most 60 bytes.
const TAG_LOOKUP_TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u16;
        table[i] = match b & 0b11 {
            0b00 => {
                let lit_len = (b >> 2) + 1;
                if lit_len <= 60 {
                    lit_len
                } else {
                    (lit_len - 60) << 11
                }
            },
            0b01 => {
                let len = 4 + ((b >> 2) & 0b111);
                let offset = (b >> 5) & 0b111;
                (1 << 11) | (offset << 8) | len
            },
            0b10 => (2 << 11) | (1 + (b >> 2)),
            _ => (4 << 11) | (1 + (b >> 2)),
        };
        i += 1;
    }
    table
};

#[cold]
fn corrupt(msg: &str) -> ParquetError {
    ParquetError::oos(format!("Invalid snappy data: {msg}"))
}

/// Decompresses the Snappy compressed `input` into the start of `output`.
///
/// Returns the number of bytes written, which is the decompressed size from the header. Those
/// bytes of `output` are initialized.
pub fn decompress(input: &[u8], output: &mut [MaybeUninit<u8>]) -> ParquetResult<usize> {
    if input.is_empty() {
        return Err(corrupt("empty input"));
    }
    let hdr = Header::read(input)?;
    if hdr.decompress_len > output.len() {
        return Err(corrupt("output buffer is too small"));
    }
    let dst = &mut output[..hdr.decompress_len];
    let mut dec = Decompress {
        src: &input[hdr.len..],
        s: 0,
        dst,
        d: 0,
    };
    dec.decompress()?;
    Ok(dec.dst.len())
}

struct Decompress<'s, 'd> {
    /// The compressed bytes without the header.
    src: &'s [u8],
    /// The current position in `src`.
    s: usize,
    /// The output buffer. Every byte before `d` is initialized.
    dst: &'d mut [MaybeUninit<u8>],
    /// The current position in `dst`.
    d: usize,
}

impl Decompress<'_, '_> {
    fn decompress(&mut self) -> ParquetResult<()> {
        while self.s < self.src.len() {
            let byte = self.src[self.s];
            self.s += 1;
            if byte & 0b11 == 0 {
                let len = (byte >> 2) as usize + 1;
                self.read_literal(len)?;
            } else {
                self.read_copy(byte)?;
            }
        }
        // Every byte of `dst` must be written, so that all of it is initialized.
        if self.d != self.dst.len() {
            return Err(corrupt("decompressed length does not match the header"));
        }
        Ok(())
    }

    /// Copies a literal from `src` to `dst`. `s` points to the byte after the tag byte.
    ///
    /// `len` is the length of the literal if it is at most 60. Otherwise, `len - 60` is the number
    /// of bytes at `src[s..]` that hold the little endian length minus one.
    #[inline(always)]
    fn read_literal(&mut self, len: usize) -> ParquetResult<()> {
        debug_assert!(len <= 64);
        let mut len = len as u64;
        // A short literal with enough room on both sides is copied with one 16 byte load and
        // store. The bytes after the literal are overwritten later.
        if len <= 16 && self.s + 16 <= self.src.len() && self.d + 16 <= self.dst.len() {
            unsafe {
                // SAFETY: both `src` and `dst` have at least 16 bytes after `s` and `d`.
                let srcp = self.src.as_ptr().add(self.s);
                let dstp = self.dst.as_mut_ptr().add(self.d).cast::<u8>();
                ptr::copy_nonoverlapping(srcp, dstp, 16);
            }
            self.d += len as usize;
            self.s += len as usize;
            return Ok(());
        }
        if len >= 61 {
            let byte_count = len as usize - 60;
            if self.src.len() - self.s < byte_count {
                return Err(corrupt("literal length is out of bounds"));
            }
            if self.src.len() - self.s >= 4 {
                len = read_u32_le(&self.src[self.s..]) as u64;
                len = (len & (WORD_MASK[byte_count] as u64)) + 1;
            } else {
                len = 0;
                for i in 0..byte_count {
                    len |= (self.src[self.s + i] as u64) << (8 * i);
                }
                len += 1;
            }
            self.s += byte_count;
        }
        if ((self.src.len() - self.s) as u64) < len || ((self.dst.len() - self.d) as u64) < len {
            return Err(corrupt("literal is out of bounds"));
        }
        unsafe {
            // SAFETY: the bounds are checked above.
            let srcp = self.src.as_ptr().add(self.s);
            let dstp = self.dst.as_mut_ptr().add(self.d).cast::<u8>();
            ptr::copy_nonoverlapping(srcp, dstp, len as usize);
        }
        self.s += len as usize;
        self.d += len as usize;
        Ok(())
    }

    /// Copies earlier output to `dst`. `s` points to the byte after the tag byte.
    #[inline(always)]
    fn read_copy(&mut self, tag_byte: u8) -> ParquetResult<()> {
        let entry = TagEntry(TAG_LOOKUP_TABLE[tag_byte as usize] as usize);
        let offset = entry.offset(self.src, self.s)?;
        let len = entry.len();
        self.s += entry.num_tag_bytes();

        // Checks `d == 0 || d < offset` with one branch: if `offset` is 0, `offset - 1` wraps to
        // `usize::MAX`.
        if self.d <= offset.wrapping_sub(1) {
            return Err(corrupt("copy offset is out of bounds"));
        }
        let end = self.d + len;
        // A short copy at least 8 bytes back is done with two 8 byte loads and stores.
        if offset >= 8 && len <= 16 && self.d + 16 <= self.dst.len() {
            unsafe {
                // SAFETY: `dstp` has at least 16 bytes after it and `offset` bytes before it.
                // Each 8 byte copy has a source at least 8 bytes before its destination, so they
                // do not overlap. The second copy may read bytes written by the first.
                let dstp = self.dst.as_mut_ptr().add(self.d);
                let srcp = dstp.sub(offset);
                ptr::copy_nonoverlapping(srcp, dstp, 8);
                ptr::copy_nonoverlapping(srcp.add(8), dstp.add(8), 8);
            }
        // With enough room after the copy, copy 16 bytes at a time. This writes up to 24 bytes
        // past `end`, which are overwritten later, and may read bytes of `dst` that were not
        // written yet. Raw pointer copies of uninitialized bytes are allowed, and those bytes
        // only end up after `end`.
        } else if end + 24 <= self.dst.len() {
            unsafe {
                // SAFETY: `dstp` has `offset` bytes before it and at least `len + 24` after it.
                let dest_len = self.dst.len();
                let mut dstp = self.dst.as_mut_ptr().add(self.d);
                let mut srcp = dstp.sub(offset);
                // While the source and destination overlap, repeat the pattern with copies of 16
                // bytes that each move `dstp` forward by the distance between them.
                loop {
                    debug_assert!(dstp >= srcp);
                    let diff = (dstp as usize) - (srcp as usize);
                    if diff >= 16 {
                        break;
                    }
                    debug_assert!(self.d + 16 <= dest_len);
                    ptr::copy(srcp, dstp, 16);
                    self.d += diff;
                    dstp = dstp.add(diff);
                }
                while self.d < end {
                    ptr::copy_nonoverlapping(srcp, dstp, 16);
                    srcp = srcp.add(16);
                    dstp = dstp.add(16);
                    self.d += 16;
                }
            }
        } else {
            if end > self.dst.len() {
                return Err(corrupt("copy is out of bounds"));
            }
            // Byte by byte, for the last few bytes of the output.
            while self.d != end {
                self.dst[self.d] = self.dst[self.d - offset];
                self.d += 1;
            }
        }
        self.d = end;
        Ok(())
    }
}

/// The varint at the start of every Snappy compressed block.
struct Header {
    /// The length of the header in bytes.
    len: usize,
    /// The decompressed length.
    decompress_len: usize,
}

impl Header {
    #[inline(always)]
    fn read(input: &[u8]) -> ParquetResult<Header> {
        let (decompress_len, header_len) = read_varu64(input);
        // The decompressed length is a u32 varint, which takes at most 5 bytes.
        if header_len == 0 || header_len > 5 {
            return Err(corrupt("invalid header"));
        }
        if decompress_len > MAX_DECOMPRESSED_LEN {
            return Err(corrupt("decompressed length is too large"));
        }
        Ok(Header {
            len: header_len,
            decompress_len: decompress_len as usize,
        })
    }
}

/// An entry of [`TAG_LOOKUP_TABLE`].
struct TagEntry(usize);

impl TagEntry {
    /// The number of bytes after the tag byte that encode the copy offset.
    fn num_tag_bytes(&self) -> usize {
        self.0 >> 11
    }

    /// The copy length.
    fn len(&self) -> usize {
        self.0 & 0xFF
    }

    /// The copy offset. `s` points to the byte after the tag byte.
    fn offset(&self, src: &[u8], s: usize) -> ParquetResult<usize> {
        let num_tag_bytes = self.num_tag_bytes();
        let trailer = if s + 4 <= src.len() {
            // Read 4 bytes and mask out the ones that are not part of the offset, which saves a
            // branch on `num_tag_bytes`.
            read_u32_le(&src[s..]) as usize & WORD_MASK[num_tag_bytes]
        } else if num_tag_bytes == 1 {
            if s >= src.len() {
                return Err(corrupt("copy offset is missing"));
            }
            src[s] as usize
        } else if num_tag_bytes == 2 {
            if s + 1 >= src.len() {
                return Err(corrupt("copy offset is missing"));
            }
            u16::from_le_bytes([src[s], src[s + 1]]) as usize
        } else {
            return Err(corrupt("copy offset is missing"));
        };
        Ok((self.0 & 0b0000_0111_0000_0000) | trailer)
    }
}

#[inline(always)]
fn read_u32_le(slice: &[u8]) -> u32 {
    u32::from_le_bytes(slice[..4].try_into().unwrap())
}

/// Reads a varint as described in
/// <https://developers.google.com/protocol-buffers/docs/encoding#varints>. Returns the value and
/// the number of bytes read, or `(0, 0)` if the varint is invalid.
fn read_varu64(data: &[u8]) -> (u64, usize) {
    let mut n: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in data.iter().enumerate() {
        if b < 0b1000_0000 {
            return match (b as u64).checked_shl(shift) {
                None => (0, 0),
                Some(b) => (n | b, i + 1),
            };
        }
        match ((b as u64) & 0b0111_1111).checked_shl(shift) {
            None => return (0, 0),
            Some(b) => n |= b,
        }
        shift += 7;
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let compressed = snap::raw::Encoder::new().compress_vec(data).unwrap();
        let mut out = Vec::with_capacity(data.len());
        let n = decompress(&compressed, out.spare_capacity_mut()).unwrap();
        assert_eq!(n, data.len());
        unsafe { out.set_len(n) };
        assert_eq!(out, data);
    }

    #[test]
    fn test_roundtrip() {
        roundtrip(b"");
        roundtrip(b"a");
        roundtrip(b"abcabcabcabcabcabcabcabcabcabcabcabc");
        let mut data = Vec::new();
        let mut x: u64 = 1;
        for i in 0..200_000u64 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Mix runs, short repeats and noise to hit all copy paths.
            let byte = match (i / 1000) % 4 {
                0 => b'a',
                1 => (i % 3) as u8,
                2 => (i % 37) as u8,
                _ => (x >> 56) as u8,
            };
            data.push(byte);
            if i % 50_000 == 0 {
                roundtrip(&data);
            }
        }
        roundtrip(&data);
    }

    #[test]
    fn test_corrupt() {
        let data = b"abcabcabcabcabcabcabcabcabcabcabcabc".repeat(100);
        let compressed = snap::raw::Encoder::new().compress_vec(&data).unwrap();
        let mut out = Vec::<u8>::with_capacity(data.len());
        // Truncated input must fail, as it cannot initialize all output bytes.
        for cut in 1..compressed.len() {
            assert!(decompress(&compressed[..cut], out.spare_capacity_mut()).is_err());
        }
        assert!(decompress(&compressed, &mut out.spare_capacity_mut()[..data.len() - 1]).is_err());
    }

    /// Compares the result with `snap` on random data and on randomly changed compressed data.
    #[test]
    fn test_same_as_snap() {
        let mut x: u64 = 7;
        let mut next = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as usize
        };

        for _ in 0..2000 {
            let len = next() % 2000;
            let alphabet = 1 + next() % 255;
            let data: Vec<u8> = (0..len).map(|_| (next() % alphabet) as u8).collect();
            let mut input = snap::raw::Encoder::new().compress_vec(&data).unwrap();

            for _ in 0..next() % 4 {
                if input.is_empty() {
                    break;
                }
                let i = next() % input.len();
                match next() % 3 {
                    0 => input[i] = next() as u8,
                    1 => input.truncate(i),
                    _ => input.insert(i, next() as u8),
                }
            }

            let expected = match snap::raw::decompress_len(&input) {
                Ok(len) if len <= 1 << 16 => snap::raw::Decoder::new().decompress_vec(&input).ok(),
                _ => None,
            };

            let mut out = Vec::<u8>::with_capacity(expected.as_ref().map_or(1 << 16, Vec::len));
            let result = decompress(&input, out.spare_capacity_mut());
            match expected {
                Some(expected) => {
                    assert_eq!(result.unwrap(), expected.len());
                    unsafe { out.set_len(expected.len()) };
                    assert_eq!(out, expected);
                },
                None => assert!(result.is_err()),
            }
        }
    }
}
