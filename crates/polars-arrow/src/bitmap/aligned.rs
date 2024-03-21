use std::iter::Copied;
use std::slice::Iter;

use crate::bitmap::utils::BitChunk;

fn load_chunk_le<T: BitChunk>(src: &[u8]) -> T {
    if let Ok(chunk) = src.try_into() {
        return T::from_le_bytes(chunk);
    }

    let mut chunk = T::Bytes::default();
    let len = src.len().min(chunk.as_ref().len());
    chunk.as_mut()[..len].copy_from_slice(&src[..len]);
    T::from_le_bytes(chunk)
}

/// Represents a bitmap split in three portions, a prefix, a suffix and an
/// aligned bulk section in the middle.
#[derive(Default, Clone, Debug)]
pub struct AlignedBitmapSlice<'a, T: BitChunk> {
    prefix: T,
    prefix_len: u32,
    bulk: &'a [T],
    suffix: T,
    suffix_len: u32,
}

impl<'a, T: BitChunk> AlignedBitmapSlice<'a, T> {
    #[inline(always)]
    pub fn prefix(&self) -> T {
        self.prefix
    }

    #[inline(always)]
    pub fn bulk_iter(&self) -> Copied<Iter<'a, T>> {
        self.bulk.iter().copied()
    }

    #[inline(always)]
    pub fn bulk(&self) -> &'a [T] {
        self.bulk
    }

    #[inline(always)]
    pub fn suffix(&self) -> T {
        self.suffix
    }

    /// The length (in bits) of the portion of the bitmap found in prefix.
    #[inline(always)]
    pub fn prefix_bitlen(&self) -> usize {
        self.prefix_len as usize
    }

    /// The length (in bits) of the portion of the bitmap found in bulk.
    #[inline(always)]
    pub fn bulk_bitlen(&self) -> usize {
        8 * std::mem::size_of::<T>() * self.bulk.len()
    }

    /// The length (in bits) of the portion of the bitmap found in suffix.
    #[inline(always)]
    pub fn suffix_bitlen(&self) -> usize {
        self.suffix_len as usize
    }

    pub fn new(mut bytes: &'a [u8], mut offset: usize, len: usize) -> Self {
        if len == 0 {
            return Self::default();
        }

        assert!(bytes.len() * 8 >= offset + len);

        // Strip off useless bytes from start.
        let start_byte_idx = offset / 8;
        bytes = &bytes[start_byte_idx..];
        offset %= 8;

        // Fast-path: fits entirely in one chunk.
        let chunk_len = std::mem::size_of::<T>();
        let chunk_len_bits = 8 * chunk_len;
        if offset + len <= chunk_len_bits {
            let mut prefix = load_chunk_le::<T>(bytes) >> offset;
            if len < chunk_len_bits {
                prefix &= (T::one() << len) - T::one();
            }
            return Self {
                prefix,
                prefix_len: len as u32,
                ..Self::default()
            };
        }

        // Find how many bytes from the start our aligned section would start.
        let mut align_offset = bytes.as_ptr().align_offset(chunk_len);
        let mut align_offset_bits = 8 * align_offset;

        // Oops, the original pointer was already aligned, but our offset means
        // we can't start there, start one chunk later.
        if offset > align_offset_bits {
            align_offset_bits += chunk_len_bits;
            align_offset += chunk_len;
        }

        // Calculate based on this the lengths of our sections (in bits).
        let prefix_len = (align_offset_bits - offset).min(len);
        let rest_len = len - prefix_len;
        let suffix_len = rest_len % chunk_len_bits;
        let bulk_len = rest_len - suffix_len;
        debug_assert!(prefix_len < chunk_len_bits);
        debug_assert!(bulk_len % chunk_len_bits == 0);
        debug_assert!(suffix_len < chunk_len_bits);

        // Now we just have to load.
        let (prefix_bytes, rest_bytes) = bytes.split_at(align_offset);
        let (bulk_bytes, suffix_bytes) = rest_bytes.split_at(bulk_len / 8);
        let mut prefix = load_chunk_le::<T>(prefix_bytes) >> offset;
        let mut suffix = load_chunk_le::<T>(suffix_bytes);
        prefix &= (T::one() << prefix_len) - T::one();
        suffix &= (T::one() << suffix_len) - T::one();
        Self {
            prefix,
            bulk: bytemuck::cast_slice(bulk_bytes),
            suffix,
            prefix_len: prefix_len as u32,
            suffix_len: suffix_len as u32,
        }
    }
}
