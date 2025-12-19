use std::cmp;
use std::io::Read;

use polars_core::prelude::*;
use polars_error::{feature_gated, to_compute_err};
use polars_utils::mmap::MemSlice;
use self_cell::self_cell;

/// Represents the compression algorithms that we have decoders for
pub enum SupportedCompression {
    GZIP,
    ZLIB,
    ZSTD,
}

impl SupportedCompression {
    /// If the given byte slice starts with the "magic" bytes for a supported compression family, return
    /// that family, for unsupported/uncompressed slices, return None.
    /// Based on <https://en.wikipedia.org/wiki/List_of_file_signatures>.
    pub fn check(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            // not enough bytes to perform prefix checks
            return None;
        }
        match bytes[..4] {
            [0x1f, 0x8b, _, _] => Some(Self::GZIP),
            // Different zlib compression levels without preset dictionary.
            [0x78, 0x01, _, _] => Some(Self::ZLIB),
            [0x78, 0x5e, _, _] => Some(Self::ZLIB),
            [0x78, 0x9c, _, _] => Some(Self::ZLIB),
            [0x78, 0xda, _, _] => Some(Self::ZLIB),
            [0x28, 0xb5, 0x2f, 0xfd] => Some(Self::ZSTD),
            _ => None,
        }
    }
}

/// Decompress `bytes` if compression is detected, otherwise simply return it.
/// An `out` vec must be given for ownership of the decompressed data.
#[allow(clippy::ptr_arg)]
pub fn maybe_decompress_bytes<'a>(bytes: &'a [u8], out: &'a mut Vec<u8>) -> PolarsResult<&'a [u8]> {
    assert!(out.is_empty());

    let Some(algo) = SupportedCompression::check(bytes) else {
        return Ok(bytes);
    };

    feature_gated!("decompress", {
        match algo {
            SupportedCompression::GZIP => {
                flate2::read::MultiGzDecoder::new(bytes)
                    .read_to_end(out)
                    .map_err(to_compute_err)?;
            },
            SupportedCompression::ZLIB => {
                flate2::read::ZlibDecoder::new(bytes)
                    .read_to_end(out)
                    .map_err(to_compute_err)?;
            },
            SupportedCompression::ZSTD => {
                zstd::Decoder::with_buffer(bytes)?.read_to_end(out)?;
            },
        }

        Ok(out)
    })
}

enum CompressedReaderDecoder<'a> {
    Uncompressed { slice: MemSlice, offset: usize },
    Gzip(flate2::bufread::MultiGzDecoder<&'a [u8]>),
    Zlib(flate2::bufread::ZlibDecoder<&'a [u8]>),
    Zstd(zstd::Decoder<'a, &'a [u8]>),
}

impl<'a> CompressedReaderDecoder<'a> {
    fn try_new(slice: &MemSlice, bytes: &'a [u8]) -> PolarsResult<Self> {
        let algo = SupportedCompression::check(bytes);

        Ok(match algo {
            None => CompressedReaderDecoder::Uncompressed {
                slice: slice.clone(),
                offset: 0,
            },
            Some(SupportedCompression::GZIP) => {
                CompressedReaderDecoder::Gzip(flate2::bufread::MultiGzDecoder::new(bytes))
            },
            Some(SupportedCompression::ZLIB) => {
                CompressedReaderDecoder::Zlib(flate2::bufread::ZlibDecoder::new(bytes))
            },
            Some(SupportedCompression::ZSTD) => {
                CompressedReaderDecoder::Zstd(zstd::Decoder::with_buffer(bytes)?)
            },
        })
    }
}

self_cell!(
    struct OwnedCompressedReaderDecoder {
        owner: MemSlice,
        #[covariant]
        dependent: CompressedReaderDecoder,
    }
);

/// Reader that implements a streaming read trait for uncompressed, gzip, zlib and zstd
/// compression.
///
/// This allows handling decompression transparently in a streaming fashion.
pub struct CompressedReader {
    decoder: OwnedCompressedReaderDecoder,
}

impl CompressedReader {
    pub fn try_new(mem_slice: MemSlice) -> PolarsResult<Self> {
        let decoder = OwnedCompressedReaderDecoder::try_new(mem_slice, |mem_slice| {
            CompressedReaderDecoder::try_new(mem_slice, mem_slice)
        })?;

        Ok(Self { decoder })
    }

    pub fn is_compressed(&self) -> bool {
        !matches!(
            &self.decoder.borrow_dependent(),
            CompressedReaderDecoder::Uncompressed { .. }
        )
    }

    /// Reads exactly `read_size` bytes if possible from the internal readers and creates a new
    /// [`MemSlice`] with the content `concat(prev_leftover, new_bytes)`.
    ///
    /// Returns the new slice and the number of bytes read, which will be 0 when eof is reached and
    /// this function is called again.
    ///
    /// If the underlying reader is uncompressed the operation is a cheap zero-copy
    /// [`MemSlice::slice`] operation.
    ///
    /// By handling slice concatenation at this level we can implement zero-copy reading *and* make
    /// the interface easier to use.
    ///
    /// It's a logic bug if `prev_leftover` is neither empty nor the last slice returned by this
    /// function.
    pub fn read_next_slice(
        &mut self,
        prev_leftover: &MemSlice,
        read_size: usize,
    ) -> std::io::Result<(MemSlice, usize)> {
        // Assuming that callers of this function correctly handle re-trying, by continuously growing
        // prev_leftover if it doesn't contain a single row, this abstraction supports arbitrarily
        // sized rows.
        let prev_len = prev_leftover.len();

        debug_assert!(read_size.is_power_of_two());

        let mut buf = Vec::new();
        if self.is_compressed() {
            buf.reserve_exact(prev_len + read_size);
            buf.extend_from_slice(prev_leftover);
        }

        let new_slice_from_read = |read_n: usize, mut buf: Vec<u8>| {
            buf.truncate(prev_len + read_n);
            Ok((MemSlice::from_vec(buf), read_n))
        };

        self.decoder.with_dependent_mut(|_, decoder| match decoder {
            CompressedReaderDecoder::Uncompressed { slice, offset } => {
                let read_n = cmp::min(read_size, slice.len() - *offset);
                let new_slice = slice.slice((*offset - prev_len)..(*offset + read_n));
                *offset += read_n;
                Ok((new_slice, read_n))
            },
            CompressedReaderDecoder::Gzip(decoder) => {
                new_slice_from_read(decoder.take(read_size as u64).read_to_end(&mut buf)?, buf)
            },
            CompressedReaderDecoder::Zlib(decoder) => {
                new_slice_from_read(decoder.take(read_size as u64).read_to_end(&mut buf)?, buf)
            },
            CompressedReaderDecoder::Zstd(decoder) => {
                new_slice_from_read(decoder.take(read_size as u64).read_to_end(&mut buf)?, buf)
            },
        })
    }
}
