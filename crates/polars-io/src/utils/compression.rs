use std::cmp;
use std::io::Read;

use polars_core::prelude::*;
use polars_error::{feature_gated, to_compute_err};
use polars_utils::mmap::{MemReader, MemSlice};

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

/// Reader that implements a streaming read trait for uncompressed, gzip, zlib and zstd
/// compression.
///
/// This allows handling decompression transparently in a streaming fashion.
pub enum CompressedReader {
    Uncompressed {
        slice: MemSlice,
        offset: usize,
    },
    #[cfg(feature = "decompress")]
    Gzip(flate2::bufread::MultiGzDecoder<MemReader>),
    #[cfg(feature = "decompress")]
    Zlib(flate2::bufread::ZlibDecoder<MemReader>),
    #[cfg(feature = "decompress")]
    Zstd(zstd::Decoder<'static, MemReader>),
}

impl CompressedReader {
    pub fn try_new(slice: MemSlice) -> PolarsResult<Self> {
        let algo = SupportedCompression::check(&slice);

        Ok(match algo {
            None => CompressedReader::Uncompressed { slice, offset: 0 },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::GZIP) => {
                CompressedReader::Gzip(flate2::bufread::MultiGzDecoder::new(MemReader::new(slice)))
            },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::ZLIB) => {
                CompressedReader::Zlib(flate2::bufread::ZlibDecoder::new(MemReader::new(slice)))
            },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::ZSTD) => {
                CompressedReader::Zstd(zstd::Decoder::with_buffer(MemReader::new(slice))?)
            },
            #[cfg(not(feature = "decompress"))]
            _ => panic!("activate 'decompress' feature"),
        })
    }

    pub fn is_compressed(&self) -> bool {
        !matches!(&self, CompressedReader::Uncompressed { .. })
    }

    /// If possible returns the total number of bytes that will be produced by reading from the
    /// start to finish.
    pub fn total_len_estimate(&self) -> usize {
        const ESTIMATED_DEFLATE_RATIO: usize = 3;
        const ESTIMATED_ZSTD_RATIO: usize = 5;

        match self {
            CompressedReader::Uncompressed { slice, .. } => slice.len(),
            #[cfg(feature = "decompress")]
            CompressedReader::Gzip(reader) => {
                reader.get_ref().total_len() * ESTIMATED_DEFLATE_RATIO
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Zlib(reader) => {
                reader.get_ref().total_len() * ESTIMATED_DEFLATE_RATIO
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Zstd(reader) => reader.get_ref().total_len() * ESTIMATED_ZSTD_RATIO,
        }
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

        let mut buf = Vec::new();
        if self.is_compressed() {
            let reserve_size = cmp::min(
                prev_len.saturating_add(read_size),
                self.total_len_estimate().saturating_mul(2),
            );
            buf.reserve_exact(reserve_size);
            buf.extend_from_slice(prev_leftover);
        }

        let new_slice_from_read =
            |read_n: usize, mut buf: Vec<u8>| -> std::io::Result<(MemSlice, usize)> {
                buf.truncate(prev_len + read_n);
                Ok((MemSlice::from_vec(buf), read_n))
            };

        match self {
            CompressedReader::Uncompressed { slice, offset, .. } => {
                let read_n = cmp::min(read_size, slice.len() - *offset);
                let new_slice = slice.slice((*offset - prev_len)..(*offset + read_n));
                *offset += read_n;
                Ok((new_slice, read_n))
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Gzip(decoder) => {
                new_slice_from_read(decoder.take(read_size as u64).read_to_end(&mut buf)?, buf)
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Zlib(decoder) => {
                new_slice_from_read(decoder.take(read_size as u64).read_to_end(&mut buf)?, buf)
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Zstd(decoder) => {
                new_slice_from_read(decoder.take(read_size as u64).read_to_end(&mut buf)?, buf)
            },
        }
    }
}
