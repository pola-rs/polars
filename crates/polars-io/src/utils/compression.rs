use std::io::Read;

use polars_core::prelude::*;
use polars_error::{feature_gated, to_compute_err};

/// Represents the compression algorithms that we have decoders for
pub enum SupportedCompression {
    GZIP,
    ZLIB,
    ZSTD,
}

impl SupportedCompression {
    /// If the given byte slice starts with the "magic" bytes for a supported compression family, return
    /// that family, for unsupported/uncompressed slices, return None
    pub fn check(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            // not enough bytes to perform prefix checks
            return None;
        }
        match bytes[..4] {
            [31, 139, _, _]          => Some(Self::GZIP),
            [0x78, 0x01, _, _] | // ZLIB0
            [0x78, 0x9C, _, _] | // ZLIB1
            [0x78, 0xDA, _, _]   // ZLIB2
                                     => Some(Self::ZLIB),
            [0x28, 0xB5, 0x2F, 0xFD] => Some(Self::ZSTD),
            _ => None,
        }
    }
}

/// Decompress `bytes` if compression is detected, otherwise simply return it.
/// An `out` vec must be given for ownership of the decompressed data.
pub fn maybe_decompress_bytes<'a>(bytes: &'a [u8], out: &'a mut Vec<u8>) -> PolarsResult<&'a [u8]> {
    assert!(out.is_empty());

    if let Some(algo) = SupportedCompression::check(bytes) {
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
    } else {
        Ok(bytes)
    }
}
