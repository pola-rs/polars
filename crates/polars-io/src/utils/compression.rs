use std::cmp;
use std::io::{BufRead, Cursor, Read, Write};

use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_error::{feature_gated, to_compute_err};

use crate::utils::file::{Writeable, WriteableTrait};
#[cfg(feature = "async")]
use crate::utils::stream_buf_reader::ReaderSource;
use crate::utils::sync_on_close::SyncOnCloseType;

/// Represents the compression algorithms that we have decoders for
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
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
#[deprecated(note = "may cause OOM, use CompressedReader instead")]
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
        slice: Buffer<u8>,
        offset: usize,
    },
    #[cfg(feature = "decompress")]
    Gzip(flate2::bufread::MultiGzDecoder<Cursor<Buffer<u8>>>),
    #[cfg(feature = "decompress")]
    Zlib(flate2::bufread::ZlibDecoder<Cursor<Buffer<u8>>>),
    #[cfg(feature = "decompress")]
    Zstd(zstd::Decoder<'static, Cursor<Buffer<u8>>>),
}

impl CompressedReader {
    pub fn try_new(slice: Buffer<u8>) -> PolarsResult<Self> {
        let algo = SupportedCompression::check(&slice);

        Ok(match algo {
            None => CompressedReader::Uncompressed { slice, offset: 0 },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::GZIP) => {
                CompressedReader::Gzip(flate2::bufread::MultiGzDecoder::new(Cursor::new(slice)))
            },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::ZLIB) => {
                CompressedReader::Zlib(flate2::bufread::ZlibDecoder::new(Cursor::new(slice)))
            },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::ZSTD) => {
                CompressedReader::Zstd(zstd::Decoder::with_buffer(Cursor::new(slice))?)
            },
            #[cfg(not(feature = "decompress"))]
            _ => panic!("activate 'decompress' feature"),
        })
    }

    pub fn is_compressed(&self) -> bool {
        !matches!(&self, CompressedReader::Uncompressed { .. })
    }

    pub const fn initial_read_size() -> usize {
        // We don't want to read too much at the beginning to keep decompression to a minimum if for
        // example only the schema is needed or a slice op is used. Keep in sync with
        // `ideal_read_size` so that `initial_read_size * N * 4 == ideal_read_size`.
        32 * 1024
    }

    pub const fn ideal_read_size() -> usize {
        // Somewhat conservative guess for L2 size, which performs the best on most machines and is
        // nearly always core exclusive. The loss of going larger and accidentally hitting L3 is not
        // recouped by amortizing the block processing cost even further.
        //
        // It's possible that callers use or need a larger `read_size` if for example a single row
        // doesn't fit in the 512KB.
        512 * 1024
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
                reader.get_ref().get_ref().len() * ESTIMATED_DEFLATE_RATIO
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Zlib(reader) => {
                reader.get_ref().get_ref().len() * ESTIMATED_DEFLATE_RATIO
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Zstd(reader) => {
                reader.get_ref().get_ref().len() * ESTIMATED_ZSTD_RATIO
            },
        }
    }

    /// Reads exactly `read_size` bytes if possible from the internal readers and creates a new
    /// [`Buffer`] with the content `concat(prev_leftover, new_bytes)`.
    ///
    /// Returns the new slice and the number of bytes read, which will be 0 when eof is reached and
    /// this function is called again.
    ///
    /// If the underlying reader is uncompressed the operation is a cheap zero-copy
    /// [`Buffer::sliced`] operation.
    ///
    /// By handling slice concatenation at this level we can implement zero-copy reading *and* make
    /// the interface easier to use.
    ///
    /// It's a logic bug if `prev_leftover` is neither empty nor the last slice returned by this
    /// function.
    pub fn read_next_slice(
        &mut self,
        prev_leftover: &Buffer<u8>,
        read_size: usize,
    ) -> std::io::Result<(Buffer<u8>, usize)> {
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
            |bytes_read: usize, mut buf: Vec<u8>| -> std::io::Result<(Buffer<u8>, usize)> {
                buf.truncate(prev_len + bytes_read);
                Ok((Buffer::from_vec(buf), bytes_read))
            };

        match self {
            CompressedReader::Uncompressed { slice, offset, .. } => {
                let bytes_read = cmp::min(read_size, slice.len() - *offset);
                let new_slice = slice
                    .clone()
                    .sliced(*offset - prev_len..*offset + bytes_read);
                *offset += bytes_read;
                Ok((new_slice, bytes_read))
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

/// This implementation is meant for compatibility. Use [`Self::read_next_slice`] for best
/// performance.
impl Read for CompressedReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            CompressedReader::Uncompressed { slice, offset, .. } => {
                let bytes_read = cmp::min(buf.len(), slice.len() - *offset);
                buf[..bytes_read].copy_from_slice(&slice[*offset..(*offset + bytes_read)]);
                *offset += bytes_read;
                Ok(bytes_read)
            },
            #[cfg(feature = "decompress")]
            CompressedReader::Gzip(decoder) => decoder.read(buf),
            #[cfg(feature = "decompress")]
            CompressedReader::Zlib(decoder) => decoder.read(buf),
            #[cfg(feature = "decompress")]
            CompressedReader::Zstd(decoder) => decoder.read(buf),
        }
    }
}

/// A byte source that abstracts over in-memory buffers and streaming
/// readers, with optional transparent decompression and buffering.
///
/// Implements `BufRead`, allowing uniform access regardless of whether
/// the underlying data is an in-memory slice, a raw stream, or a
/// compressed stream (gzip/zlib/zstd).
///
/// This is the generic successor to [`CompressedReader`], which only
/// supports in-memory (`Buffer<u8>`) sources.
#[cfg(feature = "async")]
pub enum ByteSourceReader<R: BufRead> {
    UncompressedMemory {
        slice: Buffer<u8>,
        offset: usize,
    },
    UncompressedStream(R),
    #[cfg(feature = "decompress")]
    Gzip(flate2::bufread::MultiGzDecoder<R>),
    #[cfg(feature = "decompress")]
    Zlib(flate2::bufread::ZlibDecoder<R>),
    #[cfg(feature = "decompress")]
    Zstd(zstd::Decoder<'static, R>),
}

#[cfg(feature = "async")]
impl<R: BufRead> ByteSourceReader<R> {
    pub fn try_new(reader: R, compression: Option<SupportedCompression>) -> PolarsResult<Self> {
        Ok(match compression {
            None => Self::UncompressedStream(reader),
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::GZIP) => {
                Self::Gzip(flate2::bufread::MultiGzDecoder::new(reader))
            },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::ZLIB) => {
                Self::Zlib(flate2::bufread::ZlibDecoder::new(reader))
            },
            #[cfg(feature = "decompress")]
            Some(SupportedCompression::ZSTD) => Self::Zstd(zstd::Decoder::with_buffer(reader)?),
            #[cfg(not(feature = "decompress"))]
            _ => panic!("activate 'decompress' feature"),
        })
    }

    pub fn is_compressed(&self) -> bool {
        !matches!(
            &self,
            Self::UncompressedMemory { .. } | Self::UncompressedStream(_)
        )
    }

    pub const fn initial_read_size() -> usize {
        // We don't want to read too much at the beginning to keep decompression to a minimum if for
        // example only the schema is needed or a slice op is used. Keep in sync with
        // `ideal_read_size` so that `initial_read_size * N * 4 == ideal_read_size`.
        32 * 1024
    }

    pub const fn ideal_read_size() -> usize {
        // Somewhat conservative guess for L2 size, which performs the best on most machines and is
        // nearly always core exclusive. The loss of going larger and accidentally hitting L3 is not
        // recouped by amortizing the block processing cost even further.
        //
        // It's possible that callers use or need a larger `read_size` if for example a single row
        // doesn't fit in the 512KB.
        512 * 1024
    }

    /// Reads exactly `read_size` bytes if possible from the internal readers and creates a new
    /// [`Buffer`] with the content `concat(prev_leftover, new_bytes)`.
    ///
    /// Returns the new slice and the number of bytes read, which will be 0 when eof is reached and
    /// this function is called again.
    ///
    /// If the underlying reader is uncompressed the operation is a cheap zero-copy
    /// [`Buffer::sliced`] operation.
    ///
    /// By handling slice concatenation at this level we can implement zero-copy reading *and* make
    /// the interface easier to use.
    ///
    /// It's a logic bug if `prev_leftover` is neither empty nor the last slice returned by this
    /// function.
    pub fn read_next_slice(
        &mut self,
        prev_leftover: &Buffer<u8>,
        read_size: usize,
        uncompressed_size_hint: Option<usize>,
    ) -> std::io::Result<(Buffer<u8>, usize)> {
        // Assuming that callers of this function correctly handle re-trying, by continuously growing
        // prev_leftover if it doesn't contain a single row, this abstraction supports arbitrarily
        // sized rows.
        let prev_len = prev_leftover.len();

        let reader: &mut dyn Read = match self {
            // Zero-copy fast-path — no allocation required
            Self::UncompressedMemory { slice, offset } => {
                let bytes_read = cmp::min(read_size, slice.len() - *offset);
                let new_slice = slice
                    .clone()
                    .sliced(*offset - prev_len..*offset + bytes_read);
                *offset += bytes_read;
                return Ok((new_slice, bytes_read));
            },
            Self::UncompressedStream(reader) => reader,
            #[cfg(feature = "decompress")]
            Self::Gzip(reader) => reader,
            #[cfg(feature = "decompress")]
            Self::Zlib(reader) => reader,
            #[cfg(feature = "decompress")]
            Self::Zstd(reader) => reader,
        };

        let mut buf = Vec::new();

        // Cap the reserve_size, for the scenario where read_size == usize::MAX
        let max_reserve_size = uncompressed_size_hint.unwrap_or(4 * 1024 * 1024);
        let reserve_size = cmp::min(prev_len.saturating_add(read_size), max_reserve_size);
        buf.reserve_exact(reserve_size);
        buf.extend_from_slice(prev_leftover);

        let bytes_read = reader.take(read_size as u64).read_to_end(&mut buf)?;
        buf.truncate(prev_len + bytes_read);
        Ok((Buffer::from_vec(buf), bytes_read))
    }
}

#[cfg(feature = "async")]
impl ByteSourceReader<ReaderSource> {
    pub fn from_memory(
        slice: Buffer<u8>,
        compression: Option<SupportedCompression>,
    ) -> PolarsResult<Self> {
        match compression {
            None => Ok(Self::UncompressedMemory { slice, offset: 0 }),
            _ => Self::try_new(ReaderSource::Memory(Cursor::new(slice)), compression),
        }
    }
}

/// Constructor for `WriteableTrait` compressed encoders.
pub enum CompressedWriter {
    #[cfg(feature = "decompress")]
    Gzip(Option<flate2::write::GzEncoder<Writeable>>),
    #[cfg(feature = "decompress")]
    Zstd(Option<zstd::Encoder<'static, Writeable>>),
}

impl CompressedWriter {
    pub fn gzip(writer: Writeable, level: Option<u32>) -> Self {
        feature_gated!("decompress", {
            Self::Gzip(Some(flate2::write::GzEncoder::new(
                writer,
                level.map(flate2::Compression::new).unwrap_or_default(),
            )))
        })
    }

    pub fn zstd(writer: Writeable, level: Option<u32>) -> std::io::Result<Self> {
        feature_gated!("decompress", {
            zstd::Encoder::new(writer, level.unwrap_or(3) as i32)
                .map(Some)
                .map(Self::Zstd)
        })
    }
}

impl Write for CompressedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        feature_gated!("decompress", {
            match self {
                Self::Gzip(encoder) => encoder.as_mut().unwrap().write(buf),
                Self::Zstd(encoder) => encoder.as_mut().unwrap().write(buf),
            }
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        feature_gated!("decompress", {
            match self {
                Self::Gzip(encoder) => encoder.as_mut().unwrap().flush(),
                Self::Zstd(encoder) => encoder.as_mut().unwrap().flush(),
            }
        })
    }
}

impl WriteableTrait for CompressedWriter {
    fn close(&mut self) -> std::io::Result<()> {
        feature_gated!("decompress", {
            let writer = match self {
                Self::Gzip(encoder) => encoder.take().unwrap().finish()?,
                Self::Zstd(encoder) => encoder.take().unwrap().finish()?,
            };

            writer.close(SyncOnCloseType::All)
        })
    }

    fn sync_all(&self) -> std::io::Result<()> {
        feature_gated!("decompress", {
            match self {
                Self::Gzip(encoder) => encoder.as_ref().unwrap().get_ref().sync_all(),
                Self::Zstd(encoder) => encoder.as_ref().unwrap().get_ref().sync_all(),
            }
        })
    }

    fn sync_data(&self) -> std::io::Result<()> {
        feature_gated!("decompress", {
            match self {
                Self::Gzip(encoder) => encoder.as_ref().unwrap().get_ref().sync_data(),
                Self::Zstd(encoder) => encoder.as_ref().unwrap().get_ref().sync_data(),
            }
        })
    }
}
