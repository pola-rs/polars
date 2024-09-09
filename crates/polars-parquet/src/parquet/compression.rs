//! Functionality to compress and decompress data according to the parquet specification
pub use super::parquet_bridge::{
    BrotliLevel, Compression, CompressionOptions, GzipLevel, ZstdLevel,
};
use crate::parquet::error::{ParquetError, ParquetResult};

#[cfg(any(feature = "snappy", feature = "lz4"))]
fn inner_compress<
    G: Fn(usize) -> ParquetResult<usize>,
    F: Fn(&[u8], &mut [u8]) -> ParquetResult<usize>,
>(
    input: &[u8],
    output: &mut Vec<u8>,
    get_length: G,
    compress: F,
) -> ParquetResult<()> {
    let original_length = output.len();
    let max_required_length = get_length(input.len())?;

    output.resize(original_length + max_required_length, 0);
    let compressed_size = compress(input, &mut output[original_length..])?;

    output.truncate(original_length + compressed_size);
    Ok(())
}

/// Compresses data stored in slice `input_buf` and writes the compressed result
/// to `output_buf`.
///
/// Note that you'll need to call `clear()` before reusing the same `output_buf`
/// across different `compress` calls.
#[allow(unused_variables)]
pub fn compress(
    compression: CompressionOptions,
    input_buf: &[u8],
    #[allow(clippy::ptr_arg)] output_buf: &mut Vec<u8>,
) -> ParquetResult<()> {
    match compression {
        #[cfg(feature = "brotli")]
        CompressionOptions::Brotli(level) => {
            use std::io::Write;
            const BROTLI_DEFAULT_BUFFER_SIZE: usize = 4096;
            const BROTLI_DEFAULT_LG_WINDOW_SIZE: u32 = 22; // recommended between 20-22

            let q = level.unwrap_or_default();
            let mut encoder = brotli::CompressorWriter::new(
                output_buf,
                BROTLI_DEFAULT_BUFFER_SIZE,
                q.compression_level(),
                BROTLI_DEFAULT_LG_WINDOW_SIZE,
            );
            encoder.write_all(input_buf)?;
            encoder.flush().map_err(|e| e.into())
        },
        #[cfg(not(feature = "brotli"))]
        CompressionOptions::Brotli(_) => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Brotli,
            "compress to brotli".to_string(),
        )),
        #[cfg(feature = "gzip")]
        CompressionOptions::Gzip(level) => {
            use std::io::Write;
            let level = level.unwrap_or_default();
            let mut encoder = flate2::write::GzEncoder::new(output_buf, level.into());
            encoder.write_all(input_buf)?;
            encoder.try_finish().map_err(|e| e.into())
        },
        #[cfg(not(feature = "gzip"))]
        CompressionOptions::Gzip(_) => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Gzip,
            "compress to gzip".to_string(),
        )),
        #[cfg(feature = "snappy")]
        CompressionOptions::Snappy => inner_compress(
            input_buf,
            output_buf,
            |len| Ok(snap::raw::max_compress_len(len)),
            |input, output| Ok(snap::raw::Encoder::new().compress(input, output)?),
        ),
        #[cfg(not(feature = "snappy"))]
        CompressionOptions::Snappy => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Snappy,
            "compress to snappy".to_string(),
        )),
        #[cfg(feature = "lz4")]
        CompressionOptions::Lz4Raw => inner_compress(
            input_buf,
            output_buf,
            |len| Ok(lz4::block::compress_bound(len)?),
            |input, output| {
                let compressed_size = lz4::block::compress_to_buffer(input, None, false, output)?;
                Ok(compressed_size)
            },
        ),
        #[cfg(all(not(feature = "lz4"), not(feature = "lz4_flex")))]
        CompressionOptions::Lz4Raw => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Lz4,
            "compress to lz4".to_string(),
        )),
        #[cfg(feature = "zstd")]
        CompressionOptions::Zstd(level) => {
            let level = level.map(|v| v.compression_level()).unwrap_or_default();
            // Make sure the buffer is large enough; the interface assumption is
            // that decompressed data is appended to the output buffer.
            let old_len = output_buf.len();
            output_buf.resize(
                old_len + zstd::zstd_safe::compress_bound(input_buf.len()),
                0,
            );
            match zstd::bulk::compress_to_buffer(input_buf, &mut output_buf[old_len..], level) {
                Ok(written_size) => {
                    output_buf.truncate(old_len + written_size);
                    Ok(())
                },
                Err(e) => Err(e.into()),
            }
        },
        #[cfg(not(feature = "zstd"))]
        CompressionOptions::Zstd(_) => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Zstd,
            "compress to zstd".to_string(),
        )),
        CompressionOptions::Uncompressed => Err(ParquetError::InvalidParameter(
            "Compressing uncompressed".to_string(),
        )),
        _ => Err(ParquetError::FeatureNotSupported(format!(
            "Compression {:?} is not supported",
            compression,
        ))),
    }
}

/// Decompresses data stored in slice `input_buf` and writes output to `output_buf`.
/// Returns the total number of bytes written.
#[allow(unused_variables)]
pub fn decompress(
    compression: Compression,
    input_buf: &[u8],
    output_buf: &mut [u8],
) -> ParquetResult<()> {
    match compression {
        #[cfg(feature = "brotli")]
        Compression::Brotli => {
            use std::io::Read;
            const BROTLI_DEFAULT_BUFFER_SIZE: usize = 4096;
            brotli::Decompressor::new(input_buf, BROTLI_DEFAULT_BUFFER_SIZE)
                .read_exact(output_buf)
                .map_err(|e| e.into())
        },
        #[cfg(not(feature = "brotli"))]
        Compression::Brotli => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Brotli,
            "decompress with brotli".to_string(),
        )),
        #[cfg(feature = "gzip")]
        Compression::Gzip => {
            use std::io::Read;
            let mut decoder = flate2::read::GzDecoder::new(input_buf);
            decoder.read_exact(output_buf).map_err(|e| e.into())
        },
        #[cfg(not(feature = "gzip"))]
        Compression::Gzip => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Gzip,
            "decompress with gzip".to_string(),
        )),
        #[cfg(feature = "snappy")]
        Compression::Snappy => {
            use snap::raw::{decompress_len, Decoder};

            let len = decompress_len(input_buf)?;
            if len > output_buf.len() {
                return Err(ParquetError::oos("snappy header out of spec"));
            }
            Decoder::new()
                .decompress(input_buf, output_buf)
                .map_err(|e| e.into())
                .map(|_| ())
        },
        #[cfg(not(feature = "snappy"))]
        Compression::Snappy => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Snappy,
            "decompress with snappy".to_string(),
        )),
        #[cfg(all(feature = "lz4_flex", not(feature = "lz4")))]
        Compression::Lz4Raw => lz4_flex::block::decompress_into(input_buf, output_buf)
            .map(|_| {})
            .map_err(|e| e.into()),
        #[cfg(feature = "lz4")]
        Compression::Lz4Raw => {
            lz4::block::decompress_to_buffer(input_buf, Some(output_buf.len() as i32), output_buf)
                .map(|_| {})
                .map_err(|e| e.into())
        },
        #[cfg(all(not(feature = "lz4"), not(feature = "lz4_flex")))]
        Compression::Lz4Raw => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Lz4,
            "decompress with lz4".to_string(),
        )),

        #[cfg(any(feature = "lz4_flex", feature = "lz4"))]
        Compression::Lz4 => try_decompress_hadoop(input_buf, output_buf).or_else(|_| {
            lz4_decompress_to_buffer(input_buf, Some(output_buf.len() as i32), output_buf)
                .map(|_| {})
        }),

        #[cfg(all(not(feature = "lz4_flex"), not(feature = "lz4")))]
        Compression::Lz4 => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Lz4,
            "decompress with legacy lz4".to_string(),
        )),

        #[cfg(feature = "zstd")]
        Compression::Zstd => {
            use std::io::Read;
            let mut decoder = zstd::Decoder::new(input_buf)?;
            decoder.read_exact(output_buf).map_err(|e| e.into())
        },
        #[cfg(not(feature = "zstd"))]
        Compression::Zstd => Err(ParquetError::FeatureNotActive(
            crate::parquet::error::Feature::Zstd,
            "decompress with zstd".to_string(),
        )),
        Compression::Uncompressed => Err(ParquetError::InvalidParameter(
            "Compressing uncompressed".to_string(),
        )),
        _ => Err(ParquetError::FeatureNotSupported(format!(
            "Compression {:?} is not supported",
            compression,
        ))),
    }
}

/// Try to decompress the buffer as if it was compressed with the Hadoop Lz4Codec.
/// Translated from the apache arrow c++ function [TryDecompressHadoop](https://github.com/apache/arrow/blob/bf18e6e4b5bb6180706b1ba0d597a65a4ce5ca48/cpp/src/arrow/util/compression_lz4.cc#L474).
/// Returns error if decompression failed.
#[cfg(any(feature = "lz4", feature = "lz4_flex"))]
fn try_decompress_hadoop(input_buf: &[u8], output_buf: &mut [u8]) -> ParquetResult<()> {
    // Parquet files written with the Hadoop Lz4Codec use their own framing.
    // The input buffer can contain an arbitrary number of "frames", each
    // with the following structure:
    // - bytes 0..3: big-endian uint32_t representing the frame decompressed size
    // - bytes 4..7: big-endian uint32_t representing the frame compressed size
    // - bytes 8...: frame compressed data
    //
    // The Hadoop Lz4Codec source code can be found here:
    // https://github.com/apache/hadoop/blob/trunk/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-nativetask/src/main/native/src/codec/Lz4Codec.cc

    const SIZE_U32: usize = std::mem::size_of::<u32>();
    const PREFIX_LEN: usize = SIZE_U32 * 2;
    let mut input_len = input_buf.len();
    let mut input = input_buf;
    let mut output_len = output_buf.len();
    let mut output: &mut [u8] = output_buf;
    while input_len >= PREFIX_LEN {
        let mut bytes = [0; SIZE_U32];
        bytes.copy_from_slice(&input[0..4]);
        let expected_decompressed_size = u32::from_be_bytes(bytes);
        let mut bytes = [0; SIZE_U32];
        bytes.copy_from_slice(&input[4..8]);
        let expected_compressed_size = u32::from_be_bytes(bytes);
        input = &input[PREFIX_LEN..];
        input_len -= PREFIX_LEN;

        if input_len < expected_compressed_size as usize {
            return Err(ParquetError::oos("Not enough bytes for Hadoop frame"));
        }

        if output_len < expected_decompressed_size as usize {
            return Err(ParquetError::oos(
                "Not enough bytes to hold advertised output",
            ));
        }
        let decompressed_size = lz4_decompress_to_buffer(
            &input[..expected_compressed_size as usize],
            Some(output_len as i32),
            output,
        )?;
        if decompressed_size != expected_decompressed_size as usize {
            return Err(ParquetError::oos("unexpected decompressed size"));
        }
        input_len -= expected_compressed_size as usize;
        output_len -= expected_decompressed_size as usize;
        if input_len > expected_compressed_size as usize {
            input = &input[expected_compressed_size as usize..];
            output = &mut output[expected_decompressed_size as usize..];
        } else {
            break;
        }
    }
    if input_len == 0 {
        Ok(())
    } else {
        Err(ParquetError::oos("Not all input are consumed"))
    }
}

#[cfg(feature = "lz4")]
#[inline]
fn lz4_decompress_to_buffer(
    src: &[u8],
    uncompressed_size: Option<i32>,
    buffer: &mut [u8],
) -> ParquetResult<usize> {
    let size = lz4::block::decompress_to_buffer(src, uncompressed_size, buffer)?;
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_roundtrip(c: CompressionOptions, data: &[u8]) {
        let offset = 2048;

        // Compress to a buffer that already has data is possible
        let mut compressed = vec![2; offset];
        compress(c, data, &mut compressed).expect("Error when compressing");

        // data is compressed...
        assert!(compressed.len() - offset < data.len());

        let mut decompressed = vec![0; data.len()];
        decompress(c.into(), &compressed[offset..], &mut decompressed)
            .expect("Error when decompressing");
        assert_eq!(data, decompressed.as_slice());
    }

    fn test_codec(c: CompressionOptions) {
        let sizes = vec![1000, 10000, 100000];
        for size in sizes {
            let data = (0..size).map(|x| (x % 255) as u8).collect::<Vec<_>>();
            test_roundtrip(c, &data);
        }
    }

    #[test]
    fn test_codec_snappy() {
        test_codec(CompressionOptions::Snappy);
    }

    #[test]
    fn test_codec_gzip_default() {
        test_codec(CompressionOptions::Gzip(None));
    }

    #[test]
    fn test_codec_gzip_low_compression() {
        test_codec(CompressionOptions::Gzip(Some(
            GzipLevel::try_new(1).unwrap(),
        )));
    }

    #[test]
    fn test_codec_brotli_default() {
        test_codec(CompressionOptions::Brotli(None));
    }

    #[test]
    fn test_codec_brotli_low_compression() {
        test_codec(CompressionOptions::Brotli(Some(
            BrotliLevel::try_new(1).unwrap(),
        )));
    }

    #[test]
    fn test_codec_brotli_high_compression() {
        test_codec(CompressionOptions::Brotli(Some(
            BrotliLevel::try_new(11).unwrap(),
        )));
    }

    #[test]
    fn test_codec_lz4_raw() {
        test_codec(CompressionOptions::Lz4Raw);
    }

    #[test]
    fn test_codec_zstd_default() {
        test_codec(CompressionOptions::Zstd(None));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_codec_zstd_low_compression() {
        test_codec(CompressionOptions::Zstd(Some(
            ZstdLevel::try_new(1).unwrap(),
        )));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_codec_zstd_high_compression() {
        test_codec(CompressionOptions::Zstd(Some(
            ZstdLevel::try_new(21).unwrap(),
        )));
    }
}
