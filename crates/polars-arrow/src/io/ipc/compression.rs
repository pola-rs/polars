use polars_error::PolarsResult;
#[cfg(feature = "io_ipc_compression")]
use polars_error::{polars_ensure, to_compute_err};
use polars_utils::compression::ZstdLevel;

/// Decompresses `input_buf`, appending the first `len` decompressed bytes to `output_buf`.
#[cfg(feature = "io_ipc_compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc_compression")))]
pub fn decompress_lz4(input_buf: &[u8], output_buf: &mut Vec<u8>, len: usize) -> PolarsResult<()> {
    use std::io::Read;
    let start = output_buf.len();
    output_buf.try_reserve_exact(len)?;
    lz4::Decoder::new(input_buf)?
        .take(len as u64)
        .read_to_end(output_buf)?;
    polars_ensure!(
        output_buf.len() - start == len,
        ComputeError: "lz4 data decompresses to fewer than {len} bytes"
    );
    Ok(())
}

/// Decompresses `input_buf`, appending the first `len` decompressed bytes to `output_buf`.
#[cfg(feature = "io_ipc_compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc_compression")))]
pub fn decompress_zstd(input_buf: &[u8], output_buf: &mut Vec<u8>, len: usize) -> PolarsResult<()> {
    polars_utils::compression::zstd::decompress_prefix(input_buf, output_buf, len)
}

#[cfg(not(feature = "io_ipc_compression"))]
pub fn decompress_lz4(
    _input_buf: &[u8],
    _output_buf: &mut Vec<u8>,
    _len: usize,
) -> PolarsResult<()> {
    panic!(
        "The crate was compiled without IPC compression. Use `io_ipc_compression` to read compressed IPC."
    );
}

#[cfg(not(feature = "io_ipc_compression"))]
pub fn decompress_zstd(
    _input_buf: &[u8],
    _output_buf: &mut Vec<u8>,
    _len: usize,
) -> PolarsResult<()> {
    panic!(
        "The crate was compiled without IPC compression. Use `io_ipc_compression` to read compressed IPC."
    );
}

/// Compresses `input_buf`, appending the result to `output_buf`.
#[cfg(feature = "io_ipc_compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc_compression")))]
pub fn compress_lz4(input_buf: &[u8], output_buf: &mut Vec<u8>) -> PolarsResult<()> {
    use std::io::Write;

    let mut encoder = lz4::EncoderBuilder::new()
        .build(output_buf)
        .map_err(to_compute_err)?;
    encoder.write_all(input_buf)?;
    encoder.finish().1.map_err(|e| e.into())
}

/// Compresses `input_buf`, appending the result to `output_buf`.
#[cfg(feature = "io_ipc_compression")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc_compression")))]
pub fn compress_zstd(
    input_buf: &[u8],
    output_buf: &mut Vec<u8>,
    level: ZstdLevel,
) -> PolarsResult<()> {
    polars_utils::compression::zstd::compress(input_buf, output_buf, level.compression_level())
}

#[cfg(not(feature = "io_ipc_compression"))]
pub fn compress_lz4(_input_buf: &[u8], _output_buf: &mut Vec<u8>) -> PolarsResult<()> {
    panic!(
        "The crate was compiled without IPC compression. Use `io_ipc_compression` to write compressed IPC."
    )
}

#[cfg(not(feature = "io_ipc_compression"))]
pub fn compress_zstd(
    _input_buf: &[u8],
    _output_buf: &mut Vec<u8>,
    _level: ZstdLevel,
) -> PolarsResult<()> {
    panic!(
        "The crate was compiled without IPC compression. Use `io_ipc_compression` to write compressed IPC."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "io_ipc_compression")]
    #[test]
    #[cfg_attr(miri, ignore)] // ZSTD uses foreign calls that miri does not support
    fn round_trip_zstd() {
        let data: Vec<u8> = (0..200u8).map(|x| x % 10).collect();
        let mut buffer = vec![];
        compress_zstd(&data, &mut buffer, Default::default()).unwrap();

        let mut result = vec![];
        decompress_zstd(&buffer, &mut result, 200).unwrap();
        assert_eq!(data, result);
    }

    #[cfg(feature = "io_ipc_compression")]
    #[test]
    #[cfg_attr(miri, ignore)] // LZ4 uses foreign calls that miri does not support
    fn round_trip_lz4() {
        let data: Vec<u8> = (0..200u8).map(|x| x % 10).collect();
        let mut buffer = vec![];
        compress_lz4(&data, &mut buffer).unwrap();

        let mut result = vec![];
        decompress_lz4(&buffer, &mut result, 200).unwrap();
        assert_eq!(data, result);
    }
}
