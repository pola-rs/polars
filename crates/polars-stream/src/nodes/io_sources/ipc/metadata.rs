use arrow::io::ipc::read::OutOfSpecKind;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};

use crate::metrics::OptIOMetrics;

/// Read the metadata bytes of a parquet file, does not decode the bytes. If during metadata fetch
/// the bytes of the entire file are loaded, it is returned in the second return value.
pub async fn read_ipc_metadata_bytes(
    byte_source: &DynByteSource,
    verbose: bool,
    io_metrics: &OptIOMetrics,
) -> PolarsResult<(Buffer<u8>, Option<Buffer<u8>>)> {
    const FOOTER_HEADER_SIZE: usize = 10;
    const ARROW_MAGIC_V1: [u8; 4] = [b'F', b'E', b'A', b'1'];
    const ARROW_MAGIC_V2: [u8; 6] = [b'A', b'R', b'R', b'O', b'W', b'1'];

    let file_size = io_metrics
        .record_download(1, byte_source.get_size())
        .await?;

    polars_ensure!(
        file_size >= FOOTER_HEADER_SIZE,
        ComputeError: "ipc file size is smaller than the minimum"
    );

    let estimated_metadata_size = if let DynByteSource::Buffer(_) = byte_source {
        // Mmapped or in-memory, reads are free.
        file_size
    } else {
        (file_size / 2048).clamp(16_384, 131_072).min(file_size)
    };

    let range = (file_size - estimated_metadata_size)..file_size;
    let fut = byte_source.get_range(range.clone());
    let bytes = match byte_source {
        DynByteSource::Buffer(_) => fut.await?,
        DynByteSource::Cloud(_) => io_metrics.record_download(range.len() as u64, fut).await?,
    };

    let footer_header_bytes = bytes.clone().sliced((bytes.len() - FOOTER_HEADER_SIZE)..);

    if footer_header_bytes[4..] != ARROW_MAGIC_V2 {
        if footer_header_bytes[..4] == ARROW_MAGIC_V1 {
            polars_bail!(ComputeError: "feather v1 not supported");
        }
        return Err(polars_err!(oos = OutOfSpecKind::InvalidFooter));
    }

    let footer_size = u32::from_le_bytes(footer_header_bytes[..4].try_into().unwrap());
    let footer_size = footer_size as usize + FOOTER_HEADER_SIZE;

    polars_ensure!(
        file_size >= footer_size,
        ComputeError:
        "file size ({file_size}) is less than the indicated footer size ({footer_size})",
    );

    if bytes.len() < footer_size {
        debug_assert!(!matches!(byte_source, DynByteSource::Buffer(_)));
        if verbose {
            eprintln!(
                "[IpcFileReader]: Extra {} bytes need to be fetched for metadata \
                (initial estimate = {}, actual size = {})",
                footer_size - estimated_metadata_size,
                bytes.len(),
                footer_size,
            );
        }

        let mut out = Vec::with_capacity(footer_size);
        let offset = file_size - footer_size;
        let len = footer_size - bytes.len();

        let range = offset..(offset + len);
        let fut = byte_source.get_range(range.clone());

        let delta_bytes = match byte_source {
            DynByteSource::Buffer(_) => fut.await?,
            DynByteSource::Cloud(_) => io_metrics.record_download(range.len() as u64, fut).await?,
        };

        debug_assert!(out.capacity() >= delta_bytes.len() + bytes.len());

        out.extend_from_slice(&delta_bytes);
        out.extend_from_slice(&bytes);

        Ok((Buffer::from_vec(out), None))
    } else {
        if verbose && !matches!(byte_source, DynByteSource::Buffer(_)) {
            eprintln!(
                "[IpcFileReader]: Fetched all bytes for metadata on first try \
                (initial estimate = {}, actual size = {}, excess = {}, total file size = {})",
                bytes.len(),
                footer_size,
                estimated_metadata_size - footer_size,
                file_size,
            );
        }

        let metadata_bytes = bytes.clone().sliced((bytes.len() - footer_size)..);

        if bytes.len() == file_size {
            Ok((metadata_bytes, Some(bytes)))
        } else {
            debug_assert!(!matches!(byte_source, DynByteSource::Buffer(_)));
            let metadata_bytes = if bytes.len() - footer_size >= bytes.len() {
                // Re-allocate to drop the excess bytes
                Buffer::from_vec(metadata_bytes.to_vec())
            } else {
                metadata_bytes
            };

            Ok((metadata_bytes, None))
        }
    }
}
