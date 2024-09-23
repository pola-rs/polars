use polars_core::prelude::{ArrowSchema, DataType};
use polars_error::{polars_bail, PolarsResult};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_utils::mmap::MemSlice;

/// Read the metadata bytes of a parquet file, does not decode the bytes. If during metadata fetch
/// the bytes of the entire file are loaded, it is returned in the second return value.
pub(super) async fn read_parquet_metadata_bytes(
    byte_source: &DynByteSource,
    verbose: bool,
) -> PolarsResult<(MemSlice, Option<MemSlice>)> {
    use polars_parquet::parquet::error::ParquetError;
    use polars_parquet::parquet::PARQUET_MAGIC;

    const FOOTER_HEADER_SIZE: usize = polars_parquet::parquet::FOOTER_SIZE as usize;

    let file_size = byte_source.get_size().await?;

    if file_size < FOOTER_HEADER_SIZE {
        return Err(ParquetError::OutOfSpec(format!(
            "file size ({}) is less than minimum size required to store parquet footer ({})",
            file_size, FOOTER_HEADER_SIZE
        ))
        .into());
    }

    let estimated_metadata_size = if let DynByteSource::MemSlice(_) = byte_source {
        // Mmapped or in-memory, reads are free.
        file_size
    } else {
        (file_size / 2048).clamp(16_384, 131_072).min(file_size)
    };

    let bytes = byte_source
        .get_range((file_size - estimated_metadata_size)..file_size)
        .await?;

    let footer_header_bytes = bytes.slice((bytes.len() - FOOTER_HEADER_SIZE)..bytes.len());

    let (v, remaining) = footer_header_bytes.split_at(4);
    let footer_size = i32::from_le_bytes(v.try_into().unwrap());

    if remaining != PARQUET_MAGIC {
        return Err(ParquetError::OutOfSpec(format!(
            r#"expected parquet magic bytes "{}" in footer, got "{}" instead"#,
            std::str::from_utf8(&PARQUET_MAGIC).unwrap(),
            String::from_utf8_lossy(remaining)
        ))
        .into());
    }

    if footer_size < 0 {
        return Err(ParquetError::OutOfSpec(format!(
            "expected positive footer size, got {} instead",
            footer_size
        ))
        .into());
    }

    let footer_size = footer_size as usize + FOOTER_HEADER_SIZE;

    if file_size < footer_size {
        return Err(ParquetError::OutOfSpec(format!(
            "file size ({}) is less than the indicated footer size ({})",
            file_size, footer_size
        ))
        .into());
    }

    if bytes.len() < footer_size {
        debug_assert!(!matches!(byte_source, DynByteSource::MemSlice(_)));
        if verbose {
            eprintln!(
                "[ParquetSource]: Extra {} bytes need to be fetched for metadata \
            (initial estimate = {}, actual size = {})",
                footer_size - estimated_metadata_size,
                bytes.len(),
                footer_size,
            );
        }

        let mut out = Vec::with_capacity(footer_size);
        let offset = file_size - footer_size;
        let len = footer_size - bytes.len();
        let delta_bytes = byte_source.get_range(offset..(offset + len)).await?;

        debug_assert!(out.capacity() >= delta_bytes.len() + bytes.len());

        out.extend_from_slice(&delta_bytes);
        out.extend_from_slice(&bytes);

        Ok((MemSlice::from_vec(out), None))
    } else {
        if verbose && !matches!(byte_source, DynByteSource::MemSlice(_)) {
            eprintln!(
                "[ParquetSource]: Fetched all bytes for metadata on first try \
                (initial estimate = {}, actual size = {}, excess = {})",
                bytes.len(),
                footer_size,
                estimated_metadata_size - footer_size,
            );
        }

        let metadata_bytes = bytes.slice((bytes.len() - footer_size)..bytes.len());

        if bytes.len() == file_size {
            Ok((metadata_bytes, Some(bytes)))
        } else {
            debug_assert!(!matches!(byte_source, DynByteSource::MemSlice(_)));
            let metadata_bytes = if bytes.len() - footer_size >= bytes.len() {
                // Re-allocate to drop the excess bytes
                MemSlice::from_vec(metadata_bytes.to_vec())
            } else {
                metadata_bytes
            };

            Ok((metadata_bytes, None))
        }
    }
}

/// Ensures that a parquet file has all the necessary columns for a projection with the correct
/// dtype. There are no ordering requirements and extra columns are permitted.
pub(super) fn ensure_schema_has_projected_fields(
    schema: &ArrowSchema,
    projected_fields: &ArrowSchema,
) -> PolarsResult<()> {
    for field in projected_fields.iter_values() {
        // Note: We convert to Polars-native dtypes for timezone normalization.
        let expected_dtype = DataType::from_arrow(&field.dtype, true);
        let dtype = {
            let Some(field) = schema.get(&field.name) else {
                polars_bail!(SchemaMismatch: "did not find column: {}", field.name)
            };
            DataType::from_arrow(&field.dtype, true)
        };

        if dtype != expected_dtype {
            polars_bail!(SchemaMismatch: "data type mismatch for column {}: found: {}, expected: {}",
                &field.name, dtype, expected_dtype
            )
        }
    }

    Ok(())
}
