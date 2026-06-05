//! Read parquet files in parallel from the Object Store without a third party crate.

use arrow::datatypes::ArrowSchemaRef;
use object_store::path::Path as ObjectPath;
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_parquet::parquet::error::ParquetError;
use polars_parquet::parquet::read::{deserialize_metadata, deserialize_num_rows};
use polars_parquet::parquet::{DEFAULT_FOOTER_READ_SIZE, FOOTER_SIZE, PARQUET_MAGIC};
use polars_parquet::write::FileMetadata;
use polars_utils::pl_path::PlRefPath;

use crate::cloud::{
    CloudLocation, CloudOptions, ConcurrencyStrategy, FetchConfig, PolarsObjectStore,
    build_object_store, object_path_from_str,
};
use crate::parquet::metadata::FileMetadataRef;

pub struct ParquetObjectStore {
    store: PolarsObjectStore,
    path: ObjectPath,
    length: Option<usize>,
    metadata: Option<FileMetadataRef>,
    schema: Option<ArrowSchemaRef>,
}

impl ParquetObjectStore {
    pub async fn from_uri(
        uri: PlRefPath,
        options: Option<&CloudOptions>,
        metadata: Option<FileMetadataRef>,
    ) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, store) = build_object_store(uri, options, false).await?;
        let path = object_path_from_str(&prefix)?;

        Ok(ParquetObjectStore {
            store,
            path,
            length: None,
            metadata,
            schema: None,
        })
    }

    /// Initialize the length property of the object, unless it has already been fetched.
    async fn length(&mut self) -> PolarsResult<usize> {
        if self.length.is_none() {
            self.length = Some(
                self.store
                    .head(&self.path, ConcurrencyStrategy::BytesBased)
                    .await?
                    .size as usize,
            );
        }
        Ok(self.length.unwrap())
    }

    /// Number of rows in the parquet file.
    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata().await?;
        Ok(metadata.num_rows)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    async fn fetch_metadata(&mut self) -> PolarsResult<FileMetadata> {
        let length = self.length().await?;
        fetch_metadata(&self.store, &self.path, length).await
    }

    /// Fetch and memoize the metadata of the parquet file.
    pub async fn get_metadata(&mut self) -> PolarsResult<&FileMetadataRef> {
        if self.metadata.is_none() {
            self.metadata = Some(Arc::new(self.fetch_metadata().await?));
        }
        Ok(self.metadata.as_ref().unwrap())
    }

    /// Decode only `FileMetaData.num_rows` from the remote footer.
    /// Not memoized. Used by `RowCounts` resolve mode.
    pub async fn num_rows_only(&mut self) -> PolarsResult<i64> {
        let length = self.length().await?;
        fetch_num_rows(&self.store, &self.path, length).await
    }

    pub async fn schema(&mut self) -> PolarsResult<ArrowSchemaRef> {
        self.schema = Some(match self.schema.as_ref() {
            Some(schema) => Arc::clone(schema),
            None => {
                let metadata = self.get_metadata().await?;
                let arrow_schema = polars_parquet::arrow::read::infer_schema(metadata)?;
                Arc::new(arrow_schema)
            },
        });

        Ok(self.schema.clone().unwrap())
    }
}

fn read_n<const N: usize>(reader: &mut &[u8]) -> Option<[u8; N]> {
    if N <= reader.len() {
        let (head, tail) = reader.split_at(N);
        *reader = tail;
        Some(head.try_into().unwrap())
    } else {
        None
    }
}

fn read_i32le(reader: &mut &[u8]) -> Option<i32> {
    read_n(reader).map(i32::from_le_bytes)
}

/// Speculatively read `DEFAULT_FOOTER_READ_SIZE` from the tail. If the
/// footer fits in the prefetch (the common case), we're done in one range
/// request; otherwise re-fetch the full footer. Mirrors the sync
/// `fetch_footer_buf` strategy.
async fn fetch_footer_bytes(
    store: &PolarsObjectStore,
    path: &ObjectPath,
    file_byte_length: usize,
) -> PolarsResult<Buffer<u8>> {
    let out_of_spec = |msg: &str| ParquetError::OutOfSpec(msg.to_string());

    let prefetch_len = std::cmp::min(DEFAULT_FOOTER_READ_SIZE as usize, file_byte_length);
    let prefetched = store
        .get_range(
            path,
            file_byte_length
                .checked_sub(prefetch_len)
                .ok_or_else(|| out_of_spec("not enough bytes to contain parquet footer"))?
                ..file_byte_length,
            FetchConfig::random_access(),
        )
        .await?;

    if prefetched.len() < FOOTER_SIZE as usize {
        return Err(out_of_spec("not enough bytes to contain parquet footer").into());
    }

    // Trailing 8 bytes: footer size (i32 LE) + magic.
    let footer_byte_length: usize = {
        let tail_start = prefetched.len() - FOOTER_SIZE as usize;
        let reader = &mut &prefetched.as_ref()[tail_start..];
        let footer_byte_size = read_i32le(reader).unwrap();
        let magic = read_n(reader).unwrap();
        debug_assert!(reader.is_empty());
        if magic != PARQUET_MAGIC {
            return Err(out_of_spec("incorrect magic in parquet footer").into());
        }
        footer_byte_size
            .try_into()
            .map_err(|_| out_of_spec("negative footer byte length"))?
    };

    let footer_len = FOOTER_SIZE as usize + footer_byte_length;
    if footer_len <= prefetched.len() {
        // Common case: footer already in the prefetch; zero extra round trips.
        let start = prefetched.len() - footer_len;
        Ok(prefetched.sliced(start..))
    } else {
        // Fallback: footer larger than the prefetch; re-fetch the full footer.
        store
            .get_range(
                path,
                file_byte_length
                    .checked_sub(footer_len)
                    .ok_or_else(|| out_of_spec("not enough bytes to contain parquet footer"))?
                    ..file_byte_length,
                FetchConfig::random_access(),
            )
            .await
    }
}

/// Asynchronously reads the files' metadata.
pub async fn fetch_metadata(
    store: &PolarsObjectStore,
    path: &ObjectPath,
    file_byte_length: usize,
) -> PolarsResult<FileMetadata> {
    let footer = fetch_footer_bytes(store, path, file_byte_length).await?;
    Ok(deserialize_metadata(footer)?)
}

/// Fetch only `FileMetaData.num_rows` from a remote parquet footer.
pub async fn fetch_num_rows(
    store: &PolarsObjectStore,
    path: &ObjectPath,
    file_byte_length: usize,
) -> PolarsResult<i64> {
    let footer = fetch_footer_bytes(store, path, file_byte_length).await?;
    Ok(deserialize_num_rows(footer)?)
}

// //kdn TODO REFACTOR
// {
//     let footer_header_bytes = store
//         .get_range(
//             path,
//             file_byte_length
//                 .checked_sub(polars_parquet::parquet::FOOTER_SIZE as usize)
//                 .ok_or_else(|| {
//                     polars_parquet::parquet::error::ParquetError::OutOfSpec(
//                         "not enough bytes to contain parquet footer".to_string(),
//                     )
//                 })?..file_byte_length,
//             FetchConfig::random_access(),
//         )
//         .await?;

//     let footer_byte_length: usize = {
//         let reader = &mut footer_header_bytes.as_ref();
//         let footer_byte_size = read_i32le(reader).unwrap();
//         let magic = read_n(reader).unwrap();
//         debug_assert!(reader.is_empty());
//         if magic != polars_parquet::parquet::PARQUET_MAGIC {
//             return Err(polars_parquet::parquet::error::ParquetError::OutOfSpec(
//                 "incorrect magic in parquet footer".to_string(),
//             )
//             .into());
//         }
//         footer_byte_size.try_into().map_err(|_| {
//             polars_parquet::parquet::error::ParquetError::OutOfSpec(
//                 "negative footer byte length".to_string(),
//             )
//         })?
//     };

//     let footer_bytes = store
//         .get_range(
//             path,
//             file_byte_length
//                 .checked_sub(polars_parquet::parquet::FOOTER_SIZE as usize + footer_byte_length)
//                 .ok_or_else(|| {
//                     polars_parquet::parquet::error::ParquetError::OutOfSpec(
//                         "not enough bytes to contain parquet footer".to_string(),
//                     )
//                 })?..file_byte_length,
//             FetchConfig::random_access(),
//         )
//         .await?;

//     Ok(polars_parquet::parquet::read::deserialize_metadata(
//         footer_bytes,
//     )?)
// }
