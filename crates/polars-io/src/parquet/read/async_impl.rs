//! Read parquet files in parallel from the Object Store without a third party crate.

use arrow::datatypes::ArrowSchemaRef;
use object_store::path::Path as ObjectPath;
use polars_core::prelude::*;
use polars_parquet::write::FileMetadata;
use polars_utils::pl_path::PlRefPath;

use crate::cloud::{
    CloudLocation, CloudOptions, PolarsObjectStore, build_object_store, object_path_from_str,
};
use crate::parquet::metadata::FileMetadataRef;

pub struct ParquetObjectStore {
    store: PolarsObjectStore,
    path: ObjectPath,
    length: Option<usize>,
    footer_length: Option<usize>,
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
            footer_length: None,
            metadata,
            schema: None,
        })
    }

    /// Set the known length of the object.
    pub fn set_length(&mut self, length: usize) {
        self.length = Some(length);
    }

    /// Set the known footer byte length (Thrift metadata size).
    pub fn set_footer_length(&mut self, footer_length: usize) {
        self.footer_length = Some(footer_length);
    }

    /// Initialize the length property of the object, unless it is already available.
    async fn length(&mut self) -> PolarsResult<usize> {
        if self.length.is_none() {
            self.length = Some(self.store.head(&self.path).await?.size as usize);
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
        let (metadata, footer_byte_length) =
            fetch_metadata(&self.store, &self.path, length, self.footer_length).await?;
        self.footer_length = Some(footer_byte_length);
        Ok(metadata)
    }

    /// Fetch and memoize the metadata of the parquet file.
    pub async fn get_metadata(&mut self) -> PolarsResult<&FileMetadataRef> {
        if self.metadata.is_none() {
            self.metadata = Some(Arc::new(self.fetch_metadata().await?));
        }
        Ok(self.metadata.as_ref().unwrap())
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

/// Asynchronously reads the files' metadata
pub async fn fetch_metadata(
    store: &PolarsObjectStore,
    path: &ObjectPath,
    file_byte_length: usize,
    footer_byte_length_hint: Option<usize>,
) -> PolarsResult<(FileMetadata, usize)> {
    let footer_byte_length = if let Some(hint) = footer_byte_length_hint {
        hint
    } else {
        let footer_header_bytes = store
            .get_range(
                path,
                file_byte_length
                    .checked_sub(polars_parquet::parquet::FOOTER_SIZE as usize)
                    .ok_or_else(|| {
                        polars_parquet::parquet::error::ParquetError::OutOfSpec(
                            "not enough bytes to contain parquet footer".to_string(),
                        )
                    })?..file_byte_length,
            )
            .await?;

        let reader = &mut footer_header_bytes.as_ref();
        let footer_byte_size = read_i32le(reader).unwrap();
        let magic = read_n(reader).unwrap();
        debug_assert!(reader.is_empty());
        if magic != polars_parquet::parquet::PARQUET_MAGIC {
            return Err(polars_parquet::parquet::error::ParquetError::OutOfSpec(
                "incorrect magic in parquet footer".to_string(),
            )
            .into());
        }
        footer_byte_size.try_into().map_err(|_| {
            polars_parquet::parquet::error::ParquetError::OutOfSpec(
                "negative footer byte length".to_string(),
            )
        })?
    };

    let footer_bytes = store
        .get_range(
            path,
            file_byte_length
                .checked_sub(polars_parquet::parquet::FOOTER_SIZE as usize + footer_byte_length)
                .ok_or_else(|| {
                    polars_parquet::parquet::error::ParquetError::OutOfSpec(
                        "not enough bytes to contain parquet footer".to_string(),
                    )
                })?..file_byte_length,
        )
        .await?;

    let metadata = polars_parquet::parquet::read::deserialize_metadata(
        std::io::Cursor::new(footer_bytes.as_ref()),
        // TODO: Describe why this makes sense. Taken from the previous
        // implementation which said "a highly nested but sparse struct could
        // result in many allocations".
        footer_bytes.as_ref().len() * 2 + 1024,
    )?;
    Ok((metadata, footer_byte_length))
}
