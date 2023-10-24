//! Read parquet files in parallel from the Object Store without a third party crate.
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow::io::parquet::read::{self as parquet2_read, RowGroupMetaData};
use arrow::io::parquet::write::FileMetaData;
use bytes::Bytes;
use futures::future::try_join_all;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use polars_core::config::verbose;
use polars_core::datatypes::PlHashMap;
use polars_core::error::{to_compute_err, PolarsResult};
use polars_core::prelude::*;
use polars_core::schema::Schema;
use smartstring::alias::String as SmartString;

use super::cloud::{build_object_store, CloudLocation, CloudReader};
use super::mmap;
use super::mmap::ColumnStore;
use crate::cloud::CloudOptions;
use crate::predicates::PhysicalIoExpr;
use crate::prelude::predicates::read_this_row_group;

pub struct ParquetObjectStore {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    length: Option<u64>,
    metadata: Option<Arc<FileMetaData>>,
}

impl ParquetObjectStore {
    pub async fn from_uri(
        uri: &str,
        options: Option<&CloudOptions>,
        metadata: Option<Arc<FileMetaData>>,
    ) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, store) = build_object_store(uri, options).await?;

        Ok(ParquetObjectStore {
            store,
            path: ObjectPath::from_url_path(prefix).map_err(to_compute_err)?,
            length: None,
            metadata,
        })
    }

    /// Initialize the length property of the object, unless it has already been fetched.
    async fn initialize_length(&mut self) -> PolarsResult<()> {
        if self.length.is_some() {
            return Ok(());
        }
        self.length = Some(
            self.store
                .head(&self.path)
                .await
                .map_err(to_compute_err)?
                .size as u64,
        );
        Ok(())
    }

    pub async fn schema(&mut self) -> PolarsResult<Schema> {
        let metadata = self.get_metadata().await?;

        let arrow_schema = parquet2_read::infer_schema(metadata)?;

        Ok(Schema::from_iter(&arrow_schema.fields))
    }

    /// Number of rows in the parquet file.
    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata().await?;
        Ok(metadata.num_rows)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    async fn fetch_metadata(&mut self) -> PolarsResult<FileMetaData> {
        self.initialize_length().await?;
        let object_store = self.store.clone();
        let path = self.path.clone();
        let length = self.length;
        let mut reader = CloudReader::new(length, object_store, path);

        parquet2_read::read_metadata_async(&mut reader)
            .await
            .map_err(to_compute_err)
    }

    /// Fetch and memoize the metadata of the parquet file.
    pub async fn get_metadata(&mut self) -> PolarsResult<&Arc<FileMetaData>> {
        if self.metadata.is_none() {
            self.metadata = Some(Arc::new(self.fetch_metadata().await?));
        }
        Ok(self.metadata.as_ref().unwrap())
    }
}

async fn read_single_column_async(
    async_reader: &ParquetObjectStore,
    start: usize,
    length: usize,
) -> PolarsResult<(u64, Bytes)> {
    let chunk = async_reader
        .store
        .get_range(&async_reader.path, start..start + length)
        .await
        .map_err(to_compute_err)?;
    Ok((start as u64, chunk))
}

async fn read_columns_async(
    async_reader: &ParquetObjectStore,
    ranges: &[(u64, u64)],
) -> PolarsResult<Vec<(u64, Bytes)>> {
    let futures = ranges.iter().map(|(start, length)| async {
        read_single_column_async(async_reader, *start as usize, *length as usize).await
    });

    try_join_all(futures).await
}

/// Download rowgroups for the column whose indexes are given in `projection`.
/// We concurrently download the columns for each field.
async fn download_projection(
    fields: &[SmartString],
    row_groups: &[RowGroupMetaData],
    async_reader: &Arc<ParquetObjectStore>,
) -> PolarsResult<Vec<Vec<(u64, Bytes)>>> {
    // Build the cartesian product of the fields and the row groups.
    let product_futures = fields
        .iter()
        .flat_map(|name| row_groups.iter().map(move |r| (name.clone(), r)))
        .map(|(name, row_group)| async move {
            let columns = row_group.columns();
            let ranges = columns
                .iter()
                .filter_map(|meta| {
                    if meta.descriptor().path_in_schema[0] == name.as_str() {
                        Some(meta.byte_range())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let async_reader = async_reader.clone();
            let handle =
                tokio::spawn(async move { read_columns_async(&async_reader, &ranges).await });
            handle.await.unwrap()
        });

    // Download concurrently
    futures::future::try_join_all(product_futures).await
}

pub struct FetchRowGroupsFromObjectStore {
    reader: Arc<ParquetObjectStore>,
    row_groups_metadata: Vec<RowGroupMetaData>,
    projected_fields: Vec<SmartString>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    schema: SchemaRef,
    logging: bool,
}

impl FetchRowGroupsFromObjectStore {
    pub fn new(
        reader: ParquetObjectStore,
        metadata: &FileMetaData,
        schema: SchemaRef,
        projection: Option<&[usize]>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
    ) -> PolarsResult<Self> {
        let logging = verbose();

        let projected_fields = projection
            .map(|projection| {
                projection
                    .iter()
                    .map(|i| schema.get_at_index(*i).unwrap().0.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| schema.iter().map(|tpl| tpl.0).cloned().collect());

        Ok(FetchRowGroupsFromObjectStore {
            reader: Arc::new(reader),
            row_groups_metadata: metadata.row_groups.to_owned(),
            projected_fields,
            predicate,
            schema,
            logging,
        })
    }

    pub(crate) async fn fetch_row_groups(
        &mut self,
        row_groups: Range<usize>,
    ) -> PolarsResult<ColumnStore> {
        if row_groups.start == row_groups.end {
            return Ok(ColumnStore::Fetched(Default::default()));
        }
        // Fetch the required row groups.
        let row_groups = self
            .row_groups_metadata
            .get(row_groups.clone())
            .map_or_else(
                || Err(polars_err!(
                    ComputeError: "cannot access slice {0}..{1}", row_groups.start, row_groups.end,
                )),
                Ok,
            )?;

        let row_groups = if let Some(pred) = self.predicate.as_deref() {
            Cow::Owned(
                row_groups
                    .iter()
                    .filter(|rg| {
                        matches!(read_this_row_group(Some(pred), rg, &self.schema), Ok(true))
                    })
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        } else {
            Cow::Borrowed(row_groups)
        };

        // Package in the format required by ColumnStore.
        let downloaded =
            download_projection(&self.projected_fields, &row_groups, &self.reader).await?;

        if self.logging {
            eprintln!(
                "BatchedParquetReader: fetched {} row_groups for {} fields, yielding {} column chunks.",
                row_groups.len(),
                self.projected_fields.len(),
                downloaded.len(),
            );
        }
        let downloaded_per_filepos = downloaded
            .into_iter()
            .flat_map(|rg| rg.into_iter())
            .collect::<PlHashMap<_, _>>();

        if self.logging {
            eprintln!(
                "BatchedParquetReader: column chunks start & len: {}.",
                downloaded_per_filepos
                    .iter()
                    .map(|(pos, data)| format!("({}, {})", pos, data.len()))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }

        Ok(mmap::ColumnStore::Fetched(downloaded_per_filepos))
    }
}
