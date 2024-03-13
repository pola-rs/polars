use std::ops::Range;
use std::sync::Arc;

use arrow::io::ipc::read::{get_row_count, FileMetadata, OutOfSpecKind};
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use polars_core::datatypes::IDX_DTYPE;
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_error::{polars_bail, polars_err, to_compute_err, PolarsResult};

use crate::cloud::{build_object_store, CloudLocation, CloudOptions};
use crate::pl_async::{
    tune_with_concurrency_budget, with_concurrency_budget, MAX_BUDGET_PER_REQUEST,
};
use crate::predicates::PhysicalIoExpr;
use crate::prelude::{materialize_projection, IpcReader};
use crate::RowIndex;

/// Polars specific wrapper for `Arc<dyn ObjectStore>` that limits the number of
/// concurrent requests for the entire application.
#[derive(Debug, Clone)]
pub struct PolarsObjectStore(Arc<dyn ObjectStore>);

impl PolarsObjectStore {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self(store)
    }

    pub async fn get(&self, path: &Path) -> PolarsResult<Bytes> {
        tune_with_concurrency_budget(1, || async {
            self.0
                .get(path)
                .await
                .map_err(to_compute_err)?
                .bytes()
                .await
                .map_err(to_compute_err)
        })
        .await
    }

    pub async fn get_range(&self, path: &Path, range: Range<usize>) -> PolarsResult<Bytes> {
        tune_with_concurrency_budget(1, || self.0.get_range(path, range))
            .await
            .map_err(to_compute_err)
    }

    pub async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<usize>],
    ) -> PolarsResult<Vec<Bytes>> {
        tune_with_concurrency_budget(
            (ranges.len() as u32).clamp(0, MAX_BUDGET_PER_REQUEST as u32),
            || self.0.get_ranges(path, ranges),
        )
        .await
        .map_err(to_compute_err)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn head(&self, path: &Path) -> PolarsResult<ObjectMeta> {
        with_concurrency_budget(1, || self.0.head(path))
            .await
            .map_err(to_compute_err)
    }
}

/// An Arrow IPC reader implemented on top of PolarsObjectStore.
pub struct IpcReaderAsync {
    store: PolarsObjectStore,
    path: Path,
}

#[derive(Default, Clone)]
pub struct IpcReadOptions {
    // Names of the columns to include in the output.
    projection: Option<Vec<String>>,

    // The maximum number of rows to include in the output.
    row_limit: Option<usize>,

    // Include a column with the row number under the provided name  starting at the provided index.
    row_index: Option<RowIndex>,

    // Only include rows that pass this predicate.
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl IpcReadOptions {
    pub fn with_projection(mut self, indices: impl Into<Option<Vec<String>>>) -> Self {
        self.projection = indices.into();
        self
    }

    pub fn with_row_limit(mut self, row_limit: impl Into<Option<usize>>) -> Self {
        self.row_limit = row_limit.into();
        self
    }

    pub fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.row_index = row_index.into();
        self
    }

    pub fn with_predicate(mut self, predicate: impl Into<Option<Arc<dyn PhysicalIoExpr>>>) -> Self {
        self.predicate = predicate.into();
        self
    }
}

impl IpcReaderAsync {
    pub async fn from_uri(
        uri: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<IpcReaderAsync> {
        let (
            CloudLocation {
                prefix, expansion, ..
            },
            store,
        ) = build_object_store(uri, cloud_options).await?;

        let path = {
            // Any wildcards should already have been resolved here. Without this assertion they would
            // be ignored.
            debug_assert!(expansion.is_none(), "path should not contain wildcards");
            Path::from_url_path(prefix).map_err(to_compute_err)?
        };

        Ok(Self {
            store: PolarsObjectStore::new(store),
            path,
        })
    }

    async fn object_metadata(&self) -> PolarsResult<ObjectMeta> {
        self.store.head(&self.path).await
    }

    async fn file_size(&self) -> PolarsResult<usize> {
        Ok(self.object_metadata().await?.size)
    }

    pub async fn metadata(&self) -> PolarsResult<FileMetadata> {
        let file_size = self.file_size().await?;

        // TODO: Do a larger request and hope that the entire footer is contained within it to save one round-trip.
        let footer_metadata =
            self.store
                .get_range(
                    &self.path,
                    file_size.checked_sub(FOOTER_METADATA_SIZE).ok_or_else(|| {
                        to_compute_err("ipc file size is smaller than the minimum")
                    })?..file_size,
                )
                .await?;

        let footer_size = deserialize_footer_metadata(
            footer_metadata
                .as_ref()
                .try_into()
                .map_err(to_compute_err)?,
        )?;

        let footer = self
            .store
            .get_range(
                &self.path,
                file_size
                    .checked_sub(FOOTER_METADATA_SIZE + footer_size)
                    .ok_or_else(|| {
                        to_compute_err("invalid ipc footer metadata: footer size too large")
                    })?..file_size,
            )
            .await?;

        arrow::io::ipc::read::deserialize_footer(
            footer.as_ref(),
            footer_size.try_into().map_err(to_compute_err)?,
        )
    }

    pub async fn data(
        &self,
        metadata: Option<&FileMetadata>,
        options: IpcReadOptions,
        verbose: bool,
    ) -> PolarsResult<DataFrame> {
        // TODO: Only download what is needed rather than the entire file by
        // making use of the projection, row limit, predicate and such.
        let bytes = self.store.get(&self.path).await?;

        let projection = match options.projection.as_deref() {
            Some(projection) => {
                fn prepare_schema(mut schema: Schema, row_index: Option<&RowIndex>) -> Schema {
                    if let Some(rc) = row_index {
                        let _ = schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
                    }
                    schema
                }

                // Retrieve the metadata for the schema so we can map column names to indices.
                let fetched_metadata;
                let metadata = if let Some(metadata) = metadata {
                    metadata
                } else {
                    // This branch is  happens when _metadata is None, which can happen if we Deserialize the execution plan.
                    fetched_metadata = self.metadata().await?;
                    &fetched_metadata
                };

                let schema = prepare_schema((&metadata.schema).into(), options.row_index.as_ref());

                let hive_partitions = None;

                materialize_projection(
                    Some(projection),
                    &schema,
                    hive_partitions,
                    options.row_index.is_some(),
                )
            },
            None => None,
        };

        let reader =
            <IpcReader<_> as crate::SerReader<_>>::new(std::io::Cursor::new(bytes.as_ref()))
                .with_row_index(options.row_index)
                .with_n_rows(options.row_limit)
                .with_projection(projection);
        reader.finish_with_scan_ops(options.predicate, verbose)
    }

    pub async fn count_rows(&self, _metadata: Option<&FileMetadata>) -> PolarsResult<i64> {
        // TODO: Only download what is needed rather than the entire file by
        // making use of the projection, row limit, predicate and such.
        let bytes = self.store.get(&self.path).await?;
        get_row_count(&mut std::io::Cursor::new(bytes.as_ref()))
    }
}

const FOOTER_METADATA_SIZE: usize = 10;

// TODO: Move to polars-arrow and deduplicate parsing of footer metadata in
// sync and async readers.
fn deserialize_footer_metadata(bytes: [u8; FOOTER_METADATA_SIZE]) -> PolarsResult<usize> {
    let footer_size: usize =
        i32::from_le_bytes(bytes[0..4].try_into().unwrap_or_else(|_| unreachable!()))
            .try_into()
            .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    if &bytes[4..] != b"ARROW1" {
        polars_bail!(oos = OutOfSpecKind::InvalidFooter);
    }

    Ok(footer_size)
}
