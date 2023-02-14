//! Read parquet files in parallel from the Object Store without a third party crate.
use std::ops::Range;
use std::sync::Arc;

use arrow::io::parquet::read::{
    self as parquet2_read, read_columns_async, ColumnChunkMetaData, RowGroupMetaData,
};
use arrow::io::parquet::write::FileMetaData;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use futures::{stream, StreamExt, TryFutureExt, TryStreamExt};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use polars_core::cloud::CloudOptions;
use polars_core::config::verbose;
use polars_core::datatypes::PlHashMap;
use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_core::schema::Schema;

use super::cloud::{build, CloudLocation, CloudReader};
use super::mmap;
use super::mmap::ColumnStore;
use super::read_impl::FetchRowGroups;

pub struct ParquetObjectStore {
    store: Arc<Mutex<Box<dyn ObjectStore>>>,
    path: ObjectPath,
    length: Option<u64>,
    metadata: Option<FileMetaData>,
}

impl ParquetObjectStore {
    pub fn from_uri(uri: &str, options: Option<&CloudOptions>) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, store) = build(uri, options)?;
        let store = Arc::new(Mutex::from(store));

        Ok(ParquetObjectStore {
            store,
            path: prefix.into(),
            length: None,
            metadata: None,
        })
    }

    /// Initialize the length property of the object, unless it has already been fetched.
    async fn initialize_length(&mut self) -> PolarsResult<()> {
        if self.length.is_some() {
            return Ok(());
        }
        let path = self.path.clone();
        let locked_store = self.store.lock().await;
        self.length = Some({
            locked_store
                .head(&path)
                .await
                .map_err(anyhow::Error::from)?
                .size as u64
        });
        Ok(())
    }

    pub async fn schema(&mut self) -> PolarsResult<Schema> {
        let metadata = self.get_metadata().await?;

        let arrow_schema = parquet2_read::infer_schema(metadata).map_err(anyhow::Error::from)?;

        Ok(arrow_schema.fields.iter().into())
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
            .map_err(|e| anyhow::Error::from(e).into())
    }

    /// Fetch and memoize the metadata of the parquet file.
    pub async fn get_metadata(&mut self) -> PolarsResult<&FileMetaData> {
        self.initialize_length().await?;
        if self.metadata.is_none() {
            self.metadata = Some(self.fetch_metadata().await?);
        }
        Ok(self.metadata.as_ref().unwrap())
    }
}

/// A vector of downloaded RowGroups.
/// A RowGroup will have 1 or more columns, for each column we store:
///   - a reference to its metadata
///   - the actual content as downloaded from object storage (generally cloud).
type RowGroupChunks<'a> = Vec<Vec<(&'a ColumnChunkMetaData, Vec<u8>)>>;

/// Download rowgroups for the column whose indexes are given in `projection`.
/// We concurrently download the columns for each field.
#[tokio::main(flavor = "current_thread")]
async fn download_projection<'a: 'b, 'b>(
    projection: &[usize],
    row_groups: &'a [RowGroupMetaData],
    schema: &ArrowSchema,
    async_reader: &'b ParquetObjectStore,
) -> anyhow::Result<RowGroupChunks<'a>> {
    let fields = projection
        .iter()
        .map(|i| schema.fields[*i].name.clone())
        .collect::<Vec<_>>();

    let reader_factory = || {
        let object_store = async_reader.store.clone();
        let path = async_reader.path.clone();
        Box::pin(futures::future::ready(Ok(CloudReader::new(
            async_reader.length,
            object_store,
            path,
        ))))
    }
        as BoxFuture<'static, std::result::Result<CloudReader, std::io::Error>>;

    // Build the cartesian product of the fields and the row groups.
    let product = fields
        .into_iter()
        .flat_map(|f| row_groups.iter().map(move |r| (f.clone(), r)));

    // Download them all concurrently.
    stream::iter(product)
        .then(move |(name, row_group)| async move {
            let columns = row_group.columns();
            read_columns_async(reader_factory, columns, name.as_ref())
                .map_err(anyhow::Error::from)
                .await
        })
        .try_collect()
        .await
}

pub(crate) struct FetchRowGroupsFromObjectStore {
    reader: ParquetObjectStore,
    row_groups_metadata: Vec<RowGroupMetaData>,
    projection: Vec<usize>,
    logging: bool,
    schema: ArrowSchema,
}

impl FetchRowGroupsFromObjectStore {
    pub fn new(
        reader: ParquetObjectStore,
        metadata: &FileMetaData,
        projection: &Option<Vec<usize>>,
    ) -> PolarsResult<Self> {
        let schema = parquet2_read::schema::infer_schema(metadata)?;
        let logging = verbose();

        let projection = projection
            .to_owned()
            .unwrap_or_else(|| (0usize..schema.fields.len()).collect::<Vec<_>>());

        Ok(FetchRowGroupsFromObjectStore {
            reader,
            row_groups_metadata: metadata.row_groups.to_owned(),
            projection,
            logging,
            schema,
        })
    }
}

impl FetchRowGroups for FetchRowGroupsFromObjectStore {
    fn fetch_row_groups(&mut self, row_groups: Range<usize>) -> PolarsResult<ColumnStore> {
        // Fetch the required row groups.
        let row_groups = &self
            .row_groups_metadata
            .get(row_groups.clone())
            .map_or_else(
                || {
                    PolarsResult::Err(PolarsError::ComputeError(
                        format!(
                            "cannot acess slice {0}..{1}",
                            row_groups.start, row_groups.end
                        )
                        .into(),
                    ))
                },
                Ok,
            )?;

        // Package in the format required by ColumnStore.
        let downloaded =
            download_projection(&self.projection, row_groups, &self.schema, &self.reader)?;
        if self.logging {
            eprintln!(
                "BatchedParquetReader: fetched {} row_groups for {} fields, yielding {} column chunks.",
                row_groups.len(),
                self.projection.len(),
                downloaded.len(),
            );
        }
        let downloaded_per_filepos = downloaded
            .into_iter()
            .flat_map(|rg| {
                rg.into_iter()
                    .map(|(meta, data)| (meta.byte_range().0, data))
            })
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
