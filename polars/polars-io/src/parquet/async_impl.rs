//! Read parquet files in parallel from the Object Store without a third party crate.
use std::collections::VecDeque;
use std::sync::Arc;

use ahash::HashMap;
use arrow::io::parquet::read::{
    self as parquet2_read, read_columns_async, ColumnChunkMetaData, RowGroupMetaData,
};
use arrow::io::parquet::write::FileMetaData;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use futures::{stream, StreamExt, TryFutureExt, TryStreamExt};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use polars_core::error::{PolarsError, PolarsResult};
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_core::utils::split_df;
use polars_core::POOL;

use super::read_impl::{rg_to_dfs, rg_to_dfs_par, HasNextBatches};
use super::{mmap, ParallelStrategy};
use crate::object_store::{build, CloudReader};
use crate::RowCount;

pub struct ParquetObjectStore {
    store: Arc<Mutex<Box<dyn ObjectStore>>>,
    path: ObjectPath,
    length: Option<u64>,
    metadata: Option<FileMetaData>,
}

impl ParquetObjectStore {
    pub fn from_uri(uri: &str) -> PolarsResult<Self> {
        let (path, store) = build(uri)?;
        let store = Arc::new(Mutex::from(store));

        Ok(ParquetObjectStore {
            store,
            path,
            length: None,
            metadata: None,
        })
    }

    /// Initialize the lenght property of the object, unless it has already been fetched.
    async fn initialize_lenght(&mut self) -> PolarsResult<()> {
        if self.length.is_some() {
            return Ok(());
        }
        let path = self.path.clone();
        let locked_store = self.store.lock().await;
        self.length = Some({
            locked_store
                .head(&path)
                .await
                .map_err(|e| PolarsError::External("reading object lenght".into(), Box::new(e)))?
                .size as u64
        });
        Ok(())
    }

    pub async fn schema(&mut self) -> PolarsResult<Schema> {
        let metadata = self.get_metadata().await?;

        let arrow_schema = parquet2_read::infer_schema(metadata)
            .map_err(|e| PolarsError::External("infer schema".into(), Box::new(e)))?;

        Ok(arrow_schema.fields.iter().into())
    }

    /// Number of rows in the parquet file.
    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata().await?;
        Ok(metadata.num_rows)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn fetch_metadata(&mut self) -> PolarsResult<FileMetaData> {
        self.initialize_lenght().await?;
        let object_store = self.store.clone();
        let path = self.path.clone();
        let length = self.length;
        let mut reader = CloudReader::new(length, object_store, path);
        parquet2_read::read_metadata_async(&mut reader)
            .await
            .map_err(|e| PolarsError::External("read metadata async".into(), Box::new(e)))
    }

    /// Fetch and memoize the metadata of the parquet file.
    async fn get_metadata(&mut self) -> PolarsResult<&FileMetaData> {
        self.initialize_lenght().await?;
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
) -> PolarsResult<RowGroupChunks<'a>> {
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

    // Download them all concurently.
    stream::iter(product)
        .then(move |(name, row_group)| async move {
            let columns = row_group.columns();
            read_columns_async(reader_factory, columns, name.as_ref())
                .map_err(|e| PolarsError::External("parquet read_columns_async".into(), e.into()))
                .await
        })
        .try_collect()
        .await
}

pub struct BatchedParquetReader {
    // use to keep ownership
    reader: ParquetObjectStore,
    limit: usize,
    projection: Vec<usize>,
    schema: ArrowSchema,
    metadata: FileMetaData,
    row_count: Option<RowCount>,
    rows_read: IdxSize,
    row_group_offset: usize,
    n_row_groups: usize,
    chunks_fifo: VecDeque<DataFrame>,
    parallel: ParallelStrategy,
    chunk_size: usize,
    logging: bool,
}

impl BatchedParquetReader {
    pub fn new(
        reader: ParquetObjectStore,
        metadata: FileMetaData,
        limit: usize,
        projection: Option<Vec<usize>>,
        row_count: Option<RowCount>,
        chunk_size: usize,
        logging: bool,
    ) -> PolarsResult<Self> {
        let schema = parquet2_read::schema::infer_schema(&metadata)?;
        let n_row_groups = metadata.row_groups.len();
        let projection =
            projection.unwrap_or_else(|| (0usize..schema.fields.len()).collect::<Vec<_>>());

        let parallel =
            if n_row_groups > projection.len() || n_row_groups > POOL.current_num_threads() {
                ParallelStrategy::RowGroups
            } else {
                ParallelStrategy::Columns
            };

        Ok(BatchedParquetReader {
            reader,
            limit,
            projection,
            schema,
            metadata,
            row_count,
            rows_read: 0,
            row_group_offset: 0,
            n_row_groups,
            chunks_fifo: VecDeque::with_capacity(POOL.current_num_threads()),
            parallel,
            chunk_size,
            logging,
        })
    }
}

impl HasNextBatches for BatchedParquetReader {
    fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        // fill up fifo stack
        if self.row_group_offset <= self.n_row_groups && self.chunks_fifo.len() < n {
            let row_group_start = self.row_group_offset;
            let row_group_end = std::cmp::min(self.row_group_offset + n, self.n_row_groups);

            // Fetch the required row groups.
            let row_groups = self
                .metadata
                .row_groups
                .get(row_group_start..row_group_end)
                .map_or_else(
                    || {
                        PolarsResult::Err(PolarsError::External(
                            "get row groups".into(),
                            format!("cannot acess slice {}..{}", row_group_start, row_group_end)
                                .into(),
                        ))
                    },
                    Ok,
                )?;

            // Package in a format that is usable by the helper functions rg_to_dfs.
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
                .collect::<HashMap<_, _>>();

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

            let store = mmap::ColumnStore::Fetched(&downloaded_per_filepos);

            // Deserialize and build dataframes.
            let dfs = match self.parallel {
                ParallelStrategy::Columns => {
                    let dfs = rg_to_dfs(
                        &store,
                        &mut self.rows_read,
                        row_group_start,
                        row_group_end,
                        &mut self.limit,
                        &self.metadata,
                        &self.schema,
                        None,
                        self.row_count.clone(),
                        ParallelStrategy::Columns,
                        &self.projection,
                    )?;
                    self.row_group_offset += n;
                    dfs
                }
                ParallelStrategy::RowGroups => {
                    let dfs = rg_to_dfs_par(
                        &store,
                        row_group_start,
                        row_group_end,
                        &mut self.rows_read,
                        &mut self.limit,
                        &self.metadata,
                        &self.schema,
                        None,
                        self.row_count.clone(),
                        &self.projection,
                    )?;
                    self.row_group_offset += n;
                    dfs
                }
                _ => unimplemented!(),
            };

            // TODO! this is slower than it needs to be
            // we also need to parallelize over row groups here.

            for mut df in dfs {
                // make sure that the chunks are not too large
                let n = df.shape().0 / self.chunk_size;
                if n > 1 {
                    for df in split_df(&mut df, n)? {
                        self.chunks_fifo.push_back(df)
                    }
                } else {
                    self.chunks_fifo.push_back(df)
                }
            }
        };

        if self.chunks_fifo.is_empty() {
            Ok(None)
        } else {
            let mut chunks = Vec::with_capacity(n);
            let mut i = 0;
            while let Some(df) = self.chunks_fifo.pop_front() {
                chunks.push(df);
                i += 1;
                if i == n {
                    break;
                }
            }

            Ok(Some(chunks))
        }
    }
}
