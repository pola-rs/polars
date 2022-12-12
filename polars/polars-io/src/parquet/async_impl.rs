//! Read parquet files in parallel from the Object Store without a third party crate.
use std::borrow::Cow;
use std::io::{self};
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use arrow::io::parquet::read::{
    self as parquet2_read, read_columns_async, to_deserializer, ColumnChunkMetaData,
    RowGroupMetaData,
};
use arrow::io::parquet::write::FileMetaData;
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use futures::{
    stream, AsyncRead, AsyncSeek, Future, Stream, StreamExt, TryFutureExt, TryStreamExt,
};
use object_store::aws::AmazonS3Builder;
use object_store::path::{Path, Path as ObjectPath};
use object_store::ObjectStore;
use polars_core::error::{PolarsError, PolarsResult};
use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::POOL;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use url::Url;

use super::predicates::read_this_row_group;
use super::read_impl::array_iter_to_series;
use super::ParallelStrategy;
use crate::predicates::{apply_predicate, arrow_schema_to_empty_df, PhysicalIoExpr};
use crate::prelude::apply_projection;
use crate::RowCount;

pub struct AsyncCloudObject {
    pos: u64,
    length: Option<u64>, // total size
    object_store: Arc<Mutex<dyn ObjectStore>>,
    path: Path,
    active: Arc<Mutex<Option<BoxFuture<'static, std::io::Result<Vec<u8>>>>>>,
}

impl AsyncCloudObject {
    pub fn new(length: Option<u64>, object_store: Arc<Mutex<dyn ObjectStore>>, path: Path) -> Self {
        Self {
            pos: 0,
            length,
            object_store,
            path,
            active: Arc::new(Mutex::new(None)),
        }
    }

    async fn create_future_once(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        lenght: usize,
    ) -> std::task::Poll<std::io::Result<Vec<u8>>> {
        let start = self.pos as usize;

        // If we already have a future just poll it.
        if let Some(fut) = self.active.lock().await.as_mut() {
            return Future::poll(fut.as_mut(), cx);
        }

        // Create the future.
        let future = {
            let path = self.path.clone();
            let arc = self.object_store.clone();
            // Use an async move block to get our owned objects.
            async move {
                let object_store = arc.lock().await;
                object_store
                    .get_range(&path, start..start + lenght)
                    .map_ok(|r| r.to_vec())
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("object store error {}", e),
                        )
                    })
                    .await
            }
        };
        let mut future = Box::pin(future);

        // Need to poll it once to get the pump going.
        let polled = Future::poll(future.as_mut(), cx);

        // Save for next time.
        let mut state = self.active.lock().await;
        *state = Some(future);
        polled
    }
}

impl AsyncRead for AsyncCloudObject {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        match block_on(self.create_future_once(cx, buf.len())) {
            Poll::Ready(Ok(bytes)) => {
                buf.copy_from_slice(&bytes[..]);
                Poll::Ready(Ok(bytes.len()))
            }
            Poll::Ready(Err(e)) => {
                Poll::Ready(Err(e))
            }
            Poll::Pending => {
                Poll::Pending
            }
        }
    }
}
impl AsyncSeek for AsyncCloudObject {
    fn poll_seek(
        mut self: Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        pos: io::SeekFrom,
    ) -> std::task::Poll<std::io::Result<u64>> {
        match pos {
            io::SeekFrom::Start(pos) => self.pos = pos,
            io::SeekFrom::End(pos) => {
                let length = self.length.ok_or::<io::Error>(io::Error::new(
                    std::io::ErrorKind::Other,
                    "Cannot seek from end of stream when length is unknown.",
                ))?;
                self.pos = (length as i64 + pos) as u64
            }
            io::SeekFrom::Current(pos) => self.pos = (self.pos as i64 + pos) as u64,
        };
        std::task::Poll::Ready(Ok(self.pos))
    }
}

pub(crate) struct ParquetImpl {
    store: Arc<Mutex<dyn ObjectStore>>,
    path: ObjectPath,
    length: Option<u64>,
    metadata: Option<FileMetaData>,
}

impl ParquetImpl {
    pub fn from_s3_path(path: &str) -> PolarsResult<Self> {
        let parsed =
            Url::parse(path).map_err(|e| PolarsError::External("url parse".into(), Box::new(e)))?;
        let s3 = AmazonS3Builder::from_env()
            //.with_access_key_id(cred.access_key.unwrap())
            //.with_secret_access_key(cred.secret_key.unwrap())
            .with_region("us-west-2")
            .with_bucket_name(
                parsed
                    .host()
                    .ok_or(PolarsError::External(
                        "wrong host".into(),
                        format!("Cannot parse host from {}", path).into(),
                    ))?
                    .to_string(),
            )
            .build()
            .map_err(|e| {
                PolarsError::External("object store amazon s3 builder".into(), Box::new(e))
            })?;
        let store = Arc::new(Mutex::new(s3));
        let path = ObjectPath::from(parsed.path());

        Ok(ParquetImpl {
            store,
            path,
            length: None,
            metadata: None,
        })
    }

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

        let arrow_schema = parquet2_read::infer_schema(&metadata)
            .map_err(|e| PolarsError::External("infer schema".into(), Box::new(e)))?;

        Ok(arrow_schema.fields.iter().into())
    }

    /// Number of rows in the parquet file.
    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata().await?;
        Ok(metadata.num_rows)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn fetch_metadata(&self) -> PolarsResult<FileMetaData> {
        let object_store = self.store.clone();
        let path = self.path.clone();
        let length = self.length.clone();
        let mut reader = AsyncCloudObject::new(length, object_store, path);
        parquet2_read::read_metadata_async(&mut reader)
            .await
            .map_err(|e| PolarsError::External("read metadata async".into(), Box::new(e)))
    }

    /// Fetch and memoize the metadata of the parquet file.
    pub async fn get_metadata(&mut self) -> PolarsResult<&FileMetaData> {
        self.initialize_lenght().await?;
        if self.metadata.is_none() {
            self.metadata = Some(self.fetch_metadata().await?);
        }
        Ok(self.metadata.as_ref().unwrap())
    }
}

fn column_idx_to_series(
    column_i: usize,
    md: &RowGroupMetaData,
    remaining_rows: usize,
    schema: &ArrowSchema,
    downloaded_columns: Vec<(&ColumnChunkMetaData, Vec<u8>)>,
    chunk_size: usize,
) -> PolarsResult<Series> {
    let mut field = schema.fields[column_i].clone();

    match field.data_type {
        ArrowDataType::Utf8 => {
            field.data_type = ArrowDataType::LargeUtf8;
        }
        ArrowDataType::List(fld) => field.data_type = ArrowDataType::LargeList(fld),
        _ => {}
    }

    let iter = to_deserializer(
        downloaded_columns,
        field.clone(),
        remaining_rows,
        Some(chunk_size),
        None,
    )?;

    if remaining_rows < md.num_rows() {
        array_iter_to_series(iter, &field, Some(remaining_rows))
    } else {
        array_iter_to_series(iter, &field, None)
    }
}

/// Download rowgroups for the column whose indexes are given in `projection`.
///
/// Each rowgroup may be made out of 1 or more chunks. The function return a vector of vectors of tuples:
///
///   - the top level vector contains  one entry for each column, in the order of the columns.
///   - each entry is in turn a vector containing the columns for the given rowgroup.
///   - each tuple has a reference to the column metadata and then the content of the column.
#[allow(unused_variables)]
fn download_projection<'a: 'b, 'b>(
    projection: &[usize],
    md: &'a RowGroupMetaData,
    remaining_rows: usize,
    schema: &ArrowSchema,
    async_reader: &'b ParquetImpl,
    chunk_size: usize,
) -> impl Stream<Item = PolarsResult<Vec<(&'a ColumnChunkMetaData, Vec<u8>)>>> + 'b {
    let fields = projection
        .iter()
        .map(|i| schema.fields[*i].name.clone())
        .collect::<Vec<_>>();
    let columns = md.columns();

    let reader_factory = || {
        let object_store = async_reader.store.clone();
        let path = async_reader.path.clone();
        Box::pin(futures::future::ready(Ok(AsyncCloudObject::new(
            async_reader.length,
            object_store,
            path,
        ))))
    }
        as BoxFuture<'static, std::result::Result<AsyncCloudObject, std::io::Error>>;
    stream::iter(fields.into_iter()).then(move |name| async move {
        let reader_factory = reader_factory.clone();
        let columns = columns.clone();
        read_columns_async(reader_factory, columns, name.as_ref())
            .map_err(|e| PolarsError::External("parquet read_columns_async".into(), e.into()))
            .await
    })
}

#[allow(clippy::too_many_arguments)]
// might parallelize over columns
async fn rg_to_dfs(
    async_reader: &ParquetImpl,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    remaining_rows: &mut usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchema,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    row_count: Option<RowCount>,
    parallel: ParallelStrategy,
    projection: &[usize],
) -> PolarsResult<Vec<DataFrame>> {
    let mut dfs = Vec::with_capacity(row_group_end - row_group_start);

    for rg in row_group_start..row_group_end {
        let md = &file_metadata.row_groups[rg];
        let current_row_count = md.num_rows() as IdxSize;

        if !read_this_row_group(predicate.as_ref(), file_metadata, schema, rg)? {
            *previous_row_count += current_row_count;
            continue;
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        // First fetch all the columns.

        let chunk_size = md.num_rows();
        let downloaded_columns = download_projection(
            projection,
            md,
            *remaining_rows,
            schema,
            async_reader,
            chunk_size,
        )
        .try_collect::<Vec<_>>()
        .await?;
        let columns = if let ParallelStrategy::Columns = parallel {
            POOL.install(|| {
                downloaded_columns
                    .into_par_iter()
                    .enumerate()
                    .map(|(index, columns)| {
                        column_idx_to_series(
                            projection[index],
                            md,
                            *remaining_rows,
                            schema,
                            columns,
                            chunk_size,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            downloaded_columns
                .into_iter()
                .enumerate()
                .map(|(index, columns)| {
                    column_idx_to_series(
                        projection[index],
                        md,
                        *remaining_rows,
                        schema,
                        columns,
                        chunk_size,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?
        };

        *remaining_rows = remaining_rows.saturating_sub(file_metadata.row_groups[rg].num_rows());

        let mut df = DataFrame::new_no_checks(columns);
        if let Some(rc) = &row_count {
            df.with_row_count_mut(&rc.name, Some(*previous_row_count + rc.offset));
        }

        apply_predicate(&mut df, predicate.as_deref(), true)?;

        *previous_row_count += current_row_count;
        dfs.push(df);

        if *remaining_rows == 0 {
            break;
        }
    }
    Ok(dfs)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn read_parquet_async(
    async_reader: &ParquetImpl,
    mut limit: usize,
    projection: Option<&[usize]>,
    schema: &ArrowSchema,
    metadata: FileMetaData,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    mut parallel: ParallelStrategy,
    row_count: Option<RowCount>,
) -> PolarsResult<DataFrame> {
    let row_group_len = metadata.row_groups.len();

    let projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..schema.fields.len()).collect::<Vec<_>>()));

    if let ParallelStrategy::Auto = parallel {
        if row_group_len > projection.len() || row_group_len > POOL.current_num_threads() {
            parallel = ParallelStrategy::RowGroups;
        } else {
            parallel = ParallelStrategy::Columns;
        }
    }

    if let (ParallelStrategy::Columns, true) = (parallel, projection.len() == 1) {
        parallel = ParallelStrategy::None;
    }

    let dfs = match parallel {
        ParallelStrategy::Columns | ParallelStrategy::None => {
            rg_to_dfs(
                &async_reader,
                &mut 0,
                0,
                row_group_len,
                &mut limit,
                &metadata,
                schema,
                predicate,
                row_count,
                parallel,
                &projection,
            )
            .await?
        }
        // TODO: implement row group parallelism.
        ParallelStrategy::RowGroups => {
            rg_to_dfs(
                &async_reader,
                &mut 0,
                metadata.row_groups.len(),
                row_group_len,
                &mut limit,
                &metadata,
                schema,
                predicate,
                row_count,
                parallel,
                &projection,
            )
            .await?
        }
        // auto should already be replaced by Columns or RowGroups
        ParallelStrategy::Auto => unimplemented!(),
    };

    if dfs.is_empty() {
        let schema = if let Cow::Borrowed(_) = projection {
            Cow::Owned(apply_projection(schema, &projection))
        } else {
            Cow::Borrowed(schema)
        };
        Ok(arrow_schema_to_empty_df(&schema))
    } else {
        accumulate_dataframes_vertical(dfs.into_iter())
    }
}
