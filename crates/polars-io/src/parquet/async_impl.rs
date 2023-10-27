//! Read parquet files in parallel from the Object Store without a third party crate.
use std::ops::Range;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

use arrow::datatypes::ArrowSchemaRef;
use bytes::Bytes;
use futures::future::try_join_all;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use polars_core::datatypes::PlHashMap;
use polars_core::error::{to_compute_err, PolarsResult};
use polars_core::prelude::*;
use polars_parquet::read::{self as parquet2_read, RowGroupMetaData};
use polars_parquet::write::FileMetaData;
use smartstring::alias::String as SmartString;

use super::cloud::{build_object_store, CloudLocation, CloudReader};
use super::mmap::ColumnStore;
use crate::cloud::CloudOptions;
use crate::parquet::read_impl::compute_row_group_range;
use crate::pl_async::get_runtime;
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

    pub async fn schema(&mut self) -> PolarsResult<ArrowSchemaRef> {
        let metadata = self.get_metadata().await?;

        let arrow_schema = parquet2_read::infer_schema(metadata)?;

        Ok(Arc::new(arrow_schema))
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
        .enumerate()
        .map(|(i, (name, row_group))| async move {
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

            if i < 10 {
                let handle =
                    tokio::spawn(async move { read_columns_async(&async_reader, &ranges).await });
                handle.await.unwrap()
            } else {
                read_columns_async(&async_reader, &ranges).await
            }
        });

    // Download concurrently
    futures::future::try_join_all(product_futures).await
}

type DownloadedRowGroup = Vec<Vec<(u64, Bytes)>>;

pub struct FetchRowGroupsFromObjectStore {
    row_groups: Arc<Mutex<Receiver<PolarsResult<DownloadedRowGroup>>>>,
}

impl FetchRowGroupsFromObjectStore {
    pub fn new(
        reader: ParquetObjectStore,
        schema: ArrowSchemaRef,
        projection: Option<&[usize]>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        row_groups: &[RowGroupMetaData],
        limit: usize,
    ) -> PolarsResult<Self> {
        let projected_fields = projection
            .map(|projection| {
                projection
                    .iter()
                    .map(|i| SmartString::from(schema.fields[*i].name.as_str()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                schema
                    .fields
                    .iter()
                    .map(|fld| SmartString::from(fld.name.as_str()))
                    .collect()
            });

        let row_groups_end = compute_row_group_range(0, row_groups.len(), limit, row_groups);
        let row_groups = &row_groups[0..row_groups_end];

        let row_groups = if let Some(pred) = predicate.as_deref() {
            row_groups
                .iter()
                .filter(|rg| matches!(read_this_row_group(Some(pred), rg, &schema), Ok(true)))
                .cloned()
                .collect::<Vec<_>>()
        } else {
            row_groups.to_vec()
        };
        let reader = Arc::new(reader);

        let (snd, rcv) = sync_channel(5);

        let _ = std::thread::spawn(move || {
            get_runtime().block_on(async {
                'loop_rg: for rg in row_groups {
                    let fetched = download_projection(&projected_fields, &[rg], &reader).await;

                    match fetched {
                        Ok(fetched) => {
                            let payload = PolarsResult::Ok(fetched);
                            if snd.send(payload).is_err() {
                                break 'loop_rg;
                            }
                        },
                        Err(err) => {
                            let payload = Err(err);
                            let _ = snd.send(payload);
                            break 'loop_rg;
                        },
                    }
                }
            })
        });

        Ok(FetchRowGroupsFromObjectStore {
            row_groups: Arc::new(Mutex::new(rcv)),
        })
    }

    pub(crate) fn fetch_row_groups(
        &mut self,
        row_groups: Range<usize>,
    ) -> PolarsResult<ColumnStore> {
        let mut received = PlHashMap::new();
        let guard = self.row_groups.lock().unwrap();
        for _ in row_groups {
            let downloaded = guard.recv().unwrap()?;

            if received.is_empty() {
                let downloaded_per_filepos = downloaded
                    .into_iter()
                    .flat_map(|rg| rg.into_iter())
                    .collect::<PlHashMap<_, _>>();
                received = downloaded_per_filepos
            } else {
                let downloaded_per_filepos = downloaded.into_iter().flat_map(|rg| rg.into_iter());

                received.extend(downloaded_per_filepos)
            }
        }
        Ok(ColumnStore::Fetched(received))
    }
}
