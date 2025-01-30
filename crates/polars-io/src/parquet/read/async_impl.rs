//! Read parquet files in parallel from the Object Store without a third party crate.
use std::ops::Range;

use arrow::datatypes::ArrowSchemaRef;
use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use polars_core::config::{get_rg_prefetch_size, verbose};
use polars_core::prelude::*;
use polars_parquet::read::RowGroupMetadata;
use polars_parquet::write::FileMetadata;
use polars_utils::pl_str::PlSmallStr;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::sync::Mutex;

use super::mmap::ColumnStore;
use super::predicates::read_this_row_group;
use crate::cloud::{
    build_object_store, object_path_from_str, CloudLocation, CloudOptions, PolarsObjectStore,
};
use crate::parquet::metadata::FileMetadataRef;
use crate::pl_async::get_runtime;
use crate::predicates::ScanIOPredicate;

type DownloadedRowGroup = PlHashMap<u64, Bytes>;
type QueuePayload = (usize, DownloadedRowGroup);
type QueueSend = Arc<Sender<PolarsResult<QueuePayload>>>;

pub struct ParquetObjectStore {
    store: PolarsObjectStore,
    path: ObjectPath,
    length: Option<usize>,
    metadata: Option<FileMetadataRef>,
}

impl ParquetObjectStore {
    pub async fn from_uri(
        uri: &str,
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
        })
    }

    async fn get_ranges(&self, ranges: &mut [Range<usize>]) -> PolarsResult<PlHashMap<u64, Bytes>> {
        self.store.get_ranges_sort(&self.path, ranges).await
    }

    /// Initialize the length property of the object, unless it has already been fetched.
    async fn length(&mut self) -> PolarsResult<usize> {
        if self.length.is_none() {
            self.length = Some(self.store.head(&self.path).await?.size);
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
) -> PolarsResult<FileMetadata> {
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

    let footer_byte_length: usize = {
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

    Ok(polars_parquet::parquet::read::deserialize_metadata(
        std::io::Cursor::new(footer_bytes.as_ref()),
        // TODO: Describe why this makes sense. Taken from the previous
        // implementation which said "a highly nested but sparse struct could
        // result in many allocations".
        footer_bytes.as_ref().len() * 2 + 1024,
    )?)
}

/// Download rowgroups for the column whose indexes are given in `projection`.
/// We concurrently download the columns for each field.
async fn download_projection(
    fields: Arc<[PlSmallStr]>,
    row_group: RowGroupMetadata,
    async_reader: Arc<ParquetObjectStore>,
    sender: QueueSend,
    rg_index: usize,
) -> bool {
    let async_reader = &async_reader;
    let row_group = &row_group;
    let fields = fields.as_ref();

    let mut ranges = Vec::with_capacity(fields.len());
    let mut offsets = Vec::with_capacity(fields.len());
    fields.iter().for_each(|name| {
        // A single column can have multiple matches (structs).
        let iter = row_group
            .columns_under_root_iter(name)
            .unwrap()
            .map(|meta| {
                let byte_range = meta.byte_range();
                let offset = byte_range.start;
                let byte_range = byte_range.start as usize..byte_range.end as usize;
                (offset, byte_range)
            });

        for (offset, range) in iter {
            offsets.push(offset);
            ranges.push(range);
        }
    });

    let result = async_reader
        .get_ranges(&mut ranges)
        .await
        .map(|bytes_map| (rg_index, bytes_map));
    sender.send(result).await.is_ok()
}

async fn download_row_group(
    rg: RowGroupMetadata,
    async_reader: Arc<ParquetObjectStore>,
    sender: QueueSend,
    rg_index: usize,
) -> bool {
    if rg.n_columns() == 0 {
        return true;
    }

    let mut ranges = rg
        .byte_ranges_iter()
        .map(|x| x.start as usize..x.end as usize)
        .collect::<Vec<_>>();

    sender
        .send(
            async_reader
                .get_ranges(&mut ranges)
                .await
                .map(|bytes_map| (rg_index, bytes_map)),
        )
        .await
        .is_ok()
}

pub struct FetchRowGroupsFromObjectStore {
    rg_q: Arc<Mutex<Receiver<PolarsResult<QueuePayload>>>>,
    prefetched_rg: PlHashMap<usize, DownloadedRowGroup>,
}

impl FetchRowGroupsFromObjectStore {
    pub fn new(
        reader: ParquetObjectStore,
        schema: ArrowSchemaRef,
        projection: Option<&[usize]>,
        predicate: Option<ScanIOPredicate>,
        row_group_range: Range<usize>,
        row_groups: &[RowGroupMetadata],
    ) -> PolarsResult<Self> {
        let projected_fields: Option<Arc<[PlSmallStr]>> = projection.map(|projection| {
            projection
                .iter()
                .map(|i| (schema.get_at_index(*i).as_ref().unwrap().0.clone()))
                .collect()
        });

        let mut prefetched: PlHashMap<usize, DownloadedRowGroup> = PlHashMap::new();

        let mut row_groups = if let Some(pred) = predicate.as_ref() {
            row_group_range
                .filter_map(|i| {
                    let rg = &row_groups[i];

                    let should_be_read =
                        matches!(read_this_row_group(Some(pred), rg, &schema), Ok(true));

                    // Already add the row groups that will be skipped to the prefetched data.
                    if !should_be_read {
                        prefetched.insert(i, Default::default());
                    }

                    should_be_read.then(|| (i, rg.clone()))
                })
                .collect::<Vec<_>>()
        } else {
            row_groups.iter().cloned().enumerate().collect()
        };
        let reader = Arc::new(reader);
        let msg_limit = get_rg_prefetch_size();

        if verbose() {
            eprintln!("POLARS ROW_GROUP PREFETCH_SIZE: {}", msg_limit)
        }

        let (snd, rcv) = channel(msg_limit);
        let snd = Arc::new(snd);

        get_runtime().spawn(async move {
            let chunk_len = msg_limit;
            let mut handles = Vec::with_capacity(chunk_len.clamp(0, row_groups.len()));
            for chunk in row_groups.chunks_mut(chunk_len) {
                // Start downloads concurrently
                for (i, rg) in chunk {
                    let rg = std::mem::take(rg);

                    match &projected_fields {
                        Some(projected_fields) => {
                            let handle = tokio::spawn(download_projection(
                                projected_fields.clone(),
                                rg,
                                reader.clone(),
                                snd.clone(),
                                *i,
                            ));
                            handles.push(handle)
                        },
                        None => {
                            let handle = tokio::spawn(download_row_group(
                                rg,
                                reader.clone(),
                                snd.clone(),
                                *i,
                            ));
                            handles.push(handle)
                        },
                    }
                }

                // Wait n - 3 tasks, so we already start the next downloads earlier.
                for task in handles.drain(..handles.len().saturating_sub(3)) {
                    let succeeded = task.await.unwrap();
                    if !succeeded {
                        return;
                    }
                }
            }

            // Drain remaining tasks.
            for task in handles.drain(..) {
                let succeeded = task.await.unwrap();
                if !succeeded {
                    return;
                }
            }
        });

        Ok(FetchRowGroupsFromObjectStore {
            rg_q: Arc::new(Mutex::new(rcv)),
            prefetched_rg: Default::default(),
        })
    }

    pub(crate) async fn fetch_row_groups(
        &mut self,
        row_groups: Range<usize>,
    ) -> PolarsResult<ColumnStore> {
        let mut guard = self.rg_q.lock().await;

        while !row_groups
            .clone()
            .all(|i| self.prefetched_rg.contains_key(&i))
        {
            let Some(fetched) = guard.recv().await else {
                break;
            };
            let (rg_i, payload) = fetched?;

            self.prefetched_rg.insert(rg_i, payload);
        }

        let received = row_groups
            .flat_map(|i| self.prefetched_rg.remove(&i))
            .flat_map(|rg| rg.into_iter())
            .collect::<PlHashMap<_, _>>();

        Ok(ColumnStore::Fetched(received))
    }
}
