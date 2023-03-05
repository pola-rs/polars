//! Polars is a heavily multi-threaded library. Some IO operations, specially cloud based ones,
//! are best served by an async module. The AsyncManager owns a multi-threaded Tokio runtime
//! and is responsible for managing the async calls to the object store and the associated state.

use std::ops::Range;

use arrow::io::parquet::read::RowGroupMetaData;
use arrow::io::parquet::write::FileMetaData;
use futures::channel::mpsc::Sender;
use futures::channel::oneshot;
use once_cell::sync::Lazy;
use polars_core::prelude::*;
use tokio::runtime::Handle;

use super::mmap::ColumnMapper;

static GLOBAL_ASYNC_MANAGER: Lazy<AsyncManager> = Lazy::new(AsyncManager::default);

enum AsyncParquetReaderMessage {
    /// Fetch the metadata of the parquet file, do not memoize it.
    FetchMetadata {
        /// The channel to send the result to.
        tx: oneshot::Sender<PolarsResult<FileMetaData>>,
    },
    /// Fetch and memoize the metadata of the parquet file.
    GetMetadata {
        /// The channel to send the result to.
        tx: oneshot::Sender<PolarsResult<FileMetaData>>,
    },
    /// Fetch the number of rows of the parquet file.
    NumRows {
        /// The channel to send the result to.
        tx: oneshot::Sender<PolarsResult<usize>>,
    },
    /// Fetch the schema of the parquet file.
    Schema {
        /// The channel to send the result to.
        tx: oneshot::Sender<PolarsResult<ArrowSchema>>,
    },
    /// Fetch the row groups of the parquet file.
    RowGroups {
        /// The channel to send the result to.
        tx: oneshot::Sender<PolarsResult<Vec<RowGroupMetaData>>>,
    },
    /// Fetch the row groups of the parquet file.
    FetchRowGroups {
        /// The row groups to fetch.
        row_groups: Range<usize>,
        /// The channel to send the result to.
        tx: oneshot::Sender<PolarsResult<ColumnMapper>>,
    },
}

/// Separate the async calls in their own manager and interact with the rest of the code with a channel.
pub(crate) struct AsyncManager {
    /// The channel to communicate with the manager.
    tx: Sender<AsyncParquetReaderMessage>,
    /// A handle to the Tokio runtime running the manager.
    handle: Handle,
    /// Opened readers.
    readers: PlHashMap<String, Arc<ParquetObjectStore>>,
}

impl AsyncManager {
    /// Create a new async manager.
    pub fn new() -> AsyncManager {
        let (tx, rx) = futures::channel::mpsc::channel(1);
        let handle = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        handle.spawn(async move {
            let mut reader = None;
            while let Some(message) = rx.next().await {
                match message {
                    AsyncParquetReaderMessage::FetchMetadata { tx } => {
                        let reader = reader.as_mut().unwrap();
                        let result = reader.fetch_metadata().await;
                        tx.send(result).unwrap();
                    }
                    AsyncParquetReaderMessage::GetMetadata { tx } => {
                        let reader = reader.as_mut().unwrap();
                        let result = reader.get_metadata().await;
                        tx.send(result).unwrap();
                    }
                    AsyncParquetReaderMessage::NumRows { tx } => {
                        let reader = reader.as_mut().unwrap();
                        let result = reader.num_rows().await;
                        tx.send(result).unwrap();
                    }
                    AsyncParquetReaderMessage::Schema { tx } => {
                        let reader = reader.as_mut().unwrap();
                        let result = reader.schema().await;
                        tx.send(result).unwrap();
                    }
                    AsyncParquetReaderMessage::RowGroups { tx } => {
                        let reader = reader.as_mut().unwrap();
                        let result = reader.row_groups().await;
                        tx.send(result).unwrap();
                    }
                    AsyncParquetReaderMessage::FetchRowGroups { row_groups, tx } => {
                        let reader = reader.as_mut().unwrap();
                        let result = reader.fetch_row_groups(row_groups).await;
                        tx.send(result).unwrap();
                    }
                }
            }
        });
        AsyncManager { tx, handle }
    }
}

impl Default for AsyncManager {
    fn default() -> Self {
        AsyncManager::new()
    }
}
