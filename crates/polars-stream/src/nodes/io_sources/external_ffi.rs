//! Multi-scan readers of I/O plugins (`polars.io_plugin.file_reader.v1`).
//!
//! The plugin's functions block: they run on blocking threads. A reader pulls its source's Arrow
//! C stream on a blocking thread and turns the batches into morsels; it applies positive slices
//! itself, and gets row counts from the plugin.
//!
//! A plugin call still running when its task is dropped (an early stop of the query, an error
//! elsewhere) is handed to the query, which waits for it before it returns: a plugin's cleanup
//! must not run on, say, a user's connection while the user does something else with it.
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use polars_arrow::array::{Array, StructArray};
use polars_arrow::ffi::ArrowArrayStreamReader;
use polars_async::executor::{JoinHandle, TaskMetricAggregator, TaskPriority};
use polars_async::{ASYNC, executor};
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::io_plugin_ffi::{ffi_err_to_polars, import_struct_schema};
use polars_io_ext_ffi::common::FfiStr;
use polars_io_ext_ffi::file_reader_v1::{BeginRead, Plugin, ReaderHandle, ScanHandle, flags};
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::index::idxsize_try_from;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

use crate::execute::StreamingExecutionState;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_scan::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use crate::nodes::io_sources::multi_scan::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};

/// Batches buffered between the fetch thread and the morsel task.
const FETCH_CHANNEL_CAPACITY: usize = 2;

/// The query's tasks are awaited before it returns.
type QueryTasks = Option<Arc<StreamingExecutionState>>;

/// A plugin call running on a blocking thread, off the async executor's threads. Dropped before
/// it is joined, it hands the call to the query to wait for.
struct Pending<T: Send + 'static> {
    handle: Option<tokio::task::JoinHandle<T>>,
    query_tasks: QueryTasks,
}

impl<T: Send + 'static> Pending<T> {
    fn spawn(query_tasks: &QueryTasks, f: impl FnOnce() -> T + Send + 'static) -> Self {
        Self {
            handle: Some(ASYNC.spawn_blocking(f)),
            query_tasks: query_tasks.clone(),
        }
    }

    async fn join(mut self) -> T {
        // Awaited in place: if this future is dropped, `self` still holds the call.
        let out = self.handle.as_mut().unwrap().await;
        // (joined: a panic of the call isn't awaited again on drop)
        self.handle = None;
        out.unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))
    }
}

impl<T: Send + 'static> Drop for Pending<T> {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        let wait = async move {
            let _ = handle.await;
            Ok(())
        };
        match &self.query_tasks {
            Some(state) => state.spawn_query_task(wait),
            None => drop(ASYNC.spawn(wait)),
        }
    }
}

/// Runs a plugin call that may block, off the async executor's threads, in a task.
async fn blocking<T: Send + 'static>(
    query_tasks: &QueryTasks,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    Pending::spawn(query_tasks, f).join().await
}

/// One execution of the scan (`ScanHandle`). Releasing it stops readers waiting for the scan.
struct Scan {
    package: PlSmallStr,
    plugin: &'static Plugin,
    handle: ScanHandle,
}

impl Drop for Scan {
    fn drop(&mut self) {
        unsafe { (self.plugin.scan_release)(self.handle) }
    }
}

/// The reader of one source (`ReaderHandle`).
struct Reader {
    scan: Arc<Scan>,
    handle: ReaderHandle,
    source: PlSmallStr,
}

// The contract makes handles thread-safe; a reader is used by one thread at a time, except for
// `reader_cancel`.
unsafe impl Send for Reader {}
unsafe impl Sync for Reader {}
unsafe impl Send for Scan {}
unsafe impl Sync for Scan {}

impl Reader {
    fn plugin(&self) -> &'static Plugin {
        self.scan.plugin
    }

    fn err(&self, e: polars_io_ext_ffi::common::FfiError) -> PolarsError {
        ffi_err_to_polars(e, &self.scan.package)
    }

    fn schema(&self) -> PolarsResult<SchemaRef> {
        let schema = unsafe { (self.plugin().reader_schema)(self.handle) }
            .into_result()
            .map_err(|e| self.err(e))?;
        Ok(Arc::new(import_struct_schema(schema, &self.scan.package)?))
    }

    fn n_rows(&self) -> PolarsResult<IdxSize> {
        let n = unsafe { (self.plugin().reader_n_rows)(self.handle) }
            .into_result()
            .map_err(|e| self.err(e))?;
        idxsize_try_from(n).map_err(|_| {
            polars_err!(
                ComputeError:
                "source '{}' has {n} rows, which exceeds IdxSize; use bigidx", self.source
            )
        })
    }

    fn cancel(&self) {
        unsafe { (self.plugin().reader_cancel)(self.handle) }
    }

    /// The error of the stream's last failed batch, with its kind, if the plugin kept it.
    fn take_error(&self) -> Option<PolarsError> {
        unsafe { (self.plugin().reader_take_error)(self.handle) }
            .into_result()
            .err()
            .map(|e| self.err(e))
    }

    /// Reads `columns` and sends every batch to `tx`, until the end, an error, or `tx` closes.
    fn fetch(&self, columns: Vec<String>, tx: tokio::sync::mpsc::Sender<PolarsResult<DataFrame>>) {
        let begin = BeginRead {
            columns,
            verbose: polars_core::config::verbose(),
        };
        let stream = begin
            .with_ffi(|args| unsafe { (self.plugin().reader_begin_read)(self.handle, args) })
            .into_result()
            .map_err(|e| self.err(e));
        let stream = match stream {
            // SAFETY: the contract's struct has the C stream interface layout.
            Ok(s) => Box::new(unsafe { s.transmute_into::<polars_arrow::ffi::ArrowArrayStream>() }),
            Err(e) => {
                let _ = tx.blocking_send(Err(e));
                return;
            },
        };
        let mut reader = match unsafe { ArrowArrayStreamReader::try_new(stream) } {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.blocking_send(Err(e));
                return;
            },
        };
        loop {
            // SAFETY: the plugin produces arrays of the stream's schema.
            let batch = match unsafe { reader.next() } {
                None => return,
                Some(Ok(array)) => batch_to_df(array),
                Some(Err(e)) => Err(self.take_error().unwrap_or(e)),
            };
            let is_err = batch.is_err();
            // Releasing the stream (dropping `reader`) when nobody reads any more stops the read.
            if tx.blocking_send(batch).is_err() || is_err {
                return;
            }
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        unsafe { (self.plugin().reader_release)(self.handle) }
    }
}

/// Cancels the reader's read on drop, unless disarmed: when the task is dropped mid-read (an
/// early stop of the whole query, an error elsewhere), the plugin stops its statement and a
/// fetch thread waiting in the plugin returns. The query waits for the cancel.
struct CancelOnDrop(Option<Arc<Reader>>, QueryTasks);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(reader) = self.0.take() {
            // The plugin may block to cancel.
            drop(Pending::spawn(&self.1, move || reader.cancel()));
        }
    }
}

fn batch_to_df(array: Box<dyn Array>) -> PolarsResult<DataFrame> {
    let Some(array) = array.as_any().downcast_ref::<StructArray>() else {
        polars_bail!(ComputeError: "I/O plugin returned a non-struct batch: {:?}", array.dtype());
    };
    // A record batch has no top-level nulls.
    DataFrame::try_from(array.clone().with_validity(None))
}

pub struct FfiFileReaderBuilder {
    builder: polars_io::external_reader::ffi::FfiFileReaderBuilder,
    scan: Arc<Scan>,
    flags: u32,
    max_concurrent_scans: Option<usize>,
    preferred_remote_pre_init: Option<usize>,
    task_metrics: OnceLock<Arc<TaskMetricAggregator>>,
    execution_state: OnceLock<Arc<StreamingExecutionState>>,
}

impl FfiFileReaderBuilder {
    /// Starts an execution of the scan.
    pub fn new(
        builder: polars_io::external_reader::ffi::FfiFileReaderBuilder,
    ) -> PolarsResult<Self> {
        let plugin = builder.plugin();
        let package = builder.package().clone();
        let caps = builder.capabilities()?;
        let handle = unsafe { (plugin.scan_new)(builder.handle()) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &package))?;
        let to_usize = |v: Option<u64>| v.map(|v| usize::try_from(v).unwrap_or(usize::MAX));
        Ok(Self {
            scan: Arc::new(Scan {
                package,
                plugin,
                handle,
            }),
            flags: caps.flags,
            max_concurrent_scans: to_usize(caps.max_concurrent_scans.into_option()),
            preferred_remote_pre_init: to_usize(caps.preferred_remote_pre_init.into_option()),
            builder,
            task_metrics: OnceLock::new(),
            execution_state: OnceLock::new(),
        })
    }
}

impl std::fmt::Debug for FfiFileReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiFileReaderBuilder")
            .field("builder", &self.builder)
            .finish()
    }
}

impl FileReaderBuilder for FfiFileReaderBuilder {
    fn reader_name(&self) -> PolarsResult<PlSmallStr> {
        Ok(self.builder.name())
    }

    fn reader_capabilities(&self) -> PolarsResult<ReaderCapabilities> {
        use ReaderCapabilities as RC;
        let mut caps = RC::PRE_SLICE;
        if self.flags & flags::UNORDERED != 0 {
            caps |= RC::UNORDERED_FILES;
        }
        Ok(caps)
    }

    fn set_execution_state(&self, execution_state: &StreamingExecutionState) {
        if let Some(task_metrics) = execution_state.task_metrics.clone() {
            let _ = self.task_metrics.set(task_metrics);
        }
        let _ = self.execution_state.set(Arc::new(execution_state.clone()));
    }

    fn preferred_remote_pre_init(&self) -> Option<usize> {
        self.preferred_remote_pre_init
    }

    fn sources_are_remote(&self) -> Option<bool> {
        (self.flags & flags::SOURCES_REMOTE != 0).then_some(true)
    }

    fn max_concurrent_scans(&self) -> Option<usize> {
        self.max_concurrent_scans
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        _cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> PolarsResult<Box<dyn FileReader>> {
        let ScanSource::Path(path) = source else {
            polars_bail!(ComputeError: "I/O plugin scan sources must be ids");
        };
        let id = path.as_str();
        let scan = &self.scan;
        let handle = unsafe { (scan.plugin.reader_new)(scan.handle, FfiStr::new(id)) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &scan.package))?;
        Ok(Box::new(FfiFileReader {
            reader: Arc::new(Reader {
                scan: scan.clone(),
                handle,
                source: id.into(),
            }),
            schema: None,
            n_rows: None,
            task_metrics: self.task_metrics.get().cloned(),
            query_tasks: self.execution_state.get().cloned(),
        }))
    }
}

pub struct FfiFileReader {
    reader: Arc<Reader>,
    schema: Option<SchemaRef>,
    n_rows: Option<IdxSize>,
    task_metrics: Option<Arc<TaskMetricAggregator>>,
    query_tasks: QueryTasks,
}

impl FfiFileReader {
    async fn count(&mut self) -> PolarsResult<IdxSize> {
        if let Some(n) = self.n_rows {
            return Ok(n);
        }
        let reader = self.reader.clone();
        // A count may run a statement as long as the read: stopped as a read is.
        let mut cancel_guard = CancelOnDrop(Some(reader.clone()), self.query_tasks.clone());
        let n = blocking(&self.query_tasks, move || reader.n_rows()).await;
        cancel_guard.0 = None;
        let n = n?;
        self.n_rows = Some(n);
        Ok(n)
    }

    async fn schema(&mut self) -> PolarsResult<SchemaRef> {
        if let Some(s) = &self.schema {
            return Ok(s.clone());
        }
        let reader = self.reader.clone();
        let mut cancel_guard = CancelOnDrop(Some(reader.clone()), self.query_tasks.clone());
        let schema = blocking(&self.query_tasks, move || reader.schema()).await;
        cancel_guard.0 = None;
        let schema = schema?;
        self.schema = Some(schema.clone());
        Ok(schema)
    }
}

/// Positive-slice state over the stream of rows of one source.
struct SliceState {
    skip: usize,
    remaining: usize,
}

impl SliceState {
    fn new(pre_slice: Option<Slice>) -> PolarsResult<Self> {
        Ok(match pre_slice {
            None => Self {
                skip: 0,
                remaining: usize::MAX,
            },
            Some(Slice::Positive { offset, len }) => Self {
                skip: offset,
                remaining: len,
            },
            Some(Slice::Negative { .. }) => {
                polars_bail!(ComputeError: "I/O plugin reader was given a negative slice")
            },
        })
    }

    fn done(&self) -> bool {
        self.remaining == 0
    }

    fn apply(&mut self, df: DataFrame) -> DataFrame {
        let h = df.height();
        if self.skip >= h {
            self.skip -= h;
            return df.slice(0, 0);
        }
        let take = (h - self.skip).min(self.remaining);
        let out = df.slice(self.skip as i64, take);
        self.skip = 0;
        self.remaining -= take;
        out
    }
}

#[async_trait]
impl FileReader for FfiFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        // Plugins do their I/O when counting or reading.
        Ok(())
    }

    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        self.schema().await
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self.count().await
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let BeginReadArgs {
            projection,
            row_index: _,
            pre_slice,
            predicate: _,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
            ..
        } = args;

        let projected = match projection {
            Projection::Plain(schema) => schema,
            Projection::Mapped { .. } => {
                polars_bail!(ComputeError: "I/O plugin reader was given a mapped projection")
            },
        };
        let slice_end = pre_slice.as_ref().map(|s| s.end_position());
        let mut slice = SliceState::new(pre_slice)?;
        let (mut tx, rx) = FileReaderOutputSend::new_serial();

        let mut slf = FfiFileReader {
            reader: self.reader.clone(),
            schema: self.schema.clone(),
            n_rows: self.n_rows,
            task_metrics: self.task_metrics.clone(),
            query_tasks: self.query_tasks.clone(),
        };
        let task_metrics = self.task_metrics.clone();

        let handle = executor::spawn(TaskPriority::Low, task_metrics.as_deref(), async move {
            if let Some(tx) = file_schema_tx {
                let _ = tx.send(slf.schema().await?);
            }
            if let Some(tx) = n_rows_in_file_tx {
                let _ = tx.send(slf.count().await?);
            }

            let source_token = SourceToken::new();

            // Only the row count is needed: no rows are read.
            if projected.is_empty() {
                let n = slf.count().await?;
                let df = slice.apply(DataFrame::empty_with_height(n as usize));
                if let Some(tx) = row_position_on_end_tx {
                    let _ = tx.send(n);
                }
                if df.height() > 0 {
                    let _ = tx
                        .send_morsel(Morsel::new_unregistered(
                            df,
                            MorselSeq::new(0),
                            source_token,
                        ))
                        .await;
                }
                return Ok(());
            }

            let columns: Vec<String> = projected.iter_names().map(|c| c.to_string()).collect();
            let (batch_tx, mut batch_rx) =
                tokio::sync::mpsc::channel::<PolarsResult<DataFrame>>(FETCH_CHANNEL_CAPACITY);
            let reader = slf.reader.clone();
            let mut cancel_guard = CancelOnDrop(Some(reader.clone()), slf.query_tasks.clone());
            // Plugins block: the stream is pulled on its own thread.
            let fetch = Pending::spawn(&slf.query_tasks, move || reader.fetch(columns, batch_tx));

            let morsel_size = get_ideal_morsel_size().max(1);
            let mut rows_read: u64 = 0;
            let mut seq: u64 = 0;
            let mut stopped = false;
            let mut error = None;

            'outer: while let Some(batch) = batch_rx.recv().await {
                let df = match batch {
                    Ok(df) => df,
                    Err(e) => {
                        error = Some(e);
                        break;
                    },
                };
                if let Err(e) = check_columns(&df, &projected) {
                    error = Some(e);
                    break;
                }
                rows_read += df.height() as u64;
                let df = slice.apply(df);
                let mut offset = 0;
                while offset < df.height() {
                    let chunk = df.slice(offset as i64, morsel_size);
                    offset += chunk.height();
                    let morsel =
                        Morsel::new_unregistered(chunk, MorselSeq::new(seq), source_token.clone());
                    seq += 1;
                    if tx.send_morsel(morsel).await.is_err() || source_token.stop_requested() {
                        stopped = true;
                        break 'outer;
                    }
                }
                if slice.done() {
                    stopped = true;
                    break;
                }
            }

            if !(stopped || error.is_some()) {
                cancel_guard.0 = None;
            }
            // Stops the read now, while the fetch thread may still be blocked in the plugin.
            drop(cancel_guard);
            drop(batch_rx);
            fetch.join().await;

            if let Some(e) = error {
                return Err(e);
            }
            if let Some(tx) = row_position_on_end_tx {
                let end = match slice_end {
                    // Reading stopped at the end of the slice.
                    Some(end) if slice.done() => (end as u64).min(rows_read),
                    _ => rows_read,
                };
                let _ = tx.send(idxsize_try_from(end).unwrap_or(IdxSize::MAX));
            }
            Ok(())
        });

        Ok((rx, handle))
    }
}

/// Plugins return exactly the requested columns.
fn check_columns(df: &DataFrame, projected: &SchemaRef) -> PolarsResult<()> {
    polars_ensure!(
        df.width() == projected.len()
            && df
                .columns()
                .iter()
                .zip(projected.iter_names())
                .all(|(c, name)| c.name() == name),
        SchemaMismatch:
        "I/O plugin returned columns {:?}, expected {:?}",
        df.get_column_names(),
        projected.iter_names().collect::<Vec<_>>()
    );
    Ok(())
}
