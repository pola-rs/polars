//! `polars.io_plugin.file_reader.v1`: reading the partitions of a resolved database scan.
//!
//! Each source is a partition id. The SQL of every partition was rendered when the scan was
//! resolved; a reader only opens a session, runs that SQL and casts the batches to the planned
//! schema. The sessions of one scan execution are pooled ([`Shared`]), never more than the scan's
//! `max_connections`.
use std::collections::VecDeque;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use polars_arrow::array::{Array, StructArray};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_core::prelude::*;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_err};
use polars_io_ext_ffi::common::{
    ArrowArrayStream, ArrowSchema, FfiBuf, FfiResult, FfiSlice, FfiStr, FfiVec,
};
use polars_io_ext_ffi::file_reader_v1::{
    BeginRead, BeginReadArgs, BuilderHandle, Capabilities, Plugin, ReaderHandle, ScanHandle, flags,
};
use serde::{Deserialize, Serialize};

use crate::database::backend::{
    BatchIter, CancelHandle, DatabaseBackend, DatabaseSession, RETRY_BACKOFF_MS, is_transient,
    open_backend, open_session_with_retries, retry_wait,
};
use crate::database::config::CompiledPartition;
use crate::database::planner::check_unplanned_volatility;
use crate::database::schema::cast_to_schema;
use crate::database::{DatabaseScanConfig, catalog};
use crate::ffi::{
    arrow_fields, borrow_handle, clone_handle, export_schema, ffi_call, ffi_call_infallible,
    into_handle, release_handle,
};

/// Prefix of serialized builders, so that a change of the format fails clearly.
const SERDE_MAGIC: &[u8] = b"PLDBRB1";

/// The scan executions whose state ([`Shared`]) is alive in this process: their sessions may
/// still be read, stopped or closed.
static LIVE_SCANS: Mutex<usize> = Mutex::new(0);
static LIVE_SCANS_CHANGED: std::sync::Condvar = std::sync::Condvar::new();

/// Waits up to `timeout` for every scan execution to end, its sessions closed; whether they
/// did. A query can outlive the call that started it: `collect_batches()` runs on until it
/// notices that its iterator was dropped, and its readers then stop their statements.
pub fn wait_for_scans(timeout: std::time::Duration) -> bool {
    let live = LIVE_SCANS.lock().unwrap();
    let (live, _) = LIVE_SCANS_CHANGED
        .wait_timeout_while(live, timeout, |n| *n > 0)
        .unwrap();
    *live == 0
}

/// The cancel handles of every partition reader, to stop their statements at exit.
static RUNNING_STATEMENTS: Mutex<Vec<std::sync::Weak<Mutex<Option<CancelHandle>>>>> =
    Mutex::new(vec![]);

fn register_statement(cancel: &Arc<Mutex<Option<CancelHandle>>>) {
    let mut running = RUNNING_STATEMENTS.lock().unwrap();
    running.retain(|w| w.strong_count() > 0);
    running.push(Arc::downgrade(cancel));
}

/// Cancels the statement of every partition reader still running one: a thread inside a
/// driver call when Python finalizes is ended by Python 3.12 and 3.13 through Rust frames,
/// which aborts the process.
pub fn cancel_running_statements() {
    let running: Vec<_> = RUNNING_STATEMENTS
        .lock()
        .unwrap()
        .iter()
        .filter_map(|w| w.upgrade())
        .collect();
    for cancel in running {
        let handle = cancel.lock().unwrap().clone();
        if let Some(handle) = handle {
            let _ = handle();
        }
    }
}

/// The plan of a resolved scan: its config, with the SQL of every partition.
pub struct ReaderBuilder {
    config: Arc<DatabaseScanConfig>,
    /// The columns of the scan: the projected ones of the planned schema.
    schema: SchemaRef,
}

#[derive(Serialize, Deserialize)]
struct SerializedBuilder {
    config: Arc<DatabaseScanConfig>,
    schema: SchemaRef,
}

impl ReaderBuilder {
    pub fn new(config: Arc<DatabaseScanConfig>, schema: SchemaRef) -> Self {
        Self { config, schema }
    }

    fn fingerprint(&self) -> u64 {
        let mut h = polars_utils::aliases::PlFixedStateQuality::default().build_hasher();
        self.config.hash(&mut h);
        for name in self.schema.iter_names() {
            name.hash(&mut h);
        }
        h.finish()
    }

    fn serialize(&self) -> PolarsResult<Vec<u8>> {
        let mut out = SERDE_MAGIC.to_vec();
        out.extend(polars_utils::pl_serialize::serialize_to_bytes::<_, false>(
            &SerializedBuilder {
                config: self.config.clone(),
                schema: self.schema.clone(),
            },
        )?);
        Ok(out)
    }

    fn deserialize(bytes: &[u8]) -> PolarsResult<Self> {
        let bytes = bytes.strip_prefix(SERDE_MAGIC).ok_or_else(|| {
            polars_err!(
                ComputeError:
                "cannot deserialize a scan_database plan: it was made by an incompatible \
                version of {}", crate::PACKAGE_NAME
            )
        })?;
        let s: SerializedBuilder =
            polars_utils::pl_serialize::deserialize_from_reader::<_, _, false>(bytes)?;
        Ok(Self {
            config: s.config,
            schema: s.schema,
        })
    }

    fn name(&self) -> String {
        format!(
            "database[{}/{}]",
            self.config.dialect, self.config.backend.kind
        )
    }
}

/// A session leased from the scan's pool. Dropping it closes the session and frees its slot;
/// [`Shared::release`] keeps it open for the next partition instead.
struct Session {
    inner: Option<Box<dyn DatabaseSession>>,
    shared: Arc<Shared>,
}

impl std::ops::Deref for Session {
    type Target = dyn DatabaseSession;

    fn deref(&self) -> &Self::Target {
        self.inner.as_deref().unwrap()
    }
}

impl std::ops::DerefMut for Session {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_deref_mut().unwrap()
    }
}

impl Session {
    /// Closes the session and opens a new one in the same slot, so no other partition can
    /// take the slot in between.
    fn reconnect(&mut self) -> PolarsResult<()> {
        drop(self.inner.take());
        match self.shared.open_new() {
            Ok(s) => {
                self.inner = Some(s);
                Ok(())
            },
            Err(e) => {
                self.shared.free_slot();
                Err(e)
            },
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(s) = self.inner.take() {
            drop(s);
            self.shared.free_slot();
        }
    }
}

/// Open sessions of one scan: never more than `max_connections` (idle ones included).
///
/// The host starts at most `max_connections` readers at once (`max_concurrent_scans`, exact in
/// ordered mode too), and each holds one session at a time: a reader waits only for a session
/// that is being given back, never for one held by a reader whose output waits behind it.
#[derive(Default)]
struct Pool {
    idle: Vec<Box<dyn DatabaseSession>>,
    open: usize,
}

/// State shared by the readers of one scan execution.
pub struct Shared {
    config: Arc<DatabaseScanConfig>,
    /// Instantiated once per execution: re-checks that the planned backend exists here.
    backend: OnceLock<Arc<dyn DatabaseBackend>>,
    /// The outcome of [`check_unplanned_volatility`], run on the first session opened:
    /// whether the source turned out volatile.
    volatility: OnceLock<Result<bool, String>>,
    pool: Mutex<Pool>,
    pool_changed: std::sync::Condvar,
    /// Set when the scan is torn down: threads still waiting for a session give up instead
    /// of connecting for a query that is gone.
    aborted: AtomicBool,
}

impl Shared {
    fn new(config: Arc<DatabaseScanConfig>) -> Self {
        *LIVE_SCANS.lock().unwrap() += 1;
        Self {
            config,
            backend: OnceLock::new(),
            volatility: OnceLock::new(),
            pool: Mutex::new(Pool::default()),
            pool_changed: std::sync::Condvar::new(),
            aborted: AtomicBool::new(false),
        }
    }

    fn open_new(&self) -> PolarsResult<Box<dyn DatabaseSession>> {
        let backend = self.backend()?;
        let mut session = open_session_with_retries(
            backend.as_ref(),
            &self.config.options.connection,
            self.config.dialect,
        )?;
        let checked = self.volatility.get_or_init(|| {
            check_unplanned_volatility(session.as_mut(), &self.config).map_err(|e| match e {
                PolarsError::InvalidOperation(msg) => msg.to_string(),
                e => e.to_string(),
            })
        });
        if let Err(e) = checked {
            polars_bail!(InvalidOperation: "{e}");
        }
        Ok(session)
    }

    /// The source turned out volatile when read (`schema=`): it isn't repeatable.
    fn found_volatile(&self) -> bool {
        matches!(self.volatility.get(), Some(Ok(true)))
    }

    fn free_slot(&self) {
        self.pool.lock().unwrap().open -= 1;
        self.pool_changed.notify_all();
    }

    fn abort(&self) {
        self.aborted.store(true, Ordering::Relaxed);
        // under the lock, so a waiter can't miss the wakeup between its check and its wait
        let _pool = self.pool.lock().unwrap();
        self.pool_changed.notify_all();
    }

    fn backend(&self) -> PolarsResult<Arc<dyn DatabaseBackend>> {
        if let Some(b) = self.backend.get() {
            return Ok(b.clone());
        }
        let b = open_backend(&self.config.backend, self.config.options.fetch.batch_size)?;
        Ok(self.backend.get_or_init(|| b).clone())
    }

    /// An idle session, or a new one once fewer than `max_connections` are open. Blocks
    /// until then, or until the scan is aborted or the reader (`cancelled`) is cancelled.
    fn acquire(self: &Arc<Self>, cancelled: &AtomicBool) -> PolarsResult<Session> {
        let max = self.config.max_connections();
        let mut pool = self.pool.lock().unwrap();
        loop {
            if self.aborted.load(Ordering::Relaxed) {
                polars_bail!(ComputeError: "the database scan was stopped");
            }
            if cancelled.load(Ordering::Relaxed) {
                polars_bail!(ComputeError: "the database read was cancelled");
            }
            if let Some(s) = pool.idle.pop() {
                return Ok(Session {
                    inner: Some(s),
                    shared: self.clone(),
                });
            }
            if pool.open < max {
                pool.open += 1;
                drop(pool);
                return match self.open_new() {
                    Ok(s) => Ok(Session {
                        inner: Some(s),
                        shared: self.clone(),
                    }),
                    Err(e) => {
                        self.free_slot();
                        Err(e)
                    },
                };
            }
            pool = self.pool_changed.wait(pool).unwrap();
        }
    }

    /// Keeps a session open for the next partition.
    fn release(&self, mut session: Session) {
        if let Some(s) = session.inner.take() {
            self.pool.lock().unwrap().idle.push(s);
            self.pool_changed.notify_all();
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // Closed before the scan counts as ended (`wait_for_scans`).
        drop(std::mem::take(&mut self.pool.get_mut().unwrap().idle));
        drop(self.backend.take());
        *LIVE_SCANS.lock().unwrap() -= 1;
        LIVE_SCANS_CHANGED.notify_all();
    }
}

/// One execution of a scan. Releasing it stops readers that wait for a session.
struct Scan {
    shared: Arc<Shared>,
    schema: SchemaRef,
}

impl Drop for Scan {
    fn drop(&mut self) {
        self.shared.abort();
    }
}

/// The reader of one partition.
pub struct PartitionReader {
    shared: Arc<Shared>,
    /// The columns of the scan.
    schema: SchemaRef,
    index: usize,
    partition: Arc<CompiledPartition>,
    n_rows: Mutex<Option<u64>>,
    /// The rows of a source that isn't repeatable, read to count them: the read returns
    /// these rows, the ones the count (and the slices planned from it) saw.
    counted_rows: Mutex<Option<DataFrame>>,
    /// The running statement's cancel handle; replaced when a retry reconnects.
    cancel: Arc<Mutex<Option<CancelHandle>>>,
    cancelled: Arc<AtomicBool>,
    /// The error of the last failed batch, with its kind (`reader_take_error`).
    error: Arc<Mutex<Option<PolarsError>>>,
}

impl PartitionReader {
    fn context(&self) -> String {
        format!(
            "database partition {} via {}",
            self.index, self.shared.config.backend.kind
        )
    }

    /// Runs the partition's `COUNT(*)`. A source that isn't repeatable (`TABLESAMPLE`,
    /// `LIMIT` without `ORDER BY`, …) returns other rows each time: its rows are read and
    /// kept for the read instead.
    fn count(&self) -> PolarsResult<u64> {
        if let Some(n) = *self.n_rows.lock().unwrap() {
            return Ok(n);
        }
        let repeatable = self
            .shared
            .config
            .compiled
            .as_ref()
            .is_none_or(|c| c.repeatable);
        let mut session = self.shared.acquire(&self.cancelled)?;
        let repeatable = repeatable && !self.shared.found_volatile();
        {
            // cancellable as the read is (see `PartitionStream::start`)
            let mut cancel = self.cancel.lock().unwrap();
            if self.cancelled.load(Ordering::Relaxed) {
                drop(cancel);
                self.shared.release(session);
                polars_bail!(ComputeError: "the database read was cancelled");
            }
            *cancel = session.cancel_handle();
        }
        let r = if repeatable {
            session
                .query(&self.partition.count_sql)
                .and_then(|df| catalog::scalar_u64(&df))
        } else {
            self.read_counted(&mut session).map(|df| {
                let n = df.height() as u64;
                *self.counted_rows.lock().unwrap() = Some(df);
                n
            })
        };
        // Back to the pool: holding it until this partition is read would let readers that
        // are counted out of order (a negative slice counts from the end) take every
        // session while the partition that is read next waits. A failed session may be in
        // any state: it is closed, as is one whose statement was cancelled. Closed here,
        // before the count returns (see `PartitionStream::drop`).
        let cancelled = {
            let mut cancel = self.cancel.lock().unwrap();
            *cancel = None;
            self.cancelled.load(Ordering::Relaxed)
        };
        if r.is_ok() && !cancelled {
            self.shared.release(session);
        } else {
            drop(session);
        }
        let n = r.map_err(|e| out_of_range_date_hint(e).context(self.context().into()))?;
        // A cancel that came as the statement started may have found nothing to stop.
        if cancelled {
            polars_bail!(ComputeError: "the database read was cancelled");
        }
        *self.n_rows.lock().unwrap() = Some(n);
        Ok(n)
    }

    /// The partition's rows, each batch cast to the planned schema as the read casts
    /// it: drivers type each batch on its own (a decimal's scale, a time zone), and
    /// batches of other dtypes can't be stacked.
    fn read_counted(&self, session: &mut Session) -> PolarsResult<DataFrame> {
        let config = &self.shared.config;
        let overrides = config.options.schema_overrides.as_deref();
        let mut out: Option<DataFrame> = None;
        for batch in session.execute(&self.partition.sql, config.options.fetch.batch_size)? {
            let batch = batch?;
            let names = batch
                .get_column_names()
                .into_iter()
                .filter(|n| config.schema.contains(n))
                .cloned()
                .collect::<Vec<_>>();
            let batch = crate::database::planner::decode_sqlite_reals(batch, config)?;
            let batch = cast_to_schema(&batch, &config.schema.try_project(&names)?, overrides)?;
            match &mut out {
                None => out = Some(batch),
                Some(acc) => {
                    acc.vstack_mut_owned(batch)?;
                },
            }
        }
        out.ok_or_else(|| {
            polars_err!(ComputeError: "database query returned no result: {}", self.partition.sql)
        })
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
        // Wakes the reader if it waits for a session.
        {
            let _pool = self.shared.pool.lock().unwrap();
            self.shared.pool_changed.notify_all();
        }
        // Under the lock that the stream takes to give its session back: a session is never
        // returned to the pool while its statement is being cancelled.
        let cancel = self.cancel.lock().unwrap();
        if let Some(cancel) = &*cancel
            && let Err(e) = cancel()
            && polars_core::config::verbose()
        {
            eprintln!("[scan_database]: cancelling a statement failed: {e}");
        }
    }

    fn begin_read(&self, args: &BeginRead) -> PolarsResult<polars_arrow::ffi::ArrowArrayStream> {
        let columns: Vec<PlSmallStr> = args.columns.iter().map(|c| c.as_str().into()).collect();
        let projected = self.shared.config.schema.try_project(&columns)?;
        let field = ArrowField::new(
            PlSmallStr::EMPTY,
            ArrowDataType::Struct(arrow_fields(&projected)),
            false,
        );
        *self.error.lock().unwrap() = None;
        let config = &self.shared.config;
        let stream = PartitionStream {
            shared: self.shared.clone(),
            sql: self.partition.sql.clone(),
            batch_size: config.options.fetch.batch_size,
            // Reconnecting to a live connection gives back the same connection: no retries.
            live: config.options.connection.is_live(),
            // A source that isn't repeatable was counted from the rows it returns (if this
            // read takes them); a pushed LIMIT counts more rows than it reads.
            limited: config
                .compiled
                .as_ref()
                .is_some_and(|c| c.limit.is_some() || !c.repeatable)
                || self.shared.found_volatile(),
            expected_rows: *self.n_rows.lock().unwrap(),
            counted_rows: self.counted_rows.lock().unwrap().take(),
            context: self.context(),
            projected,
            dtype: field.dtype().clone(),
            cancel: self.cancel.clone(),
            cancelled: self.cancelled.clone(),
            error: self.error.clone(),
            batches: None,
            session: None,
            pending: VecDeque::new(),
            attempt: 0,
            sent: false,
            rows_read: 0,
            done: false,
            complete: false,
        };
        Ok(polars_arrow::ffi::export_iterator(Box::new(stream), field))
    }
}

/// The batches of one partition, pulled by the host through an Arrow C stream.
struct PartitionStream {
    shared: Arc<Shared>,
    sql: String,
    batch_size: Option<usize>,
    live: bool,
    limited: bool,
    /// The partition's `COUNT(*)`, if the host asked for it.
    expected_rows: Option<u64>,
    /// The rows read by the count of a source that isn't repeatable, returned as they are.
    counted_rows: Option<DataFrame>,
    context: String,
    projected: Schema,
    dtype: ArrowDataType,
    cancel: Arc<Mutex<Option<CancelHandle>>>,
    cancelled: Arc<AtomicBool>,
    error: Arc<Mutex<Option<PolarsError>>>,
    // Field order is drop order: the result is released before its session.
    batches: Option<BatchIter>,
    session: Option<Session>,
    pending: VecDeque<Box<dyn Array>>,
    attempt: usize,
    sent: bool,
    rows_read: u64,
    done: bool,
    /// The statement ran to its end.
    complete: bool,
}

impl PartitionStream {
    fn retryable(&self, e: &PolarsError) -> bool {
        !self.live && !self.sent && self.attempt < RETRY_BACKOFF_MS.len() && is_transient(e)
    }

    fn check_cancelled(&self) -> PolarsResult<()> {
        if self.cancelled.load(Ordering::Relaxed) {
            polars_bail!(ComputeError: "the database read was cancelled");
        }
        Ok(())
    }

    /// Starts the statement, on a new session from the pool.
    fn start(&mut self) -> PolarsResult<()> {
        let mut session = self.shared.acquire(&self.cancelled)?;
        {
            let mut cancel = self.cancel.lock().unwrap();
            // Cancelled while taking the session: nothing ran on it.
            if self.cancelled.load(Ordering::Relaxed) {
                drop(cancel);
                self.shared.release(session);
                polars_bail!(ComputeError: "the database read was cancelled");
            }
            *cancel = session.cancel_handle();
        }
        let batches = session.execute(&self.sql, self.batch_size);
        self.session = Some(session);
        self.batches = Some(batches?);
        // A cancel between publishing the handle and the statement running had nothing to
        // stop (the session also refuses to start one once cancelled).
        self.check_cancelled()
    }

    /// Nothing was sent: retries on a new session, in the same pool slot.
    fn retry(&mut self, e: &PolarsError) -> PolarsResult<()> {
        self.batches = None;
        // The old cancel handle holds the old connection open: drop it first.
        *self.cancel.lock().unwrap() = None;
        retry_wait(self.attempt, e);
        self.attempt += 1;
        self.check_cancelled()?;
        let session = self.session.as_mut().unwrap();
        // `reconnect` retries connecting itself.
        if let Err(e) = session.reconnect() {
            // `reconnect` freed the slot already
            self.session = None;
            return Err(e);
        }
        *self.cancel.lock().unwrap() = session.cancel_handle();
        // A cancel between the check above and setting the handle had nothing to stop.
        if self.cancelled.load(Ordering::Relaxed) {
            polars_bail!(ComputeError: "the database read was cancelled");
        }
        self.batches = Some(session.execute(&self.sql, self.batch_size)?);
        self.check_cancelled()
    }

    fn cast(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let config = &self.shared.config;
        let overrides = config.options.schema_overrides.as_deref();
        let df = crate::database::planner::decode_sqlite_reals(df.clone(), config)?;
        cast_to_schema(&df, &self.projected, overrides)
    }

    /// The next non-empty batch, cast to the planned schema; `None` at the end.
    fn next_batch(&mut self) -> PolarsResult<Option<DataFrame>> {
        self.check_cancelled()?;
        if let Some(df) = self.counted_rows.take() {
            self.complete = true;
            self.sent = true;
            self.rows_read = df.height() as u64;
            if df.height() == 0 {
                return Ok(None);
            }
            return self.cast(&df).map(Some);
        }
        if self.complete {
            return Ok(None);
        }
        if self.session.is_none() {
            if let Err(e) = self.start() {
                if !self.retryable(&e) || self.session.is_none() {
                    return Err(e);
                }
                self.retry(&e)?;
            }
        }
        loop {
            self.check_cancelled()?;
            let Some(batch) = self.batches.as_mut().unwrap().next() else {
                self.complete = true;
                return Ok(None);
            };
            // The first `next` is where a Python statement starts: a cancel that landed
            // before it is stopped here, if the session didn't refuse the statement.
            self.check_cancelled()?;
            let batch = match batch {
                Ok(b) => b,
                Err(e) if self.retryable(&e) => {
                    self.retry(&e)?;
                    continue;
                },
                Err(e) => return Err(e),
            };
            self.sent = true;
            self.rows_read += batch.height() as u64;
            if batch.height() == 0 {
                continue;
            }
            return self.cast(&batch).map(Some);
        }
    }

    fn next_array(&mut self) -> PolarsResult<Option<Box<dyn Array>>> {
        loop {
            self.check_cancelled()?;
            if let Some(a) = self.pending.pop_front() {
                return Ok(Some(a));
            }
            let Some(mut df) = self.next_batch()? else {
                // The count and the rows come from separate statements. Slices were planned
                // from the count: if the rows changed in between, they took the wrong rows.
                if !self.limited
                    && let Some(n) = self.expected_rows
                    && self.rows_read != n
                {
                    polars_bail!(
                        ComputeError:
                        "the partition returned {} rows, but COUNT(*) found {n}: the table \
                        changed during the scan", self.rows_read
                    );
                }
                return Ok(None);
            };
            df.rechunk_mut();
            for batch in df.iter_chunks(CompatLevel::newest(), false) {
                let len = batch.height();
                let arrays = batch.into_arrays();
                self.pending
                    .push_back(StructArray::new(self.dtype.clone(), len, arrays, None).to_boxed());
            }
        }
    }
}

/// Python's `datetime` holds years 1..9999 only: a Python driver fails on an Oracle BC
/// date with a bare `ValueError`. Says which values and what to do.
fn out_of_range_date_hint(e: PolarsError) -> PolarsError {
    let msg = e.to_string();
    if !(msg.contains("year must be in 1..9999")
        || (msg.contains("year") && msg.contains("is out of range")))
    {
        return e;
    }
    e.wrap_msg(|m| {
        format!(
            "{m}\n\nthe result has a date or timestamp outside years 1..9999 (e.g. a BC \
            date), which the Python driver can't convert; read the column as text in a query \
            (e.g. Oracle TO_CHAR(col, 'SYYYY-MM-DD HH24:MI:SS')), or with another driver"
        )
    })
}

impl Iterator for PartitionStream {
    type Item = PolarsResult<Box<dyn Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // Called from the stream's `extern "C"` `get_next`: a panic must not unwind.
        let next = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.next_array()))
            .unwrap_or_else(|payload| {
                Err(polars_err!(
                    ComputeError: "{} panicked: {}",
                    crate::PACKAGE_NAME, crate::ffi::panic_message(&*payload)
                ))
            });
        match next {
            Ok(Some(a)) => Some(Ok(a)),
            Ok(None) => {
                self.done = true;
                None
            },
            Err(e) => {
                self.done = true;
                let e = out_of_range_date_hint(e).context(self.context.clone().into());
                *self.error.lock().unwrap() = Some(e.clone());
                Some(Err(e))
            },
        }
    }
}

impl Drop for PartitionStream {
    fn drop(&mut self) {
        // Under the cancel lock: a concurrent `reader_cancel` either finished cancelling or
        // finds no statement, and a cancelled read never gives its session back to the pool.
        let (cancel, complete) = {
            let mut cancel = self.cancel.lock().unwrap();
            let complete = self.complete
                && self.error.lock().unwrap().is_none()
                && !self.cancelled.load(Ordering::Relaxed);
            (cancel.take(), complete)
        };
        let batches = self.batches.take();
        let session = self.session.take();
        if complete {
            drop(batches);
            if let Some(session) = session {
                self.shared.release(session);
            }
        } else {
            // Released before the end: stop the statement, so the database doesn't keep
            // working for nobody. A cancelled or failed session may be in any state: it is
            // closed, not reused.
            //
            // Both are done here, before the stream's release returns, even though they may
            // block: the host releases the stream on the blocking thread that read it, and
            // its query waits for that thread. Left to a thread of its own, the cleanup
            // outlived `collect()`: a pooled connection went back to the user's pool late,
            // and driver calls made while Python finalized aborted the process (psycopg2,
            // MySQL over ADBC). The statement is cancelled first, so that closing it doesn't
            // wait for the rest of its result.
            if batches.is_some()
                && let Some(cancel) = cancel
                && let Err(e) = cancel()
                && polars_core::config::verbose()
            {
                eprintln!("[scan_database]: cancelling a statement failed: {e}");
            }
            drop(batches);
            drop(session);
        }
    }
}

unsafe fn builder<'a>(h: BuilderHandle) -> &'a ReaderBuilder {
    unsafe { borrow_handle::<ReaderBuilder>(h.0) }
}

unsafe fn reader<'a>(h: ReaderHandle) -> &'a PartitionReader {
    unsafe { borrow_handle::<PartitionReader>(h.0) }
}

unsafe extern "C" fn builder_deserialize(bytes: FfiSlice<'_, u8>) -> FfiResult<BuilderHandle> {
    ffi_call(|| {
        Ok(BuilderHandle(into_handle(Arc::new(
            ReaderBuilder::deserialize(bytes.as_slice())?,
        ))))
    })
}

unsafe extern "C" fn builder_serialize(h: BuilderHandle) -> FfiResult<FfiBuf> {
    ffi_call(|| Ok(FfiBuf::from_vec(unsafe { builder(h) }.serialize()?)))
}

unsafe extern "C" fn builder_clone(h: BuilderHandle) -> BuilderHandle {
    BuilderHandle(unsafe { clone_handle::<ReaderBuilder>(h.0) })
}

unsafe extern "C" fn builder_release(h: BuilderHandle) {
    ffi_call_infallible(|| {
        unsafe { release_handle::<ReaderBuilder>(h.0) };
        #[cfg(feature = "python")]
        crate::python::release_queued_objects();
    })
}

unsafe extern "C" fn builder_name(h: BuilderHandle) -> FfiBuf {
    ffi_call_infallible(|| FfiBuf::from_string(unsafe { builder(h) }.name()))
}

unsafe extern "C" fn builder_fingerprint(h: BuilderHandle) -> u64 {
    ffi_call_infallible(|| unsafe { builder(h) }.fingerprint())
}

unsafe extern "C" fn builder_explain(h: BuilderHandle) -> FfiResult<FfiVec<FfiBuf>> {
    ffi_call(|| {
        let props = unsafe { builder(h) }.config.explain_properties();
        Ok(FfiVec::from_vec(
            props
                .into_iter()
                .flat_map(|(k, v)| [FfiBuf::from_string(k.to_string()), FfiBuf::from_string(v)])
                .collect(),
        ))
    })
}

unsafe extern "C" fn builder_capabilities(h: BuilderHandle) -> FfiResult<Capabilities> {
    ffi_call(|| {
        let max = unsafe { builder(h) }.config.max_connections() as u64;
        Ok(Capabilities {
            flags: flags::UNORDERED | flags::SOURCES_REMOTE,
            max_concurrent_scans: Some(max).into(),
            // Readers take a session only to count or read, so pre-initializing is free, but
            // more readers than sessions only queue.
            preferred_remote_pre_init: Some(max).into(),
        })
    })
}

unsafe extern "C" fn scan_new(h: BuilderHandle) -> FfiResult<ScanHandle> {
    ffi_call(|| {
        let b = unsafe { builder(h) };
        let config = b.config.clone();
        polars_ensure_compiled(&config)?;
        Ok(ScanHandle(into_handle(Arc::new(Scan {
            schema: b.schema.clone(),
            shared: Arc::new(Shared::new(config)),
        }))))
    })
}

fn polars_ensure_compiled(config: &DatabaseScanConfig) -> PolarsResult<()> {
    if config.compiled.is_none() {
        polars_bail!(ComputeError: "database scan was not resolved");
    }
    Ok(())
}

unsafe extern "C" fn scan_release(h: ScanHandle) {
    ffi_call_infallible(|| {
        unsafe { release_handle::<Scan>(h.0) };
        #[cfg(feature = "python")]
        crate::python::release_queued_objects();
    })
}

unsafe extern "C" fn reader_new(h: ScanHandle, source: FfiStr<'_>) -> FfiResult<ReaderHandle> {
    ffi_call(|| {
        let scan = unsafe { borrow_handle::<Scan>(h.0) };
        let shared = scan.shared.clone();
        let id = source
            .as_str()
            .map_err(|e| polars_err!(ComputeError: "invalid source: {e}"))?;
        let compiled = shared.config.compiled.as_ref().unwrap();
        // An absolute index into the partitions, never a position in the current sources.
        let (index, partition) = id
            .parse::<usize>()
            .ok()
            .and_then(|i| compiled.partitions.get(i).map(|p| (i, p.clone())))
            .ok_or_else(|| {
                polars_err!(
                    ComputeError:
                    "invalid database partition id '{id}' ({} partitions)",
                    compiled.partitions.len()
                )
            })?;
        let cancel = Arc::new(Mutex::new(None));
        register_statement(&cancel);
        Ok(ReaderHandle(into_handle(Arc::new(PartitionReader {
            shared,
            schema: scan.schema.clone(),
            index,
            partition: Arc::new(partition),
            n_rows: Mutex::new(None),
            counted_rows: Mutex::new(None),
            cancel,
            cancelled: Arc::new(AtomicBool::new(false)),
            error: Arc::new(Mutex::new(None)),
        }))))
    })
}

unsafe extern "C" fn reader_release(h: ReaderHandle) {
    ffi_call_infallible(|| unsafe { release_handle::<PartitionReader>(h.0) })
}

unsafe extern "C" fn reader_schema(h: ReaderHandle) -> FfiResult<ArrowSchema> {
    ffi_call(|| Ok(export_schema(&unsafe { reader(h) }.schema)))
}

unsafe extern "C" fn reader_n_rows(h: ReaderHandle) -> FfiResult<u64> {
    ffi_call(|| unsafe { reader(h) }.count())
}

unsafe extern "C" fn reader_begin_read(
    h: ReaderHandle,
    args: *const BeginReadArgs<'_>,
) -> FfiResult<ArrowArrayStream> {
    ffi_call(|| {
        let args = BeginRead::from_ffi(unsafe { &*args })
            .map_err(|e| polars_err!(ComputeError: "invalid read arguments: {e}"))?;
        let stream = unsafe { reader(h) }.begin_read(&args)?;
        // SAFETY: the contract's struct has the C stream interface layout.
        Ok(unsafe { ArrowArrayStream::transmute_from(stream) })
    })
}

unsafe extern "C" fn reader_cancel(h: ReaderHandle) {
    ffi_call_infallible(|| unsafe { reader(h) }.cancel())
}

unsafe extern "C" fn reader_take_error(h: ReaderHandle) -> FfiResult<()> {
    ffi_call(|| match unsafe { reader(h) }.error.lock().unwrap().take() {
        Some(e) => Err(e),
        None => Ok(()),
    })
}

pub static PLUGIN_V1: Plugin = Plugin {
    builder_deserialize,
    builder_serialize,
    builder_clone,
    builder_release,
    builder_name,
    builder_fingerprint,
    builder_explain,
    builder_capabilities,
    scan_new,
    scan_release,
    reader_new,
    reader_release,
    reader_schema,
    reader_n_rows,
    reader_begin_read,
    reader_cancel,
    reader_take_error,
};
