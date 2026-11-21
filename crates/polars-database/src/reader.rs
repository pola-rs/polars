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
use crate::database::dialect::DialectName;
use crate::database::planner::{
    check_mysql_session_zone, check_time_with_zone, check_unplanned_volatility,
    time_text_risk_columns, truncating_columns,
};
use crate::database::schema::cast_columns;
use crate::database::{DatabaseScanConfig, catalog};
use crate::ffi::{
    arrow_fields, borrow_handle, clone_handle, export_schema, ffi_call, ffi_call_infallible,
    into_handle, release_handle,
};

/// Kind prefix of serialized builders (see [`crate::serde_header`]).
const SERDE_KIND: &str = "PLDBRB";

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
    /// [`Self::compute_fingerprint`], taken once: it hashes every partition's SQL, and is asked
    /// on every comparison of plans.
    fingerprint: u64,
}

#[derive(Serialize, Deserialize)]
struct SerializedBuilder {
    config: Arc<DatabaseScanConfig>,
    schema: SchemaRef,
}

impl ReaderBuilder {
    pub fn new(config: Arc<DatabaseScanConfig>, schema: SchemaRef) -> Self {
        let fingerprint = Self::compute_fingerprint(&config, &schema);
        Self {
            config,
            schema,
            fingerprint,
        }
    }

    fn compute_fingerprint(config: &DatabaseScanConfig, schema: &SchemaRef) -> u64 {
        let mut h = polars_utils::aliases::PlFixedStateQuality::default().build_hasher();
        config.hash(&mut h);
        for name in schema.iter_names() {
            name.hash(&mut h);
        }
        h.finish()
    }

    fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    fn serialize(&self) -> PolarsResult<Vec<u8>> {
        let mut out = crate::serde_header(SERDE_KIND);
        out.extend(polars_utils::pl_serialize::serialize_to_bytes::<_, false>(
            &SerializedBuilder {
                config: self.config.clone(),
                schema: self.schema.clone(),
            },
        )?);
        Ok(out)
    }

    fn deserialize(bytes: &[u8]) -> PolarsResult<Self> {
        let bytes = crate::strip_serde_header(bytes, SERDE_KIND)?;
        let s: SerializedBuilder =
            polars_utils::pl_serialize::deserialize_from_reader::<_, _, false>(bytes)?;
        Ok(Self::new(s.config, s.schema))
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
    /// The columns that [`decode_sqlite_reals`](crate::database::planner::decode_sqlite_reals)
    /// decodes.
    sqlite_text_columns: Vec<PlSmallStr>,
    /// [`mysql_timestamp_text_columns`](crate::database::planner::mysql_timestamp_text_columns)
    mysql_timestamp_text: Vec<PlSmallStr>,
    /// [`time_text_risk_columns`]
    time_text_risk: PlHashMap<PlSmallStr, String>,
    /// [`truncating_columns`]
    truncating: PlHashSet<PlSmallStr>,
    /// The SQLite Binary columns whose batches are checked for blobs only.
    sqlite_blob_columns: Vec<(PlSmallStr, bool)>,
    /// [`numeric_temporal_conversions`](crate::database::planner::numeric_temporal_conversions)
    own_numbers: PlHashMap<PlSmallStr, DataType>,
    /// [`converted_columns`](crate::database::planner::converted_columns)
    converted: PlHashSet<PlSmallStr>,
    /// Instantiated once per execution: re-checks that the planned backend exists here.
    backend: OnceLock<Arc<dyn DatabaseBackend>>,
    /// The outcome of [`check_unplanned_volatility`], run on the first session opened:
    /// whether the source turned out volatile.
    volatility: OnceLock<Result<bool, String>>,
    /// The outcome of [`check_mysql_session_zone`], run on the first session opened.
    session_zone: OnceLock<Result<(), String>>,
    /// The first session's resolution of the names, for a source planned without one.
    resolution_baseline: Mutex<Option<Vec<String>>>,
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
            sqlite_text_columns: crate::database::planner::sqlite_text_columns(&config),
            sqlite_blob_columns: crate::database::planner::sqlite_blob_columns(&config),
            mysql_timestamp_text: crate::database::planner::mysql_timestamp_text_columns(&config),
            time_text_risk: time_text_risk_columns(&config),
            truncating: truncating_columns(&config),
            own_numbers: crate::database::planner::numeric_temporal_conversions(&config),
            converted: crate::database::planner::converted_columns(&config),
            config,
            backend: OnceLock::new(),
            volatility: OnceLock::new(),
            session_zone: OnceLock::new(),
            resolution_baseline: Mutex::new(None),
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
        crate::database::planner::check_resolution(
            session.as_mut(),
            &self.config,
            &self.resolution_baseline,
        )?;
        let zone = self.session_zone.get_or_init(|| {
            check_mysql_session_zone(session.as_mut(), &self.config).map_err(|e| match e {
                PolarsError::InvalidOperation(msg) => msg.to_string(),
                e => e.to_string(),
            })
        });
        if let Err(e) = zone {
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

/// How much of a non-repeatable source's rows a count keeps for the read; a bigger result is
/// counted and dropped.
const COUNT_BUFFER_BYTES: usize = 256 << 20;

/// [`COUNT_BUFFER_BYTES`], or `POLARS_DATABASE_COUNT_BUFFER_BYTES` (a test hook).
fn count_buffer_bytes() -> PolarsResult<usize> {
    match std::env::var("POLARS_DATABASE_COUNT_BUFFER_BYTES") {
        Ok(v) => v.trim().parse().map_err(|_| {
            polars_err!(
                ComputeError:
                "POLARS_DATABASE_COUNT_BUFFER_BYTES must be a number of bytes, got '{v}'"
            )
        }),
        Err(_) => Ok(COUNT_BUFFER_BYTES),
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
    /// The count of a source that isn't repeatable passed `COUNT_BUFFER_BYTES`: its rows weren't
    /// kept, so the read runs the query again and must find as many rows.
    count_unbuffered: Mutex<bool>,
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
        let config = &self.shared.config;
        let r = if repeatable
            && config.dialect == DialectName::Sqlite
            && !config.options.connection.opened_by_plugin()
        {
            // The user's connection may have redefined `count` (`create_aggregate`), and a wrong
            // count slices the wrong rows without failing: count the rows themselves.
            let sql = format!(
                "SELECT 1 FROM ({}) pl_count",
                crate::database::planner::wrappable(&self.partition.sql)
            );
            session
                .execute(&sql, config.options.fetch.batch_size)
                .and_then(|batches| {
                    let mut n = 0u64;
                    for batch in batches {
                        n += batch?.height() as u64;
                    }
                    Ok(n)
                })
        } else if repeatable {
            session
                .query(&self.partition.count_sql)
                .and_then(|df| catalog::scalar_u64(&df))
        } else {
            self.read_counted(&mut session).map(|(n, df)| {
                *self.count_unbuffered.lock().unwrap() = df.is_none();
                *self.counted_rows.lock().unwrap() = df;
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
        let n = r.map_err(|e| {
            crate::database::planner::explain_sqlite_error(
                &self.shared.config,
                out_of_range_date_hint(e),
            )
            .context(self.context().into())
        })?;
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
    ///
    /// Returns the number of rows, and the rows unless they passed `COUNT_BUFFER_BYTES`: a result
    /// that big is counted and dropped (the read runs the query again, and checks that it finds as
    /// many rows), so a `tail(n)` can't hold the whole result. A source that may write
    /// ([`source_may_write`](crate::database::planner::source_may_write)) keeps all its rows.
    fn read_counted(&self, session: &mut Session) -> PolarsResult<(u64, Option<DataFrame>)> {
        let config = &self.shared.config;
        let overrides = config.options.schema_overrides.as_deref();
        let mut out: Option<DataFrame> = None;
        let (mut rows, mut kept_bytes, mut dropped) = (0u64, 0usize, false);
        let buffer_bytes = count_buffer_bytes()?;
        let batches = session.execute(&self.partition.sql, config.options.fetch.batch_size)?;
        let (mut checked, mut keep_all) = (false, false);
        for batch in batches {
            let batch = batch?;
            // (a Python statement runs at the first batch: its description is known now)
            if !std::mem::replace(&mut checked, true) {
                check_time_with_zone(&**session, config)?;
            }
            let names = batch
                .get_column_names()
                .into_iter()
                .filter(|n| config.schema.contains(n))
                .cloned()
                .collect::<Vec<_>>();
            crate::database::planner::check_sqlite_blobs(&batch, &self.shared.sqlite_blob_columns)?;
            let batch = crate::database::planner::decode_sqlite_reals(
                batch,
                &self.shared.sqlite_text_columns,
            )?;
            let batch = crate::database::planner::as_utc_instants(
                batch,
                &self.shared.mysql_timestamp_text,
            )?;
            let batch = cast_columns(
                &batch,
                &config.schema.try_project(&names)?,
                overrides,
                config.user_schema,
                &self.shared.own_numbers,
                &self.shared.converted,
                &self.shared.time_text_risk,
                &self.shared.truncating,
            )?;
            rows += batch.height() as u64;
            if dropped {
                continue;
            }
            kept_bytes += batch.estimated_size(false);
            if kept_bytes > buffer_bytes && !keep_all {
                // a source that may write must not run again: all its rows are kept
                if crate::database::planner::source_may_write(config, self.shared.found_volatile())
                {
                    keep_all = true;
                } else {
                    dropped = true;
                    out = None;
                    continue;
                }
            }
            match &mut out {
                None => out = Some(batch),
                Some(acc) => {
                    acc.vstack_mut_owned(batch)?;
                },
            }
        }
        if dropped {
            return Ok((rows, None));
        }
        let df = out.ok_or_else(|| {
            polars_err!(ComputeError: "database query returned no result: {}", self.partition.sql)
        })?;
        Ok((rows, Some(df)))
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
            must_match_count: *self.count_unbuffered.lock().unwrap(),
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
            zone_checked: false,
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
    /// The rows of a count weren't kept: this read must find as many (see `count_unbuffered`).
    must_match_count: bool,
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
    /// [`check_time_with_zone`] ran on this statement.
    zone_checked: bool,
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

    fn cast(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        let config = &self.shared.config;
        let overrides = config.options.schema_overrides.as_deref();
        crate::database::planner::check_sqlite_blobs(&df, &self.shared.sqlite_blob_columns)?;
        let df =
            crate::database::planner::decode_sqlite_reals(df, &self.shared.sqlite_text_columns)?;
        let df = crate::database::planner::as_utc_instants(df, &self.shared.mysql_timestamp_text)?;
        cast_columns(
            &df,
            &self.projected,
            overrides,
            config.user_schema,
            &self.shared.own_numbers,
            &self.shared.converted,
            &self.shared.time_text_risk,
            &self.shared.truncating,
        )
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
            // (decoded by `read_counted` already: decoding twice would unescape twice)
            let overrides = self.shared.config.options.schema_overrides.as_deref();
            let user_schema = self.shared.config.user_schema;
            let own_numbers = &self.shared.own_numbers;
            let converted = &self.shared.converted;
            let df =
                crate::database::planner::as_utc_instants(df, &self.shared.mysql_timestamp_text)?;
            return cast_columns(
                &df,
                &self.projected,
                overrides,
                user_schema,
                own_numbers,
                converted,
                &self.shared.time_text_risk,
                &self.shared.truncating,
            )
            .map(Some);
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
            // (a Python statement runs at the first batch: its description is known now)
            if !std::mem::replace(&mut self.zone_checked, true)
                && let Some(session) = &self.session
            {
                check_time_with_zone(&**session, &self.shared.config)?;
            }
            return self.cast(batch).map(Some);
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
                if (!self.limited || self.must_match_count)
                    && let Some(n) = self.expected_rows
                    && self.rows_read != n
                {
                    if self.must_match_count {
                        // only a source that isn't repeatable is counted and dropped
                        polars_bail!(
                            ComputeError:
                            "the source isn't repeatable (it returns other rows each time) \
                            and its result is larger than the count buffer ({} bytes), so the \
                            rows it returned when counted were dropped, and running it again for the \
                            read found {} rows where the count found {n}. Make the source \
                            repeatable: a LIMIT / OFFSET needs an ORDER BY on a unique key, and a \
                            TABLESAMPLE or a volatile function (random(), now()) has to go. Or \
                            raise POLARS_DATABASE_COUNT_BUFFER_BYTES",
                            count_buffer_bytes().unwrap_or(COUNT_BUFFER_BYTES), self.rows_read
                        );
                    }
                    polars_bail!(
                        ComputeError:
                        "the partition returned {} rows, but COUNT(*) found {n}: the table \
                        changed during the scan", self.rows_read
                    );
                }
                return Ok(None);
            };
            // (no copy where the columns' chunks already line up)
            df.align_chunks();
            for batch in df.iter_chunks(CompatLevel::newest(), false) {
                let len = batch.height();
                if len == 0 {
                    continue;
                }
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
                let e = crate::database::planner::explain_sqlite_error(
                    &self.shared.config,
                    out_of_range_date_hint(e),
                )
                .context(self.context.clone().into());
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
            count_unbuffered: Mutex::new(false),
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
