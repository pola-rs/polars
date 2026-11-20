//! Backends turn SQL into batches. One dialect can be served by several backends; the SQL is
//! the same whichever backend runs it.

pub mod adbc;
pub mod resolve;

use std::ffi::CString;
use std::fmt;
use std::sync::{Arc, OnceLock};

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_utils::pl_str::PlSmallStr;
use serde::{Deserialize, Serialize};

use super::connection::ConnectionSpec;
use super::dialect::DialectName;

/// Which backend moves the bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// An ADBC driver, found by Rust and loaded through Python's `adbc_driver_manager`.
    Adbc { driver: PlSmallStr },
    /// ConnectorX (Python package).
    ConnectorX,
    /// A Python DB-API (or vendor) driver module, e.g. `psycopg`, `oracledb`, `duckdb`.
    Python { module: PlSmallStr },
    /// SQLAlchemy, using the URL's dialect and driver.
    Sqlalchemy,
    /// A Python ODBC module: `arrow_odbc`, `turbodbc` or `pyodbc`.
    Odbc { module: PlSmallStr },
}

impl BackendKind {
    pub fn parse(s: &str) -> PolarsResult<Self> {
        let (head, tail) = s.split_once(':').unwrap_or((s, ""));
        Ok(match (head, tail) {
            ("adbc", d) if !d.is_empty() => Self::Adbc { driver: d.into() },
            ("connectorx", "") => Self::ConnectorX,
            ("sqlalchemy", "") => Self::Sqlalchemy,
            ("odbc", m) if !m.is_empty() => Self::Odbc { module: m.into() },
            ("python", m) if !m.is_empty() => Self::Python { module: m.into() },
            _ => polars_bail!(InvalidOperation: "invalid database backend: '{s}'"),
        })
    }

    /// Whether statements run in a session that later queries share, so that the dialect's
    /// session setup applies to them. ConnectorX and arrow-odbc open one per query.
    pub fn has_sessions(&self) -> bool {
        match self {
            Self::ConnectorX => false,
            Self::Odbc { module } => module != "arrow_odbc",
            _ => true,
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adbc { driver } => write!(f, "adbc:{driver}"),
            Self::ConnectorX => f.write_str("connectorx"),
            Self::Python { module } => write!(f, "python:{module}"),
            Self::Sqlalchemy => f.write_str("sqlalchemy"),
            Self::Odbc { module } => write!(f, "odbc:{module}"),
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BackendCapabilities: u16 {
        /// Produces Arrow, not rows.
        const ARROW_NATIVE = 1 << 0;
        /// Bounded-memory batches, not fetch-all.
        const STREAMING = 1 << 1;
        /// Batches are pulled without holding the GIL.
        const NO_GIL = 1 << 2;
        const CANCEL = 1 << 3;
        /// Needs a Python interpreter and a module on the machine that reads.
        const NEEDS_PYTHON = 1 << 4;
        /// Can run the per-session setup statements.
        const SESSION_SETUP = 1 << 5;
    }
}

/// The backend chosen during planning. Recorded in the plan and re-checked wherever a
/// connection is opened.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendChoice {
    pub kind: BackendKind,
    /// Informational; a different version on a worker is allowed.
    pub version: Option<PlSmallStr>,
    /// Where the driver was found (ADBC library or manifest), as a hint for other machines.
    pub location: Option<PlSmallStr>,
    /// Why this backend was chosen, for verbose output.
    pub reason: PlSmallStr,
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if let Some(v) = &self.version {
            write!(f, " {v}")?;
        }
        Ok(())
    }
}

/// Batches from one query. Blocking: pull it from a blocking thread.
pub type BatchIter = Box<dyn Iterator<Item = PolarsResult<DataFrame>> + Send>;

/// Cancels the running statement of a session, from another thread.
pub type CancelHandle = Arc<dyn Fn() -> PolarsResult<()> + Send + Sync>;

pub trait DatabaseBackend: Send + Sync {
    fn choice(&self) -> &BackendChoice;
    fn capabilities(&self) -> BackendCapabilities;
    /// Opens a session. Blocking.
    fn connect(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>>;
}

pub trait DatabaseSession: Send {
    /// Runs a statement and returns its batches. Blocking.
    fn execute(&mut self, sql: &str, batch_size: Option<usize>) -> PolarsResult<BatchIter>;

    /// Runs a statement whose result is ignored (session setup).
    fn execute_update(&mut self, sql: &str) -> PolarsResult<()> {
        for batch in self.execute(sql, None)? {
            batch?;
        }
        Ok(())
    }

    /// Runs a query and collects its result.
    fn query(&mut self, sql: &str) -> PolarsResult<DataFrame> {
        collect_batches(self.execute(sql, None)?, sql)
    }

    /// Runs a query of the plugin's own (a catalog lookup) and collects its result. It
    /// holds none of the placeholders of the user's query: its parameters (`params=`) aren't
    /// bound to it.
    fn catalog_query(&mut self, sql: &str) -> PolarsResult<DataFrame> {
        self.query(sql)
    }

    fn cancel_handle(&self) -> Option<CancelHandle> {
        None
    }
}

/// The batches of a query's result as one frame.
pub fn collect_batches(batches: BatchIter, sql: &str) -> PolarsResult<DataFrame> {
    let mut out: Option<DataFrame> = None;
    for batch in batches {
        let batch = batch?;
        match &mut out {
            None => out = Some(batch),
            Some(acc) => {
                // Python drivers type each batch on its own: a column that is all NULL in
                // one batch has the Null dtype there.
                let (acc_df, batch) = align_null_columns(std::mem::take(acc), batch)?;
                *acc = acc_df;
                acc.vstack_mut_owned(batch)?;
            },
        }
    }
    out.ok_or_else(|| polars_err!(ComputeError: "database query returned no result: {sql}"))
}

/// Availability of a Python-side backend.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub version: Option<PlSmallStr>,
    /// For ADBC pip packages: the driver library path.
    pub location: Option<PlSmallStr>,
}

/// Python-side drivers, provided by `polars-python`. Absent when Polars runs without Python.
pub trait PythonDriverBridge: Send + Sync {
    /// Whether a Python-side backend is installed. For [`BackendKind::Adbc`], whether an
    /// `adbc_driver_<name>` pip package is installed, with its library path.
    fn probe(&self, kind: &BackendKind) -> Option<ProbeResult>;

    /// The dialect and the Python module of a factory or live connection object. A factory is
    /// called for it, so ask once per scan.
    fn connection_info(&self, spec: &ConnectionSpec) -> PolarsResult<(DialectName, PlSmallStr)>;

    /// Opens a session with a Python-side backend.
    fn connect(
        &self,
        kind: &BackendKind,
        spec: &ConnectionSpec,
        dialect: DialectName,
        batch_size: Option<usize>,
    ) -> PolarsResult<Box<dyn DatabaseSession>>;

    /// Whether Python's `adbc_driver_manager` is installed, at a version that can load ADBC
    /// drivers for [`Self::adbc_connect`].
    fn has_adbc_driver_manager(&self) -> bool;

    /// Opens an ADBC connection through Python's `adbc_driver_manager`. `options` are the
    /// database options, `uri` first; `execute_options` are set on every statement.
    fn adbc_connect(
        &self,
        driver: &adbc::FoundDriver,
        options: &[(CString, CString)],
        dialect: DialectName,
        execute_options: &[(CString, CString)],
    ) -> PolarsResult<Box<dyn DatabaseSession>>;

    /// Evaluates the connection's credential provider, if any: `(user, password)`.
    fn credentials(
        &self,
        spec: &ConnectionSpec,
    ) -> PolarsResult<Option<(Option<String>, Option<String>)>>;
}

static PYTHON_BRIDGE: OnceLock<Arc<dyn PythonDriverBridge>> = OnceLock::new();

pub fn register_python_bridge(bridge: Arc<dyn PythonDriverBridge>) {
    let _ = PYTHON_BRIDGE.set(bridge);
}

pub fn python_bridge() -> Option<&'static Arc<dyn PythonDriverBridge>> {
    PYTHON_BRIDGE.get()
}

/// A backend served by the Python bridge.
pub struct PythonBackend {
    choice: BackendChoice,
    bridge: Arc<dyn PythonDriverBridge>,
    batch_size: Option<usize>,
}

impl DatabaseBackend for PythonBackend {
    fn choice(&self) -> &BackendChoice {
        &self.choice
    }

    fn capabilities(&self) -> BackendCapabilities {
        let mut caps = BackendCapabilities::NEEDS_PYTHON | BackendCapabilities::SESSION_SETUP;
        match &self.choice.kind {
            BackendKind::ConnectorX => caps |= BackendCapabilities::ARROW_NATIVE,
            BackendKind::Python { module }
                if matches!(
                    module.as_str(),
                    "duckdb" | "oracledb" | "mssql_python" | "snowflake" | "databricks"
                ) =>
            {
                caps |= BackendCapabilities::ARROW_NATIVE
                    | BackendCapabilities::STREAMING
                    | BackendCapabilities::CANCEL
            },
            BackendKind::Odbc { module } if module != "pyodbc" => {
                caps |= BackendCapabilities::ARROW_NATIVE | BackendCapabilities::STREAMING
            },
            _ => caps |= BackendCapabilities::STREAMING | BackendCapabilities::CANCEL,
        }
        if !self.choice.kind.has_sessions() {
            // Setup statements can't apply to later queries.
            caps.remove(BackendCapabilities::SESSION_SETUP);
        }
        caps
    }

    fn connect(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        self.bridge
            .connect(&self.choice.kind, spec, dialect, self.batch_size)
    }
}

/// Opens a session and runs the dialect's session setup (read-only transactions, UTC, the
/// application name, binary string comparisons). An optional setup statement the server
/// rejects is skipped: it only tightens session hygiene. A required one fails the session:
/// the pushed SQL was written for it, and would mean something else without it.
pub fn open_session(
    backend: &dyn DatabaseBackend,
    spec: &ConnectionSpec,
    dialect: DialectName,
) -> PolarsResult<Box<dyn DatabaseSession>> {
    let mut session = backend.connect(spec, dialect)?;
    if backend
        .capabilities()
        .contains(BackendCapabilities::SESSION_SETUP)
    {
        for stmt in dialect.session_setup() {
            match session.execute_update(stmt.sql) {
                Ok(()) => {},
                Err(e) if stmt.required => {
                    return Err(e.context(
                        format!(
                            "session setup '{}' failed; the {dialect} queries rely on it",
                            stmt.sql
                        )
                        .into(),
                    ));
                },
                Err(e) => {
                    if polars_core::config::verbose() {
                        eprintln!(
                            "[scan_database]: session setup '{}' failed, skipped: {e}",
                            stmt.sql
                        );
                    }
                },
            }
        }
    }
    Ok(session)
}

/// Casts columns that are `Null` on one side to the other side's dtype, numeric columns
/// typed apart to their supertype, and numbers next to text to text, so the frames stack.
fn align_null_columns(mut a: DataFrame, mut b: DataFrame) -> PolarsResult<(DataFrame, DataFrame)> {
    use polars_core::prelude::DataType;
    use polars_core::utils::try_get_supertype;

    let numeric = |d: &DataType| d.is_primitive_numeric() || d.is_decimal();

    if a.width() != b.width() {
        return Ok((a, b));
    }
    for i in 0..a.width() {
        let (da, db) = (
            a.columns()[i].dtype().clone(),
            b.columns()[i].dtype().clone(),
        );
        if da == db {
            continue;
        }
        if da == DataType::Null {
            let c = a.columns()[i].cast(&db)?;
            a.replace_column(i, c)?;
        } else if db == DataType::Null {
            let c = b.columns()[i].cast(&da)?;
            b.replace_column(i, c)?;
        } else if let Some(st) = numeric(&da)
            .then(|| numeric(&db).then(|| try_get_supertype(&da, &db).ok()))
            .flatten()
            .flatten()
        {
            // Numbers typed per batch (a SQLite column of integers, then floats; a
            // decimal's scale): both widened, as one result would be typed.
            let c = a.columns()[i].cast(&st)?;
            a.replace_column(i, c)?;
            let c = b.columns()[i].cast(&st)?;
            b.replace_column(i, c)?;
        } else if (da == DataType::String && numeric(&db))
            || (db == DataType::String && numeric(&da))
        {
            // A SQLite column of numbers and text, typed per batch: read as text
            let c = a.columns()[i].cast(&DataType::String)?;
            a.replace_column(i, c)?;
            let c = b.columns()[i].cast(&DataType::String)?;
            b.replace_column(i, c)?;
        }
    }
    Ok((a, b))
}

/// Whether a failed connect or query is worth retrying: lost or refused connections, busy
/// servers and listeners, connect timeouts. Authentication, permission and SQL errors fail
/// the same way again, and retrying a failed login can lock the account. Statement and lock
/// timeouts and cancelled queries are the user's limits: a retry would only defeat them.
pub fn is_transient(error: &polars_error::PolarsError) -> bool {
    // The plugin's own errors (bad arguments, values it can't read): never the server's.
    if matches!(error, polars_error::PolarsError::InvalidOperation(_)) {
        return false;
    }
    // Drivers echo their input (a DSN, which holds `connect_timeout=`; a value or name):
    // URIs are dropped, and the keywords below are phrases that input rarely holds.
    let msg = error
        .to_string()
        .to_ascii_lowercase()
        .split_whitespace()
        .filter(|w| !w.contains("://"))
        .collect::<Vec<_>>()
        .join(" ");
    const PERMANENT: &[&str] = &[
        "password",
        "authenticat",
        "access denied",
        "permission denied",
        "login failed",
        "ora-01017",
        "ora-28000",
        "28000",
        "28p01",
        "syntax error",
        "does not exist",
        "unknown database",
        // ConnectorX's own pool already waited ~30s, retrying each connect itself
        "timed out waiting for connection",
        // SQLAlchemy's pool waited its `pool_timeout` for a connection of the user's pool
        "queuepool limit",
        // cancelled statements, statement and lock timeouts: SQLSTATE 57014 (Postgres
        // `QueryCanceled`), 55P03, ODBC HYT00, ADBC's CANCELLED status, MySQL 1205 / 3024 /
        // 1317, Oracle ORA-01013 and python-oracledb's call timeout
        "57014",
        "55p03",
        "hyt00",
        "querycanceled",
        "canceling statement",
        "cancelled",
        "canceled",
        "statement timeout",
        "lock timeout",
        "lock wait timeout",
        "query timeout",
        "max_execution_time",
        "maximum statement execution time",
        "query execution was interrupted",
        "ora-01013",
        "dpy-4024",
    ];
    const TRANSIENT: &[&str] = &[
        "connection refused",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "could not connect",
        "can't connect",
        "cannot connect",
        "lost connection",
        "server closed the connection",
        "server has gone away",
        "too many connections",
        "too many clients",
        "the database system is starting up",
        "the database system is shutting down",
        // connect timeouts (libpq's "timeout expired", sockets, ODBC HYT01)
        "timed out",
        "timeout expired",
        "connect timeout",
        "connection timeout",
        "connecttimeout",
        "hyt01",
        "temporarily unavailable",
        "ora-12514",
        "ora-12516",
        "ora-12519",
        "ora-12520",
        "ora-12528",
        "ora-12541",
        "dpy-6000",
        "dpy-6005",
        // Python's ConnectionError family (refused, reset, aborted)
        "connectionerror",
        "connectionrefusederror",
        "connectionreseterror",
        "database is locked",
    ];
    !PERMANENT.iter().any(|p| msg.contains(p)) && TRANSIENT.iter().any(|t| msg.contains(t))
}

/// Backoff before each retry of a failed connect, or of a query that failed before its first
/// batch. Connect storms (many partitions, a server spawning a process per session) are
/// often transient.
pub const RETRY_BACKOFF_MS: [u64; 3] = [100, 500, 2000];

/// Sleeps before retry number `attempt` (0-based).
pub fn retry_wait(attempt: usize, error: &polars_error::PolarsError) {
    if polars_core::config::verbose() {
        eprintln!(
            "[scan_database]: retrying in {}ms after: {error}",
            RETRY_BACKOFF_MS[attempt]
        );
    }
    std::thread::sleep(std::time::Duration::from_millis(RETRY_BACKOFF_MS[attempt]));
}

/// [`open_session`], retried with backoff.
pub fn open_session_with_retries(
    backend: &dyn DatabaseBackend,
    spec: &ConnectionSpec,
    dialect: DialectName,
) -> PolarsResult<Box<dyn DatabaseSession>> {
    let mut attempt = 0;
    loop {
        match open_session(backend, spec, dialect) {
            Ok(s) => return Ok(s),
            Err(e) if attempt < RETRY_BACKOFF_MS.len() && is_transient(&e) => {
                retry_wait(attempt, &e);
                attempt += 1;
            },
            Err(e) => {
                return Err(e.context(
                    format!(
                        "connecting to {} via {} ({} attempts)",
                        spec.target,
                        backend.choice().kind,
                        attempt + 1
                    )
                    .into(),
                ));
            },
        }
    }
}

/// Instantiates the chosen backend on this machine. Re-checks availability: a worker without
/// the planned backend fails here, before any query runs.
pub fn open_backend(
    choice: &BackendChoice,
    batch_size: Option<usize>,
) -> PolarsResult<Arc<dyn DatabaseBackend>> {
    match &choice.kind {
        BackendKind::Adbc { driver } => Ok(Arc::new(adbc::AdbcBackend::load(
            driver,
            choice.location.as_deref(),
            choice.clone(),
        )?)),
        kind => {
            let Some(bridge) = python_bridge() else {
                polars_bail!(
                    ComputeError:
                    "database backend '{kind}' needs Python, which is not available on this machine"
                )
            };
            if bridge.probe(kind).is_none() {
                polars_bail!(
                    ComputeError:
                    "database backend '{kind}' was chosen during planning but is not installed \
                    on this machine; install it, or pin another driver with `driver=`"
                )
            }
            Ok(Arc::new(PythonBackend {
                choice: choice.clone(),
                bridge: bridge.clone(),
                batch_size,
            }))
        },
    }
}

#[cfg(test)]
mod tests {
    use polars_error::polars_err;

    use super::*;
    use crate::database::connection::ConnectionTarget;

    #[test]
    fn session_setup_needs_sessions() {
        for (kind, sessions) in [
            ("odbc:arrow_odbc", false),
            ("connectorx", false),
            ("odbc:pyodbc", true),
            ("python:pymysql", true),
            ("adbc:postgresql", true),
        ] {
            assert_eq!(
                BackendKind::parse(kind).unwrap().has_sessions(),
                sessions,
                "{kind}"
            );
        }
    }

    /// A backend whose sessions reject every statement holding `reject`.
    struct Rejecting {
        choice: BackendChoice,
        reject: &'static str,
    }

    struct RejectingSession(&'static str);

    impl DatabaseSession for RejectingSession {
        fn execute(&mut self, sql: &str, _: Option<usize>) -> PolarsResult<BatchIter> {
            if sql.contains(self.0) {
                polars_bail!(ComputeError: "rejected: {sql}");
            }
            Ok(Box::new(std::iter::empty()))
        }
    }

    impl DatabaseBackend for Rejecting {
        fn choice(&self) -> &BackendChoice {
            &self.choice
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities::SESSION_SETUP
        }

        fn connect(
            &self,
            _: &ConnectionSpec,
            _: DialectName,
        ) -> PolarsResult<Box<dyn DatabaseSession>> {
            Ok(Box::new(RejectingSession(self.reject)))
        }
    }

    #[test]
    fn required_session_setup() {
        let spec = ConnectionSpec::new(ConnectionTarget::Uri("mysql://h/db".into()));
        let backend = |reject| Rejecting {
            choice: BackendChoice {
                kind: BackendKind::Adbc {
                    driver: "mysql".into(),
                },
                version: None,
                location: None,
                reason: "test".into(),
            },
            reject,
        };
        // the read-only transaction only tightens hygiene
        assert!(open_session(&backend("READ ONLY"), &spec, DialectName::Mysql).is_ok());
        // literals are written for the time zone and backslash escapes
        for reject in ["time_zone", "sql_mode"] {
            let e = open_session(&backend(reject), &spec, DialectName::Mysql)
                .err()
                .unwrap();
            assert!(e.to_string().contains("session setup"), "{e}");
        }
        assert!(open_session(&backend("NLS_COMP"), &spec, DialectName::Oracle).is_err());
        assert!(open_session(&backend("TIME ZONE"), &spec, DialectName::Postgresql).is_ok());
    }

    #[test]
    fn transient_errors() {
        let e = |m: &str| polars_err!(ComputeError: "{m}");
        for m in [
            "connection to server at \"10.0.0.1\", port 5432 failed: timeout expired",
            "ORA-12520: TNS:listener could not find available handler",
            "ConnectionRefusedError: [Errno 111] Connection refused",
            "FATAL: sorry, too many clients already",
            "OperationalError: (2003, \"Can't connect to MySQL server on '127.0.0.1' (timed out)\")",
            "[HYT01] [Microsoft][ODBC Driver 18 for SQL Server]Login timeout expired",
        ] {
            assert!(is_transient(&e(m)), "{m}");
        }
        for m in [
            "FATAL: password authentication failed for user \"polars\"",
            "ORA-01017: invalid username/password; logon denied",
            "(1045, \"Access denied for user 'polars'@'localhost'\")",
            "syntax error at end of input",
            "division by zero",
            "timed out waiting for connection: password authentication failed",
            "TimeoutError: QueuePool limit of size 1 overflow 0 reached, connection timed \
             out, timeout 3.00",
            // statement and lock timeouts, cancels
            "QueryCanceled: canceling statement due to statement timeout",
            "ADBC execute failed (CANCELLED) in 'libadbc_driver_postgresql.so': ERROR: \
             canceling statement due to statement timeout (SQLSTATE 57014)",
            "OperationalError: (1205, 'Lock wait timeout exceeded; try restarting transaction')",
            "(3024, 'Query execution was interrupted, maximum statement execution time exceeded')",
            "ORA-01013: user requested cancel of current operation",
            "[HYT00] [Microsoft][ODBC Driver 18 for SQL Server]Query timeout expired",
            // echoed input
            "invalid connection option \"redshift://h:5439/db?connect_timeout\"",
            "invalid input syntax for type integer: \"timeout\"",
        ] {
            assert!(!is_transient(&e(m)), "{m}");
        }
    }
}
