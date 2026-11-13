//! The Python driver bridge: database backends implemented by Python packages (ConnectorX,
//! DB-API drivers, SQLAlchemy, ODBC modules, connection factories and live connections),
//! behind the Rust backend traits.
//!
//! The Python side is `polars_database._bridge`. Sessions return Polars DataFrames, so every
//! driver the eager `read_database` supports is reachable. The DataFrames cross as Arrow C
//! streams (`__arrow_c_stream__`): this module is not built with the Polars of the session.
//!
//! ADBC drivers are loaded and called through Python's `adbc_driver_manager`. Those sessions
//! return the driver's Arrow C stream, which is imported without the GIL.

use std::ffi::CString;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use polars_arrow::ffi::ArrowArrayStream;
use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_err};
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyCapsule;

use crate::database::backend::adbc::{self, FoundDriver, StreamOwner};
use crate::database::backend::{
    BackendKind, BatchIter, CancelHandle, DatabaseSession, ProbeResult, PythonDriverBridge,
    collect_batches,
};
use crate::database::connection::{ConnectionSpec, ConnectionTarget, scrub_secrets};
use crate::database::dialect::DialectName;

fn bridge_module(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    MODULE
        .get_or_try_init(py, || Ok(py.import("polars_database._bridge")?.unbind()))
        .map(|m| m.bind(py))
}

fn err(e: PyErr) -> polars_error::PolarsError {
    polars_err!(ComputeError: "{e}")
}

/// A driver error with the connection's secrets removed: some drivers echo the DSN, password
/// included.
fn scrubbed_err(e: PyErr, secrets: &[String]) -> polars_error::PolarsError {
    polars_err!(ComputeError: "{}", scrub_secrets(&e.to_string(), secrets))
}

/// Moves the Arrow C stream out of an `arrow_array_stream` capsule, leaving a released stream
/// for the capsule's destructor.
pub(crate) fn take_stream(capsule: &Bound<'_, PyAny>) -> PyResult<Box<ArrowArrayStream>> {
    let capsule = capsule.cast::<PyCapsule>()?;
    // SAFETY: an `arrow_array_stream` capsule holds an Arrow C stream.
    Ok(unsafe {
        Box::new(std::ptr::replace(
            capsule
                .pointer_checked(Some(c"arrow_array_stream"))?
                .as_ptr() as *mut ArrowArrayStream,
            ArrowArrayStream::empty(),
        ))
    })
}

/// Set at interpreter exit (`_at_exit`), after the scans still running had time to end: the
/// driver calls of a query that runs on (a dropped `collect_batches()` iterator held until
/// exit) are skipped from then on. Python ends a thread that takes the GIL while it finalizes,
/// which aborts the process inside Rust frames.
pub(crate) static EXITING: AtomicBool = AtomicBool::new(false);

/// Bounds the retries of the cancels that follow (`_at_exit`): a driver call that ignores the
/// interrupt must not hold the exit.
pub(crate) fn begin_exit_cancel() {
    let _ = Python::attach(|py| -> PyResult<()> {
        bridge_module(py)?.call_method0("begin_exit_cancel")?;
        Ok(())
    });
}

/// Lifts the bound of `begin_exit_cancel`: a scan in a user's `atexit` hook that runs after
/// `_at_exit` cancels as usual.
pub(crate) fn end_exit_cancel() {
    let _ = Python::attach(|py| -> PyResult<()> {
        bridge_module(py)?.call_method0("end_exit_cancel")?;
        Ok(())
    });
}

fn exiting() -> bool {
    EXITING.load(Ordering::Acquire)
}

/// Releases the Python objects dropped without the GIL, which pyo3 queues until this module's
/// next use of it: the host releases plans on its own threads, or with a GIL this module
/// doesn't know it holds. A scan's live connection (a DuckDB file kept open), connection
/// factory or credential provider would stay referenced until the next scan.
pub(crate) fn release_queued_objects() {
    // SAFETY: only reads whether the interpreter exists.
    if !exiting() && unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
        Python::attach(|_| {});
    }
}

fn exiting_err() -> polars_error::PolarsError {
    polars_err!(ComputeError: "the database read was stopped: Python is exiting")
}

/// A Python object released with the GIL. A `Py` dropped without it is only queued for pyo3's
/// next use of the GIL, which may not come: a scan's last session (a closed connection, the
/// user's live one) stayed referenced until the next scan, or to the end of the process,
/// where DuckDB's module reported it leaked.
struct Released(Option<Py<PyAny>>);

impl Released {
    fn new(obj: Py<PyAny>) -> Self {
        Self(Some(obj))
    }
}

impl std::ops::Deref for Released {
    type Target = Py<PyAny>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

impl Drop for Released {
    fn drop(&mut self) {
        if let Some(obj) = self.0.take() {
            if exiting() {
                std::mem::forget(obj);
            } else {
                Python::attach(|py| drop(obj.into_bound(py)));
            }
        }
    }
}

struct NoOwner;

impl StreamOwner for NoOwner {}

/// A Python Polars DataFrame as a DataFrame of this module, through its Arrow C stream.
fn python_df_to_rust(py_df: &Bound<'_, PyAny>) -> PyResult<PolarsResult<DataFrame>> {
    let stream = take_stream(&py_df.call_method0("__arrow_c_stream__")?)?;
    // SAFETY: Polars' own stream, usable from any thread.
    let batches = match unsafe { adbc::import_stream(stream, NoOwner) } {
        Ok(b) => b,
        Err(e) => return Ok(Err(e)),
    };
    let mut out: Option<DataFrame> = None;
    for batch in batches {
        let batch = match batch {
            Ok(b) => b,
            Err(e) => return Ok(Err(e)),
        };
        match &mut out {
            None => out = Some(batch),
            Some(acc) => {
                if let Err(e) = acc.vstack_mut_owned(batch) {
                    return Ok(Err(e));
                }
            },
        }
    }
    Ok(out.ok_or_else(|| polars_err!(ComputeError: "Python driver returned no DataFrame")))
}

/// The target of a connection as the Python side expects it: `(kind, object)`.
fn target<'py>(
    py: Python<'py>,
    spec: &ConnectionSpec,
) -> PyResult<(&'static str, Bound<'py, PyAny>)> {
    Ok(match &spec.target {
        ConnectionTarget::Uri(uri) => ("uri", uri.as_str().into_pyobject(py)?.into_any()),
        ConnectionTarget::Odbc(s) => ("odbc", s.as_str().into_pyobject(py)?.into_any()),
        ConnectionTarget::Factory(f) => ("factory", f.0.bind(py).clone()),
        ConnectionTarget::Live(c) => ("live", c.0.0.bind(py).clone()),
    })
}

fn options(opts: &[(PlSmallStr, PlSmallStr)]) -> Vec<(&str, &str)> {
    opts.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

/// ADBC options as strings: they were built from strings.
fn c_options(opts: &[(CString, CString)]) -> Vec<(String, String)> {
    opts.iter()
        .map(|(k, v)| {
            (
                k.to_string_lossy().into_owned(),
                v.to_string_lossy().into_owned(),
            )
        })
        .collect()
}

pub struct PyDriverBridge;

impl PyDriverBridge {
    fn connection_info(&self, spec: &ConnectionSpec) -> PolarsResult<(DialectName, PlSmallStr)> {
        Python::attach(|py| {
            let (kind, obj) = target(py, spec)?;
            let (dialect, module): (String, String) = bridge_module(py)?
                .call_method1("connection_info", (obj, kind))?
                .extract()?;
            PyResult::Ok((dialect, module))
        })
        .map_err(err)
        .map(|(dialect, module)| {
            (
                DialectName::parse(&dialect)
                    .unwrap_or_else(|_| DialectName::from_python_module(&module)),
                module.into(),
            )
        })
    }
}

impl PythonDriverBridge for PyDriverBridge {
    fn probe(&self, kind: &BackendKind) -> Option<ProbeResult> {
        Python::attach(|py| {
            let out = bridge_module(py)
                .ok()?
                .call_method1("probe", (kind.to_string(),))
                .ok()?;
            if out.is_none() {
                return None;
            }
            let (version, location): (Option<String>, Option<String>) = out.extract().ok()?;
            Some(ProbeResult {
                version: version.map(Into::into),
                location: location.map(Into::into),
            })
        })
    }

    fn connection_dialect(&self, spec: &ConnectionSpec) -> PolarsResult<DialectName> {
        Ok(self.connection_info(spec)?.0)
    }

    fn connection_module(&self, spec: &ConnectionSpec) -> PolarsResult<PlSmallStr> {
        Ok(self.connection_info(spec)?.1)
    }

    fn connect(
        &self,
        kind: &BackendKind,
        spec: &ConnectionSpec,
        dialect: DialectName,
        batch_size: Option<usize>,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        if exiting() {
            return Err(exiting_err());
        }
        let secrets: Arc<[String]> = spec.secrets().into();
        Python::attach(|py| {
            let (target_kind, obj) = target(py, spec)?;
            let session = bridge_module(py)?.call_method1(
                "connect",
                (
                    kind.to_string(),
                    target_kind,
                    obj,
                    options(&spec.connect_options),
                    options(&spec.execute_options),
                    dialect.to_string(),
                    spec.credential_provider
                        .as_ref()
                        .map(|p| p.0.bind(py).clone()),
                    spec.query_params.as_ref().map(|p| p.0.bind(py).clone()),
                    spec.python_options.as_ref().map(|p| p.0.bind(py).clone()),
                ),
            )?;
            let can_cancel: bool = session.getattr("can_cancel")?.extract()?;
            PyResult::Ok(Box::new(PySession {
                session: Arc::new(Released::new(session.unbind())),
                can_cancel,
                batch_size,
                secrets: secrets.clone(),
            }) as Box<dyn DatabaseSession>)
        })
        .map_err(|e| scrubbed_err(e, &secrets))
    }

    fn has_adbc_driver_manager(&self) -> bool {
        Python::attach(|py| {
            bridge_module(py)
                .and_then(|m| m.call_method0("has_adbc_driver_manager")?.extract())
                .unwrap_or(false)
        })
    }

    fn adbc_connect(
        &self,
        driver: &FoundDriver,
        options: &[(CString, CString)],
        dialect: DialectName,
        execute_options: &[(CString, CString)],
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        Python::attach(|py| {
            let session = bridge_module(py)?.call_method1(
                "adbc_connect",
                (
                    driver.library.to_string_lossy().as_ref(),
                    driver.entrypoint.as_deref(),
                    c_options(options),
                    c_options(execute_options),
                ),
            )?;
            PyResult::Ok(Box::new(PyAdbcSession {
                session: Arc::new(Released::new(session.unbind())),
                dialect,
            }) as Box<dyn DatabaseSession>)
        })
        .map_err(err)
    }

    fn credentials(
        &self,
        spec: &ConnectionSpec,
    ) -> PolarsResult<Option<(Option<String>, Option<String>)>> {
        let Some(provider) = &spec.credential_provider else {
            return Ok(None);
        };
        Python::attach(|py| {
            bridge_module(py)?
                .call_method1("credentials", (provider.0.bind(py),))?
                .extract::<Option<(Option<String>, Option<String>)>>()
        })
        .map_err(err)
    }
}

/// A session of a Python backend.
struct PySession {
    session: Arc<Released>,
    can_cancel: bool,
    batch_size: Option<usize>,
    /// Scrubbed from driver errors.
    secrets: Arc<[String]>,
}

impl PySession {
    /// `bind`: the query parameters (`params=`) are bound to the statement.
    fn execute_bound(
        &mut self,
        sql: &str,
        batch_size: Option<usize>,
        bind: bool,
    ) -> PolarsResult<BatchIter> {
        if exiting() {
            return Err(exiting_err());
        }
        let iter = Python::attach(|py| {
            self.session
                .call_method1(py, "execute", (sql, batch_size.or(self.batch_size), bind))
        })
        .map_err(|e| scrubbed_err(e, &self.secrets))?;
        Ok(Box::new(PyBatches {
            iter: Some(Released::new(iter)),
            secrets: self.secrets.clone(),
        }))
    }
}

impl DatabaseSession for PySession {
    fn execute(&mut self, sql: &str, batch_size: Option<usize>) -> PolarsResult<BatchIter> {
        self.execute_bound(sql, batch_size, true)
    }

    fn catalog_query(&mut self, sql: &str) -> PolarsResult<DataFrame> {
        collect_batches(self.execute_bound(sql, None, false)?, sql)
    }

    fn execute_update(&mut self, sql: &str) -> PolarsResult<()> {
        if exiting() {
            return Err(exiting_err());
        }
        Python::attach(|py| self.session.call_method1(py, "execute_update", (sql,)))
            .map_err(|e| scrubbed_err(e, &self.secrets))?;
        Ok(())
    }

    fn cancel_handle(&self) -> Option<CancelHandle> {
        if !self.can_cancel {
            return None;
        }
        let session = self.session.clone();
        Some(Arc::new(move || {
            if exiting() {
                return Ok(());
            }
            Python::attach(|py| session.call_method0(py, "cancel"))
                .map_err(err)
                .map(|_| ())
        }))
    }
}

impl Drop for PySession {
    fn drop(&mut self) {
        if exiting() {
            return;
        }
        Python::attach(|py| {
            if let Err(e) = self.session.call_method0(py, "close") {
                if polars_core::config::verbose() {
                    eprintln!("[scan_database]: closing a Python session failed: {e}");
                }
            }
        });
    }
}

/// Batches from a Python iterator of DataFrames.
struct PyBatches {
    iter: Option<Released>,
    secrets: Arc<[String]>,
}

impl Iterator for PyBatches {
    type Item = PolarsResult<DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        let iter = self.iter.as_ref()?;
        if exiting() {
            self.iter = None;
            return Some(Err(exiting_err()));
        }
        let out = Python::attach(|py| {
            let builtins = py.import("builtins")?;
            let sentinel = py.None();
            let df = builtins.call_method1("next", (iter.bind(py), &sentinel))?;
            if df.is_none() {
                return PyResult::Ok(None);
            }
            Ok(Some(python_df_to_rust(&df)?))
        });
        match out {
            Ok(Some(df)) => Some(df),
            Ok(None) => {
                self.iter = None;
                None
            },
            Err(e) => {
                self.iter = None;
                Some(Err(scrubbed_err(e, &self.secrets)))
            },
        }
    }
}

/// An ADBC connection opened through Python's `adbc_driver_manager`.
struct PyAdbcSession {
    session: Arc<Released>,
    dialect: DialectName,
}

impl DatabaseSession for PyAdbcSession {
    fn execute(&mut self, sql: &str, batch_size: Option<usize>) -> PolarsResult<BatchIter> {
        let hint = batch_size
            .and_then(|n| adbc::batch_size_option(self.dialect, n))
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v));
        if exiting() {
            return Err(exiting_err());
        }
        let (stream, stmt) = Python::attach(|py| {
            let out = self.session.call_method1(py, "execute", (sql, hint))?;
            let (capsule, stmt): (Bound<'_, PyAny>, Py<PyAny>) = out.extract(py)?;
            PyResult::Ok((take_stream(&capsule)?, stmt))
        })
        .map_err(err)?;
        let owner = PyAdbcStatement {
            session: self.session.clone(),
            stmt: Some(stmt),
        };
        // SAFETY: a C stream, from the driver or the driver manager's wrapper around it (for
        // ADBC 1.1 errors), never a Python object.
        let batches = unsafe { adbc::import_stream(stream, owner) }?;
        Ok(adbc::check_batches(self.dialect, batches))
    }

    fn execute_update(&mut self, sql: &str) -> PolarsResult<()> {
        if exiting() {
            return Err(exiting_err());
        }
        Python::attach(|py| self.session.call_method1(py, "execute_update", (sql,)))
            .map_err(err)?;
        Ok(())
    }

    fn cancel_handle(&self) -> Option<CancelHandle> {
        let session = self.session.clone();
        Some(Arc::new(move || {
            if exiting() {
                return Ok(());
            }
            Python::attach(|py| session.call_method0(py, "cancel"))
                .map_err(err)
                .map(|_| ())
        }))
    }
}

impl Drop for PyAdbcSession {
    fn drop(&mut self) {
        if exiting() {
            return;
        }
        Python::attach(|py| {
            if let Err(e) = self.session.call_method0(py, "close") {
                if polars_core::config::verbose() {
                    eprintln!("[scan_database]: closing an ADBC session failed: {e}");
                }
            }
        });
    }
}

/// The statement of a result stream; closed after the stream is released.
struct PyAdbcStatement {
    session: Arc<Released>,
    stmt: Option<Py<PyAny>>,
}

impl StreamOwner for PyAdbcStatement {}

impl Drop for PyAdbcStatement {
    fn drop(&mut self) {
        let Some(stmt) = self.stmt.take() else {
            return;
        };
        if exiting() {
            std::mem::forget(stmt);
            return;
        }
        Python::attach(|py| {
            if let Err(e) = self.session.call_method1(py, "release", (stmt,)) {
                if polars_core::config::verbose() {
                    eprintln!("[scan_database]: closing an ADBC statement failed: {e}");
                }
            }
        });
    }
}
