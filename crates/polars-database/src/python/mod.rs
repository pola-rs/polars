//! Python extension module `polars_database._lib`.
//!
//! * The plugin contracts as PyCapsules, with the handshake shared by all Polars I/O plugins
//!   (`_polars_io_plugin_ids`, `_capsule(id)`; see `polars_io_ext_ffi`). Polars calls the
//!   contracts directly, without going through Python.
//! * `scan_resolver(...)`: the `scan_database` arguments → a resolver, handed to Polars as a
//!   capsule (`lazyframe_resolver_v1::HANDLE_CAPSULE_NAME`).
//! * The functions behind `show_drivers()` and `read_database`'s ADBC driver lookup.
//! * The Python driver backends ([`bridge`]) and the sqlglot SQL renderer ([`sqlglot`]).
mod bridge;
mod sqlglot;

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Once};

pub(crate) use bridge::release_queued_objects;
use polars_arrow::ffi::ArrowArrayStreamReader;
use polars_core::prelude::*;
use polars_core::scalar::Scalar;
use polars_io_ext_ffi::file_reader_v1;
use polars_io_ext_ffi::lazyframe_resolver_v1::{self, ResolverHandle};
use polars_utils::python_function::PythonObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyCapsule, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::database::backend::adbc::{file_path_params, file_rest};
use crate::database::backend::resolve::{candidate_statuses, install_hint, probe};
use crate::database::backend::{BackendKind, adbc, register_python_bridge};
use crate::database::connection::{
    ConnectionSpec, ConnectionTarget, LiveConnection, percent_encode, scrub_secrets,
    secret_userinfo_end, userinfo_end as userinfo_end_of,
};
use crate::database::sql::register_sql_renderer;
use crate::database::{
    Consistency, DatabaseRelation, DatabaseScanOptions, DialectName, FetchOptions, PartitionSpec,
    PartitionStrategy,
};
use crate::ffi::struct_field_to_schema;
use crate::resolver::{Resolver, into_resolver_handle, resolver_release};

/// Plugin IDs exported by this module, oldest first.
const PLUGIN_IDS: &[&CStr] = &[file_reader_v1::ID, lazyframe_resolver_v1::ID];

/// Registers the sqlglot renderer and the Python driver bridge. Idempotent.
fn register() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        register_sql_renderer("sqlglot", || {
            Ok(Arc::new(sqlglot::SqlglotRenderer::new()?) as _)
        });
        register_python_bridge(Arc::new(bridge::PyDriverBridge));
        // Like the ADBC driver manager: drivers installed into this Python environment
        // (`dbc install` in a venv) are found without `VIRTUAL_ENV` being set.
        let prefix = Python::attach(|py| -> PyResult<Option<String>> {
            let sys = py.import("sys")?;
            let prefix: String = sys.getattr("prefix")?.extract()?;
            let base: String = sys.getattr("base_prefix")?.extract()?;
            Ok((prefix != base).then_some(prefix))
        });
        if let Ok(Some(prefix)) = prefix {
            adbc::manifest::add_manifest_search_dir(
                std::path::Path::new(&prefix)
                    .join("etc")
                    .join("adbc")
                    .join("drivers"),
            );
        }
    });
}

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn polars_err_to_py(e: polars_error::PolarsError) -> PyErr {
    value_err(e)
}

/// Returns the PyCapsule of the contract with the given plugin ID.
#[pyfunction]
fn _capsule<'py>(py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyCapsule>> {
    let Some(&plugin_id) = PLUGIN_IDS.iter().find(|i| i.to_str() == Ok(id)) else {
        return Err(value_err(format!(
            "{} does not provide plugin ID {id:?}",
            crate::PACKAGE_NAME
        )));
    };
    let plugin: NonNull<c_void> = match plugin_id {
        id if id == file_reader_v1::ID => NonNull::from(&crate::reader::PLUGIN_V1).cast(),
        _ => NonNull::from(&crate::resolver::PLUGIN_V1).cast(),
    };
    // SAFETY: The struct is a static; the extension module is never unloaded.
    unsafe { PyCapsule::new_with_pointer(py, plugin, plugin_id) }
}

/// The schema of an object with an Arrow C stream (an empty Polars DataFrame).
fn arrow_schema_of(obj: &Bound<'_, PyAny>) -> PyResult<Schema> {
    let stream = bridge::take_stream(&obj.call_method0("__arrow_c_stream__")?)?;
    // SAFETY: Polars' own stream.
    let reader = unsafe { ArrowArrayStreamReader::try_new(stream) }.map_err(polars_err_to_py)?;
    struct_field_to_schema(reader.field()).map_err(polars_err_to_py)
}

/// The two values of a Polars Series, as scalars.
fn range_of(obj: &Bound<'_, PyAny>) -> PyResult<(Scalar, Scalar)> {
    let stream = bridge::take_stream(&obj.call_method0("__arrow_c_stream__")?)?;
    // SAFETY: Polars' own stream.
    let mut reader =
        unsafe { ArrowArrayStreamReader::try_new(stream) }.map_err(polars_err_to_py)?;
    let field = reader.field().clone();
    let mut chunks = vec![];
    // SAFETY: Polars produces arrays matching its schema.
    while let Some(array) = unsafe { reader.next() } {
        chunks.push(array.map_err(polars_err_to_py)?);
    }
    let s = Series::try_from((&field, chunks)).map_err(polars_err_to_py)?;
    if s.len() != 2 {
        return Err(value_err("partition_range must be (low, high)"));
    }
    let get = |i: usize| -> PyResult<Scalar> {
        let av = s.get(i).map_err(polars_err_to_py)?.into_static();
        Ok(Scalar::new(s.dtype().clone(), av))
    };
    Ok((get(0)?, get(1)?))
}

/// A stable identity of `params=`, for scan deduplication: the `repr` of plain values, which
/// shows them whole. Any other object's (a NumPy array's elides its middle) is the object
/// itself, which the scan holds, so two scans merge only if they share it.
///
/// Plain values are copied (deeply): the scan runs with the values its key was taken of,
/// whatever the caller's list or dict holds by then.
fn params_key<'py>(params: Bound<'py, PyAny>) -> PyResult<(PlSmallStr, Bound<'py, PyAny>)> {
    fn plain(v: &Bound<'_, PyAny>) -> PyResult<bool> {
        if v.is_none()
            || v.is_exact_instance_of::<PyBool>()
            || v.is_exact_instance_of::<PyInt>()
            || v.is_exact_instance_of::<PyFloat>()
            || v.is_exact_instance_of::<PyString>()
            || v.is_exact_instance_of::<PyBytes>()
        {
            return Ok(true);
        }
        if v.is_exact_instance_of::<PyList>() || v.is_exact_instance_of::<PyTuple>() {
            for item in v.try_iter()? {
                if !plain(&item?)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if let Ok(d) = v.cast_exact::<PyDict>() {
            for (k, item) in d.iter() {
                if !plain(&k)? || !plain(&item)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        // `Decimal`, `date`, `datetime`, `time`, `timedelta`: their repr is whole
        let module = v.get_type().module()?;
        Ok(matches!(module.to_str()?, "decimal" | "datetime"))
    }
    Ok(if plain(&params)? {
        let params = params
            .py()
            .import("copy")?
            .call_method1("deepcopy", (params,))?;
        (params.repr()?.to_string().into(), params)
    } else {
        let key = format!("<object at {:#x}>", params.as_ptr() as usize).into();
        (key, params)
    })
}

/// The `scan_database` arguments, checked by Polars, as a resolver capsule for
/// `LazyFrame._from_io_plugin_resolver`.
#[pyfunction]
#[pyo3(signature = (
    source, source_type, connection, connection_kind, dialect, driver, connect_options,
    execute_options, python_options, credential_provider, params, schema, schema_overrides,
    partition_on, partition_auto, partition_num, partition_range, partition_predicates,
    order_by, consistency, max_connections, batch_size, pushdown,
))]
#[allow(clippy::too_many_arguments)]
fn scan_resolver<'py>(
    py: Python<'py>,
    source: &str,
    source_type: Option<&str>,
    connection: Py<PyAny>,
    connection_kind: &str,
    dialect: Option<&str>,
    driver: Option<String>,
    connect_options: Vec<(String, String)>,
    execute_options: Vec<(String, String)>,
    python_options: Option<Py<PyAny>>,
    credential_provider: Option<Py<PyAny>>,
    params: Option<Py<PyAny>>,
    schema: Option<Bound<'py, PyAny>>,
    schema_overrides: Option<Bound<'py, PyAny>>,
    partition_on: Option<String>,
    partition_auto: bool,
    partition_num: Option<usize>,
    partition_range: Option<Bound<'py, PyAny>>,
    partition_predicates: Option<Vec<String>>,
    order_by: Vec<String>,
    consistency: &str,
    max_connections: Option<usize>,
    batch_size: Option<usize>,
    pushdown: bool,
) -> PyResult<Bound<'py, PyCapsule>> {
    register();

    let (query_params_key, query_params) = match params {
        Some(p) => {
            let (key, p) = params_key(p.into_bound(py))?;
            (Some(key), Some(Arc::new(PythonObject(p.unbind()))))
        },
        None => (None, None),
    };
    let (python_options_key, python_options) = match python_options {
        Some(p) => {
            let (key, p) = params_key(p.into_bound(py))?;
            (Some(key), Some(Arc::new(PythonObject(p.unbind()))))
        },
        None => (None, None),
    };
    let target = match connection_kind {
        "uri" => ConnectionTarget::Uri(connection.extract::<String>(py)?.into()),
        "odbc" => ConnectionTarget::Odbc(connection.extract::<String>(py)?.into()),
        "factory" => ConnectionTarget::Factory(Arc::new(PythonObject(connection))),
        "live" => ConnectionTarget::Live(Arc::new(LiveConnection(PythonObject(connection)))),
        k => return Err(value_err(format!("invalid connection kind: {k}"))),
    };
    let connection = ConnectionSpec {
        target,
        connect_options: connect_options
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect(),
        execute_options: execute_options
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect(),
        credential_provider: credential_provider.map(|p| Arc::new(PythonObject(p))),
        query_params_key,
        query_params,
        python_options,
        python_options_key,
    };

    let source_type = source_type.filter(|s| *s != "auto");
    let relation = DatabaseRelation::parse(source, source_type);
    // A bad value of an argument is a ValueError, as for the other arguments.
    let dialect = dialect
        .map(DialectName::parse)
        .transpose()
        .map_err(value_err)?;

    // Polars has checked the combinations.
    let partitioning = match (partition_predicates, partition_on.as_deref()) {
        (Some(preds), _) => PartitionSpec::Predicates(preds.into_iter().map(Into::into).collect()),
        (None, None) => PartitionSpec::None,
        (None, Some(_)) if partition_auto => PartitionSpec::Auto {
            num_partitions: partition_num,
        },
        (None, Some(column)) => PartitionSpec::Column {
            column: column.into(),
            num_partitions: partition_num,
            range: partition_range
                .map(|r| range_of(&r))
                .transpose()?
                .map(Box::new),
            strategy: PartitionStrategy::MinMax,
        },
    };

    let consistency = match consistency {
        "none" => Consistency::None,
        "snapshot" => Consistency::Snapshot,
        c => return Err(value_err(format!("invalid consistency: '{c}'"))),
    };

    let options = DatabaseScanOptions {
        connection,
        relation,
        dialect,
        driver: driver.map(Into::into),
        partitioning,
        order_by: order_by.into_iter().map(Into::into).collect(),
        consistency,
        fetch: FetchOptions {
            batch_size,
            max_connections,
        },
        pushdown,
        schema_overrides: schema_overrides
            .map(|s| arrow_schema_of(&s))
            .transpose()?
            .map(Arc::new),
    };
    let user_schema = schema
        .map(|s| arrow_schema_of(&s))
        .transpose()?
        .map(Arc::new);

    let handle = into_resolver_handle(Resolver::new(options, user_schema));
    // The capsule holds a reference of its own: Polars takes another one.
    PyCapsule::new_with_value_and_destructor(
        py,
        handle,
        lazyframe_resolver_v1::HANDLE_CAPSULE_NAME,
        |h: ResolverHandle, _| unsafe { resolver_release(h) },
    )
}

/// Rows of `show_drivers()`: `(dialect, backend, available, version, location, install)`.
#[pyfunction]
#[pyo3(signature = (dialect=None))]
#[allow(clippy::type_complexity)]
fn driver_statuses(
    py: Python<'_>,
    dialect: Option<&str>,
) -> PyResult<Vec<(String, String, bool, Option<String>, Option<String>, String)>> {
    register();
    let dialects = match dialect {
        Some(d) => vec![DialectName::parse(d).map_err(value_err)?],
        None => DialectName::all().collect(),
    };
    py.detach(|| {
        Ok(dialects
            .into_iter()
            .flat_map(candidate_statuses)
            .map(|s| {
                (
                    s.dialect.to_string(),
                    s.kind.to_string(),
                    s.probe.is_some(),
                    s.probe
                        .as_ref()
                        .and_then(|p| p.version.as_ref().map(|v| v.to_string())),
                    s.probe
                        .as_ref()
                        .and_then(|p| p.location.as_ref().map(|v| v.to_string())),
                    install_hint(&s.kind),
                )
            })
            .collect())
    })
}

/// The dialect a URI scheme maps to.
#[pyfunction]
fn dialect_from_scheme(scheme: &str) -> String {
    DialectName::from_scheme(scheme).to_string()
}

/// Finds an installed ADBC driver (manifest search, then pip packages): `(library, entrypoint)`.
#[pyfunction]
fn find_adbc_driver(py: Python<'_>, name: &str) -> PyResult<Option<(String, Option<String>)>> {
    register();
    let name = name.to_string();
    py.detach(move || {
        if let Some(found) = adbc::find_driver(&name) {
            return Ok(Some((
                found.library.to_string_lossy().into_owned(),
                found.entrypoint,
            )));
        }
        let kind = BackendKind::Adbc {
            driver: name.into(),
        };
        Ok(probe(&kind)
            .and_then(|p| p.location)
            .map(|l| (l.to_string(), None)))
    })
}

/// The index (in characters) of the `@` that ends the userinfo of `rest`, the part of a URI
/// after `scheme://`; with `secret`, for hiding the password (see `userinfo_end` and
/// `secret_userinfo_end`).
#[pyfunction]
#[pyo3(signature = (rest, secret=false))]
fn userinfo_end(rest: &str, secret: bool) -> Option<usize> {
    let at = if secret {
        secret_userinfo_end(rest)
    } else {
        userinfo_end_of(rest)
    }?;
    Some(rest[..at].chars().count())
}

/// The secrets of a connection (`kind`: `"uri"` or `"odbc"`), for scrubbing driver messages:
/// see [`ConnectionSpec::secrets`]. `password`: one a credential provider gave. `options`:
/// the string values of the connect options.
#[pyfunction]
#[pyo3(signature = (target, kind, password, options))]
fn connection_secrets(
    target: &str,
    kind: &str,
    password: Option<&str>,
    options: Vec<(String, String)>,
) -> PyResult<Vec<String>> {
    let target = match kind {
        "uri" => ConnectionTarget::Uri(target.into()),
        "odbc" => ConnectionTarget::Odbc(target.into()),
        k => return Err(value_err(format!("invalid connection kind: {k}"))),
    };
    let mut spec = ConnectionSpec::new(target);
    spec.connect_options = options
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
    let mut out = spec.secrets();
    if let Some(p) = password.filter(|p| !p.is_empty()) {
        out.push(p.to_string());
        out.push(percent_encode(p));
        out.sort_by_key(|s| std::cmp::Reverse(s.len()));
        out.dedup();
    }
    Ok(out)
}

/// `message` with every secret (from `connection_secrets`) replaced by `***`.
#[pyfunction]
fn scrub(message: &Bound<'_, PyString>, secrets: Vec<String>) -> String {
    // (lossy: a message may hold surrogates, from a non-UTF-8 name echoed by a driver)
    scrub_secrets(&message.to_string_lossy(), &secrets)
}

/// The path and the parameters of a `sqlite:` or `duckdb:` URI (any `scheme[+driver]:`), as
/// SQLAlchemy reads it. A `file:` URI is returned whole as the path.
#[pyfunction]
fn file_target(uri: &str) -> (String, Vec<(String, String)>) {
    // `^\w+(\+\w+)?:`
    let word = |s: &str| {
        let n = s
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(s.len());
        (n > 0).then_some(n)
    };
    let scheme_end = word(uri).and_then(|n| {
        let after = &uri[n..];
        let n = match after.strip_prefix('+') {
            Some(driver) => n + 1 + word(driver)?,
            None => n,
        };
        uri[n..].starts_with(':').then_some(n + 1)
    });
    let rest = file_rest(&uri[scheme_end.unwrap_or(0)..]);
    if rest.starts_with("file:") {
        return (rest.to_string(), vec![]);
    }
    let (path, params) = file_path_params(rest);
    (path, params)
}

/// The ADBC database options of a URI for a driver loaded through the generic driver manager,
/// as `scan_database` passes them; `None` if the driver takes the URI as it is.
#[pyfunction]
fn adbc_db_kwargs(driver: &str, uri: &str) -> Option<Vec<(String, String)>> {
    adbc::driver_db_kwargs(driver, uri)
}

/// The default `max_connections` of a scan for a dialect named as SQLAlchemy names it.
#[pyfunction]
fn default_max_connections(dialect: &str) -> usize {
    DialectName::all()
        .find(|d| dialect.starts_with(<&str>::from(*d)))
        .unwrap_or(DialectName::Generic)
        .default_max_connections()
}

/// The module name must match `module-name` of the wheel (see `pyproject.toml`).
/// Registered with `atexit` on import: waits a moment for queries still running (a
/// `collect_batches()` iterator dropped just before exit stops its query only when the query
/// notices), so their sessions are closed while Python still runs; then stops every driver
/// call of a query that runs on. Python ends a thread that takes the GIL while it finalizes,
/// which aborts the process inside Rust frames.
///
/// A query still running after the wait has its statement cancelled (a thread inside a
/// driver call would be ended inside Rust frames too), and is given a moment more. Only
/// then are driver calls stopped: with every query ended, a scan in a user's `atexit` hook
/// that runs after this one (registered before the first scan imported this module) runs
/// as usual.
#[pyfunction]
fn _at_exit(py: Python<'_>) {
    use std::time::Duration;
    let ended = py.detach(|| {
        crate::reader::wait_for_scans(Duration::from_secs(2)) || {
            // (a driver call that ignores the interrupt is asked for two seconds, not forever)
            bridge::begin_exit_cancel();
            crate::reader::cancel_running_statements();
            // (every cancel has returned: the bound ends with them)
            bridge::end_exit_cancel();
            crate::reader::wait_for_scans(Duration::from_secs(2))
        }
    });
    if !ended {
        bridge::EXITING.store(true, std::sync::atomic::Ordering::Release);
    }
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Plans may be deserialized and run in a process that never called `scan_resolver`.
    register();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "_polars_io_plugin_ids",
        PyTuple::new(m.py(), PLUGIN_IDS.iter().map(|id| id.to_str().unwrap()))?,
    )?;
    m.add_function(wrap_pyfunction!(_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(scan_resolver, m)?)?;
    m.add_function(wrap_pyfunction!(driver_statuses, m)?)?;
    m.add_function(wrap_pyfunction!(dialect_from_scheme, m)?)?;
    m.add_function(wrap_pyfunction!(find_adbc_driver, m)?)?;
    m.add_function(wrap_pyfunction!(userinfo_end, m)?)?;
    m.add_function(wrap_pyfunction!(connection_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(scrub, m)?)?;
    m.add_function(wrap_pyfunction!(file_target, m)?)?;
    m.add_function(wrap_pyfunction!(adbc_db_kwargs, m)?)?;
    m.add_function(wrap_pyfunction!(default_max_connections, m)?)?;
    m.add_function(wrap_pyfunction!(_at_exit, m)?)?;
    Ok(())
}
