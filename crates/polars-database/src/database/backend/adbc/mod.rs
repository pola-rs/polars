//! The ADBC backend: finds ADBC drivers and reads query results as Arrow C streams.
//!
//! Drivers are found through ADBC driver manifests (see [`manifest`]) or given as a library
//! path, and loaded and called through Python's `adbc_driver_manager`, over the Python bridge
//! ([`PythonDriverBridge::adbc_connect`]). This module builds the database and statement
//! options ([`connect_options`], [`batch_size_option`]) and imports the result streams
//! ([`import_stream`]).

pub mod manifest;

use std::ffi::{CStr, CString};
use std::sync::Arc;

use polars_arrow::array::{Array, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi::{ArrowArrayStream, ArrowArrayStreamReader};
use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

pub use self::manifest::{FoundDriver, find_driver, manifest_search_dirs};
use super::{
    BackendCapabilities, BackendChoice, BatchIter, DatabaseBackend, DatabaseSession,
    PythonDriverBridge,
};
use crate::database::connection::{
    ConnectionSpec, ConnectionTarget, password_secrets, percent_decode, percent_encode,
    scrub_secrets, secret_userinfo_end, userinfo_end,
};
use crate::database::dialect::DialectName;

const ADBC_OPTION_URI: &CStr = c"uri";
const ADBC_OPTION_USERNAME: &CStr = c"username";
const ADBC_OPTION_PASSWORD: &CStr = c"password";

pub struct AdbcBackend {
    bridge: Arc<dyn PythonDriverBridge>,
    found: FoundDriver,
    choice: BackendChoice,
}

impl AdbcBackend {
    /// `driver`: an ADBC driver name (`"sqlite"`, `"postgresql"`, …), or a path to a shared
    /// library or a `.toml` manifest.
    /// `location_hint`: a library or manifest path found during planning, possibly on another
    /// machine. Tried first when it exists here.
    pub fn load(
        driver: &str,
        location_hint: Option<&str>,
        choice: BackendChoice,
    ) -> PolarsResult<Self> {
        let found = location_hint
            .and_then(find_driver)
            .or_else(|| find_driver(driver));
        let Some(found) = found else {
            let dirs = manifest_search_dirs()
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let hint = location_hint
                .map(|h| format!("'{h}' (from planning) does not exist here; "))
                .unwrap_or_default();
            if driver.contains(['/', '\\']) || driver.ends_with(".toml") {
                polars_bail!(ComputeError: "ADBC driver library or manifest '{driver}' does not exist");
            }
            polars_bail!(
                ComputeError:
                "ADBC driver '{driver}' not found: {hint}looked for '{driver}.toml' in [{dirs}]. \
                Install it with: {}  (or pass the path of the driver library or manifest)",
                super::resolve::adbc_install_hint(driver)
            )
        };
        let Some(bridge) = super::python_bridge().filter(|b| b.has_adbc_driver_manager()) else {
            polars_bail!(
                ComputeError:
                "ADBC drivers are loaded through Python's adbc_driver_manager, which is not \
                installed on this machine: pip install adbc-driver-manager"
            )
        };
        if polars_core::config::verbose() {
            eprintln!(
                "[scan_database]: {} ('{}') is loaded through adbc_driver_manager",
                choice.kind,
                found.library.display()
            );
        }
        Ok(Self {
            bridge: bridge.clone(),
            found,
            choice,
        })
    }

    /// Opens a connection, like [`DatabaseBackend::connect`].
    pub fn open(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        let options = connect_options(spec, dialect)?;
        let execute_options = spec
            .execute_options
            .iter()
            .map(|(k, v)| Ok((cstring(k, "option key")?, cstring(v, k)?)))
            .collect::<PolarsResult<Vec<_>>>()?;
        let session = self
            .bridge
            .adbc_connect(&self.found, &options, dialect, &execute_options);
        // Some drivers quote the URI or rejected options, credentials included.
        // The spec's: the URI's query (`?password=`, `?sslpassword=`) as written.
        session.map_err(|e| {
            let mut secrets = secrets(&options);
            for s in spec.secrets() {
                secrets.push(percent_decode(&s));
                secrets.push(s);
            }
            secrets.retain(|s| !s.is_empty());
            secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
            secrets.dedup();
            e.wrap_msg(|m| redact(m, &secrets))
        })
    }
}

/// The credentials among the options of a connection: password-like options, and the
/// password of the URI as written and decoded.
fn secrets(options: &[(CString, CString)]) -> Vec<String> {
    let mut out = vec![];
    for (k, v) in options {
        let (k, v) = (
            k.to_string_lossy().to_ascii_lowercase(),
            v.to_string_lossy(),
        );
        if ["password", "passwd", "pwd", "passphrase", "secret", "token"]
            .iter()
            .any(|s| k.contains(s))
        {
            out.push(v.into_owned());
        } else if k == ADBC_OPTION_URI.to_string_lossy()
            && let Some((_, rest)) = v.split_once("://")
            && let Some(at) = secret_userinfo_end(rest)
            && let Some((_, password)) = rest[..at].split_once(':')
        {
            out.extend(password_secrets(password));
            out.push(percent_decode(password));
        }
    }
    out.retain(|s| !s.is_empty());
    // Longest first, so that one secret containing another is replaced whole.
    out.sort_by_key(|s| std::cmp::Reverse(s.len()));
    out.dedup();
    out
}

fn redact(message: &str, secrets: &[String]) -> String {
    scrub_secrets(message, secrets)
}

/// The database options for a connection, `uri` first.
pub fn connect_options(
    spec: &ConnectionSpec,
    dialect: DialectName,
) -> PolarsResult<Vec<(CString, CString)>> {
    let mut uri = adbc_uri(spec, dialect)?.map(|u| encoded_userinfo(&u).unwrap_or(u));
    // Redshift's ADBC driver isn't libpq.
    if dialect == DialectName::Postgresql
        && let Some(u) = &mut uri
        && !u.contains("connect_timeout=")
    {
        // libpq otherwise waits for the OS TCP timeout (minutes), and connects are retried.
        u.push(if u.contains('?') { '&' } else { '?' });
        u.push_str("connect_timeout=10");
    }
    let mut options: Vec<(CString, CString)> = vec![];
    if dialect == DialectName::Duckdb
        && let Some(u) = uri.take()
    {
        // The DuckDB driver reads `duckdb:///d.db` as `/d.db`: it gets the path, as the
        // Python backends read it, and the parameters as settings.
        for (k, v) in duckdb_options(&u) {
            options.push((cstring(&k, "option key")?, cstring(&v, &k)?));
        }
    }
    let mut creds = credentials(spec)?;
    if dialect == DialectName::Clickhouse
        && let Some(u) = &uri
        && let Some(http) = clickhouse_http_uri(u)
    {
        // The ClickHouse driver takes the HTTP endpoint, with credentials as options.
        let (user, password) = creds.take().unwrap_or_default();
        // the provider's credentials win over the query's, which ClickHouse would refuse
        // next to them ("authentication via parameters simultaneously")
        uri = Some(without_credential_params(
            &http.uri,
            user.is_some(),
            password.is_some(),
        ));
        for (key, value) in [
            (ADBC_OPTION_USERNAME, user.or(http.user)),
            (ADBC_OPTION_PASSWORD, password.or(http.password)),
        ] {
            if let Some(v) = value {
                options.push((key.to_owned(), cstring(&v, "credential")?));
            }
        }
    }
    if let Some((user, password)) = creds {
        // Drivers that take a `scheme://` URI (libpq, MySQL, …) read credentials from it
        // and may reject the `username`/`password` options.
        // and from its query, after the userinfo: the provider's must win
        match uri
            .as_deref()
            .and_then(|u| with_userinfo(u, user.as_deref(), password.as_deref()))
            .map(|u| without_credential_params(&u, user.is_some(), password.is_some()))
        {
            Some(with_creds) => uri = Some(with_creds),
            None => {
                if let Some(u) = user {
                    options.push((ADBC_OPTION_USERNAME.to_owned(), cstring(&u, "credential")?));
                }
                if let Some(p) = password {
                    options.push((ADBC_OPTION_PASSWORD.to_owned(), cstring(&p, "credential")?));
                }
            },
        }
    }
    if let Some(uri) = uri {
        options.insert(0, (ADBC_OPTION_URI.to_owned(), cstring(&uri, "uri")?));
    }
    for (k, v) in &spec.connect_options {
        options.push((cstring(k, "option key")?, cstring(v, k)?));
    }
    Ok(options)
}

impl DatabaseBackend for AdbcBackend {
    fn choice(&self) -> &BackendChoice {
        &self.choice
    }

    fn capabilities(&self) -> BackendCapabilities {
        // `CANCEL` also for ADBC 1.0 drivers: the driver manager answers their cancel with an
        // error, which the cancel path tolerates.
        BackendCapabilities::ARROW_NATIVE
            | BackendCapabilities::STREAMING
            | BackendCapabilities::NO_GIL
            | BackendCapabilities::SESSION_SETUP
            | BackendCapabilities::CANCEL
            | BackendCapabilities::NEEDS_PYTHON
    }

    fn connect(
        &self,
        spec: &ConnectionSpec,
        dialect: DialectName,
    ) -> PolarsResult<Box<dyn DatabaseSession>> {
        self.open(spec, dialect)
    }
}

fn cstring(s: &str, what: &str) -> PolarsResult<CString> {
    CString::new(s).map_err(|_| polars_err!(ComputeError: "ADBC {what} contains a NUL byte"))
}

/// The `uri` database option for a connection target.
fn adbc_uri(spec: &ConnectionSpec, dialect: DialectName) -> PolarsResult<Option<String>> {
    match &spec.target {
        ConnectionTarget::Uri(_) => {
            let uri = spec.uri_without_scheme_driver().unwrap_or_default();
            if dialect == DialectName::Sqlite {
                return Ok(Some(sqlite_path(&uri)?));
            }
            // an alias scheme (`pgsql://`, `tidb://`) as the one the driver reads
            let canonical = match dialect {
                DialectName::Postgresql => &["postgresql", "postgres"][..],
                DialectName::Mysql => &["mysql"],
                _ => &[],
            };
            if let Some((scheme, rest)) = uri.split_once(':')
                && let Some(first) = canonical.first()
                && !canonical.iter().any(|c| scheme.eq_ignore_ascii_case(c))
            {
                return Ok(Some(format!("{first}:{rest}")));
            }
            Ok(Some(uri))
        },
        ConnectionTarget::Odbc(_) => polars_bail!(
            InvalidOperation: "an ODBC connection string can't be used with an ADBC driver"
        ),
        #[allow(unreachable_patterns)]
        _ => polars_bail!(
            InvalidOperation: "a Python connection object can't be used with an ADBC driver"
        ),
    }
}

struct ClickhouseHttp {
    uri: String,
    user: Option<String>,
    password: Option<String>,
}

/// `clickhouse://user:pass@host:8123/db` → `http://host:8123/?database=db` and the
/// credentials; `clickhouses://` (or `?secure=true`) → `https://`. `http(s)://` passes through.
fn clickhouse_http_uri(uri: &str) -> Option<ClickhouseHttp> {
    let (scheme, rest) = uri.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme == "http" || scheme == "https" {
        return None;
    }
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (userinfo, host) = match authority.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, authority),
    };
    let (user, password) = match userinfo {
        Some(u) => match u.split_once(':') {
            Some((u, p)) => (Some(percent_decode(u)), Some(percent_decode(p))),
            None => (Some(percent_decode(u)), None),
        },
        None => (None, None),
    };
    let secure = scheme.ends_with('s')
        || query
            .split('&')
            .any(|kv| kv.eq_ignore_ascii_case("secure=true") || kv == "secure=1");
    let mut params: Vec<String> = query
        .split('&')
        .filter(|kv| !kv.is_empty() && !kv.to_ascii_lowercase().starts_with("secure="))
        .map(str::to_string)
        .collect();
    let database = path.trim_matches('/');
    if !database.is_empty() {
        params.insert(0, format!("database={database}"));
    }
    let mut out = format!("{}://{host}", if secure { "https" } else { "http" });
    if !params.is_empty() {
        out.push_str("/?");
        out.push_str(&params.join("&"));
    }
    Some(ClickhouseHttp {
        uri: out,
        user,
        password,
    })
}

/// The path and the parameters of a `sqlite:` or `duckdb:` URI, read as SQLAlchemy (and the
/// Python backends, `_file_target`) read it: `sqlite:///rel.db` → `rel.db`,
/// `sqlite:////abs.db` → `/abs.db`, `sqlite://` → `:memory:`. `None` for a `file:` URI after
/// the scheme, or a string without the scheme, which pass through.
fn file_target(uri: &str, schemes: &[&str]) -> Option<(String, Vec<(String, String)>)> {
    let lower = uri.to_ascii_lowercase();
    let prefix = schemes.iter().find(|p| lower.starts_with(*p))?;
    let rest = file_rest(&uri[prefix.len()..]);
    if rest.starts_with("file:") {
        return None;
    }
    Some(file_path_params(rest))
}

/// What follows a file URI's scheme, without the `//` (and the `/` after it) that starts an
/// authority: `sqlite://rel.db`, ConnectorX's form, is read as part of the path.
pub(crate) fn file_rest(rest: &str) -> &str {
    match rest.strip_prefix("//") {
        Some(r) => r.strip_prefix('/').unwrap_or(r),
        None => rest,
    }
}

/// The path and the parameters of a [`file_rest`] that isn't a `file:` URI.
pub(crate) fn file_path_params(rest: &str) -> (String, Vec<(String, String)>) {
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    let mut path = match percent_decode(path) {
        p if matches!(p.as_str(), "" | ":memory:" | "/:memory:") => ":memory:".to_string(),
        p => p,
    };
    // `sqlite:////C:/path` → `C:/path`: on Windows, `/C:/path` is a network path.
    let b = path.as_bytes();
    if b.len() >= 4
        && b[0] == b'/'
        && b[1].is_ascii_alphabetic()
        && b[2] == b':'
        && matches!(b[3], b'/' | b'\\')
    {
        path.remove(0);
    }
    let params = query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (
                percent_decode(&k.replace('+', " ")),
                percent_decode(&v.replace('+', " ")),
            )
        })
        .collect();
    (path, params)
}

/// SQLite URI filename parameters: they make the path a `file:` URI.
const SQLITE_URI_PARAMS: [&str; 7] = [
    "cache",
    "immutable",
    "mode",
    "modeof",
    "nolock",
    "psow",
    "vfs",
];

/// The file of a `sqlite:` URI ([`file_target`]). URI filename parameters (`?mode=ro`) make a
/// `file:` URI; others are an error, as the ADBC driver has no option for them.
fn sqlite_path(uri: &str) -> PolarsResult<String> {
    let Some((path, params)) = file_target(uri, &["sqlite3:", "sqlite:"]) else {
        let lower = uri.to_ascii_lowercase();
        return Ok(
            match ["sqlite3:", "sqlite:"]
                .iter()
                .find(|p| lower.starts_with(*p))
            {
                // `sqlite:///file:a.db?mode=ro`
                Some(p) => uri[p.len()..].trim_start_matches('/').to_string(),
                None => uri.to_string(),
            },
        );
    };
    let mut file_params = vec![];
    let mut unknown = vec![];
    for (k, v) in &params {
        if SQLITE_URI_PARAMS.contains(&k.as_str()) {
            file_params.push(format!("{k}={}", percent_encode(v)));
        } else if !matches!(k.as_str(), "uri" | "check_same_thread") {
            // `uri=true` is implied by the file parameters (SQLAlchemy's form); the thread
            // check is a Python `sqlite3` option.
            unknown.push(k.as_str());
        }
    }
    polars_ensure!(
        unknown.is_empty(),
        InvalidOperation:
        "the SQLite ADBC driver has no option for URI parameter(s) {unknown:?}; \
        pin another driver with driver=\"sqlite3\""
    );
    if file_params.is_empty() {
        return Ok(path);
    }
    let mut file = String::from("file:");
    for c in path.chars() {
        match c {
            '%' => file.push_str("%25"),
            '?' => file.push_str("%3F"),
            '#' => file.push_str("%23"),
            c => file.push(c),
        }
    }
    Ok(format!("{file}?{}", file_params.join("&")))
}

/// The DuckDB driver's database options for a `duckdb:` URI: `path`, and the URI parameters as
/// DuckDB settings (`read_only=true` is `access_mode=READ_ONLY`). DuckDB rejects an unknown
/// setting itself.
fn duckdb_options(uri: &str) -> Vec<(String, String)> {
    let Some((path, params)) = file_target(uri, &["duckdb:"]) else {
        return vec![("path".into(), uri.to_string())];
    };
    duckdb_settings(path, params)
}

/// `path` and the URI parameters as DuckDB settings.
fn duckdb_settings(path: String, params: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut out = vec![("path".to_string(), path)];
    for (k, v) in params {
        if k == "read_only" {
            let read_only = matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
            out.push((
                "access_mode".into(),
                if read_only { "READ_ONLY" } else { "READ_WRITE" }.into(),
            ));
        } else {
            out.push((k.to_string(), v));
        }
    }
    out
}

/// The database options a URI gives for a driver that the generic driver manager loads (see
/// `read_database`): DuckDB's `path` and settings, ClickHouse's HTTP endpoint and credentials.
/// `None` for the others, which take the URI as it is. DuckDB's `uri` is what follows the
/// scheme (`read_database` strips it).
pub fn driver_db_kwargs(driver: &str, uri: &str) -> Option<Vec<(String, String)>> {
    match driver {
        "duckdb" => {
            let (path, params) = file_path_params(file_rest(uri));
            Some(duckdb_settings(path, params))
        },
        "clickhouse" => {
            let http = clickhouse_http_uri(uri)?;
            let mut out = vec![("uri".to_string(), http.uri)];
            out.extend(http.user.map(|u| ("username".to_string(), u)));
            out.extend(http.password.map(|p| ("password".to_string(), p)));
            Some(out)
        },
        _ => None,
    }
}

/// `username` / `password` database options from the connection's credential provider.
fn credentials(spec: &ConnectionSpec) -> PolarsResult<Option<(Option<String>, Option<String>)>> {
    #[cfg(feature = "python")]
    if spec.credential_provider.is_some() {
        let Some(bridge) = super::python_bridge() else {
            polars_bail!(ComputeError: "a credential provider needs Python, which is not available");
        };
        return bridge.credentials(spec);
    }
    let _ = spec;
    Ok(None)
}

/// `scheme://user:password@host/…` with the given credentials, replacing those in the URI.
/// `None` if the URI has no authority part.
fn with_userinfo(uri: &str, user: Option<&str>, password: Option<&str>) -> Option<String> {
    let (scheme, rest) = uri.split_once("://")?;
    // `host_and_tail`: the host, and the path and query after it.
    let (old_userinfo, host_and_tail) = match userinfo_end(rest) {
        Some(at) => (Some(&rest[..at]), &rest[at + 1..]),
        None => (None, rest),
    };
    let (old_user, old_password) = match old_userinfo.map(|u| u.split_once(':')) {
        Some(Some((u, p))) => (Some(u), Some(p)),
        Some(None) => (old_userinfo, None),
        None => (None, None),
    };
    let user = match user {
        Some(u) => Some(percent_encode(u)),
        None => old_user.map(str::to_string),
    };
    let mut userinfo = user.unwrap_or_default();
    // A provider that gives only the user keeps the URI's password.
    if let Some(p) = password
        .map(percent_encode)
        .or(old_password.map(str::to_string))
    {
        userinfo.push(':');
        userinfo.push_str(&p);
    }
    let at = if userinfo.is_empty() { "" } else { "@" };
    Some(format!("{scheme}://{userinfo}{at}{host_and_tail}"))
}

/// The URI without the query parameters that give the user (`user=`) or the password
/// (`password=`), which libpq, MySQL and SQLAlchemy read after the userinfo.
fn without_credential_params(uri: &str, user: bool, password: bool) -> String {
    let Some((head, query)) = uri.split_once('?') else {
        return uri.to_string();
    };
    let (query, fragment) = match query.split_once('#') {
        Some((q, f)) => (q, Some(f)),
        None => (query, None),
    };
    let kept: Vec<&str> = query
        .split('&')
        .filter(|kv| {
            let k = kv
                .split_once('=')
                .map_or(*kv, |(k, _)| k)
                .to_ascii_lowercase();
            !(user && matches!(k.as_str(), "user" | "username" | "uid")
                || password && matches!(k.as_str(), "password" | "passwd" | "pwd"))
        })
        .collect();
    let mut out = head.to_string();
    if !kept.is_empty() {
        out.push('?');
        out.push_str(&kept.join("&"));
    }
    if let Some(f) = fragment {
        out.push('#');
        out.push_str(f);
    }
    out
}

/// The URI with its password percent-encoded, if it holds an unencoded `/`, `?` or `#`
/// (`u:Ab#cd/Ef9@h`, which the Python backends accept): drivers end the userinfo at the
/// first of them.
fn encoded_userinfo(uri: &str) -> Option<String> {
    let (_, rest) = uri.split_once("://")?;
    let at = userinfo_end(rest)?;
    if at < rest.find(['/', '?', '#']).unwrap_or(rest.len()) {
        return None;
    }
    let (_, password) = rest[..at].split_once(':')?;
    with_userinfo(uri, None, Some(&percent_decode(password)))
}

/// The statement option that bounds the batch size, for drivers known to have one. A hint:
/// drivers that reject it still work.
pub fn batch_size_option(dialect: DialectName, rows: usize) -> Option<(&'static CStr, String)> {
    match dialect {
        DialectName::Sqlite => Some((c"adbc.sqlite.query.batch_rows", rows.to_string())),
        // A byte hint: assume narrow rows, so a batch never has far more rows than asked.
        DialectName::Postgresql | DialectName::Redshift => Some((
            c"adbc.postgresql.batch_size_hint_bytes",
            rows.saturating_mul(64).max(1024).to_string(),
        )),
        _ => None,
    }
}

/// What a result stream needs kept alive until it is released: its statement, and through
/// it the connection.
pub trait StreamOwner: Send + 'static {}

/// The batches of a query's result stream, as `DataFrame`s. `owner` is dropped after the
/// stream is released.
///
/// # Safety
/// `stream` must be a valid Arrow C stream, usable from any thread.
pub unsafe fn import_stream(
    stream: Box<ArrowArrayStream>,
    owner: impl StreamOwner,
) -> PolarsResult<BatchIter> {
    let reader = unsafe { ArrowArrayStreamReader::try_new(stream) }?;
    // Before any array is imported or built: those of these types would panic.
    check_arrow_types(&reader.field().dtype)?;
    Ok(Box::new(AdbcBatches {
        reader: Some(reader),
        _owner: owner,
        yielded: false,
    }))
}

/// The batches of one query, as `DataFrame`s.
struct AdbcBatches<O> {
    // Field order is drop order: the stream is released before its statement.
    reader: Option<ArrowArrayStreamReader<Box<ArrowArrayStream>>>,
    /// Only kept alive.
    _owner: O,
    yielded: bool,
}

// SAFETY: the stream and statement are only used from the thread that pulls the iterator.
unsafe impl<O: StreamOwner> Send for AdbcBatches<O> {}

impl<O: StreamOwner> Iterator for AdbcBatches<O> {
    type Item = PolarsResult<DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let reader = self.reader.as_mut()?;
            // SAFETY: the driver produces arrays matching its schema.
            match unsafe { reader.next() } {
                Some(Ok(array)) => {
                    if array.is_empty() {
                        continue;
                    }
                    self.yielded = true;
                    return Some(batch_to_df(array));
                },
                Some(Err(e)) => {
                    self.reader = None;
                    return Some(Err(e));
                },
                None => {
                    let reader = self.reader.take()?;
                    if self.yielded {
                        return None;
                    }
                    // Empty result: one empty frame with the result's columns.
                    self.yielded = true;
                    let dtype = reader.field().dtype.clone();
                    return Some(batch_to_df(Box::new(StructArray::new_empty(dtype))));
                },
            }
        }
    }
}

/// Checks the batches of a result for values the driver could not convert.
pub fn check_batches(dialect: DialectName, batches: BatchIter) -> BatchIter {
    Box::new(batches.map(move |df| {
        df.and_then(|df| {
            check_utf8(&df)?;
            if dialect == DialectName::Postgresql {
                check_postgres_infinity(&df)?;
            }
            Ok(df)
        })
    }))
}

/// An error for text that isn't valid UTF-8, before anything reads it as a string: drivers
/// pass on what the database holds (SQLite's `CAST(X'FF' AS TEXT)`) without checking.
fn check_utf8(df: &DataFrame) -> PolarsResult<()> {
    for c in df.columns() {
        if !c.dtype().is_string() {
            continue;
        }
        let bytes = c.str()?.as_binary();
        if bytes.downcast_iter().any(|a| a.validate_utf8().is_err()) {
            polars_bail!(
                ComputeError:
                "column '{}' holds text that is not valid UTF-8, which a Polars String can't \
                hold; read it as binary in a query, e.g. CAST(\"{}\" AS BLOB)",
                c.name(), c.name()
            );
        }
    }
    Ok(())
}

/// The earliest Postgres date and timestamp, 4714-11-24 BC (Julian day 0), from the Unix epoch.
const PG_MIN_DAYS: i32 = -2_440_588;
const PG_MIN_US: i64 = PG_MIN_DAYS as i64 * 86_400_000_000;

/// An error for a Postgres date or timestamp the ADBC Postgres driver could not convert.
///
/// The driver shifts the server's values (µs or days since 2000) to the Unix epoch, and the
/// shift wraps for `infinity` / `-infinity` (`i64::MAX` / `i64::MIN`, `i32::MAX` / `i32::MIN`)
/// and for timestamps after about year 294 000. Wrapped, they read as values before the
/// earliest Postgres value, so anything earlier than that is an error, also nested in lists
/// and structs.
fn check_postgres_infinity(df: &DataFrame) -> PolarsResult<()> {
    for c in df.columns() {
        if wraps_postgres_temporal(c.as_materialized_series())? {
            polars_bail!(
                InvalidOperation:
                "column '{}' holds a Postgres 'infinity', '-infinity' or a timestamp after year \
                294 000, which Polars can't represent; filter it out or replace it in a query, \
                e.g. CASE WHEN isfinite(\"{}\") THEN \"{}\" END",
                c.name(), c.name(), c.name()
            );
        }
    }
    Ok(())
}

fn wraps_postgres_temporal(s: &polars_core::series::Series) -> PolarsResult<bool> {
    use polars_core::prelude::{ChunkAgg, DataType, TimeUnit};

    Ok(match s.dtype() {
        DataType::Datetime(TimeUnit::Microseconds, _) => s
            .to_physical_repr()
            .i64()?
            .min()
            .is_some_and(|v| v < PG_MIN_US),
        DataType::Date => s
            .to_physical_repr()
            .i32()?
            .min()
            .is_some_and(|v| v < PG_MIN_DAYS),
        DataType::List(inner) if has_temporal(inner) => {
            wraps_postgres_temporal(&s.list()?.get_inner())?
        },
        DataType::Struct(_) => {
            for f in s.struct_()?.fields_as_series() {
                if wraps_postgres_temporal(&f)? {
                    return Ok(true);
                }
            }
            false
        },
        _ => false,
    })
}

fn has_temporal(dtype: &polars_core::prelude::DataType) -> bool {
    use polars_core::prelude::DataType;
    match dtype {
        DataType::Date | DataType::Datetime(_, _) => true,
        DataType::List(inner) => has_temporal(inner),
        DataType::Struct(_) => true,
        _ => false,
    }
}

/// Arrow types Polars has no dtype for (importing them would panic).
fn unsupported_arrow_type(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Interval(_) | ArrowDataType::Union(_) | ArrowDataType::Decimal256(_, _) => {
            true
        },
        ArrowDataType::List(f)
        | ArrowDataType::LargeList(f)
        | ArrowDataType::FixedSizeList(f, _)
        | ArrowDataType::Map(f, _) => unsupported_arrow_type(f.dtype()),
        ArrowDataType::Struct(fields) => fields.iter().any(|f| unsupported_arrow_type(f.dtype())),
        ArrowDataType::Extension(ext) => unsupported_arrow_type(&ext.inner),
        _ => false,
    }
}

/// An error for a result column of a type Polars has no dtype for.
fn check_arrow_types(dtype: &ArrowDataType) -> PolarsResult<()> {
    let ArrowDataType::Struct(fields) = dtype else {
        polars_bail!(ComputeError: "ADBC driver returned a non-struct batch: {dtype:?}");
    };
    if let Some(f) = fields.iter().find(|f| unsupported_arrow_type(f.dtype())) {
        polars_bail!(
            InvalidOperation:
            "column '{}' has Arrow type {:?}, which Polars can't read; convert it in a query, \
            e.g. CAST(\"{}\" AS TEXT)",
            f.name, f.dtype(), f.name
        );
    }
    Ok(())
}

fn batch_to_df(array: Box<dyn Array>) -> PolarsResult<DataFrame> {
    check_arrow_types(array.dtype())?;
    // A record batch has no top-level nulls; drop a validity some drivers still attach.
    let array = array.with_validity(None);
    let array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct dtype")
        .clone();
    DataFrame::try_from(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_keys_are_decoded() {
        let (_, params) = file_path_params("a.db?read%5Fonly=1&x+y=a%20b");
        assert_eq!(
            params,
            vec![
                ("read_only".to_string(), "1".to_string()),
                ("x y".to_string(), "a b".to_string())
            ]
        );
    }

    #[test]
    fn redacts_credentials() {
        let opts = |kv: &[(&str, &str)]| {
            kv.iter()
                .map(|(k, v)| (CString::new(*k).unwrap(), CString::new(*v).unwrap()))
                .collect::<Vec<_>>()
        };
        let s = secrets(&opts(&[
            ("uri", "mysql://u:p%40ss@w/rd@h:3306/db?x=1"),
            ("password", "hunter2"),
            ("adbc.snowflake.sql.client_option.auth_token", "tok3n"),
            ("username", "u"),
        ]));
        let m = redact(
            "bad uri mysql://u:p%40ss@w/rd@h:3306/db: password='hunter2' p@ss@w/rd tok3n u",
            &s,
        );
        assert_eq!(
            m,
            "bad uri mysql://u:***@h:3306/db: password='***' *** *** u"
        );
        assert!(secrets(&opts(&[("uri", "postgresql://u@h/db")])).is_empty());
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("p%40ss%2F"), "p@ss/");
        assert_eq!(percent_decode("%41"), "A");
        // not a hex escape: kept as is, also before a multi-byte character
        assert_eq!(percent_decode("%aé"), "%aé");
        assert_eq!(percent_decode("%é1"), "%é1");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%4"), "%4");
    }

    #[test]
    fn postgres_infinity() {
        use polars_core::prelude::*;

        // what the ADBC Postgres driver reads for ±infinity
        let ts = |v: i64| {
            Series::new("t".into(), [Some(0i64), Some(v)])
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap()
        };
        let date = |v: i32| {
            Series::new("d".into(), [Some(0i32), Some(v)])
                .cast(&DataType::Date)
                .unwrap()
        };
        let df = |s: Series| DataFrame::new(s.len(), vec![s.into()]).unwrap();
        for v in [-9222425352054775809, -9222425352054775808] {
            assert!(check_postgres_infinity(&df(ts(v))).is_err());
        }
        for v in [-2147472692, -2147472691] {
            assert!(check_postgres_infinity(&df(date(v))).is_err());
        }
        // the earliest Postgres values are fine
        assert!(check_postgres_infinity(&df(ts(PG_MIN_US))).is_ok());
        assert!(check_postgres_infinity(&df(date(PG_MIN_DAYS))).is_ok());
        // nested
        let list = ts(-9222425352054775809).implode().unwrap().into_series();
        assert!(check_postgres_infinity(&df(list)).is_err());
    }

    #[test]
    fn clickhouse_uris() {
        let h = clickhouse_http_uri("clickhouse://u:p%40ss@h:8123/db").unwrap();
        assert_eq!(h.uri, "http://h:8123/?database=db");
        assert_eq!(h.user.as_deref(), Some("u"));
        assert_eq!(h.password.as_deref(), Some("p@ss"));
        let h = clickhouse_http_uri("clickhouse://h:8443?secure=true").unwrap();
        assert_eq!(h.uri, "https://h:8443");
        assert!(clickhouse_http_uri("http://h:8123").is_none());
    }

    #[test]
    fn unencoded_passwords() {
        assert_eq!(
            encoded_userinfo("postgresql://polars:Ab#cd/Ef9@h:5432/db?sslmode=disable").as_deref(),
            Some("postgresql://polars:Ab%23cd%2FEf9@h:5432/db?sslmode=disable")
        );
        assert_eq!(encoded_userinfo("postgresql://u:p%40ss@h/db"), None);
        assert_eq!(encoded_userinfo("postgresql://localhost:5432/my@db"), None);
        let spec = ConnectionSpec::new(ConnectionTarget::Uri(
            "clickhouse://polars:Ab#cd/Ef9@h:8123/db".into(),
        ));
        let options = connect_options(&spec, DialectName::Clickhouse).unwrap();
        let get = |k: &str| {
            options
                .iter()
                .find(|(key, _)| key.to_str() == Ok(k))
                .map(|(_, v)| v.to_str().unwrap().to_string())
        };
        assert_eq!(get("uri").as_deref(), Some("http://h:8123/?database=db"));
        assert_eq!(get("password").as_deref(), Some("Ab#cd/Ef9"));
        let spec = ConnectionSpec::new(ConnectionTarget::Uri(
            "postgresql://polars:Ab#cd/Ef9@h:5432/db".into(),
        ));
        let options = connect_options(&spec, DialectName::Postgresql).unwrap();
        assert_eq!(
            options[0].1.to_str().unwrap(),
            "postgresql://polars:Ab%23cd%2FEf9@h:5432/db?connect_timeout=10"
        );
    }

    #[test]
    fn userinfo() {
        assert_eq!(
            with_userinfo("postgresql://u@h:5432/db", None, Some("p@ss:w/rd")).as_deref(),
            Some("postgresql://u:p%40ss%3Aw%2Frd@h:5432/db")
        );
        assert_eq!(
            with_userinfo("postgresql://old:x@h/db?a=1", Some("new"), Some("y")).as_deref(),
            Some("postgresql://new:y@h/db?a=1")
        );
        assert_eq!(
            with_userinfo("mysql://h", Some("u"), None).as_deref(),
            Some("mysql://u@h")
        );
        assert_eq!(with_userinfo("/tmp/a.db", Some("u"), None), None);
        assert_eq!(
            without_credential_params(
                "postgresql://u:p@h/db?password=x&sslmode=require",
                false,
                true
            ),
            "postgresql://u:p@h/db?sslmode=require"
        );
        assert_eq!(
            without_credential_params("mysql://h/db?User=a&password=x", true, false),
            "mysql://h/db?password=x"
        );
        // An `@` in a parameter isn't the end of the userinfo.
        assert_eq!(
            with_userinfo(
                "postgresql://polars:pw@127.0.0.1:5432/polars?application_name=a@b",
                Some("u"),
                None
            )
            .as_deref(),
            Some("postgresql://u:pw@127.0.0.1:5432/polars?application_name=a@b")
        );
        assert_eq!(
            with_userinfo("postgresql://u:12#Sekr3t@h:5432/db", None, Some("x")).as_deref(),
            Some("postgresql://u:x@h:5432/db")
        );
        // A path with an `@` isn't a userinfo.
        assert_eq!(
            with_userinfo("postgresql://localhost:5432/my@db", Some("u"), None).as_deref(),
            Some("postgresql://u@localhost:5432/my@db")
        );
        // Only a user from the provider: the URI's password stays.
        assert_eq!(
            with_userinfo("postgresql://old:p%40ss@h/db", Some("new"), None).as_deref(),
            Some("postgresql://new:p%40ss@h/db")
        );
        assert_eq!(
            with_userinfo("postgresql://old:pa#ss@h/db", Some("new"), Some("y")).as_deref(),
            Some("postgresql://new:y@h/db")
        );
    }

    #[test]
    fn sqlite_paths() {
        let sqlite_path = |uri: &str| super::sqlite_path(uri).unwrap();
        // As SQLAlchemy reads them: three slashes are relative, four absolute.
        assert_eq!(sqlite_path("sqlite:///tmp/a.db"), "tmp/a.db");
        assert_eq!(sqlite_path("sqlite:////tmp/a.db"), "/tmp/a.db");
        assert_eq!(sqlite_path("sqlite:rel.db"), "rel.db");
        assert_eq!(sqlite_path("sqlite:///a%20b.db"), "a b.db");
        assert_eq!(sqlite_path("sqlite://"), ":memory:");
        assert_eq!(sqlite_path("sqlite:///:memory:"), ":memory:");
        assert_eq!(sqlite_path("file:a.db?mode=ro"), "file:a.db?mode=ro");
        assert_eq!(
            sqlite_path("sqlite:///file:a.db?mode=ro"),
            "file:a.db?mode=ro"
        );
        assert_eq!(
            sqlite_path("sqlite:////t/a%3Fb.db?mode=ro"),
            "file:/t/a%3Fb.db?mode=ro"
        );
        assert_eq!(sqlite_path("sqlite:////C:/t/a.db"), "C:/t/a.db");
        // As the Python `sqlite3` backend: SQLAlchemy's `uri=true` is implied.
        assert_eq!(
            sqlite_path("sqlite:////t/a.db?mode=ro&uri=true"),
            "file:/t/a.db?mode=ro"
        );
        assert!(super::sqlite_path("sqlite:////t/a.db?timeout=5").is_err());
    }

    #[test]
    fn duckdb_paths() {
        let opts = |uri: &str| duckdb_options(uri);
        assert_eq!(
            opts("duckdb:///d.db"),
            [("path".to_string(), "d.db".to_string())]
        );
        assert_eq!(
            opts("duckdb:////t/d.db?read_only=true&threads=2"),
            [
                ("path".to_string(), "/t/d.db".to_string()),
                ("access_mode".to_string(), "READ_ONLY".to_string()),
                ("threads".to_string(), "2".to_string()),
            ]
        );
        assert_eq!(
            opts("duckdb://"),
            [("path".to_string(), ":memory:".to_string())]
        );
    }
}
