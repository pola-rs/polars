use std::fmt;
use std::hash::{Hash, Hasher};
#[cfg(feature = "python")]
use std::sync::Arc;

use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
use serde::{Deserialize, Serialize};

/// Where a database scan connects to.
#[derive(Clone, Serialize, Deserialize)]
pub enum ConnectionTarget {
    /// `scheme://…`, including `scheme+driver://…` and `profile://name`.
    Uri(PlSmallStr),
    /// An ODBC connection string, e.g. `Driver={…};Server=…`.
    Odbc(PlSmallStr),
    /// A zero-argument Python callable that returns a DB-API connection.
    #[cfg(feature = "python")]
    Factory(Arc<PythonObject>),
    /// A live Python connection object. Usable on the local engines only.
    #[cfg(feature = "python")]
    Live(Arc<LiveConnection>),
}

/// A live connection object can't be sent anywhere, so it refuses to serialize.
#[cfg(feature = "python")]
#[derive(Clone)]
pub struct LiveConnection(pub PythonObject);

#[cfg(feature = "python")]
impl Serialize for LiveConnection {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "scan_database with a live connection object cannot be serialized; \
            pass a URI or a connection factory instead",
        ))
    }
}

#[cfg(feature = "python")]
impl<'de> Deserialize<'de> for LiveConnection {
    fn deserialize<D: serde::Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "scan_database with a live connection object cannot be deserialized",
        ))
    }
}

/// Everything needed to open a connection, as given by the user.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub target: ConnectionTarget,
    /// Driver-specific connect options (ADBC database options, ConnectorX `protocol`, …).
    pub connect_options: Vec<(PlSmallStr, PlSmallStr)>,
    /// Driver-specific statement options.
    pub execute_options: Vec<(PlSmallStr, PlSmallStr)>,
    /// Called where the connection is opened; returns `{user, password}` overrides.
    #[cfg(feature = "python")]
    pub credential_provider: Option<Arc<PythonObject>>,
    /// Bound parameters of a query source, passed to the driver's own binding. Never
    /// interpolated into the SQL.
    #[cfg(feature = "python")]
    pub query_params: Option<Arc<PythonObject>>,
    /// A stable identity of `query_params` (their `repr`), for scan deduplication.
    pub query_params_key: Option<PlSmallStr>,
    /// `(connect_options, execute_options)` with their Python values, for Python backends:
    /// driver keyword arguments aren't always strings.
    #[cfg(feature = "python")]
    pub python_options: Option<Arc<PythonObject>>,
    /// A stable identity of `python_options` (as `query_params_key`): their string forms in
    /// `connect_options` / `execute_options` don't tell `1` from `"1"`.
    pub python_options_key: Option<PlSmallStr>,
}

impl ConnectionSpec {
    pub fn new(target: ConnectionTarget) -> Self {
        Self {
            target,
            connect_options: vec![],
            execute_options: vec![],
            #[cfg(feature = "python")]
            credential_provider: None,
            #[cfg(feature = "python")]
            query_params: None,
            query_params_key: None,
            #[cfg(feature = "python")]
            python_options: None,
            python_options_key: None,
        }
    }

    pub fn has_query_params(&self) -> bool {
        self.query_params_key.is_some()
    }

    pub fn uri(&self) -> Option<&str> {
        match &self.target {
            ConnectionTarget::Uri(uri) => Some(uri),
            _ => None,
        }
    }

    /// The URI scheme, lower-cased, without a `+driver` suffix.
    pub fn scheme(&self) -> Option<PlSmallStr> {
        let uri = self.uri()?;
        let (scheme, _) = uri.split_once(':')?;
        let base = scheme.split('+').next().unwrap_or(scheme);
        Some(base.to_ascii_lowercase().into())
    }

    /// The `+driver` suffix of a SQLAlchemy-style URI scheme, if any.
    pub fn scheme_driver(&self) -> Option<PlSmallStr> {
        let uri = self.uri()?;
        let (scheme, _) = uri.split_once(':')?;
        let (_, driver) = scheme.split_once('+')?;
        Some(driver.to_ascii_lowercase().into())
    }

    /// The URI without the `+driver` part of its scheme, for drivers that don't understand it.
    pub fn uri_without_scheme_driver(&self) -> Option<String> {
        let uri = self.uri()?;
        let (scheme, rest) = uri.split_once(':')?;
        let base = scheme.split('+').next().unwrap_or(scheme);
        Some(format!("{base}:{rest}"))
    }

    /// The `Driver=` name of an ODBC connection string.
    pub fn odbc_driver_name(&self) -> Option<PlSmallStr> {
        let ConnectionTarget::Odbc(s) = &self.target else {
            return None;
        };
        s.split(';').find_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            k.trim()
                .eq_ignore_ascii_case("driver")
                .then(|| v.trim().trim_matches(|c| c == '{' || c == '}').into())
        })
    }

    pub fn is_live(&self) -> bool {
        #[cfg(feature = "python")]
        {
            matches!(self.target, ConnectionTarget::Live(_))
        }
        #[cfg(not(feature = "python"))]
        {
            false
        }
    }

    /// A stable identity of the connection. Used for scan deduplication and caching.
    ///
    /// Secrets are hashed as they are: scans that differ only in one (a password or a client
    /// key, so another role or row-level-security identity) read different data and must not be
    /// merged. The hash is never shown.
    pub fn fingerprint(&self, state: &mut impl Hasher) {
        match &self.target {
            ConnectionTarget::Uri(uri) => {
                0u8.hash(state);
                uri.hash(state);
            },
            ConnectionTarget::Odbc(s) => {
                1u8.hash(state);
                s.hash(state);
            },
            #[cfg(feature = "python")]
            ConnectionTarget::Factory(f) => {
                2u8.hash(state);
                (Arc::as_ptr(f) as usize).hash(state);
            },
            #[cfg(feature = "python")]
            ConnectionTarget::Live(c) => {
                3u8.hash(state);
                (Arc::as_ptr(c) as usize).hash(state);
            },
        }
        // Values too: scans that differ only in an option (a search path, a role) read
        // different data and must not be merged.
        self.connect_options.hash(state);
        self.execute_options.hash(state);
        #[cfg(feature = "python")]
        self.credential_provider
            .as_ref()
            .map(|p| Arc::as_ptr(p) as usize)
            .hash(state);
        self.query_params_key.hash(state);
        self.python_options_key.hash(state);
    }

    /// The secrets in the connection as given: the URI's password and the values of
    /// secret-looking URI parameters, ODBC attributes and connect options, also
    /// percent-decoded. For scrubbing driver messages, which may echo them. Longest first, so
    /// that one secret containing another is replaced whole.
    pub fn secrets(&self) -> Vec<String> {
        let mut out = vec![];
        match &self.target {
            ConnectionTarget::Uri(uri) => {
                if let Some(range) = uri_password_range(uri) {
                    out.extend(password_secrets(&uri[range]));
                }
                if let Some((_, query)) = uri.split_once('?') {
                    for kv in query.split(['&', '#']) {
                        let Some((k, v)) = kv.split_once('=') else {
                            continue;
                        };
                        if is_secret_key(k) {
                            out.push(v.to_string());
                        } else if k.eq_ignore_ascii_case("odbc_connect")
                            || k.eq_ignore_ascii_case("odbc_connect_string")
                        {
                            for secret in odbc_secrets(&percent_decode(&v.replace('+', " "))) {
                                out.push(percent_encode_all(&secret));
                                out.push(secret);
                            }
                        }
                    }
                }
            },
            ConnectionTarget::Odbc(s) => out.extend(odbc_secrets(s)),
            #[cfg(feature = "python")]
            ConnectionTarget::Factory(_) | ConnectionTarget::Live(_) => {},
        }
        for (k, v) in &self.connect_options {
            if is_secret_key(k) {
                out.push(v.to_string());
            }
        }
        let decoded: Vec<String> = out.iter().map(|s| percent_decode(s)).collect();
        out.extend(decoded);
        out.retain(|s| !s.is_empty());
        out.sort_by_key(|s| std::cmp::Reverse(s.len()));
        out.dedup();
        out
    }
}

/// A URI password and, if it holds an unencoded `/`, `?`, `#` or `@`, its pieces: a driver
/// that misreads the URI echoes one (`invalid port "kq7"` for `kq7/Zx9=`).
pub(crate) fn password_secrets(password: &str) -> Vec<String> {
    let mut out = vec![password.to_string()];
    if password.contains(['/', '?', '#', '@']) {
        out.extend(
            password
                .split(['/', '?', '#', '@'])
                .filter(|p| !p.is_empty())
                .map(str::to_string),
        );
    }
    out
}

/// Secrets shorter than this are only replaced where they stand as a value (`:pw@`, `=pw`):
/// as bare substrings they would garble the message (`a` → `Oper***tionError`).
const MIN_BARE_SECRET_LEN: usize = 4;

/// Replaces every secret in `message` (from [`ConnectionSpec::secrets`]) with `***`.
pub fn scrub_secrets(message: &str, secrets: &[String]) -> String {
    secrets.iter().fold(message.to_string(), |m, s| {
        if s.chars().count() >= MIN_BARE_SECRET_LEN {
            return m.replace(s.as_str(), "***");
        }
        let mut out = String::with_capacity(m.len());
        let mut rest = m.as_str();
        while let Some(i) = rest.find(s.as_str()) {
            let (before, after) = (&rest[..i], &rest[i + s.len()..]);
            let opens = before.ends_with([':', '=', '"', '\'']);
            let closes = after
                .chars()
                .next()
                .is_none_or(|c| matches!(c, '@' | ';' | '&' | ' ' | '\'' | '"' | '}' | ',' | ')'));
            out.push_str(before);
            out.push_str(if opens && closes { "***" } else { s });
            rest = after;
        }
        out.push_str(rest);
        out
    })
}

/// The index of the `@` that ends the userinfo of `rest`, the part of a URI after `scheme://`.
/// Where the URI connects to: never a host found in its path or query.
///
/// The usual case: the last `@` before the first `/`, `?` or `#`. A password may also hold an
/// unencoded `/`, `?` or `#` (`u:pa/ss@host`); then there is no `@` before it, and a later `@`
/// ends the userinfo if it is followed by a host (`host`, `host:port` or `[v6]:port`), and if
/// what lies between doesn't look like a query (`h:5432/db?x=a@b` has no userinfo). Database
/// URIs have no fragments: a `#` before a later `@` is always in the password.
pub(crate) fn userinfo_end(rest: &str) -> Option<usize> {
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    if let Some(at) = rest[..end].rfind('@') {
        return Some(at);
    }
    // Without a `:` there is no password to hold the delimiter.
    if !rest[..end].contains(':') || rest.starts_with('[') {
        return None;
    }
    later_userinfo_end(rest, end, true, true)
}

/// [`userinfo_end`] for hiding the password: a password may hold both an `@` and a `/`, `?` or
/// `#` (`u:p@ss/w@host`), so a later `@` that would end the userinfo is taken too.
pub(crate) fn secret_userinfo_end(rest: &str) -> Option<usize> {
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    match rest[..end].rfind('@') {
        Some(at) => {
            Some(later_userinfo_end(rest, end, false, true).map_or(at, |later| later.max(at)))
        },
        None if !rest[..end].contains(':') || rest.starts_with('[') => None,
        None => later_userinfo_end(rest, end, false, false),
    }
}

/// The last `@` after `end` (the first `/`, `?` or `#`) that can end a userinfo. For
/// `routing`, `host:port/` ends the authority: `localhost:5432/my@db` has no userinfo (a
/// password like `12/x` is then only hidden, see [`secret_userinfo_end`]).
///
/// With `query_guard`, what lies between mustn't look like a query (`=`, `&`). Hiding a
/// password that the authority doesn't already end drops it: the password may be base64.
fn later_userinfo_end(rest: &str, end: usize, routing: bool, query_guard: bool) -> Option<usize> {
    let fragment = rest.as_bytes().get(end) == Some(&b'#');
    if routing
        && rest.as_bytes().get(end) == Some(&b'/')
        && let Some((_, port)) = rest[..end].rsplit_once(':')
        && !port.is_empty()
        && port.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    rest.match_indices('@')
        .rev()
        .map(|(at, _)| at)
        .take_while(|&at| at > end)
        .find(|&at| {
            let after = &rest[at + 1..];
            let host = &after[..after.find(['/', '?', '#']).unwrap_or(after.len())];
            is_host(host) && (!query_guard || fragment || !rest[end..at].contains(['=', '&']))
        })
}

/// `host`, `host:port`, `[v6]` or `[v6]:port` (libpq's `h1:1,h2:2` too).
fn is_host(s: &str) -> bool {
    !s.is_empty()
        && s.split(',').all(|h| {
            let (name, port) = match h.strip_prefix('[') {
                Some(v6) => match v6.split_once(']') {
                    Some((addr, port)) if !addr.is_empty() => (None, port),
                    _ => return false,
                },
                None => {
                    let (name, port) = h.split_at(h.find(':').unwrap_or(h.len()));
                    (Some(name), port)
                },
            };
            let port_ok = port.is_empty()
                || port
                    .strip_prefix(':')
                    .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
            let name_ok = name.is_none_or(|n| {
                !n.is_empty()
                    && n.bytes().all(|b| {
                        b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'%')
                    })
            });
            port_ok && name_ok
        })
}

/// Where the password of a `scheme://user:password@host` URI is.
fn uri_password_range(uri: &str) -> Option<std::ops::Range<usize>> {
    let auth_start = uri.find("://")? + 3;
    let rest = &uri[auth_start..];
    let at = secret_userinfo_end(rest)?;
    let colon = rest[..at].find(':')?;
    Some(auth_start + colon + 1..auth_start + at)
}

/// Replaces the password of a `scheme://user:password@host` URI and the values of
/// secret-looking query parameters.
pub fn redact_uri(uri: &str) -> String {
    let mut out = uri.to_string();
    if let Some(range) = uri_password_range(&out) {
        out.replace_range(range, "***");
    }
    if let Some(q) = out.find('?') {
        let (head, query) = out.split_at(q + 1);
        let redacted = query
            .split('&')
            .map(|kv| match kv.split_once('=') {
                Some((k, _)) if is_secret_key(k) => format!("{k}=***"),
                // SQLAlchemy's `mssql+pyodbc://?odbc_connect=…%3BPWD%3D…`, shown decoded
                Some((k, v))
                    if k.eq_ignore_ascii_case("odbc_connect")
                        || k.eq_ignore_ascii_case("odbc_connect_string") =>
                {
                    format!("{k}={}", redact_odbc(&percent_decode(&v.replace('+', " "))))
                },
                _ => kv.to_string(),
            })
            .collect::<Vec<_>>()
            .join("&");
        out = format!("{head}{redacted}");
    }
    out
}

/// The `key=value` attributes of an ODBC connection string, `None` for one without `=`. A value
/// in braces (`{…}`, with `}}` for `}`) may hold `;`. Values are returned as written.
fn odbc_attributes(s: &str) -> Vec<(&str, Option<&str>)> {
    let mut out = vec![];
    let mut rest = s;
    loop {
        let attr_end = rest.find(';').unwrap_or(rest.len());
        let Some(eq) = rest[..attr_end].find('=') else {
            out.push((&rest[..attr_end], None));
            match rest.get(attr_end + 1..) {
                Some(r) => rest = r,
                None => break,
            }
            continue;
        };
        let value = &rest[eq + 1..];
        let lead = value.len() - value.trim_start().len();
        let mut end = value.find(';').unwrap_or(value.len());
        if value[lead..].starts_with('{') {
            let bytes = value.as_bytes();
            let mut i = lead + 1;
            while i < bytes.len() {
                if bytes[i] == b'}' {
                    if bytes.get(i + 1) == Some(&b'}') {
                        i += 2;
                        continue;
                    }
                    break;
                }
                i += 1;
            }
            // The value ends at the first `;` after its closing brace (an unclosed one runs to
            // the end).
            let close = i.min(value.len());
            end = close + value[close..].find(';').unwrap_or(value.len() - close);
        }
        out.push((&rest[..eq], Some(&value[..end])));
        match value.get(end + 1..) {
            Some(r) => rest = r,
            None => break,
        }
    }
    out
}

/// An ODBC value without its braces, `}}` unescaped.
fn odbc_unquote(v: &str) -> String {
    match v.strip_prefix('{') {
        Some(inner) => inner.strip_suffix('}').unwrap_or(inner).replace("}}", "}"),
        None => v.to_string(),
    }
}

/// The secret values of an ODBC connection string, as written and unquoted.
fn odbc_secrets(s: &str) -> Vec<String> {
    let mut out = vec![];
    for (k, v) in odbc_attributes(s) {
        if let Some(v) = v
            && is_secret_key(k.trim())
        {
            out.push(v.trim().to_string());
            out.push(odbc_unquote(v.trim()));
        }
    }
    out
}

/// Encodes every byte but unreserved ASCII as `%XX`, as a URI query value is often written.
fn percent_encode_all(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

pub fn redact_odbc(s: &str) -> String {
    odbc_attributes(s)
        .into_iter()
        .map(|(k, v)| match v {
            Some(_) if is_secret_key(k.trim()) => format!("{k}=***"),
            Some(v) => format!("{k}={v}"),
            None => k.to_string(),
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn is_secret_key(k: &str) -> bool {
    let k = k.to_ascii_lowercase().replace(['_', '-'], "");
    // Not any `…key`: `sslkey` and `primarykey` name no secret.
    [
        "password",
        "passwd",
        "pwd",
        "passphrase",
        "secret",
        "token",
        "credential",
        "apikey",
        "accesskey",
        "privatekey",
        "accountkey",
        "encryptionkey",
        "masterkey",
        // DuckDB's `azure_storage_connection_string` holds an `AccountKey=`.
        "connectionstring",
    ]
    .iter()
    .any(|s| k.contains(s))
        // SQLCipher's `?key=`
        || k == "key"
}

/// Decodes `%XX` escapes; others are kept as they are.
pub fn percent_decode(s: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // Bytes, not `&s[..]`: the byte after `%` may start a multi-byte character.
        if bytes[i] == b'%'
            && let Some(hi) = bytes.get(i + 1).copied().and_then(hex)
            && let Some(lo) = bytes.get(i + 2).copied().and_then(hex)
        {
            out.push(hi << 4 | lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

impl fmt::Debug for ConnectionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uri(uri) => write!(f, "Uri({})", redact_uri(uri)),
            Self::Odbc(s) => write!(f, "Odbc({})", redact_odbc(s)),
            #[cfg(feature = "python")]
            Self::Factory(_) => f.write_str("Factory(<callable>)"),
            #[cfg(feature = "python")]
            Self::Live(_) => f.write_str("Live(<connection>)"),
        }
    }
}

impl fmt::Display for ConnectionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uri(uri) => f.write_str(&redact_uri(uri)),
            Self::Odbc(s) => f.write_str(&redact_odbc(s)),
            #[cfg(feature = "python")]
            Self::Factory(_) => f.write_str("<connection factory>"),
            #[cfg(feature = "python")]
            Self::Live(_) => f.write_str("<live connection>"),
        }
    }
}

impl fmt::Debug for ConnectionSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionSpec")
            .field("target", &self.target)
            .field(
                "connect_options",
                &self
                    .connect_options
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::hash::BuildHasher;

    use super::*;

    #[test]
    fn redacts_passwords() {
        assert_eq!(
            redact_uri("postgresql://u:secret@h:5432/db?sslmode=require&password=x"),
            "postgresql://u:***@h:5432/db?sslmode=require&password=***"
        );
        assert_eq!(redact_uri("sqlite:///tmp/a.db"), "sqlite:///tmp/a.db");
        assert_eq!(redact_uri("postgresql://u@h/db"), "postgresql://u@h/db");
        assert_eq!(
            redact_odbc("Driver={X};Server=s;PWD=abc;UID=u"),
            "Driver={X};Server=s;PWD=***;UID=u"
        );
    }

    #[test]
    fn redacts_passwords_with_uri_delimiters() {
        for pw in [
            "pa#Sekr3t",
            "pa?Sekr3t",
            "pa/Sekr3t",
            "pa#S@kr3t",
            "p@ss/w#rd",
        ] {
            let uri = format!("teradata://u:{pw}@h/db?x=1");
            assert_eq!(redact_uri(&uri), "teradata://u:***@h/db?x=1", "{pw}");
            let spec = ConnectionSpec::new(ConnectionTarget::Uri(uri.into()));
            assert!(spec.secrets().contains(&pw.to_string()), "{pw}");
        }
        // A password that starts like a port.
        for pw in ["12#Sekr3t", "12/Sekr3t", "12?Sekr3t"] {
            let uri = format!("postgresql://u:{pw}@h:5432/db");
            assert_eq!(redact_uri(&uri), "postgresql://u:***@h:5432/db", "{pw}");
        }
        // An `@` in a parameter isn't a password.
        for uri in [
            "postgresql://u@h/db?application_name=a@b",
            "sqlite:///tmp/a:b@c.db",
        ] {
            assert_eq!(redact_uri(uri), uri);
        }
        // After `host:port`, it may be (`u:12/x=@h`): hidden, though the URI connects to `h`.
        assert_eq!(
            redact_uri("clickhouse://h:8123/db?user=a@b"),
            "clickhouse://h:***@b"
        );
        assert_eq!(
            redact_uri("postgresql://polars:pw@127.0.0.1:5432/polars?application_name=a@b"),
            "postgresql://polars:***@127.0.0.1:5432/polars?application_name=a@b"
        );
        let spec = ConnectionSpec::new(ConnectionTarget::Uri(
            "postgresql://u:a%23b@h/db?sslkey=/k&api_key=k123&sslmode=require".into(),
        ));
        assert_eq!(spec.secrets(), ["a%23b", "k123", "a#b"]);
    }

    #[test]
    fn userinfo_for_routing() {
        let end = |rest: &'static str| userinfo_end(rest).map(|at| &rest[..at]);
        assert_eq!(end("localhost:5432/my@db"), None);
        assert_eq!(end("u:pa/ss@h:5432/db"), Some("u:pa/ss"));
        assert_eq!(end("u:12#x@h:5432/db"), Some("u:12#x"));
        // `12/x` reads as a port and a path: only hidden
        assert_eq!(end("u:12/x@h:5432/db"), None);
        assert_eq!(redact_uri("pg://u:12/Sekr3t@h/db"), "pg://u:***@h/db");
        assert_eq!(
            redact_uri("mysql://polars:kq7/Zx9+Ab==@127.0.0.1:3306/polars"),
            "mysql://polars:***@127.0.0.1:3306/polars"
        );
        assert_eq!(end("polars:kq7/Zx9+Ab==@127.0.0.1:3306/polars"), None);
        let spec = ConnectionSpec::new(ConnectionTarget::Uri(
            "mysql://polars:kq7/Zx9+Ab==@127.0.0.1:3306/polars".into(),
        ));
        assert_eq!(
            scrub_secrets(
                r#"invalid integer value "kq7" for "port"; invalid port ":kq7""#,
                &spec.secrets()
            ),
            r#"invalid integer value "***" for "port"; invalid port ":***""#
        );
        for k in [
            "key",
            "encryption_key",
            "MASTER_KEY",
            "azure_storage_connection_string",
        ] {
            assert!(is_secret_key(k), "{k}");
        }
        assert!(!is_secret_key("sslkey"));
    }

    #[test]
    fn short_secrets_are_scrubbed_only_as_values() {
        let secrets = ["a".to_string(), "Sekr3t".to_string()];
        assert_eq!(
            scrub_secrets(
                "OperationalError: bad dsn postgresql://u:a@h/db?password=a; Sekr3t",
                &secrets
            ),
            "OperationalError: bad dsn postgresql://u:***@h/db?password=***; ***"
        );
    }

    #[test]
    fn redacts_braced_odbc_values() {
        let s = "Driver={ODBC Driver 18 for SQL Server};Server=s;UID=u;PWD={a;Sek}}r3t};X=1";
        assert_eq!(
            redact_odbc(s),
            "Driver={ODBC Driver 18 for SQL Server};Server=s;UID=u;PWD=***;X=1"
        );
        let spec = ConnectionSpec::new(ConnectionTarget::Odbc(s.into()));
        assert!(spec.secrets().contains(&"a;Sek}r3t".to_string()));
        assert!(!redact_odbc(s).contains("Sek"));
        assert_eq!(redact_odbc("DSN=x;Trusted;"), "DSN=x;Trusted;");
        assert_eq!(redact_odbc("PWD={unclosed;x"), "PWD=***");
        assert_eq!(
            scrub_secrets("bad dsn PWD={a;Sek}}r3t} / a;Sek}r3t", &spec.secrets()),
            "bad dsn PWD=*** / ***"
        );
    }

    #[test]
    fn redacts_odbc_connect() {
        let uri = "mssql+pyodbc://@h/?odbc_connect=DRIVER%3D%7BX%7D%3BPWD%3D%7BHunter%3B2%7D&x=1";
        let redacted = redact_uri(uri);
        assert_eq!(
            redacted,
            "mssql+pyodbc://@h/?odbc_connect=DRIVER={X};PWD=***&x=1"
        );
        let spec = ConnectionSpec::new(ConnectionTarget::Uri(uri.into()));
        let secrets = spec.secrets();
        assert!(secrets.iter().any(|s| s == "Hunter;2"), "{secrets:?}");
        assert!(!scrub_secrets(uri, &secrets).contains("Hunter"));
    }

    #[test]
    fn fingerprints_differ_in_secrets() {
        let fp = |uri: &str| {
            let mut h = polars_utils::aliases::PlFixedStateQuality::default().build_hasher();
            ConnectionSpec::new(ConnectionTarget::Uri(uri.into())).fingerprint(&mut h);
            h.finish()
        };
        assert_ne!(
            fp("postgresql://u@h/db?sslkey=/certs/a.key"),
            fp("postgresql://u@h/db?sslkey=/certs/b.key")
        );
        assert_eq!(fp("postgresql://u:p@h/db"), fp("postgresql://u:p@h/db"));
    }

    #[test]
    fn scheme_parts() {
        let spec = ConnectionSpec::new(ConnectionTarget::Uri("PostgreSQL+psycopg://u@h/db".into()));
        assert_eq!(spec.scheme().as_deref(), Some("postgresql"));
        assert_eq!(spec.scheme_driver().as_deref(), Some("psycopg"));
        assert_eq!(
            spec.uri_without_scheme_driver().as_deref(),
            Some("PostgreSQL://u@h/db")
        );
        let odbc = ConnectionSpec::new(ConnectionTarget::Odbc(
            "Driver={ODBC Driver 18 for SQL Server};Server=x".into(),
        ));
        assert_eq!(
            odbc.odbc_driver_name().as_deref(),
            Some("ODBC Driver 18 for SQL Server")
        );
    }
}
