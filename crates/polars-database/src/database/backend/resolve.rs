//! Ranked, explained backend resolution.

use std::fmt::Write;

use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

use super::{BackendChoice, BackendKind, ProbeResult, adbc, python_bridge};
use crate::database::connection::{ConnectionSpec, ConnectionTarget};
use crate::database::dialect::DialectName;

/// Ranked candidates for a dialect, best first. The order is a static policy: correctness, then
/// bounded memory, then throughput.
pub fn candidates(dialect: DialectName) -> Vec<BackendKind> {
    use BackendKind as K;
    let adbc = |d: &str| K::Adbc { driver: d.into() };
    let py = |m: &str| K::Python { module: m.into() };
    let cx = K::ConnectorX;
    let sa = K::Sqlalchemy;
    match dialect {
        DialectName::Postgresql => vec![
            adbc("postgresql"),
            cx,
            py("psycopg"),
            py("psycopg2"),
            py("pg8000"),
            sa,
        ],
        DialectName::Redshift => vec![
            adbc("redshift"),
            cx,
            py("redshift_connector"),
            py("psycopg2"),
            sa,
        ],
        DialectName::Mysql => vec![
            adbc("mysql"),
            cx,
            py("MySQLdb"),
            py("mysql.connector"),
            py("pymysql"),
            py("mariadb"),
            sa,
        ],
        DialectName::Mssql => vec![
            adbc("mssql"),
            py("mssql_python"),
            cx,
            K::Odbc {
                module: "arrow_odbc".into(),
            },
            K::Odbc {
                module: "turbodbc".into(),
            },
            K::Odbc {
                module: "pyodbc".into(),
            },
            py("pymssql"),
            sa,
        ],
        DialectName::Oracle => vec![adbc("oracle"), py("oracledb"), cx, sa],
        DialectName::Sqlite => vec![adbc("sqlite"), cx, py("sqlite3"), sa],
        DialectName::Duckdb => vec![py("duckdb"), adbc("duckdb"), sa],
        DialectName::Snowflake => vec![adbc("snowflake"), py("snowflake.connector"), sa],
        DialectName::Bigquery => vec![adbc("bigquery"), py("google.cloud.bigquery"), cx, sa],
        DialectName::Databricks => vec![adbc("databricks"), py("databricks.sql"), sa],
        DialectName::Clickhouse => vec![
            adbc("clickhouse"),
            py("clickhouse_connect"),
            cx,
            py("clickhouse_driver"),
            sa,
        ],
        DialectName::Trino => vec![adbc("trino"), cx, py("trino"), sa],
        DialectName::Flightsql => vec![adbc("flightsql")],
        DialectName::Exasol => vec![adbc("exasol"), py("pyexasol"), sa],
        DialectName::Singlestore => vec![adbc("singlestore"), py("pymysql"), sa],
        DialectName::Teradata => vec![adbc("teradata"), py("teradatasql"), sa],
        DialectName::Generic => vec![sa],
    }
}

/// Install hint for a backend, for error messages.
pub fn install_hint(kind: &BackendKind) -> String {
    match kind {
        BackendKind::Adbc { driver } if has_adbc_driver_manager() => adbc_install_hint(driver),
        BackendKind::Adbc { driver } if adbc_driver_installed(kind, driver) => {
            "pip install adbc-driver-manager".into()
        },
        BackendKind::Adbc { driver } => format!(
            "pip install adbc-driver-manager, and: {}",
            adbc_install_hint(driver)
        ),
        BackendKind::ConnectorX => "pip install connectorx".into(),
        BackendKind::Sqlalchemy => {
            "pip install sqlalchemy  (plus the dialect's DB-API driver)".into()
        },
        BackendKind::Odbc { module } => format!("pip install {}", pip_name(module)),
        BackendKind::Python { module } if module == "sqlite3" => {
            "(part of the Python standard library)".into()
        },
        BackendKind::Python { module } => format!("pip install {}", pip_name(module)),
    }
}

/// Where an ADBC driver is published: the dbc registry (public or behind a Columnar login),
/// and, for the Apache drivers, PyPI. Checked 2026-10-24 with `dbc search` and PyPI.
pub(crate) fn adbc_install_hint(driver: &str) -> String {
    const DBC: &[&str] = &[
        "bigquery",
        "clickhouse",
        "databricks",
        "duckdb",
        "exasol",
        "flightsql",
        "mssql",
        "mysql",
        "postgresql",
        "redshift",
        "snowflake",
        "sqlite",
        "trino",
    ];
    const DBC_PRIVATE: &[&str] = &["oracle", "teradata"];
    const PYPI: &[&str] = &["bigquery", "flightsql", "postgresql", "snowflake", "sqlite"];
    let pypi = if PYPI.contains(&driver) {
        format!("   # or: pip install adbc-driver-{driver}")
    } else {
        String::new()
    };
    if DBC.contains(&driver) {
        format!("dbc install {driver}{pypi}")
    } else if DBC_PRIVATE.contains(&driver) {
        format!("dbc install {driver}   # needs a Columnar account: dbc auth login")
    } else {
        format!("(no ADBC driver for {driver} is published; pass its library or manifest path)")
    }
}

fn pip_name(module: &str) -> &str {
    match module {
        "MySQLdb" => "mysqlclient",
        "mysql.connector" => "mysql-connector-python",
        "psycopg" => "'psycopg[binary]'",
        "psycopg2" => "psycopg2-binary",
        "snowflake.connector" => "snowflake-connector-python",
        "google.cloud.bigquery" => "google-cloud-bigquery",
        "databricks.sql" => "databricks-sql-connector",
        "clickhouse_connect" => "clickhouse-connect",
        "clickhouse_driver" => "clickhouse-driver",
        "arrow_odbc" => "arrow-odbc",
        "mssql_python" => "mssql-python",
        other => other,
    }
}

/// One row of the availability report (`show_drivers`).
#[derive(Clone, Debug)]
pub struct CandidateStatus {
    pub dialect: DialectName,
    pub kind: BackendKind,
    pub probe: Option<ProbeResult>,
}

/// Whether ADBC drivers can be loaded here: through Python's `adbc_driver_manager`.
fn has_adbc_driver_manager() -> bool {
    python_bridge().is_some_and(|b| b.has_adbc_driver_manager())
}

/// Whether an ADBC driver is installed, whether or not it can be loaded.
fn adbc_driver_installed(kind: &BackendKind, driver: &str) -> bool {
    adbc::find_driver(driver).is_some() || python_bridge().and_then(|b| b.probe(kind)).is_some()
}

/// Probes one backend on this machine. Cheap and side-effect free.
pub fn probe(kind: &BackendKind) -> Option<ProbeResult> {
    match kind {
        // The driver alone can't be loaded: unavailable, so another backend is picked.
        BackendKind::Adbc { .. } if !has_adbc_driver_manager() => None,
        BackendKind::Adbc { driver } => {
            if let Some(found) = adbc::find_driver(driver) {
                return Some(ProbeResult {
                    version: found.version.map(Into::into),
                    location: Some(
                        found
                            .manifest
                            .unwrap_or(found.library)
                            .to_string_lossy()
                            .as_ref()
                            .into(),
                    ),
                });
            }
            // A pip-installed `adbc_driver_<name>` package carries the library.
            python_bridge().and_then(|b| b.probe(kind))
        },
        kind => python_bridge().and_then(|b| b.probe(kind)),
    }
}

pub fn candidate_statuses(dialect: DialectName) -> Vec<CandidateStatus> {
    candidates(dialect)
        .into_iter()
        .map(|kind| CandidateStatus {
            dialect,
            probe: probe(&kind),
            kind,
        })
        .collect()
}

/// Determines the dialect of a connection.
pub fn detect_dialect(
    spec: &ConnectionSpec,
    user: Option<DialectName>,
) -> PolarsResult<DialectName> {
    if let Some(d) = user {
        return Ok(d);
    }
    Ok(match &spec.target {
        ConnectionTarget::Uri(_) => {
            let scheme = spec.scheme().unwrap_or_default();
            if scheme == "profile" {
                // The profile names its driver; its dialect is resolved from that name.
                DialectName::Generic
            } else {
                DialectName::from_scheme(&scheme)
            }
        },
        ConnectionTarget::Odbc(_) => spec
            .odbc_driver_name()
            .map_or(DialectName::Generic, |n| DialectName::from_odbc_driver(&n)),
        #[cfg(feature = "python")]
        ConnectionTarget::Factory(_) | ConnectionTarget::Live(_) => match python_bridge() {
            Some(b) => b.connection_dialect(spec)?,
            None => DialectName::Generic,
        },
    })
}

/// Which backends can serve this connection form at all.
fn allowed_for_target(spec: &ConnectionSpec, kind: &BackendKind) -> bool {
    match &spec.target {
        ConnectionTarget::Uri(_) => !matches!(kind, BackendKind::Odbc { .. }),
        ConnectionTarget::Odbc(_) => matches!(kind, BackendKind::Odbc { .. }),
        #[cfg(feature = "python")]
        ConnectionTarget::Factory(_) | ConnectionTarget::Live(_) => {
            matches!(kind, BackendKind::Python { .. })
        },
    }
}

/// Parses the `+driver` part of a URI's scheme into a backend.
///
/// A URI often comes from configuration (`DATABASE_URL`), so its pin is a bare driver name:
/// paths to a driver library or manifest are taken only from `driver=`. `+pyodbc` and
/// `+turbodbc` URIs are SQLAlchemy's (`mssql+pyodbc://…?odbc_connect=…`).
fn parse_scheme_driver(pin: &str, dialect: DialectName) -> PolarsResult<BackendKind> {
    polars_ensure!(
        !pin.is_empty() && pin.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
        InvalidOperation:
        "the URI scheme's driver part '+{pin}' is not a driver name; a driver library or \
        manifest can be pinned only with `driver=`"
    );
    match pin {
        "pyodbc" | "turbodbc" => Ok(BackendKind::Sqlalchemy),
        _ => parse_driver_pin(pin, dialect),
    }
}

/// Parses a driver pin (`driver=` or a `+driver` URI suffix) into a backend.
pub fn parse_driver_pin(pin: &str, dialect: DialectName) -> PolarsResult<BackendKind> {
    let pin = pin.trim();
    polars_ensure!(!pin.is_empty(), InvalidOperation: "driver must not be empty");
    let lower = pin.to_ascii_lowercase();
    // Keywords and module names in any case (`"ADBC"`, `"mysqldb"`); paths as given.
    let is_path = pin.contains(['/', '\\'])
        || pin.ends_with(".so")
        || pin.ends_with(".dylib")
        || pin.ends_with(".dll")
        || pin.ends_with(".toml");
    let pin = if is_path || pin.contains(':') {
        pin
    } else {
        lower.as_str()
    };
    let kind = match pin {
        "adbc" => match dialect.adbc_driver() {
            Some(d) => BackendKind::Adbc { driver: d.into() },
            None => polars_bail!(
                InvalidOperation: "no ADBC driver is known for dialect '{dialect}'; pass `driver=\"adbc:<name>\"`"
            ),
        },
        "connectorx" => BackendKind::ConnectorX,
        "sqlalchemy" => BackendKind::Sqlalchemy,
        "odbc" => BackendKind::Odbc {
            module: "arrow_odbc".into(),
        },
        "arrow_odbc" | "turbodbc" | "pyodbc" => BackendKind::Odbc { module: pin.into() },
        // A path to a driver library or manifest.
        p if is_path => BackendKind::Adbc { driver: p.into() },
        p if p.contains(':') => BackendKind::parse(p)?,
        "native" => polars_bail!(
            InvalidOperation: "the native database readers are not available yet; use `driver=\"adbc\"`"
        ),
        module => BackendKind::Python {
            module: known_module(module),
        },
    };
    // A bare name must be one of the dialect's drivers: pinning a driver of another
    // database only fails later, with that driver's confusing error. `"python:<module>"`
    // and ADBC driver names or paths are taken as given.
    let bare = !is_path && !pin.contains(':');
    if bare && !matches!(kind, BackendKind::Adbc { .. }) {
        let ranked = candidates(dialect);
        if !ranked.contains(&kind) {
            let names = ranked
                .iter()
                .map(|k| match k {
                    BackendKind::Adbc { .. } => "adbc".to_string(),
                    BackendKind::Python { module } | BackendKind::Odbc { module } => {
                        module.to_string()
                    },
                    k => k.to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            polars_bail!(
                InvalidOperation:
                "driver='{pin}' is not a driver for dialect '{dialect}'; use one of: {names} \
                (or `driver=\"python:<module>\"` for another DB-API module)"
            );
        }
    }
    Ok(kind)
}

/// The importable module for a driver name in any case: SQLAlchemy driver names
/// (`postgresql+psycopg2`) and the modules of every dialect's candidates.
fn known_module(name: &str) -> PlSmallStr {
    let alias = python_module_alias(name);
    if alias != name {
        return alias.into();
    }
    DialectName::all()
        .flat_map(candidates)
        .find_map(|k| match k {
            BackendKind::Python { module } if module.eq_ignore_ascii_case(name) => Some(module),
            _ => None,
        })
        .unwrap_or_else(|| name.into())
}

/// Maps SQLAlchemy driver names (`postgresql+psycopg2`) to importable modules.
fn python_module_alias(name: &str) -> &str {
    match name {
        "mysqldb" | "mysqlclient" => "MySQLdb",
        "mysqlconnector" => "mysql.connector",
        "snowflake" => "snowflake.connector",
        "bigquery" => "google.cloud.bigquery",
        "databricks" => "databricks.sql",
        "oracle" | "cx_oracle" => "oracledb",
        "pysqlite" => "sqlite3",
        other => other,
    }
}

/// Picks the backend for a scan on this (the planning) machine.
pub fn resolve_backend(
    spec: &ConnectionSpec,
    dialect: DialectName,
    driver_pin: Option<&str>,
) -> PolarsResult<BackendChoice> {
    #[cfg(feature = "python")]
    if matches!(
        spec.target,
        ConnectionTarget::Factory(_) | ConnectionTarget::Live(_)
    ) {
        // A Python connection object is served by its own module, whatever its dialect.
        let Some(bridge) = python_bridge() else {
            polars_bail!(ComputeError: "connection objects need Python")
        };
        let module = bridge.connection_module(spec)?;
        return Ok(BackendChoice {
            kind: BackendKind::Python { module },
            version: None,
            location: None,
            reason: "connection object".into(),
        });
    }

    // An explicit pin is final: unavailable is an error, never a silent fallback.
    let pin = driver_pin
        .map(|p| (PlSmallStr::from(p), false))
        .or_else(|| spec.scheme_driver().map(|p| (p, true)));
    if let Some((pin, from_uri)) = pin {
        let kind = if from_uri {
            parse_scheme_driver(&pin, dialect)?
        } else {
            parse_driver_pin(&pin, dialect)?
        };
        let Some(found) = probe(&kind) else {
            if let BackendKind::Adbc { driver } = &kind
                && adbc_driver_installed(&kind, driver)
            {
                polars_bail!(
                    ComputeError:
                    "the pinned database driver '{kind}' is installed, but ADBC drivers are \
                    loaded through Python's adbc_driver_manager, which is not installed\n  \
                    install it with: pip install adbc-driver-manager"
                )
            }
            polars_bail!(
                ComputeError:
                "the pinned database driver '{kind}' is not installed\n  install it with: {}",
                install_hint(&kind)
            )
        };
        return Ok(BackendChoice {
            kind,
            version: found.version,
            location: found.location,
            reason: format!("pinned with driver='{pin}'").into(),
        });
    }

    let ranked = candidates(dialect);
    let mut checked = vec![];
    for kind in ranked.iter().filter(|k| allowed_for_target(spec, k)) {
        if let Some(found) = probe(kind) {
            let mut reason = String::from("first available of [");
            for c in &checked {
                write!(reason, "{c} ✗, ").unwrap();
            }
            write!(reason, "{kind} ✓]").unwrap();
            return Ok(BackendChoice {
                kind: kind.clone(),
                version: found.version,
                location: found.location,
                reason: reason.into(),
            });
        }
        checked.push(kind.to_string());
    }

    let mut msg = format!(
        "no driver available for database dialect '{dialect}' ({})\n  Install one of (best first):\n",
        spec.target
    );
    for kind in ranked.iter().filter(|k| allowed_for_target(spec, k)) {
        writeln!(msg, "    {}", install_hint(kind)).unwrap();
    }
    write!(
        msg,
        "  Or pin one with driver=..., or pass a connection factory.\n  Checked ADBC manifests in {:?}",
        adbc::manifest_search_dirs()
    )
    .unwrap();
    polars_bail!(ComputeError: "{msg}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_drivers_are_names() {
        let pg = DialectName::Postgresql;
        for path in ["x.toml", "/abs/x.so", "evil.so", "a-b"] {
            assert!(parse_scheme_driver(path, pg).is_err(), "{path}");
        }
        assert_eq!(
            parse_scheme_driver("psycopg2", pg).unwrap(),
            BackendKind::Python {
                module: "psycopg2".into()
            }
        );
        // SQLAlchemy's SQL Server URIs
        assert_eq!(
            parse_scheme_driver("pyodbc", DialectName::Mssql).unwrap(),
            BackendKind::Sqlalchemy
        );
        // a path still pins through `driver=`
        assert!(matches!(
            parse_driver_pin("/abs/x.so", pg).unwrap(),
            BackendKind::Adbc { .. }
        ));
    }
}
