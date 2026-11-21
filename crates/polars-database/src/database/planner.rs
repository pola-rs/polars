//! Planning-time work: resolving a scan (connection, schema, catalog) and compiling its SQL.
//!
//! Both run in the resolver (`crate::resolver`), which Polars calls on the planning machine: the
//! client for local runs, the scheduler for distributed runs. Workers only receive the finished
//! SQL.

use std::sync::Arc;

use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

use super::backend::resolve::{detect_dialect, object_info, resolve_backend};
use super::backend::{BackendKind, DatabaseSession, open_backend, open_session_with_retries};
use super::catalog::{self, CatalogInfo};
use super::config::{
    ColumnMeta, CompiledPartition, CompiledScan, DatabaseScanConfig, TableStats, session_setup_runs,
};
use super::connection::ConnectionSpec;
use super::dialect::DialectName;
use super::options::{Consistency, DatabaseRelation, DatabaseScanOptions, PartitionSpec};
use super::partition::{self, Partition};
use super::schema::{declared_dtype, normalize_dtype};
use super::sql::{
    QueryFacts, SqlBinOp, SqlExpr, SqlFrom, SqlLiteral, SqlOrderBy, SqlQuery, SqlRenderer,
    SqlSelectItem, sql_renderer,
};

/// Below this estimated size, `partition_on="auto"` reads a table with one query.
const MIN_PARTITIONED_BYTES: u64 = 128 * 1024 * 1024;
/// Target bytes per partition for `partition_on="auto"`.
const BYTES_PER_PARTITION: u64 = 256 * 1024 * 1024;
/// Rows sampled to type columns a row-wise driver could not type from an empty result.
const TYPE_SAMPLE_ROWS: u64 = 1000;

fn verbose() -> bool {
    polars_core::config::verbose()
}

/// The `FROM` target of a relation, for the fixed discovery and catalog SQL.
fn quoted_relation(dialect: DialectName, relation: &DatabaseRelation) -> String {
    match relation {
        DatabaseRelation::Table { parts, .. } => parts
            .iter()
            .map(|p| dialect.quote_ident(p))
            .collect::<Vec<_>>()
            .join("."),
        DatabaseRelation::Query { sql } => format!("({}) pl_q", wrappable(sql)),
    }
}

fn strip_statement_end(sql: &str) -> &str {
    sql.trim().trim_end_matches(';').trim_end()
}

/// The query as it goes inside `(...)`: a newline ends a trailing `--` (or MySQL `#`)
/// comment.
pub(crate) fn wrappable(sql: &str) -> String {
    let sql = strip_statement_end(sql);
    if sql.contains("--") || sql.contains('#') {
        format!("{sql}\n")
    } else {
        sql.to_string()
    }
}

/// Statement keywords that write or change state. Used when the query can't be parsed.
const WRITE_KEYWORDS: &[&str] = &[
    "ALTER", "ANALYZE", "CALL", "COPY", "CREATE", "DELETE", "DO", "DROP", "EXEC", "EXECUTE",
    "GRANT", "INSERT", "LOCK", "MERGE", "REPLACE", "REVOKE", "TRUNCATE", "UPDATE", "UPSERT", "USE",
    "VACUUM",
];

/// `scan_database` only reads. A query source that writes is rejected, whichever driver runs
/// it; sessions are read-only too where the dialect allows it.
///
/// A query that can't be parsed (`facts` is `None` then, or doesn't parse) can't be checked:
/// it must be plainly one `SELECT`, unless every session is read-only
/// ([`read_only_session`]), so the database refuses a write itself.
fn check_read_only(
    sql: &str,
    facts: Option<&QueryFacts>,
    read_only_session: bool,
    dialect: DialectName,
) -> PolarsResult<()> {
    if let Some(facts) = facts.filter(|f| f.parses) {
        polars_ensure!(
            facts.read_only,
            InvalidOperation:
            "scan_database only runs read queries: the source must be one SELECT statement \
            that doesn't write (no INSERT/UPDATE/DELETE/DDL, also not in a CTE, no SELECT INTO)"
        );
        return Ok(());
    }
    let first = first_keyword(sql).to_ascii_uppercase();
    polars_ensure!(
        !WRITE_KEYWORDS.contains(&first.as_str()),
        InvalidOperation: "{first} statements are not valid read queries for scan_database"
    );
    polars_ensure!(
        read_only_session || plain_select(sql, dialect),
        InvalidOperation:
        "scan_database only runs read queries, and this query can't be parsed to check that \
        it is one (a WITH may hold a write); write it as one plain SELECT (no INTO, no `;` \
        inside), or connect with a URI of a driver whose sessions are read-only"
    );
    Ok(())
}

/// A query that is one `SELECT` without `INTO` (`SELECT … INTO t` creates a table), as far as
/// its words show: a statement that can't be parsed is taken as a write otherwise. SQL Server
/// runs statements with no `;` between them (`SELECT … FROM c EXEC('…')`): no word of the query
/// may start a statement there ([`TSQL_STATEMENT_WORDS`]).
fn plain_select(sql: &str, dialect: DialectName) -> bool {
    let sql = strip_statement_end(sql);
    let mut words = sql.split(|c: char| !c.is_alphanumeric() && c != '_');
    first_keyword(sql).eq_ignore_ascii_case("SELECT")
        && !sql.contains(';')
        && !words.clone().any(|w| w.eq_ignore_ascii_case("INTO"))
        && !(dialect == DialectName::Mssql
            && words.any(|w| {
                TSQL_STATEMENT_WORDS
                    .iter()
                    .any(|k| k.eq_ignore_ascii_case(w))
            }))
}

/// Whether every session of a scan is read-only, so that the database refuses a write: the
/// session setup makes it so (Postgres, MySQL), where it runs (not on a live connection,
/// nor on ConnectorX or arrow-odbc, which have no sessions).
fn read_only_session(dialect: DialectName, backend: &BackendKind, live: bool) -> bool {
    !live
        && backend.has_sessions()
        && dialect
            .session_setup()
            .iter()
            .any(|s| s.sql.contains("READ ONLY"))
}

/// Words that start a Transact-SQL statement, from SQL Server's reserved keywords (it rejects
/// one as an alias, so one after a table is a statement). Not `FETCH` (`OFFSET … FETCH NEXT`).
const TSQL_STATEMENT_WORDS: &[&str] = &[
    "ALTER",
    "BACKUP",
    "BREAK",
    "BULK",
    "CHECKPOINT",
    "CLOSE",
    "COMMIT",
    "CONTINUE",
    "CREATE",
    "DBCC",
    "DEALLOCATE",
    "DECLARE",
    "DELETE",
    "DENY",
    "DROP",
    "DUMP",
    "EXEC",
    "EXECUTE",
    "GOTO",
    "GRANT",
    "IF",
    "INSERT",
    "KILL",
    "LOAD",
    "MERGE",
    "OPEN",
    "PRINT",
    "RAISERROR",
    "READTEXT",
    "RECONFIGURE",
    "RESTORE",
    "RETURN",
    "REVERT",
    "REVOKE",
    "ROLLBACK",
    "SAVE",
    "SET",
    "SETUSER",
    "SHUTDOWN",
    "TRUNCATE",
    "UPDATE",
    "UPDATETEXT",
    "USE",
    "WAITFOR",
    "WHILE",
    "WRITETEXT",
];

/// The first word of a statement, after comments.
fn first_keyword(sql: &str) -> &str {
    let mut s = sql.trim_start();
    loop {
        if let Some(rest) = s.strip_prefix("--") {
            s = rest.split_once('\n').map_or("", |(_, r)| r).trim_start();
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.split_once("*/").map_or("", |(_, r)| r).trim_start();
        } else if let Some(rest) = s.strip_prefix('(') {
            s = rest.trim_start();
        } else {
            break;
        }
    }
    let end = s
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    &s[..end]
}

/// Why a user query can't be wrapped as a derived table, if it can't.
fn wrap_blocker(facts: &QueryFacts, dialect: DialectName) -> Option<&'static str> {
    if !facts.parses {
        return Some("the query does not parse in the dialect");
    }
    if !facts.single_select {
        return Some("the query is not a single SELECT statement");
    }
    if dialect == DialectName::Mssql && facts.has_cte {
        return Some("SQL Server does not allow a CTE inside a derived table");
    }
    if dialect == DialectName::Mssql && facts.top_level_order_by_without_limit {
        return Some("SQL Server does not allow ORDER BY without TOP inside a derived table");
    }
    if let Some(cols) = &facts.output_columns {
        let mut seen = PlHashSet::with_capacity(cols.len());
        if !cols.iter().all(|c| seen.insert(c)) {
            return Some("the query has duplicate output column names");
        }
    }
    None
}

/// What schema discovery may do with a query source.
#[derive(Default)]
struct QueryDiscovery {
    /// It can go inside `SELECT … FROM (…) WHERE 1 = 0`, which returns its columns without
    /// running it (`wrap_blocker`).
    wrappable: bool,
    parses: bool,
    /// Running it can't write, nor give other rows each time (it parses as a read without
    /// volatile functions, or every session is read-only).
    pure_read: bool,
    /// It parses, and has volatile functions (`nextval`, `random()`): it may return other
    /// rows, or change something, each time it runs.
    volatile: bool,
    /// A query with a top-level `ORDER BY` and no `LIMIT`, on a dialect that may drop it
    /// inside a derived table (MariaDB among the MySQL dialect's servers; see
    /// [`DialectName::keeps_derived_table_order`]).
    derived_order_by: bool,
    /// It parses, and a `LIMIT` (or a lock clause) ends it (`QueryFacts::top_level_limit`).
    top_level_limit: bool,
    /// It parses, and has a CTE or a subquery (`QueryFacts::nested_queries`).
    nested_queries: bool,
    /// It parses, and running it may change something (`QueryFacts::side_effects`).
    side_effects: bool,
    /// `QueryFacts::side_effects_outside_rows`.
    side_effects_outside_rows: bool,
    /// `QueryFacts::volatile_columns`.
    volatile_columns: Vec<PlSmallStr>,
    /// `QueryFacts::all_columns_volatile`.
    all_columns_volatile: bool,
    /// Every session is read-only ([`read_only_session`]): the database refuses a write.
    read_only_session: bool,
    /// The SQL that finds its schema without running it, where the usual probe would
    /// (MySQL: [`mysql_query_probe`]).
    probe: Option<String>,
}

/// Why a query with a `WITH TIES` is read without pushdown: a `LIMIT` around it is applied
/// inside it by Trino, past the ties (it is still wrapped, to find its schema and convert).
const WITH_TIES: &str = "the query has a WITH TIES, which a LIMIT around it may not respect";

const MARIADB_ORDER_BY: &str = "MariaDB ignores the ORDER BY of a query inside a derived table";

/// Why a query whose `ORDER BY` the dialect may drop in a derived table isn't wrapped.
fn derived_order_by_reason(dialect: DialectName) -> PlSmallStr {
    format!(
        "{dialect:?} may ignore the ORDER BY of a query inside a derived table (without a \
        LIMIT); the query is read as it is"
    )
    .into()
}

/// The schema probe of a MySQL query: the query with a `LIMIT 0` of its own.
///
/// Whether a MySQL / MariaDB column type is a `FLOAT(M, D)` or `DOUBLE(M, D)`: a float with
/// digits declared (not the `FLOAT(p)` precision form, which has no comma).
fn declared_float_digits(db_type: &str) -> bool {
    let t = db_type.trim().to_ascii_lowercase();
    ["float", "double", "real"].iter().any(|f| {
        t.strip_prefix(f)
            .is_some_and(|r| r.trim_start().starts_with('('))
    }) && t.split(')').next().is_some_and(|args| args.contains(','))
}

/// MySQL and MariaDB evaluate a derived table of one row (`SELECT NEXTVAL(s)`, an
/// aggregate) when they plan a query, before a `WHERE 1 = 0` (or a `LIMIT 0`) around it
/// applies: the usual probe would run the query, and the read then runs it again. The
/// query's own `LIMIT 0` avoids most of that, but not all: a derived table inside it (a
/// CTE, a subquery), and on MariaDB a constant `WHERE`, a `HAVING` or a `UNION` branch,
/// still run. A query that may change something (`QueryDiscovery::side_effects`) must not
/// run twice: it is refused then, unless every session is read-only.
fn mysql_query_probe(
    session: &mut dyn DatabaseSession,
    sql: &str,
    query: &QueryDiscovery,
) -> PolarsResult<String> {
    // a MySQL executable comment (`/*! … */`) may call anything; sqlglot drops it
    let hidden = sql.contains("/*!");
    let sql = wrappable(sql);
    let probe = if query.parses && !query.top_level_limit {
        format!("{sql} LIMIT 0")
    } else {
        // `(SELECT … LIMIT 5) LIMIT 0`
        format!("({sql}) LIMIT 0")
    };
    // A query sqlglot can't parse may call anything.
    let may_change = hidden || query.side_effects || !query.parses;
    // MariaDB runs a function in a set operation, a query without FROM or a condition,
    // and may run anything in a query that doesn't parse, or one parenthesised with a LIMIT
    let mariadb_runs = !query.parses || query.side_effects_outside_rows || query.top_level_limit;
    let may_run = hidden
        || query.nested_queries
        || (may_change && mariadb_runs && !query.read_only_session && is_mariadb(session));
    polars_ensure!(
        !(may_run && may_change && !query.read_only_session),
        InvalidOperation:
        "this query calls a function that may change something (a volatile one, or one \
        that isn't known, e.g. a stored function), or can't be inspected: MySQL may run it \
        to find the query's schema, and the read would run it again; pass schema=..., or \
        connect with a URI and a driver whose sessions are read-only (e.g. driver=\"pymysql\")"
    );
    Ok(probe)
}

/// Whether a MySQL zone name (`@@session.time_zone`, or `@@system_time_zone` under `SYSTEM`)
/// is UTC itself.
///
/// By name, never by the offset now: a zone with daylight saving time is at offset 0 for
/// half a year (London in winter), and a TIMESTAMP stored in the other half would read
/// shifted by an hour without an error. `@@system_time_zone` is the server's abbreviation
/// (`GMT`, `BST`) or its OS's name, so a bare `GMT` is London's winter, no UTC; only the
/// spellings of UTC itself are (Windows writes it out). A zone that is UTC under another
/// name is refused, with the way out in the message.
fn is_utc_zone(zone: &str) -> bool {
    matches!(
        zone.trim().to_ascii_uppercase().as_str(),
        "UTC"
            | "+00:00"
            | "-00:00"
            | "0:00"
            | "Z"
            | "ETC/UTC"
            | "UCT"
            | "ETC/UCT"
            | "ZULU"
            | "ETC/ZULU"
            | "UNIVERSAL"
            | "ETC/UNIVERSAL"
            | "COORDINATED UNIVERSAL TIME"
    )
}

/// Whether the MySQL-dialect server is a MariaDB.
fn is_mariadb(session: &mut dyn DatabaseSession) -> bool {
    match session.catalog_query("SELECT VERSION()") {
        Ok(df) => df.columns().first().is_some_and(|c| {
            c.cast(&DataType::String)
                .ok()
                .and_then(|c| c.str().ok()?.get(0).map(|v| v.to_ascii_lowercase()))
                .is_some_and(|v| v.contains("mariadb"))
        }),
        // unknown: taken as a MariaDB, whose order would be lost
        Err(_) => true,
    }
}

/// Resolves a scan: dialect, backend, schema and catalog metadata. Runs once per resolver, when
/// Polars first needs the schema. Partitions and SQL are left to [`compile_scan`].
pub fn resolve_scan(
    options: Arc<DatabaseScanOptions>,
    user_schema: Option<SchemaRef>,
) -> PolarsResult<DatabaseScanConfig> {
    let object = object_info(&options.connection)?;
    let dialect = detect_dialect(&options.connection, options.dialect, object.as_ref())?;
    // `orders` names `ORDERS` on Oracle, as it would in SQL. A table that exists only as
    // written (created quoted, `"Orders"`) is still found: see the schema discovery below.
    // A user schema means no I/O, so nothing could check the folded name: the name is used as
    // written.
    let written = options.clone();
    let mut options = if user_schema.is_none() {
        options.with_relation_case_folded(dialect)
    } else {
        options
    };
    let as_written = (!Arc::ptr_eq(&written, &options)).then_some(written);
    let backend = resolve_backend(
        &options.connection,
        dialect,
        options.driver.as_deref(),
        object.as_ref(),
    )?;
    // The session setup (UTC, …) runs on every session of the scan.
    let setup_runs = session_setup_runs(&options.connection, &backend.kind);
    // (the relation stays a query or a table below)
    let is_query = matches!(options.relation, DatabaseRelation::Query { .. });
    if options.connection.has_query_params() {
        polars_ensure!(
            is_query,
            InvalidOperation: "params can only be used with a query source"
        );
        polars_ensure!(
            !matches!(backend.kind, BackendKind::Adbc { .. } | BackendKind::ConnectorX),
            InvalidOperation:
            "params need a driver that binds parameters itself, which {} doesn't do here; \
            pin a Python driver with driver=..., e.g. driver=\"psycopg\"",
            backend.kind
        );
    }
    if verbose() {
        eprintln!(
            "[scan_database]: dialect={dialect}, backend={backend} ({})",
            backend.reason
        );
    }

    // A query that can't be wrapped runs as given, without pushdown.
    let mut pushdown_disabled: Option<PlSmallStr> =
        (!options.pushdown).then(|| "pushdown=False".into());
    let mut query = QueryDiscovery {
        wrappable: true,
        ..Default::default()
    };
    let mut query_facts: Option<QueryFacts> = None;
    // The server drops the query's top-level `ORDER BY` inside a derived table (MariaDB, or
    // a MySQL dialect server not known to keep it without I/O): a read must not wrap it.
    let mut derived_order_dropped = false;
    // columns the read refuses, with why (`DatabaseScanConfig::refused_columns`)
    let mut refused: Vec<(PlSmallStr, PlSmallStr)> = Vec::new();
    if let DatabaseRelation::Query { sql } = &options.relation {
        let sql = sql.clone();
        let facts = sql_renderer().and_then(|r| r.inspect_query(&sql, dialect));
        // (one that doesn't parse still has the functions its tokens call)
        query_facts = facts.as_ref().ok().cloned();
        let read_only_session =
            read_only_session(dialect, &backend.kind, options.connection.is_live());
        check_read_only(&sql, facts.as_ref().ok(), read_only_session, dialect)?;
        // SQLite and DuckDB rename one of two names that differ only in case inside a derived
        // table (`x:1`, `x_1`), and a qualified name picks the first of them.
        if matches!(dialect, DialectName::Sqlite | DialectName::Duckdb)
            && let Ok(QueryFacts {
                output_columns: Some(cols),
                ..
            }) = &facts
        {
            let mut seen = PlHashMap::with_capacity(cols.len());
            for c in cols {
                if let Some(other) = seen.insert(c.to_lowercase(), c)
                    && other != c
                {
                    polars_bail!(
                        InvalidOperation:
                        "the query's columns '{other}' and '{c}' differ only in case, which \
                        {dialect} can't tell apart outside the query; alias one of them"
                    );
                }
            }
        }
        // Each partition runs the query again: it must pick the same rows every time.
        if let Ok(facts) = &facts
            && facts.parses
            && (!repeatable(facts, dialect, read_only_session)
                || params_time_relative(&options.connection))
            && matches!(
                options.partitioning,
                PartitionSpec::Column { .. } | PartitionSpec::Predicates(_)
            )
        {
            polars_bail!(
                InvalidOperation:
                "this query can't be partitioned: each partition runs it again, and it isn't \
                proven to return the same rows each time: only built-in, deterministic \
                functions and operators are (not a user's function, whatever its label, a \
                clock, a session variable, a random function, a table function, an \
                aggregate that depends on the order of rows), and no LIMIT, OFFSET, \
                TABLESAMPLE, or rows numbered or picked in no set order (a window function, \
                DISTINCT ON, any_value); a function that may change something would run \
                again too, unless the driver's sessions are read-only; not proven here: {}; \
                pass partition_on=None, or partition a table",
                if facts.unproven.is_empty() {
                    "the query's other constructs (LIMIT, a window, a bare column ...)".to_string()
                } else {
                    facts.unproven.iter().map(|u| u.as_str()).collect::<Vec<_>>().join(", ")
                }
            );
        }
        // Partitions are read side by side: the query's own order would hold only within each.
        if let Ok(facts) = &facts
            && facts.top_level_order_by_without_limit
            && matches!(
                options.partitioning,
                PartitionSpec::Column { .. } | PartitionSpec::Predicates(_)
            )
        {
            polars_bail!(
                InvalidOperation:
                "this query can't be partitioned: its ORDER BY would only order the rows                 within each partition; pass partition_on=None, or sort after the scan"
            );
        }
        // `SELECT …; -- done`: the trailing `;` and comments can't go inside `(...)`.
        if let Ok(QueryFacts {
            body: Some(body), ..
        }) = &facts
            && *body != sql
        {
            options = options.with_query_sql(body.clone());
        }
        let blocker = match &facts {
            Ok(facts) => wrap_blocker(facts, dialect).map(PlSmallStr::from),
            Err(e) => Some(format!("the query could not be inspected: {e}").into()),
        };
        query = QueryDiscovery {
            wrappable: blocker.is_none(),
            parses: facts.as_ref().is_ok_and(|f| f.parses),
            // running it can't change anything, nor give other rows
            pure_read: read_only_session
                || facts
                    .as_ref()
                    .is_ok_and(|f| f.parses && f.read_only && repeatable(f, dialect, false)),
            volatile: facts.as_ref().is_ok_and(|f| f.parses && !f.repeatable),
            derived_order_by: dialect.keeps_derived_table_order() != Some(true)
                && facts
                    .as_ref()
                    .is_ok_and(|f| f.top_level_order_by_without_limit),
            top_level_limit: facts.as_ref().is_ok_and(|f| f.parses && f.top_level_limit),
            nested_queries: facts.as_ref().is_ok_and(|f| f.parses && f.nested_queries),
            side_effects: facts.as_ref().is_ok_and(|f| f.parses && f.side_effects),
            side_effects_outside_rows: facts
                .as_ref()
                .is_ok_and(|f| f.parses && f.side_effects_outside_rows),
            volatile_columns: facts
                .as_ref()
                .map(|f| f.volatile_columns.clone())
                .unwrap_or_default(),
            all_columns_volatile: facts.as_ref().is_ok_and(|f| f.all_columns_volatile),
            read_only_session,
            probe: None,
        };
        if let Some(reason) = blocker
            && pushdown_disabled.is_none()
        {
            pushdown_disabled = Some(reason);
        }
        if facts.as_ref().is_ok_and(|f| f.parses && f.with_ties) && pushdown_disabled.is_none() {
            pushdown_disabled = Some(WITH_TIES.into());
        }
        if facts.as_ref().is_ok_and(|f| f.positional_join) && pushdown_disabled.is_none() {
            pushdown_disabled = Some(POSITIONAL_REASON.into());
        }
        if facts.as_ref().is_ok_and(|f| f.unsafe_settings) && pushdown_disabled.is_none() {
            pushdown_disabled = Some(SETTINGS_REASON.into());
        }
        if query.derived_order_by {
            if dialect.keeps_derived_table_order().is_none() {
                // Without I/O, the server may be a MariaDB (see below).
                if user_schema.is_some() {
                    derived_order_dropped = true;
                    if pushdown_disabled.is_none() {
                        pushdown_disabled = Some(MARIADB_ORDER_BY.into());
                    }
                }
            } else {
                // Trino drops it (as its documentation says), and the others aren't
                // known to keep it: the wrapped query would return its rows in another
                // order (and `head` other rows).
                derived_order_dropped = true;
                if pushdown_disabled.is_none() {
                    pushdown_disabled = Some(derived_order_by_reason(dialect));
                }
            }
        }
        if verbose()
            && let Some(reason) = &pushdown_disabled
        {
            eprintln!("[scan_database]: pushdown disabled: {reason}");
        }
    }

    let mut columns: Vec<ColumnMeta>;
    let mut volatility = SourceVolatility::default();
    // The type affinity of a SQLite table's columns (`sqlite_affinity_dtype`): a REAL
    // column holds no integers, an INTEGER column no integral reals.
    let mut affinity: PlHashMap<PlSmallStr, DataType> = PlHashMap::new();
    // The columns of TEXT affinity (declared CHAR, CLOB, TEXT): SQLite compares and sorts
    // their values as text, whatever they are read as.
    let mut text_affinity: PlHashSet<PlSmallStr> = PlHashSet::new();
    let mut stats = TableStats::default();
    let mut resolution: Option<Vec<String>> = None;
    let mut mixed_found: Vec<PlSmallStr> = Vec::new();
    let schema: SchemaRef;
    // the query columns SQL Server declares `money` or `smallmoney`
    let mut money_columns: PlHashMap<PlSmallStr, String> = PlHashMap::default();
    let volatility_unchecked = user_schema.is_some();
    let user_typed = user_schema.is_some();

    if let Some(schema_) = user_schema {
        // No I/O: a user schema is authoritative, and column semantics stay conservative:
        // collations are unknown, and so are some types behind the dtypes: SQL Server's
        // legacy `datetime`, a MySQL `YEAR` (see the filter), and a SQLite column's
        // affinity (a `String` may hold numbers that compare as such). Otherwise the
        // user's `String` states a text column.
        columns = schema_
            .iter()
            .map(|(name, dtype)| ColumnMeta {
                name: name.clone(),
                dtype: dtype.clone(),
                db_type: None,
                strings: catalog::unknown_string_semantics(dialect),
                is_key: false,
                comparable: match dtype {
                    DataType::String => dialect != DialectName::Sqlite,
                    DataType::Datetime(_, _) => dialect != DialectName::Mssql,
                    // A float or decimal dtype over a column of another numeric type
                    // (`real` read as Float64, `numeric` read as a float or at another
                    // scale): the database compares the values it holds, not the ones
                    // Polars reads.
                    dt if dt.is_float() || dt.is_decimal() => false,
                    _ => true,
                },
                unstable: false,
            })
            .collect();
        // No I/O to ask the database about views and the functions sqlglot doesn't know.
        schema = schema_;
    } else {
        let backend_impl = open_backend(&backend, options.fetch.batch_size)?;
        let mut session = catalog::with_describe_cache(
            open_session_with_retries(backend_impl.as_ref(), &options.connection, dialect)?,
            dialect,
        );
        // MariaDB drops the `ORDER BY` of a derived table without `LIMIT`: the wrapped
        // query would return its rows in another order (and `head` other rows).
        // (Its schema is still found wrapped: the order doesn't matter there.)
        if query.derived_order_by
            && dialect.keeps_derived_table_order().is_none()
            && is_mariadb(session.as_mut())
        {
            derived_order_dropped = true;
            if pushdown_disabled.is_none() {
                if verbose() {
                    eprintln!("[scan_database]: pushdown disabled: {MARIADB_ORDER_BY}");
                }
                pushdown_disabled = Some(MARIADB_ORDER_BY.into());
            }
        }
        if let DatabaseRelation::Query { sql } = &options.relation
            && dialect == DialectName::Mysql
        {
            query.probe = Some(mysql_query_probe(session.as_mut(), sql, &query)?);
        }
        // The catalog first: some declared types are read as text (`text_conversion`), and
        // discovery must select them the same way the scan will.
        let discover = |session: &mut dyn DatabaseSession, options: &DatabaseScanOptions| {
            let info = match &options.relation {
                DatabaseRelation::Table { parts, .. } => catalog::table_info(
                    session,
                    dialect,
                    parts,
                    options.connection.opened_by_plugin(),
                ),
                DatabaseRelation::Query { .. } => CatalogInfo::default(),
            };
            let select_list = converted_select_list(
                dialect,
                &backend.kind,
                setup_runs,
                options.connection.opened_by_plugin(),
                &info,
            );
            let (discovered, sampled, mixed) = discover_schema(
                session,
                options,
                dialect,
                &backend.kind,
                &query,
                select_list.as_deref(),
                &info,
            )?;
            PolarsResult::Ok((info, discovered, sampled, mixed))
        };
        let (mut info, mut discovered, sampled, mixed_blobs) =
            match discover(session.as_mut(), &options) {
                Ok(found) => found,
                // No table by the folded name: the name as written, on a new session (a failed
                // statement can abort the session's transaction).
                Err(e) => match as_written {
                    Some(written) => {
                        // closed first: a pool of one connection would wait for it
                        drop(session);
                        session = catalog::with_describe_cache(
                            open_session_with_retries(
                                backend_impl.as_ref(),
                                &written.connection,
                                dialect,
                            )?,
                            dialect,
                        );
                        let found = discover(session.as_mut(), &written).map_err(|_| e)?;
                        options = written;
                        found
                    },
                    None => return Err(e),
                },
            };
        stats = info.stats;
        mixed_found = mixed_blobs;
        if !options.connection.opened_by_plugin() && !options.connection.is_live() {
            resolution = session_resolution(session.as_mut(), dialect, &options);
        }
        let names = discovered.iter_names().cloned().collect::<Vec<_>>();
        volatility = source_volatility(
            session.as_mut(),
            dialect,
            &options.relation,
            query_facts.as_ref(),
            &names,
            true,
        );
        if volatility.positional && pushdown_disabled.is_none() {
            pushdown_disabled = Some(POSITIONAL_VIEW_REASON.into());
        }
        // a view's columns hold what its query gives, whatever `pragma_table_info` says
        // (the first branch's type, of a `UNION`)
        if dialect == DialectName::Sqlite && volatility.plain_table {
            text_affinity.extend(
                info.compare_types
                    .iter()
                    .filter(|(_, t)| sqlite_text_affinity(t))
                    .map(|(n, _)| n.clone()),
            );
            affinity.extend(
                info.compare_types
                    .iter()
                    .map(|(n, t)| (n.clone(), sqlite_affinity_dtype(t))),
            );
        }
        // A query's result types decide how the database compares its columns, and fix
        // their dtypes as a table's declared types do: the same column reads the same
        // from a table and a query, whichever driver reads it. They are looked up where
        // the query is wrapped to read columns through their conversions; also without
        // pushdown, so that the values don't depend on it. (MariaDB would drop the
        // `ORDER BY` of a wrapped query.)
        if is_query && read_wrapped(&options, &pushdown_disabled, &query, dialect) {
            let names = discovered.iter_names().cloned().collect::<Vec<_>>();
            let from = quoted_relation(dialect, &options.relation);
            // Python drivers refuse DDL, ConnectorX runs each statement on its own connection.
            let temp_tables = matches!(backend.kind, BackendKind::Adbc { .. });
            info = catalog::query_info(
                session.as_mut(),
                dialect,
                &from,
                query.probe.as_deref(),
                &names,
                temp_tables,
            );
            // Only where a table's types are declared ones too: SQLite and DuckDB tables
            // are typed by their values.
            if matches!(
                dialect,
                DialectName::Postgresql
                    | DialectName::Mysql
                    | DialectName::Mssql
                    | DialectName::Oracle
                    | DialectName::Clickhouse
                    | DialectName::Trino
            ) {
                info.types = names
                    .iter()
                    .filter_map(|n| Some((n.clone(), info.compare_types.get(n)?.clone())))
                    .collect();
            }
        }

        // A ConnectorX ClickHouse query that can't be wrapped has no conversions: it reads a
        // `DateTime64(7..9)` as microseconds, silently dropping the nanoseconds. ClickHouse
        // describes the query itself; a Datetime column whose precision is above 6 or unknown is
        // refused, unless the user typed it.
        if is_query
            && dialect == DialectName::Clickhouse
            && matches!(backend.kind, BackendKind::ConnectorX)
            && !read_wrapped(&options, &pushdown_disabled, &query, dialect)
            && discovered
                .iter()
                .any(|(_, d)| matches!(d, DataType::Datetime(..)))
        {
            let described = match &options.relation {
                DatabaseRelation::Query { sql } => session
                    .query(&format!("DESCRIBE TABLE ({})", strip_statement_end(sql)))
                    .ok()
                    .map(|df| {
                        (0..df.height())
                            .filter_map(|row| {
                                Some((
                                    catalog::cell_str(&df, 0, row)?,
                                    catalog::cell_str(&df, 1, row)?,
                                ))
                            })
                            .collect::<PlHashMap<String, String>>()
                    }),
                _ => None,
            };
            for (name, dtype) in discovered.iter() {
                let typed = options
                    .schema_overrides
                    .as_ref()
                    .is_some_and(|o| o.contains(name));
                if !matches!(dtype, DataType::Datetime(..)) || typed {
                    continue;
                }
                let why = match described.as_ref().and_then(|d| d.get(name.as_str())) {
                    Some(t) => match declared_dtype(dialect, t) {
                        Some(DataType::Datetime(TimeUnit::Nanoseconds, _)) => {
                            format!("it is a {t}, whose nanoseconds ConnectorX drops")
                        },
                        _ => continue,
                    },
                    None => "its precision can't be looked up, and a DateTime64(7..9) \
                        loses its nanoseconds in ConnectorX"
                        .to_string(),
                };
                refused.push((
                    name.clone(),
                    format!(
                        "ConnectorX can't read the ClickHouse column '{name}' of this query \
                        right: {why}, and the query can't be wrapped to convert it ({}); \
                        pin another driver (driver=\"clickhouse_connect\" or \"adbc\"), \
                        CAST the column in the query (`toString(c)`), or type it with \
                        schema_overrides to accept what ConnectorX reads",
                        pushdown_disabled
                            .as_deref()
                            .unwrap_or("it can't be wrapped")
                    )
                    .into(),
                ));
            }
        }

        // DuckDB returns `TIMESTAMPTZ` in the session time zone, the machine's by default:
        // the same instants, planned in UTC as on other databases. (Its Python module reads
        // each query on a new cursor, a connection with its own settings: a session setup
        // wouldn't apply.)
        if dialect == DialectName::Duckdb {
            for (_, dtype) in discovered.iter_mut() {
                *dtype = in_utc(dtype);
            }
        }
        // A declared type fixes the dtype, whichever driver read it.
        for (name, dtype) in discovered.iter_mut() {
            let Some(t) = info.declared_type(name) else {
                continue;
            };
            if dialect.text_conversion(t, name).is_some() {
                // read as text by the dialect rules; the value stays text
                *dtype = dialect.text_conversion_dtype(t);
            } else if let Some(declared) = declared_dtype(dialect, t) {
                // A query's Postgres `numeric` shows no precision (`pg_typeof` has no type
                // modifier): psycopg's result description has it, other drivers' don't.
                let precision_from_driver = is_query
                    && dialect == DialectName::Postgresql
                    && t == "numeric"
                    && !sampled.contains(name.as_str())
                    && matches!(&backend.kind, BackendKind::Python { module }
                        if module == "psycopg" || module == "psycopg2")
                    && matches!(dtype, DataType::Decimal(p, _) if *p <= 38);
                if !precision_from_driver {
                    *dtype = declared;
                }
            }
        }
        // DuckDB's Python driver types an empty result's columns `Null`, and some results
        // aren't sampled (an empty table, a query with a `LIMIT` or a volatile function):
        // such a column's DuckDB type gives its dtype, not the fallback String.
        if dialect == DialectName::Duckdb {
            for (name, dtype) in discovered.iter_mut() {
                if sampled.contains(name.as_str())
                    && let Some(t) = info.compare_types.get(name.as_str())
                    && dialect.text_conversion(t, name).is_none()
                    && let Some(declared) = duckdb_type_dtype(t)
                {
                    *dtype = declared;
                }
            }
        }
        // A DuckDB query's column of a type read as text (bit strings, `VARINT`, …) is read
        // through its conversion: text, as from a table, not the type's bytes.
        if dialect == DialectName::Duckdb && is_query {
            for (name, dtype) in discovered.iter_mut() {
                if let Some(t) = info.compare_types.get(name.as_str())
                    && dialect.text_conversion(t, name).is_some()
                {
                    *dtype = dialect.text_conversion_dtype(t);
                }
            }
        }
        columns = discovered
            .iter()
            .map(|(name, dtype)| ColumnMeta {
                name: name.clone(),
                dtype: dtype.clone(),
                db_type: info.declared_type(name).map(Into::into).or_else(|| {
                    // A query's result type, for the filter's checks, where the column is
                    // read as it is; a DuckDB column's type, which its conversion needs.
                    info.compare_types
                        .get(name.as_str())
                        .filter(|t| {
                            let converted = column_conversion(
                                dialect,
                                &backend.kind,
                                setup_runs,
                                options.connection.opened_by_plugin(),
                                t,
                                &dialect.quote_ident(name),
                            )
                            .is_some();
                            (info.types_required && !converted)
                                || (dialect == DialectName::Duckdb && converted)
                        })
                        .map(|t| t.as_str().into())
                }),
                strings: info.string_semantics_with_setup(dialect, name, setup_runs),
                is_key: info.key_column.as_ref() == Some(name),
                comparable: !info.incomplete
                    && info.compares_like_polars(dialect, name, dtype)
                    // read as text, compared as its own type
                    && !(dialect == DialectName::Duckdb
                        && info
                            .compare_types
                            .get(name.as_str())
                            .is_some_and(|t| dialect.text_conversion(t, name).is_some()))
                    && info.declared_type(name).is_none_or(|t| {
                        dialect.text_conversion(t, name).is_none()
                            && !(alias_shadows_column(dialect)
                                && column_conversion(
                                    dialect,
                                    &backend.kind,
                                    setup_runs,
                                    options.connection.opened_by_plugin(),
                                    t,
                                    &dialect.quote_ident(name),
                                )
                                .is_some())
                    }),
                unstable: false,
            })
            .collect();
        if is_query && dialect == DialectName::Oracle {
            oracle_query_timestamps(&info, &mut columns, &mut discovered);
        }
        // A MySQL query's `FLOAT` (planned by its result type on row drivers, without a
        // declared type) is read exactly too, where the query is wrapped.
        if is_query
            && matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
            && read_wrapped(&options, &pushdown_disabled, &query, dialect)
        {
            for c in columns
                .iter_mut()
                .filter(|c| c.dtype == DataType::Float32 && c.db_type.is_none())
            {
                c.db_type = Some("float".into());
            }
        }
        if dialect == DialectName::Mssql {
            for (name, t) in info.compare_types.iter() {
                if matches!(
                    t.trim().to_ascii_lowercase().as_str(),
                    "money" | "smallmoney"
                ) {
                    money_columns.insert(name.clone(), t.clone());
                }
            }
        }
        schema = Arc::new(discovered);
    }

    // ConnectorX writes every decimal as `Decimal(38, 10)`, cutting digits beyond 10
    // places. A table's declared types say which columns to read as text; a query's
    // decimal columns are read as text too, which needs the query wrapped. On ClickHouse
    // it reads no rows at all from a table with a decimal column: those are text too.
    if matches!(backend.kind, BackendKind::ConnectorX)
        && (is_query || dialect == DialectName::Clickhouse)
    {
        let wrapped = read_wrapped(&options, &pushdown_disabled, &query, dialect);
        for c in columns.iter_mut().filter(|c| c.dtype.is_decimal()) {
            polars_ensure!(
                wrapped,
                InvalidOperation:
                "ConnectorX cuts decimals to 10 decimal places, and this query can't be \
                wrapped to read column '{}' exactly ({}); pin another driver, or CAST the \
                column to text in the query",
                c.name, pushdown_disabled.as_deref().unwrap_or("")
            );
            // SQL Server's `money` as text has 2 places: its own conversion (`column_conversion`)
            let money = money_columns.get(c.name.as_str());
            c.db_type = Some(money.map_or(CONNECTORX_DECIMAL, |t| t.as_str()).into());
            c.comparable &= !alias_shadows_column(dialect);
        }
    }

    // The columns read as text when retyped to String: a sampled number (`2**53 + 1`
    // next to `1.5`) would be read as a float first, and then turned into text.
    let numeric_or_text: Vec<bool> = columns
        .iter()
        .map(|c| c.dtype == DataType::String || c.dtype.is_primitive_numeric())
        .collect();

    // A MySQL date column whose type is unknown may hold zero dates: it is read, ordered and
    // filtered on with them as NULL, as a declared `DATE` column is.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore) {
        for c in columns.iter_mut().filter(|c| {
            c.db_type.is_none() && matches!(c.dtype, DataType::Date | DataType::Datetime(_, _))
        }) {
            c.db_type = Some(MYSQL_DATE.into());
        }
    }

    // Each partition reads the source again: a volatile function in a view it reads, or one
    // sqlglot doesn't know (a user's), would give other rows each time.
    if let Some(reason) = &volatility.volatile
        && matches!(
            options.partitioning,
            PartitionSpec::Column { .. } | PartitionSpec::Predicates(_)
        )
    {
        polars_bail!(
            InvalidOperation:
            "this source can't be partitioned: each partition reads it again, and {reason} \
            may return other rows or values each time; pass partition_on=None"
        );
    }
    query.all_columns_volatile |= volatility.all_columns;
    query
        .volatile_columns
        .extend(volatility.columns.iter().cloned());

    // The columns of volatile functions or float aggregates give other values on another run:
    // not partition keys.
    for c in columns.iter_mut() {
        c.unstable |= query.all_columns_volatile
            || query
                .volatile_columns
                .iter()
                .any(|v| v.eq_ignore_ascii_case(&c.name));
    }

    // A filter on a volatile function's column (`random() AS r`) isn't pushed: SQLite merges
    // the wrapped query into the filter's and calls the function again for it, returning
    // rows whose value fails the filter. Postgres and DuckDB push no filter below a
    // volatile function: it sees the values the query returns.
    let refilters = !matches!(dialect, DialectName::Postgresql | DialectName::Duckdb);
    for c in columns.iter_mut().filter(|c| {
        refilters
            && (query.all_columns_volatile
                || query
                    .volatile_columns
                    .iter()
                    .any(|v| v.eq_ignore_ascii_case(&c.name)))
    }) {
        c.comparable = false;
    }

    let schema = match &options.schema_overrides {
        Some(overrides) if !overrides.is_empty() => {
            let mut schema = (*schema).clone();
            for (name, dtype) in overrides.iter() {
                // Overrides may only retype columns; they don't add any.
                if let Some(existing) = schema.get_mut(name) {
                    let retyped = existing != dtype;
                    *existing = dtype.clone();
                    if let Some(c) = columns.iter_mut().find(|c| c.name == name) {
                        c.dtype = dtype.clone();
                        // The database still compares the stored values, not the cast ones.
                        c.comparable &= !retyped;
                    }
                } else {
                    polars_bail!(
                        ColumnNotFound:
                        "schema_overrides names column '{name}', which the database source doesn't have"
                    );
                }
            }
            Arc::new(schema)
        },
        _ => schema,
    };

    // DuckDB `HUGEINT` / `UHUGEINT` values of 39 digits fail the `Decimal(38, 0)` they are
    // read as, but fit an `Int128` / `UInt128`, text or a float: a column retyped to one
    // (`schema_overrides`, or `schema=` naming a 128-bit integer) is read as text, which the
    // cast to its dtype parses.
    let mut schema = schema;
    if dialect == DialectName::Duckdb {
        for c in columns.iter_mut() {
            // `TIMETZ` has no Polars dtype that keeps its offset (the drivers return the local time
            // of day, `'10:00+02'` as `10:00`): it is read as text, `10:00:00+02`, unless the user
            // types it.
            if c.db_type.as_deref().is_some_and(|t| {
                matches!(
                    t.trim().to_ascii_uppercase().as_str(),
                    "TIMETZ" | "TIME WITH TIME ZONE"
                )
            }) {
                let overridden = options
                    .schema_overrides
                    .as_ref()
                    .is_some_and(|o| o.contains(&c.name));
                if !overridden && c.dtype == DataType::Time {
                    c.dtype = DataType::String;
                    if let Some(d) = Arc::make_mut(&mut schema).get_mut(&c.name) {
                        *d = DataType::String;
                    }
                }
                // Typed as a time (or anything but text), the value would lose its offset
                // (`10:00:00+02` is not a `Time`): refused, not read as another time.
                if c.dtype != DataType::String {
                    refused.push((
                        c.name.clone(),
                        format!(
                            "column '{}' is a DuckDB TIMETZ, which has no Polars dtype that keeps \
                            its offset ({} was asked for): read it as text (the default, or \
                            schema_overrides={{'{}': pl.String}}), or cast it in a query \
                            (`CAST({} AS TIME)` drops the offset by your choice)",
                            c.name, c.dtype, c.name, c.name
                        )
                        .into(),
                    ));
                    continue;
                }
                c.db_type = Some(DUCKDB_TEXT.into());
                continue;
            }
            // A `TIMETZ` inside a list, struct or map reads as a `Time` without its offset
            // (`'10:00+02'` as `10:00`), and a nested value can't be read as text: refused,
            // whatever the column is typed as.
            if c.db_type.as_deref().is_some_and(|t| {
                let t = t.to_ascii_uppercase();
                t.contains("TIMETZ") || t.contains("TIME WITH TIME ZONE")
            }) {
                refused.push((
                    c.name.clone(),
                    format!(
                        "column '{}' holds a DuckDB TIMETZ inside a nested type ({}), which \
                        drivers return as the local time of day without its offset: cast it in \
                        a query (`CAST({} AS VARCHAR)`, or `list_transform` / `struct_pack` \
                        with a cast of the field), or leave the column out of the read",
                        c.name,
                        c.db_type.as_deref().unwrap_or_default(),
                        c.name
                    )
                    .into(),
                ));
                continue;
            }
            // A `VARIANT` reaches the drivers as a struct of its raw `metadata` / `value` bytes,
            // not as its value: it is read as text, or refused.
            if c.db_type
                .as_deref()
                .is_some_and(|t| t.to_ascii_uppercase().contains("VARIANT"))
            {
                if c.dtype != DataType::String {
                    refused.push((
                        c.name.clone(),
                        format!(
                            "column '{}' is a DuckDB VARIANT, which drivers return as the raw \
                            bytes of its encoding: read it as text with `CAST({} AS VARCHAR)` in \
                            a query, or schema_overrides={{'{}': pl.String}}, or leave the \
                            column out of the read",
                            c.name, c.name, c.name
                        )
                        .into(),
                    ));
                    continue;
                }
                c.db_type = Some(DUCKDB_TEXT.into());
                continue;
            }
            let hugeint = c.db_type.as_deref().is_some_and(|t| {
                matches!(
                    t.trim().to_ascii_uppercase().as_str(),
                    "HUGEINT" | "UHUGEINT" | "INT128" | "UINT128"
                )
            });
            if (hugeint && !c.dtype.is_decimal())
                || (c.db_type.is_none() && matches!(c.dtype, DataType::Int128 | DataType::UInt128))
            {
                c.db_type = Some(DUCKDB_TEXT.into());
            }
        }
    }

    // A SQLite column has no type: one typed from a sample may hold values of several
    // storage classes (`'abc'` and `1`, `1` and `1.5`). Drivers type a batch by its own
    // values, so it would depend on the batch which of them round or fail: `2**53 + 1` next
    // to `1.5` read as `9007199254740992.0`, also when retyped to String, Decimal or an
    // integer. Each column is read converted to its dtype's class instead (ADBC also writes
    // a real as text with 6 significant digits): text, reals, integers (an integral real as
    // one), and a decimal as exact text; a REAL column holds only reals (and text), an
    // INTEGER one no integral reals, so they are read as they are. ConnectorX reads
    // a column by its declared type where the values' class allows: a `BOOLEAN`'s integers
    // as booleans (2 as true), and it fails on a `DATE` / `BLOB` / … column's integers. An
    // expression has no declared type: its values are read by their class, as the other
    // drivers read them.
    if dialect == DialectName::Sqlite {
        let opened_by_plugin = options.connection.opened_by_plugin();
        let connectorx = matches!(backend.kind, BackendKind::ConnectorX);
        let overridden = |name: &PlSmallStr| {
            options
                .schema_overrides
                .as_ref()
                .is_some_and(|o| o.contains(name))
        };
        for (c, numeric_or_text) in columns.iter_mut().zip(numeric_or_text) {
            match c.dtype {
                DataType::String if numeric_or_text => c.db_type = Some(SQLITE_TEXT.into()),
                DataType::Decimal(_, _) => c.db_type = Some(SQLITE_TEXT.into()),
                DataType::Float32 | DataType::Float64
                    if affinity.get(&c.name) != Some(&DataType::Float64) =>
                {
                    // a Float64 the user asked for casts (rounds) integers; one typed from
                    // the sample fails on an integer it would round
                    c.db_type = Some(
                        if user_typed || overridden(&c.name) {
                            SQLITE_REAL_CUT
                        } else {
                            SQLITE_REAL
                        }
                        .into(),
                    )
                },
                ref dt if dt.is_integer() && overridden(&c.name) => {
                    c.db_type = Some(SQLITE_INTEGER_CUT.into())
                },
                ref dt
                    if dt.is_integer()
                        && affinity.get(&c.name).is_some_and(|a| a != &DataType::Int64) =>
                {
                    c.db_type = Some(SQLITE_UNTYPED_INTEGER.into())
                },
                // a query's column, or one under `schema=`, may hold any value
                ref dt if dt.is_integer() && !affinity.contains_key(&c.name) => {
                    c.db_type = Some(SQLITE_UNTYPED_INTEGER.into())
                },
                ref dt if dt.is_integer() && connectorx => c.db_type = Some(SQLITE_INTEGER.into()),
                // Where the user opened the connection (a live object, a factory), its `abs` and
                // `typeof` may be their own, and the SQL guard can't be trusted: the read checks
                // each batch's dtype instead (`check_sqlite_blobs`).
                DataType::Binary if opened_by_plugin => c.db_type = Some(SQLITE_BLOB.into()),
                // (a blob next to other values, found by the sample: the read fails on it)
                DataType::Binary if mixed_found.contains(&c.name) => {
                    c.db_type = Some(SQLITE_BLOB_MIXED.into())
                },
                DataType::Binary => c.db_type = Some(SQLITE_BLOB_CLIENT.into()),
                _ => {},
            }
        }
    }

    // The wrapper SQL names the output columns, which a driver binding `params=` reads for
    // placeholders (`rate%`, `a:b`): such a query runs as given.
    if pushdown_disabled.is_none()
        && is_query
        && let Some(name) = schema.iter_names().find(|n| {
            crate::filter::name_reads_as_placeholder(&options.connection, &backend.kind, n)
        })
    {
        let reason = format!("the column '{name}' would be read as a placeholder of params");
        if verbose() {
            eprintln!("[scan_database]: pushdown disabled: {reason}");
        }
        pushdown_disabled = Some(reason.into());
    }

    // A TEXT column read as a number (`schema_overrides=`): a range or an order on it follows
    // the text.
    let sqlite_text_numbers = schema
        .iter()
        .filter(|(name, dtype)| {
            (dtype.is_integer() || dtype.is_float() || dtype.is_decimal())
                && text_affinity.contains(*name)
        })
        .map(|(name, _)| name.clone())
        .collect();

    // SQL Server under `schema=`: no I/O says a column's type, so one declared a `Datetime`
    // (any unit, with or without a zone) is read as text through one expression that is exact
    // for every source type (see `column_conversion`), where the drivers read another instant
    // (ConnectorX shifts a `datetimeoffset` twice, pyodbc not at all) or cut the 7th
    // fractional digit.
    if dialect == DialectName::Mssql && user_typed {
        let connectorx = matches!(backend.kind, BackendKind::ConnectorX);
        let wrapped = read_wrapped(&options, &pushdown_disabled, &query, dialect);
        for c in columns.iter_mut().filter(|c| c.db_type.is_none()) {
            let DataType::Datetime(_, tz) = &c.dtype else {
                continue;
            };
            if !wrapped {
                // (an aware column is read as text on every driver: pyodbc can't read a
                // `datetimeoffset` at all)
                if connectorx && tz.is_some() {
                    refused.push((
                        c.name.clone(),
                        format!(
                            "ConnectorX reads a SQL Server datetimeoffset in another zone as \
                            another instant, and this query can't be wrapped to convert column \
                            '{}', declared {}: pin another driver, or CAST the column in the \
                            query to a datetime2 (UTC) and declare it without a time zone",
                            c.name, c.dtype
                        )
                        .into(),
                    ));
                }
                continue;
            }
            c.db_type = Some(MSSQL_DATETIME.into());
        }
    }

    Ok(DatabaseScanConfig {
        sqlite_text_numbers,
        fingerprint: options.fingerprint(),
        options,
        dialect,
        backend,
        schema,
        columns: columns.into(),
        stats,
        compiled: None,
        connection_share: None,
        pushdown_disabled,
        volatile_source: volatility.volatile.is_some(),
        volatility_unchecked,
        user_schema: user_typed,
        derived_order_dropped,
        refused_columns: refused,
        resolution,
        bounds_cache: Default::default(),
    })
}

/// Under `schema=` the planning does no I/O, so what [`source_volatility`] finds wasn't
/// known when filters were pushed and partitions planned: the read checks it on its first
/// session, and returns whether the source is volatile (then its count reads its rows, as
/// for a query that isn't repeatable). A volatile source fails the read where its plan
/// relies on it not being so: several partitions (each reads it again), or a filter
/// pushed into the database on a column of a volatile function, where the database merges
/// a view or query into the filter's and calls `random()` again for it (SQLite, MySQL;
/// others don't push a filter into a query that calls a volatile function).
pub fn check_unplanned_volatility(
    session: &mut dyn DatabaseSession,
    config: &DatabaseScanConfig,
) -> PolarsResult<bool> {
    let Some(compiled) = config.compiled.as_deref() else {
        return Ok(false);
    };
    if !config.volatility_unchecked {
        return Ok(false);
    }
    let volatility = unplanned_volatility(session, config);
    let Some(reason) = volatility.volatile else {
        return Ok(false);
    };
    polars_ensure!(
        compiled.partitions.len() <= 1,
        InvalidOperation:
        "this source can't be partitioned: each partition reads it again, and {reason} may \
        return other rows or values each time (found when reading: schema= skips the check \
        when planning); pass partition_on=None"
    );
    let merges = matches!(
        config.dialect,
        DialectName::Sqlite | DialectName::Mysql | DialectName::Singlestore
    );
    let volatile_column = compiled.filter_columns.iter().find(|c| {
        volatility.all_columns || volatility.columns.iter().any(|v| v.eq_ignore_ascii_case(c))
    });
    if merges && let Some(column) = volatile_column {
        polars_bail!(
            InvalidOperation:
            "a filter on '{column}' was planned to run in the database, but {reason} may \
            return other values each time it runs, and the database may call it again for \
            the filter (found when reading: schema= skips the check when planning); filter \
            after collecting, or leave out schema="
        );
    }
    Ok(true)
}

/// The aware `Datetime` columns of a MySQL source read without a session setup (a live
/// connection) whose database type isn't known (`schema=`, or `schema_overrides` on a query):
/// a `TIMESTAMP` is read in the session's time zone, and a `DATETIME` as it is, and the SQL
/// can't tell them apart. They are right only in a UTC session, which the first session
/// checks here, so that the instants aren't shifted silently.
pub fn check_mysql_session_zone(
    session: &mut dyn DatabaseSession,
    config: &DatabaseScanConfig,
) -> PolarsResult<()> {
    // These drivers' TIMESTAMP columns are converted from the session's zone as they are read (the
    // bridge's `_mysql_timestamps_to_utc`), by the result's own field types.
    if session.converts_mysql_timestamps()
        || matches!(&config.backend.kind, BackendKind::Python { module }
            if matches!(module.as_str(), "pymysql" | "MySQLdb" | "mysql"))
    {
        return Ok(());
    }
    // (SingleStore has no session setup either way)
    if !matches!(
        config.dialect,
        DialectName::Mysql | DialectName::Singlestore
    ) || (config.dialect == DialectName::Mysql && config.session_setup_runs())
    {
        return Ok(());
    }
    let Some(column) = config.columns.iter().find(|c| {
        matches!(c.dtype, DataType::Datetime(_, Some(_)))
            && config.schema.contains(&c.name)
            && c.db_type.as_deref().is_none_or(|t| t == MYSQL_DATE)
    }) else {
        return Ok(());
    };
    // best effort: another statement, or an unreadable answer, is no reason to fail the read
    let Ok(df) = session.catalog_query("SELECT @@session.time_zone, @@system_time_zone") else {
        return Ok(());
    };
    let text = |i: usize| {
        df.columns()
            .get(i)
            .and_then(|c| c.get(0).ok())
            .and_then(|v| v.get_str().map(|s| s.to_string()))
    };
    let (Some(session_zone), Some(system_zone)) = (text(0), text(1)) else {
        return Ok(());
    };
    let zone = if session_zone.eq_ignore_ascii_case("SYSTEM") {
        system_zone
    } else {
        session_zone
    };
    polars_ensure!(
        is_utc_zone(&zone),
        InvalidOperation:
        "column '{}' is read as a Datetime in UTC, but this connection's time zone is '{zone}' \
        and the column's database type isn't known: a TIMESTAMP is read in the session's time \
        zone, so its instants would be shifted. Run SET time_zone = '+00:00' on the connection \
        first, or read the column as a Datetime without a time zone",
        column.name
    );
    Ok(())
}

/// The relations a source names (a table's name; the tables a query reads, if it parses).
fn named_relations(options: &DatabaseScanOptions, dialect: DialectName) -> Vec<Vec<PlSmallStr>> {
    match &options.relation {
        DatabaseRelation::Table { parts, .. } => vec![parts.to_vec()],
        DatabaseRelation::Query { sql } => {
            match sql_renderer().and_then(|r| r.inspect_query(sql, dialect)) {
                Ok(f) => f.tables,
                // can't tell which tables it reads
                Err(_) => vec![],
            }
        },
    }
}

/// How the session resolves the relations the source names
/// ([`catalog::resolution_fingerprint`]); `None` where there is nothing to compare.
pub fn session_resolution(
    session: &mut dyn DatabaseSession,
    dialect: DialectName,
    options: &DatabaseScanOptions,
) -> Option<Vec<String>> {
    if !catalog::has_resolution_fingerprint(dialect) {
        return None;
    }
    catalog::resolution_fingerprint(session, dialect, &named_relations(options, dialect))
}

/// Postgres, MySQL, MariaDB, SingleStore: a read over connections the user made (a factory, an
/// Engine's pool) must read the relation it planned. Pooled sessions differ in state the plugin
/// can't list (`search_path`, a temporary table that shadows a permanent one, a temporary child
/// of an inheritance parent), so planning on one checkout and reading on another can read another
/// relation. Each session opened for a read resolves the planned names again, and a different
/// answer is an error. `baseline` holds the first session's answer for a source planned without
/// I/O (`schema=`), which then has none of its own.
pub fn check_resolution(
    session: &mut dyn DatabaseSession,
    config: &DatabaseScanConfig,
    baseline: &std::sync::Mutex<Option<Vec<String>>>,
) -> PolarsResult<()> {
    if config.options.connection.opened_by_plugin()
        || config.options.connection.is_live()
        || !catalog::has_resolution_fingerprint(config.dialect)
    {
        return Ok(());
    }
    let names = named_relations(&config.options, config.dialect);
    if names.is_empty() {
        return Ok(());
    }
    let found = catalog::resolution_fingerprint(session, config.dialect, &names);
    let Some(found) = found else {
        // planned with a fingerprint, but this session can't give one: not the same state
        polars_ensure!(
            config.resolution.is_none(),
            InvalidOperation:
            "could not check how this connection resolves the table names of the scan, which \
            the scan was planned on another connection's session for"
        );
        return Ok(());
    };
    let mut guard = baseline.lock().unwrap();
    let planned = match &config.resolution {
        Some(p) => p,
        None => guard.get_or_insert_with(|| found.clone()),
    };
    if let Some(i) = planned.iter().zip(&found).position(|(a, b)| a != b) {
        let name = names[i]
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>()
            .join(".");
        polars_bail!(
            InvalidOperation:
            "this connection resolves '{name}' to {}, but the scan was planned on a session \
            where it is {}: a pooled connection's session state (search_path, temporary \
            tables, default database) differs from the one planned on. Read from one \
            connection (a live connection, or a factory that returns a configured one), or \
            qualify the name",
            found[i],
            planned[i]
        );
    }
    Ok(())
}

/// Postgres, MySQL, MariaDB: a partitioned read over connections the user made (a factory, an
/// Engine's pool) must not read a temporary table of a session. A temporary table shadows a
/// permanent one of its name for the session that made it, so the partitions that land on other
/// sessions of the pool would read the permanent table, or none. Run on each session opened.
/// (Oracle's global temporary tables are in the catalog: `catalog::oracle_session_bound`.)
pub fn check_temporary_relations(
    session: &mut dyn DatabaseSession,
    config: &DatabaseScanConfig,
) -> PolarsResult<()> {
    let partitioned = config
        .compiled
        .as_deref()
        .is_some_and(|c| c.partitions.len() > 1);
    if !partitioned
        || config.options.connection.opened_by_plugin()
        || config.max_connections() <= 1
        || !matches!(
            config.dialect,
            DialectName::Postgresql | DialectName::Mysql | DialectName::Singlestore
        )
    {
        return Ok(());
    }
    let relations = named_relations(&config.options, config.dialect);
    for parts in relations {
        if let Some(name) = catalog::temporary_relation(session, config.dialect, &parts) {
            polars_bail!(
                InvalidOperation:
                "the table '{name}' is a temporary table of this session, which the other \
                connections of the pool that the partitions read through don't see (they \
                would read a permanent table of its name, or none): read it without \
                partitioning (partition_on=None), or from one connection"
            );
        }
    }
    Ok(())
}

/// [`source_volatility`] of a `schema=` source, whose planning didn't look it up, on
/// `session`: its columns are the user's.
fn unplanned_volatility(
    session: &mut dyn DatabaseSession,
    config: &DatabaseScanConfig,
) -> SourceVolatility {
    let facts = match &config.options.relation {
        DatabaseRelation::Query { sql } => sql_renderer()
            .and_then(|r| r.inspect_query(sql, config.dialect))
            .ok(),
        DatabaseRelation::Table { .. } => None,
    };
    let names = config.schema.iter_names().cloned().collect::<Vec<_>>();
    source_volatility(
        session,
        config.dialect,
        &config.options.relation,
        facts.as_ref(),
        &names,
        false,
    )
}

/// ConnectorX reads other row counts than there are from a ClickHouse array of `Nullable(…)`,
/// `DateTime64` or decimal values (none, or more), and nothing tells, and an array of
/// `LowCardinality(…)` as one string: reading such a column is an error. Arrays of plain
/// numbers, strings, dates and booleans read right.
/// Refuses a read of a column the planner found it can't read right (it is raised when the column
/// is read, so a query that leaves it out still works).
fn check_refused_columns(config: &DatabaseScanConfig, columns: &[PlSmallStr]) -> PolarsResult<()> {
    if let Some((_, why)) = config
        .refused_columns
        .iter()
        .find(|(name, _)| columns.contains(name))
    {
        polars_bail!(ComputeError: "{why}");
    }
    Ok(())
}

fn check_connectorx_arrays(
    config: &DatabaseScanConfig,
    columns: &[PlSmallStr],
) -> PolarsResult<()> {
    if !matches!(config.backend.kind, BackendKind::ConnectorX)
        || config.dialect != DialectName::Clickhouse
    {
        return Ok(());
    }
    // its ClickHouse client reads every `?` in the SQL as a bind placeholder
    if let Some(c) = columns.iter().find(|c| c.contains('?')) {
        polars_bail!(
            InvalidOperation:
            "ConnectorX can't read the ClickHouse column '{c}': it reads '?' in the SQL as a \
            parameter; pin another driver, e.g. driver=\"adbc\" or \"clickhouse_connect\""
        );
    }
    for c in columns.iter().filter_map(|c| config.column(c)) {
        let wrong = match c.db_type.as_deref() {
            Some(t) => super::schema::clickhouse_value_type(t)
                .strip_prefix("array(")
                .and_then(|r| r.strip_suffix(')'))
                .is_some_and(|element| !connectorx_clickhouse_array_element(element)),
            // a query's column whose type is unknown
            None => matches!(c.dtype, DataType::List(_)),
        };
        polars_ensure!(
            !wrong,
            InvalidOperation:
            "ConnectorX reads this ClickHouse array wrong (column '{}'): its rows, or the \
            values; pin another driver, e.g. driver=\"adbc\" or \"clickhouse_connect\"",
            c.name
        );
    }
    Ok(())
}

/// A ClickHouse array element type (lower-cased) ConnectorX reads right. (`Array(Date)`
/// reads as garbage text, and may lose every row.)
fn connectorx_clickhouse_array_element(t: &str) -> bool {
    matches!(
        t,
        "int8"
            | "int16"
            | "int32"
            | "int64"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "float32"
            | "float64"
            | "string"
            | "bool"
    )
}

/// A dtype with its time zones (also in lists) as UTC.
/// The dtype DuckDB's scalar type `t` reads as (time zones in UTC); `None` for the others.
fn duckdb_type_dtype(t: &str) -> Option<DataType> {
    use DataType as D;
    let t = t.trim().to_ascii_uppercase();
    if let Some(args) = t.strip_prefix("DECIMAL(").and_then(|a| a.strip_suffix(')')) {
        let (p, s) = args.split_once(',')?;
        return Some(D::Decimal(p.trim().parse().ok()?, s.trim().parse().ok()?));
    }
    Some(match t.as_str() {
        "BOOLEAN" => D::Boolean,
        "TINYINT" => D::Int8,
        "SMALLINT" => D::Int16,
        "INTEGER" => D::Int32,
        "BIGINT" => D::Int64,
        "UTINYINT" => D::UInt8,
        "USMALLINT" => D::UInt16,
        "UINTEGER" => D::UInt32,
        "UBIGINT" => D::UInt64,
        "FLOAT" => D::Float32,
        "DOUBLE" => D::Float64,
        "VARCHAR" => D::String,
        "BLOB" => D::Binary,
        "DATE" => D::Date,
        "TIME" => D::Time,
        "TIMESTAMP" => D::Datetime(TimeUnit::Microseconds, None),
        "TIMESTAMP_MS" => D::Datetime(TimeUnit::Milliseconds, None),
        "TIMESTAMP_NS" => D::Datetime(TimeUnit::Nanoseconds, None),
        "TIMESTAMP WITH TIME ZONE" => D::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC)),
        _ => return None,
    })
}

fn in_utc(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Datetime(tu, Some(_)) => DataType::Datetime(*tu, Some(TimeZone::UTC)),
        DataType::List(inner) => DataType::List(Box::new(in_utc(inner))),
        DataType::Array(inner, width) => DataType::Array(Box::new(in_utc(inner)), *width),
        DataType::Map(key, value) => DataType::Map(Box::new(in_utc(key)), Box::new(in_utc(value))),
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Field::new(f.name().clone(), in_utc(f.dtype())))
                .collect(),
        ),
        dt => dt.clone(),
    }
}

/// Pushdown is off (a filter changes the rows), but the query is still wrapped for its
/// conversions, as with `pushdown=False`.
const POSITIONAL_REASON: &str = "a positional join (PASTE, POSITIONAL) pairs rows by their place, \
    which a pushed filter changes";
const POSITIONAL_VIEW_REASON: &str = "a view it reads pairs rows by their place (PASTE, \
    POSITIONAL), which a pushed filter changes";
const SETTINGS_REASON: &str = "a SETTINGS clause (an overflow mode, a row limit) may cut or \
    change the rows, which a pushed filter changes";

/// Whether the source is read with conversions in its select list: a table, or a query that
/// is wrapped for them (with pushdown, or without it where it can be: see
/// `compile_unwrapped`).
fn read_wrapped(
    options: &DatabaseScanOptions,
    pushdown_disabled: &Option<PlSmallStr>,
    query: &QueryDiscovery,
    dialect: DialectName,
) -> bool {
    // only a derived `ORDER BY` (repeated outside the wrapper, `compile_unwrapped`) keeps the
    // pushdown off
    // (and a filter that changes the rows it pairs or keeps, as `pushdown=False` does)
    let only_derived_order = pushdown_disabled.as_deref().is_some_and(|r| {
        r == MARIADB_ORDER_BY
            || r == WITH_TIES
            || r == derived_order_by_reason(dialect).as_str()
            || r == POSITIONAL_REASON
            || r == SETTINGS_REASON
            || r == POSITIONAL_VIEW_REASON
    });
    match options.relation {
        DatabaseRelation::Table { .. } => true,
        DatabaseRelation::Query { .. } => {
            pushdown_disabled.is_none()
                || (query.wrappable && (!options.pushdown || only_derived_order))
        },
    }
}

/// An Oracle query's `TIMESTAMP WITH [LOCAL] TIME ZONE` columns, by its result types
/// (`CatalogInfo::compare_types`): read as UTC through their conversion, like a table's.
/// Drivers read them as naive wall-clock times otherwise, so a temporal column whose type
/// is unknown is not compared in the database. The types are unknown when the query can't
/// be wrapped (then it can't be converted either) or the lookup fails: such columns are
/// read as the driver returns them (naive, the offset dropped).
fn oracle_query_timestamps(info: &CatalogInfo, columns: &mut [ColumnMeta], schema: &mut Schema) {
    for c in columns.iter_mut() {
        let t = info
            .compare_types
            .get(&c.name)
            .map(|t| t.to_ascii_lowercase());
        match t {
            Some(t) if t.starts_with("timestamp") && t.contains("time zone") => {
                let Some(dtype) = declared_dtype(DialectName::Oracle, &t) else {
                    continue;
                };
                c.db_type = Some(t.into());
                c.dtype = dtype.clone();
                // read as UTC like a table's column, so compared like one
                c.comparable = !info.incomplete;
                if let Some(d) = schema.get_mut(&c.name) {
                    *d = dtype;
                }
            },
            Some(_) => {},
            None if c.dtype.is_temporal() => c.comparable = false,
            None => {},
        }
    }
}

/// ClickHouse resolves a select-list alias in `WHERE`: a filter on a column read through
/// a conversion (`toString(x) AS x`) would compare the converted value.
fn alias_shadows_column(dialect: DialectName) -> bool {
    dialect == DialectName::Clickhouse
}

/// The `db_type` of a query's decimal column read through ConnectorX: selected as text.
const CONNECTORX_DECIMAL: &str = "<connectorx decimal>";
/// The `db_type` of a SQLite String / Float64 column: its values converted to text / reals.
const SQLITE_TEXT: &str = "<sqlite text>";
/// The `db_type` of a SQLite Float64 column typed from the sample: an integer a real can't
/// hold exactly fails the read instead of rounding.
const SQLITE_REAL: &str = "<sqlite real>";
/// The `db_type` of a SQLite column the user typed Float64 (`schema=`, `schema_overrides`):
/// integers are cast, as the cast to the dtype rounds them.
const SQLITE_REAL_CUT: &str = "<sqlite real cut>";
/// The `db_type` of a SQLite integer column: an integral real (or an integer's text) read
/// as an integer. An INTEGER one read through ConnectorX, which holds neither, is compared
/// as it is (an index on it serves).
const SQLITE_INTEGER: &str = "<sqlite integer>";
/// The `db_type` of a SQLite integer column of another affinity (none, NUMERIC), or of
/// unknown affinity (a query's, or under `schema=`): as [`SQLITE_INTEGER`], compared
/// through the conversion.
const SQLITE_UNTYPED_INTEGER: &str = "<sqlite untyped integer>";
/// The `db_type` of a SQLite column retyped to an integer by `schema_overrides`: a real cut
/// to an integer, as the cast to the override cuts it; an integer's text read as that
/// integer, as [`SQLITE_INTEGER`] reads it.
const SQLITE_INTEGER_CUT: &str = "<sqlite integer cut>";
/// The `db_type` of a DuckDB `HUGEINT` / `UHUGEINT` column read as text: retyped to a type
/// that holds its values of 39 digits, which the `Decimal(38, 0)` it is read as otherwise
/// doesn't (`duckdb_value_check`).
const DUCKDB_TEXT: &str = "<duckdb text>";
/// The `db_type` of a MySQL `Date` / `Datetime` column of an unknown type (a query's on a
/// driver without the type lookup, or under `schema=`): it may be a `DATE` / `DATETIME` /
/// `TIMESTAMP` holding zero dates (`mysql_zero_date_type`).
const MYSQL_DATE: &str = "<mysql date>";
/// The `db_type` of a SQL Server column declared a `Datetime` under `schema=` (any unit, with
/// or without a zone; no database type to go by): its UTC wall-clock time as text with all its
/// fractional digits, which the read parses to the declared unit (`column_conversion`).
const MSSQL_DATETIME: &str = "<mssql datetime>";
/// The `db_type` of a SQLite Binary column: its blobs as they are, and no other value (a number or
/// text, which drivers read as text, or as NULL, or lossily: an `%e` real) reads: the read fails on
/// it (`explain_sqlite_error`). As an expression, it also has no declared type for ConnectorX to
/// read the column by.
const SQLITE_BLOB: &str = "<sqlite blob>";
/// [`SQLITE_BLOB`] on a connection the user opened: read as it is, and each batch's dtype is
/// checked (`check_sqlite_blobs`).
const SQLITE_BLOB_CLIENT: &str = "<sqlite blob, checked by the read>";
/// What SQLite raises for `abs(-9223372036854775808)`: the error [`SQLITE_BLOB`] reads fail with
/// (an SQL expression has no other way to fail).
const SQLITE_BLOB_ERROR: &str = "integer overflow";

/// The marker that starts a real's exact value in a [`SQLITE_TEXT`] column, and escapes a
/// text that starts with it.
const SQLITE_REAL_MARK: char = '\u{1}';

/// SQL for a SQLite value's storage class, text and marker. On a connection the user made (a live
/// object, a factory's: `portable`) it calls no function: that connection may have redefined
/// `typeof`, `char`, `substr`, `abs` (`create_function`) or `BINARY` (`create_collation`).
/// Operators and `CAST` can't be, and comparing values of different classes (`+x` has no affinity)
/// or blobs uses neither. Where the plugin opens the connection, the functions are the builtin
/// ones, and cheaper: a volatile expression (`random()`) is evaluated once by `typeof`, where the
/// portable form evaluates it for each test.
#[derive(Clone, Copy)]
struct SqliteOps {
    portable: bool,
}

impl SqliteOps {
    fn is_real(self, x: &str) -> String {
        if !self.portable {
            return format!("typeof({x}) = 'real'");
        }
        format!("(+({x}) <= 9e999 AND CAST(CAST({x} AS INTEGER) AS BLOB) <> CAST({x} AS BLOB))")
    }

    fn is_integer(self, x: &str) -> String {
        if !self.portable {
            return format!("typeof({x}) = 'integer'");
        }
        format!("(+({x}) <= 9e999 AND CAST(CAST({x} AS INTEGER) AS BLOB) = CAST({x} AS BLOB))")
    }

    /// Text: above every number, below every blob.
    fn is_text(self, x: &str) -> String {
        if !self.portable {
            return format!("typeof({x}) = 'text'");
        }
        format!("(+({x}) > 9e999 AND +({x}) < X'')")
    }

    /// Text `x` that is an integer's (`as_type` INTEGER) or a real's text, as SQLite writes it
    /// (compared as bytes where portable: no collation).
    fn text_is(self, x: &str, as_type: &str) -> String {
        if !self.portable {
            return format!("CAST(CAST({x} AS {as_type}) AS TEXT) = {x}");
        }
        format!("CAST(CAST({x} AS {as_type}) AS BLOB) = CAST({x} AS BLOB)")
    }

    /// The storage class of `x` as `typeof` writes it.
    fn storage_class(self, x: &str) -> String {
        if !self.portable {
            return format!("typeof({x})");
        }
        format!(
            "CASE WHEN {x} IS NULL THEN 'null' WHEN {} THEN 'integer' WHEN {} THEN 'real' \
            WHEN {} THEN 'text' ELSE 'blob' END",
            self.is_integer(x),
            self.is_real(x),
            self.is_text(x)
        )
    }

    /// The text `x` starts with the marker.
    fn starts_with_mark(self, x: &str) -> String {
        if !self.portable {
            return format!("substr({x}, 1, 1) = {}", self.mark());
        }
        // The first character's bytes, in the database's encoding (`CAST AS BLOB` gives the stored
        // bytes): one byte in UTF-8, two in UTF-16 with the marker's byte first (be) or last (le).
        let enc = "(SELECT encoding FROM pragma_encoding)";
        format!(
            "(CAST({x} AS BLOB) >= CASE {enc} WHEN 'UTF-16le' THEN X'0100' \
            WHEN 'UTF-16be' THEN X'0001' ELSE X'01' END \
            AND CAST({x} AS BLOB) < CASE {enc} WHEN 'UTF-16le' THEN X'0101' \
            WHEN 'UTF-16be' THEN X'0002' ELSE X'02' END)"
        )
    }

    /// The marker as a SQL expression.
    fn mark(self) -> &'static str {
        if self.portable { "'\u{1}'" } else { "char(1)" }
    }
}

/// A SQLite real `x`, exactly, as `\u{1}[-]m:k` for `m · 2^k` (`\u{1}inf`, `\u{1}-inf`),
/// which [`decode_sqlite_reals`] writes as Polars writes a Float64. SQLite's own text of a
/// real (`printf`, `CAST`) depends on its build, some of which round wrongly (to another
/// real); scaling by powers of two is exact on any. (-0.0 reads as 0.0: SQLite doesn't
/// tell them apart.) No `abs()`: a live connection may have redefined it.
fn sqlite_real_exact(ops: SqliteOps, column: &str) -> String {
    format!(
        "CASE WHEN {column} = 9e999 THEN {m} || 'inf' \
        WHEN {column} = -9e999 THEN {m} || '-inf' \
        ELSE {m} || CASE WHEN {column} < 0 THEN '-' ELSE '' END || (\
        WITH RECURSIVE __polars_real(v, k) AS (SELECT CASE WHEN {column} < 0 THEN -{column} ELSE {column} END, 0 UNION ALL \
        SELECT CASE WHEN v >= 9223372036854775808.0 THEN v / 1024 ELSE v * 1024 END, \
        CASE WHEN v >= 9223372036854775808.0 THEN k + 10 ELSE k - 10 END FROM __polars_real \
        WHERE v >= 9223372036854775808.0 OR v <> CAST(v AS INTEGER)) \
        SELECT CAST(v AS INTEGER) || ':' || k FROM __polars_real \
        WHERE v < 9223372036854775808.0 AND v = CAST(v AS INTEGER) LIMIT 1) END",
        m = ops.mark()
    )
}

/// `x · 2^k`, exactly where the result is a real.
fn ldexp(mut x: f64, mut k: i32) -> f64 {
    let pow2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    while k > 1000 {
        x *= pow2(1000);
        k -= 1000;
    }
    while k < -1000 {
        x *= pow2(-1000);
        k += 1000;
    }
    x * pow2(k)
}

/// [`SQLITE_BLOB_CLIENT`] of a column whose sampled values were blobs and others: a driver reads it
/// as one type and the others as NULL, so a read of it is an error (see `check_sqlite_blobs`).
const SQLITE_BLOB_MIXED: &str = "<sqlite blob next to other values>";
const SQLITE_BLOB_ADVICE: &str = "Read the column as text or numbers (schema_overrides, or \
    `schema=`), or cast it in the query (`CAST(x AS BLOB)` for its text)";

/// The columns of a SQLite scan whose batches [`check_sqlite_blobs`] checks.
pub(crate) fn sqlite_blob_columns(config: &DatabaseScanConfig) -> Vec<(PlSmallStr, bool)> {
    if config.dialect != DialectName::Sqlite {
        return vec![];
    }
    config
        .columns
        .iter()
        .filter(|c| {
            matches!(
                c.db_type.as_deref(),
                Some(SQLITE_BLOB_CLIENT | SQLITE_BLOB_MIXED)
            )
        })
        .map(|c| {
            (
                c.name.clone(),
                c.db_type.as_deref() == Some(SQLITE_BLOB_MIXED),
            )
        })
        .collect()
}

/// A batch of a SQLite Binary column on a connection the user opened holds blobs and NULLs only: a
/// number or text (read as text, NULL or rounded by the driver, or cast to bytes) is an error, as
/// the SQL guard of [`SQLITE_BLOB`] makes it.
pub(crate) fn check_sqlite_blobs(df: &DataFrame, names: &[(PlSmallStr, bool)]) -> PolarsResult<()> {
    for (name, mixed) in names {
        let Ok(col) = df.column(name) else {
            continue;
        };
        polars_ensure!(
            !mixed,
            ComputeError:
            "the column '{name}' is read as Binary, but a value of it isn't a blob (it holds \
            blobs and other values, which a driver reads as one type and the others as \
            NULL): {SQLITE_BLOB_ADVICE}"
        );
        polars_ensure!(
            matches!(col.dtype(), DataType::Binary | DataType::Null),
            ComputeError:
            "the column '{name}' is read as Binary, but a value of it isn't a blob (the \
            driver read {}): {SQLITE_BLOB_ADVICE}",
            col.dtype()
        );
    }
    Ok(())
}

/// `err` of reading a SQLite scan, explained where it is the read of a [`SQLITE_BLOB`] column's
/// non-blob value.
pub(crate) fn explain_sqlite_error(config: &DatabaseScanConfig, err: PolarsError) -> PolarsError {
    if config.dialect != DialectName::Sqlite {
        return err;
    }
    let text = err.to_string();
    // On a connection the user opened, the driver's own frame of a batch holding a blob next to
    // numbers fails before the batch is checked (`check_sqlite_blobs`).
    let mixed = text.contains("supertype") && text.contains("binary");
    let wanted = if text.contains(SQLITE_BLOB_ERROR) {
        SQLITE_BLOB
    } else if mixed {
        SQLITE_BLOB_CLIENT
    } else {
        return err;
    };
    let blobs = config
        .columns
        .iter()
        .filter(|c| {
            c.db_type.as_deref() == Some(wanted)
                || (wanted == SQLITE_BLOB_CLIENT && c.db_type.as_deref() == Some(SQLITE_BLOB_MIXED))
        })
        .map(|c| format!("'{}'", c.name))
        .collect::<Vec<_>>();
    if blobs.is_empty() {
        return err;
    }
    polars_err!(
        ComputeError:
        "{err}\n\nIf the query itself doesn't fail, the column(s) {} are read as Binary, \
        but hold a value that isn't a blob (a number or text): drivers would read it as \
        text, as NULL or rounded. {SQLITE_BLOB_ADVICE}",
        blobs.join(", ")
    )
}

/// The columns of a SQLite scan that [`decode_sqlite_reals`] decodes.
pub(crate) fn sqlite_text_columns(config: &DatabaseScanConfig) -> Vec<PlSmallStr> {
    if config.dialect != DialectName::Sqlite {
        return vec![];
    }
    config
        .columns
        .iter()
        .filter(|c| c.db_type.as_deref() == Some(SQLITE_TEXT))
        .map(|c| c.name.clone())
        .collect()
}

/// A SQLite [`SQLITE_TEXT`] column's batch with its reals (`sqlite_real_exact`) written as
/// Polars writes a Float64, and escaped text unescaped.
pub(crate) fn decode_sqlite_reals(
    mut df: DataFrame,
    names: &[PlSmallStr],
) -> PolarsResult<DataFrame> {
    for name in names {
        let Ok(col) = df.column(name) else {
            continue;
        };
        // A connection with `text_factory=bytes` hands text over as bytes, the marker too.
        let as_text;
        let col = if let Ok(bytes) = col.binary() {
            if let Ok(text) = col.cast(&DataType::String) {
                as_text = text;
                &as_text
            } else {
                polars_ensure!(
                    !bytes.iter().flatten().any(|v| v.first() == Some(&1)),
                    ComputeError:
                    "database column '{name}' holds text that isn't valid UTF-8 next to a real \
                    (a connection with `text_factory=bytes`): read it with a str text factory"
                );
                continue;
            }
        } else {
            col
        };
        let Ok(ca) = col.str() else {
            // all NULL
            continue;
        };
        if !ca.iter().flatten().any(|v| v.starts_with(SQLITE_REAL_MARK)) {
            continue;
        }
        let mut reals: Vec<Option<f64>> = Vec::with_capacity(ca.len());
        let mut texts: Vec<Option<&str>> = Vec::with_capacity(ca.len());
        for v in ca.iter() {
            let (real, text) = match v.map(|v| (v, v.strip_prefix(SQLITE_REAL_MARK))) {
                None => (None, None),
                Some((v, None)) => (None, Some(v)),
                Some((_, Some(rest))) if rest.starts_with(SQLITE_REAL_MARK) => (None, Some(rest)),
                Some((_, Some(rest))) => {
                    let value = match rest {
                        "inf" => f64::INFINITY,
                        "-inf" => f64::NEG_INFINITY,
                        _ => {
                            let parsed = rest.split_once(':').and_then(|(m, k)| {
                                let negative = m.starts_with('-');
                                let m = m.trim_start_matches('-').parse::<u64>().ok()?;
                                let x = ldexp(m as f64, k.parse::<i32>().ok()?);
                                Some(if negative { -x } else { x })
                            });
                            polars_ensure!(
                                parsed.is_some(),
                                ComputeError:
                                "database column '{name}' read a real as '{rest}', which isn't \
                                the expected `m:k`"
                            );
                            parsed.unwrap()
                        },
                    };
                    (Some(value), None)
                },
            };
            reals.push(real);
            texts.push(text);
        }
        // as Polars writes them
        let reals = Float64Chunked::from_iter_options(name.clone(), reals.into_iter())
            .cast(&DataType::String)?;
        let decoded: StringChunked = reals
            .str()?
            .iter()
            .zip(texts)
            .map(|(real, text)| real.or(text))
            .collect();
        df.replace(name, decoded.with_name(name.clone()).into_column())?;
    }
    Ok(df)
}

/// SQL that converts a column of a declared type where the backend can't read it as is, or
/// `None` to select it as it is. `setup_runs`: the session setup runs (see `resolve_scan`);
/// `own_connection`: the plugin opened the connection (`ConnectionSpec::opened_by_plugin`).
fn column_conversion(
    dialect: DialectName,
    backend: &BackendKind,
    setup_runs: bool,
    own_connection: bool,
    db_type: &str,
    column: &str,
) -> Option<String> {
    if db_type == CONNECTORX_DECIMAL && dialect == DialectName::Oracle {
        return Some(oracle_number_text(column));
    }
    if db_type == CONNECTORX_DECIMAL {
        return Some(dialect.cast_to_text(column));
    }
    if db_type == DUCKDB_TEXT {
        return Some(format!("CAST({column} AS VARCHAR)"));
    }
    let ops = SqliteOps {
        portable: !own_connection,
    };
    if db_type == SQLITE_TEXT {
        // a real exactly, which the reader writes as Polars writes a Float64 (as the
        // column read as reals, then cast, would be); a text starting with the marker
        // escaped by another
        let text = format!("CAST({column} AS TEXT)");
        return Some(format!(
            "CASE WHEN {} THEN {} WHEN {} THEN {m} || {text} ELSE {text} END",
            ops.is_real(column),
            sqlite_real_exact(ops, column),
            ops.starts_with_mark(&text),
            m = ops.mark()
        ));
    }
    if db_type == SQLITE_BLOB {
        // `abs` of the smallest integer is documented to fail with an integer overflow
        return Some(format!(
            "CASE WHEN typeof({column}) IN ('blob', 'null') THEN {column} \
            ELSE abs(-9223372036854775808) END"
        ));
    }
    // Text in a column read as numbers is parsed by the read (`'1'` as 1), while SQLite
    // compares it as text, above every number: an integer's or a real's text as SQLite
    // writes it is read as that number, other text marked so that no read parses it (`'+1'`,
    // `'1e0'`; it fails, as `'abc'` does).
    let m = ops.mark();
    let is_int = ops.is_integer(column);
    let is_real = ops.is_real(column);
    let is_text = ops.is_text(column);
    let integer_text = ops.text_is(column, "INTEGER");
    let real_text = ops.text_is(column, "REAL");
    let other_text = format!("WHEN {is_text} THEN {m} || {column}");
    if db_type == SQLITE_REAL_CUT {
        return Some(format!(
            "CASE WHEN {is_int} THEN CAST({column} AS REAL) \
            WHEN {is_text} AND ({integer_text} \
            OR {real_text}) THEN CAST({column} AS REAL) \
            {other_text} ELSE {column} END"
        ));
    }
    if db_type == SQLITE_REAL {
        // an integer is a real only if that is exact (`CAST(.. AS INTEGER) = x` compares
        // integers; 2^63 casts back to the largest integer, so is excluded); another is
        // marked, so that no read parses it, and fails as other text does (rounding it would
        // lose digits silently)
        let exact = |int: &str| {
            format!(
                "CAST({int} AS REAL) < 9223372036854775808.0 \
                AND CAST(CAST({int} AS REAL) AS INTEGER) = {int}"
            )
        };
        let int_exact = exact(column);
        let text_int = format!("CAST({column} AS INTEGER)");
        let text_exact = exact(&text_int);
        return Some(format!(
            "CASE WHEN {is_int} THEN CASE WHEN {int_exact} \
            THEN CAST({column} AS REAL) ELSE {m} || {column} END \
            WHEN {is_text} AND {integer_text} THEN CASE WHEN {text_exact} \
            THEN CAST({column} AS REAL) ELSE {m} || {column} END \
            WHEN {is_text} AND {real_text} \
            THEN CAST({column} AS REAL) \
            {other_text} ELSE {column} END"
        ));
    }
    if db_type == SQLITE_INTEGER || db_type == SQLITE_UNTYPED_INTEGER {
        // another real stays a real, and fails the read (`1.5`, `1e19`): `CAST` would cut
        // it; `CAST(x AS INTEGER) = x` compares exactly, not as reals
        return Some(format!(
            "CASE WHEN {is_real} AND CAST({column} AS INTEGER) = {column} \
            THEN CAST({column} AS INTEGER) \
            WHEN {is_text} AND {integer_text} THEN CAST({column} AS INTEGER) \
            {other_text} ELSE {column} END"
        ));
    }
    if db_type == SQLITE_INTEGER_CUT {
        // a real as the cast to an override cuts it (one out of range stays one, and
        // fails); text as in [`SQLITE_INTEGER`]
        return Some(format!(
            "CASE WHEN {is_real} AND {column} >= -9223372036854775808.0 \
            AND {column} < 9223372036854775808.0 THEN CAST({column} AS INTEGER) \
            WHEN {is_text} AND {integer_text} THEN CAST({column} AS INTEGER) \
            {other_text} ELSE {column} END"
        ));
    }
    if let Some(sql) = dialect.text_conversion(db_type, column) {
        return Some(sql);
    }
    // Trino `timestamp(p) with time zone`: the instant in UTC (a named zone's repeated hour
    // reads an hour off otherwise: the value holds the zone's name, not its offset)
    if dialect == DialectName::Trino
        && declared_dtype(dialect, db_type)
            .is_some_and(|d| matches!(d, DataType::Datetime(_, Some(_))))
    {
        return Some(format!("{column} AT TIME ZONE 'UTC'"));
    }
    // ClickHouse `DateTime` (seconds) reaches Arrow as a plain UInt32; `DateTime64` is a
    // timestamp, and compares like the column. clickhouse_connect and ConnectorX cut a
    // `DateTime64(7..9)` to microseconds: its nanoseconds since the epoch keep them (its
    // text clamps instants before 1900).
    if dialect == DialectName::Clickhouse {
        let t = super::schema::clickhouse_value_type(db_type);
        if t == "datetime" || t.starts_with("datetime(") {
            return Some(format!("toDateTime64({column}, 0)"));
        }
        if !matches!(backend, BackendKind::Adbc { .. })
            && declared_dtype(dialect, db_type)
                .is_some_and(|d| matches!(d, DataType::Datetime(TimeUnit::Nanoseconds, _)))
        {
            return Some(format!("toUnixTimestamp64Nano({column})"));
        }
        // ADBC reads the elements of an `Array(DateTime)` as the plain integers too (seconds
        // read as milliseconds): as `DateTime64(0)`, like the scalar
        if matches!(backend, BackendKind::Adbc { .. })
            && t.starts_with("array(")
            && let Some(sql) = clickhouse_array_datetime(db_type, column, 0)
        {
            return Some(sql);
        }
        // ConnectorX fails on an array of decimals, even of no rows (its Arrow schema says
        // `Decimal128(38, 10)`, its arrays another scale): selected as text, so that the
        // table's other columns read, and `check_connectorx_arrays` refuses this one
        if matches!(backend, BackendKind::ConnectorX)
            && t.starts_with("array(")
            && t.contains("decimal")
        {
            return Some(dialect.cast_to_text(column));
        }
        // in arrays (any deep), as their nanoseconds since the epoch
        if !matches!(backend, BackendKind::Adbc { .. }) && t.starts_with("array(") {
            return clickhouse_array_nanoseconds(db_type, column, 0);
        }
    }
    // ConnectorX reads `money` as a double (`922337203685477.5807` as `…477.6`)
    if dialect == DialectName::Mssql && matches!(backend, BackendKind::ConnectorX) {
        match db_type.trim().to_ascii_lowercase().as_str() {
            // as exact text (a decimal reaches ConnectorX as a double too), parsed as the
            // planned decimal
            "money" => {
                return Some(dialect.cast_to_text(&format!("CAST({column} AS DECIMAL(19, 4))")));
            },
            "smallmoney" => {
                return Some(dialect.cast_to_text(&format!("CAST({column} AS DECIMAL(10, 4))")));
            },
            _ => {},
        }
    }
    // `datetime2(7)`, `time(7)` and `datetimeoffset(7)` hold 100 ns, which drivers cut to
    // microseconds (and read differently): read as text, parsed as the planned nanosecond
    // type (a `datetimeoffset` as its UTC wall-clock time: its own text has the offset)
    if dialect == DialectName::Mssql
        && let Some(base) = super::schema::mssql_nanosecond_type(db_type)
    {
        let x = if base == "datetimeoffset" {
            format!("CAST(SWITCHOFFSET({column}, '+00:00') AS DATETIME2(7))")
        } else {
            column.to_string()
        };
        return Some(format!("CONVERT(VARCHAR(40), {x}, 121)"));
    }
    // A `Datetime` declared under `schema=` (no database type to go by): the type is looked up
    // in the query itself. A `datetimeoffset` is its UTC wall-clock time; any other type its own
    // text, which keeps its exact digits (a conversion of a `datetime` to `datetime2` would
    // read `.997` as `.9966667`). Every branch is compiled for every type.
    if dialect == DialectName::Mssql && db_type == MSSQL_DATETIME {
        return Some(format!(
            "CASE WHEN SQL_VARIANT_PROPERTY(CAST({column} AS sql_variant), 'BaseType') = \
            'datetimeoffset' THEN CONVERT(VARCHAR(40), CAST(SWITCHOFFSET({column}, '+00:00') \
            AS DATETIME2(7)), 121) ELSE CONVERT(VARCHAR(40), {column}, 121) END"
        ));
    }
    // ConnectorX shifts a `datetimeoffset` of another offset than UTC twice (`01:30 +05:30`
    // reads as 14:30 UTC, not 20:00): at offset 0 its read is right.
    if dialect == DialectName::Mssql
        && matches!(backend, BackendKind::ConnectorX)
        && db_type
            .trim()
            .to_ascii_lowercase()
            .starts_with("datetimeoffset")
    {
        return Some(format!("SWITCHOFFSET({column}, '+00:00')"));
    }
    // Read as text and parse the exact value; the planned dtype stays the declared decimal.
    // ConnectorX reads decimals with at most 10 decimal places, dropping digits (and panics on
    // SQL Server decimals of more than 28 digits); an
    // unconstrained `numeric` / `NUMBER` reaches some drivers as a float (oracledb), and
    // its values may have more digits than the planned scale (an error then, not rounding).
    if let Some(DataType::Decimal(p, scale)) = declared_dtype(dialect, db_type)
        && ((matches!(backend, BackendKind::ConnectorX)
            && (scale > 10 || (dialect == DialectName::Mssql && p > 28)))
            || DataType::Decimal(p, scale) == super::schema::UNCONSTRAINED_DECIMAL)
    {
        if dialect == DialectName::Oracle {
            return Some(oracle_number_text(column));
        }
        return Some(dialect.cast_to_text(column));
    }
    // Oracle returns `TIMESTAMP WITH LOCAL TIME ZONE` in the session's time zone (a live
    // connection's is the user's), and drivers drop the offset of `TIMESTAMP WITH TIME
    // ZONE`, keeping the wall-clock time: read both as UTC, which the planned dtype says
    // they are (the cast keeps the column's fractional digits: its default precision is
    // 6). oracledb reads 7 to 9 fractional digits as microseconds, and wraps a value outside
    // the nanosecond range: they are read as text (`SYYYY` writes a BC year as negative,
    // which fails the parse), in the Gregorian calendar whatever the session's
    // `NLS_CALENDAR` (a live connection's may be another).
    if dialect == DialectName::Oracle {
        let t = db_type.trim().to_ascii_lowercase();
        // `NUMBER(p, s)` with `s > p`: drivers type it `decimal(p, s)`, which Arrow (and
        // Polars) refuse; read as text, it is parsed as the planned `Decimal(s, s)`
        if let Some((p, s)) = t
            .strip_prefix("number(")
            .and_then(|r| r.strip_suffix(')'))
            .and_then(|a| a.split_once(','))
            && let (Ok(p), Ok(s)) = (p.trim().parse::<i64>(), s.trim().parse::<i64>())
            && s > p
        {
            return Some(oracle_number_text(column));
        }
        // `NUMBER(p, -s)`: drivers fail on a negative scale; its values are integers
        if t.starts_with("number(")
            && let Some(DataType::Int64 | DataType::Decimal(_, 0)) = declared_dtype(dialect, &t)
            && t.contains('-')
        {
            let digits = match declared_dtype(dialect, &t) {
                Some(DataType::Decimal(p, _)) => p,
                _ => 18,
            };
            return Some(format!("CAST({column} AS NUMBER({digits}, 0))"));
        }
        if t.starts_with("timestamp") {
            let precision = super::schema::oracle_timestamp_precision(&t);
            let utc = if t.contains("local time zone") {
                let p = precision.unwrap_or(6);
                format!("SYS_EXTRACT_UTC(CAST({column} AS TIMESTAMP({p}) WITH TIME ZONE))")
            } else if t.contains("time zone") {
                format!("SYS_EXTRACT_UTC({column})")
            } else {
                column.to_string()
            };
            if precision.is_some_and(|p| p > 6) {
                return Some(format!(
                    "TRIM(TO_CHAR({utc}, 'SYYYY-MM-DD HH24:MI:SS.FF9', \
                    'NLS_CALENDAR=GREGORIAN'))"
                ));
            }
            if utc != column {
                return Some(utc);
            }
        }
    }
    // Postgres values Polars can't hold fail the read, naming the column (the message
    // isn't a valid value of the type; the `CASE` evaluates it only then):
    // - a `time` may be `24:00:00`, which isn't a time of day: Arrow drivers read it as NULL
    //   or `00:00:00`. In a `time[]`, any element (`ANY` looks at every dimension).
    // - `±infinity` dates and timestamps: psycopg2 reads them as `date.max` / `date.min`
    //   (ADBC checks them itself).
    // - a multi-dimensional array: ADBC flattens it (other drivers fail).
    //
    // A live connection gets no session setup: psycopg returns a `timestamptz` in the
    // session's time zone, whose instant Python recomputes from its own tzdata (wrong for
    // old dates: 1900 in Europe/Amsterdam is off by 19 minutes), and pg8000 fails on
    // such an offset. It is read as its UTC wall-clock time, which the cast to the planned
    // `Datetime(…, "UTC")` takes as UTC.
    if matches!(dialect, DialectName::Postgresql | DialectName::Redshift) {
        let t = db_type.trim().to_ascii_lowercase();
        let array = t.ends_with("[]");
        let utc = !setup_runs
            && matches!(backend, BackendKind::Python { .. })
            && t.starts_with("timestamp")
            && t.trim_end_matches("[]").ends_with("with time zone");
        let (value, value_type) = match (utc, array) {
            (true, false) => (
                format!("({column} AT TIME ZONE 'UTC')"),
                "timestamp without time zone".to_string(),
            ),
            // each element (in order; a NULL array stays NULL, which `ARRAY(…)` wouldn't)
            (true, true) => (
                format!(
                    "CASE WHEN {column} IS NULL THEN NULL ELSE ARRAY(SELECT pl_x AT TIME ZONE \
                    'UTC' FROM unnest({column}) WITH ORDINALITY AS pl_t(pl_x, pl_i) \
                    ORDER BY pl_i) END"
                ),
                "timestamp without time zone[]".to_string(),
            ),
            (false, _) => (column.to_string(), db_type.to_string()),
        };
        let fail = |what: &str| {
            let msg = dialect.quote_str(&format!("column {column} holds {what}"));
            format!("CAST({msg} || LEFT(CAST({column} AS TEXT), 0) AS {value_type})")
        };
        let declared = declared_dtype(dialect, db_type);
        let element = match &declared {
            Some(DataType::List(inner)) => Some(inner.as_ref()),
            _ => None,
        };
        let mut checks = vec![];
        if declared == Some(DataType::Time) {
            checks.push((
                format!("{column} = TIME '24:00:00'"),
                fail("24:00:00, which is not a time of day"),
            ));
        }
        if element == Some(&DataType::Time) {
            checks.push((
                format!("TIME '24:00:00' = ANY({column})"),
                fail("24:00:00, which is not a time of day"),
            ));
        }
        let postgres = dialect == DialectName::Postgresql;
        if postgres && !matches!(backend, BackendKind::Adbc { .. }) {
            let base = t.trim_end_matches("[]");
            if base == "date" || base.starts_with("timestamp") {
                let infinite = if t.ends_with("[]") {
                    format!("EXISTS (SELECT FROM unnest({column}) pl_x WHERE NOT isfinite(pl_x))")
                } else {
                    format!("NOT isfinite({column})")
                };
                checks.push((infinite, fail("infinity, which Polars can't represent")));
            }
        }
        // (`unnest` flattens: the elements of the array of arrays would read as one list)
        if utc && array {
            checks.push((
                format!("array_ndims({column}) > 1"),
                fail("a multi-dimensional array; read it as text, or unnest it in a query"),
            ));
        }
        if postgres && matches!(backend, BackendKind::Adbc { .. }) && db_type.trim().ends_with("[]")
        {
            checks.push((
                format!("array_ndims({column}) > 1"),
                fail("a multi-dimensional array; read it as text, or unnest it in a query"),
            ));
        }
        if !checks.is_empty() {
            let whens = checks
                .iter()
                .map(|(when, then)| format!("WHEN {when} THEN {then}"))
                .collect::<Vec<_>>()
                .join(" ");
            return Some(format!("CASE {whens} ELSE {value} END"));
        }
        if utc {
            return Some(value);
        }
    }
    if dialect == DialectName::Duckdb
        && let Some(check) = duckdb_value_check(db_type, column)
    {
        return Some(check);
    }
    // The ADBC MySQL driver sizes an `UNSIGNED` decimal one digit short, failing on values
    // with every digit used.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && matches!(backend, BackendKind::Adbc { .. })
        && db_type.to_ascii_lowercase().contains("unsigned")
        && declared_dtype(dialect, db_type).is_some_and(|d| d.is_decimal())
    {
        return Some(dialect.cast_to_text(column));
    }
    // MySQL's text protocol (PyMySQL, mysql.connector) and ADBC send a `FLOAT` rounded to
    // 6 significant digits (1234.567 reads 1234.57); as a `DOUBLE`, its stored value.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && declared_dtype(dialect, db_type) == Some(DataType::Float32)
    {
        return Some(format!("CAST({column} AS DOUBLE)"));
    }
    // MySQL `TIME` is a duration: Python drivers read it as one (`timedelta`), Arrow
    // drivers as a time of day, which fails (ConnectorX) or wraps beyond 24 hours. As
    // microseconds, any value.
    if matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && !matches!(backend, BackendKind::Python { .. })
        && declared_dtype(dialect, db_type) == Some(DataType::Duration(TimeUnit::Microseconds))
    {
        return Some(format!(
            "IF({column} < TIME '00:00:00', -1, 1) * ((HOUR({column}) * 3600 \
            + MINUTE({column}) * 60 + SECOND({column})) * 1000000 + MICROSECOND({column}))"
        ));
    }
    // MySQL zero dates (`'0000-00-00'`, and dates with a zero month or day) are no date:
    // read them as NULL, as Connector/J's `zeroDateTimeBehavior=convertToNull` does. Drivers
    // fail on them otherwise. Pushed filters see them as NULL too (`zero_dates_as_null`).
    //
    // A `TIMESTAMP` is read in the session time zone, UTC only after the session setup
    // (not on a live connection, nor on ConnectorX): elsewhere its stored instant
    // (`UNIX_TIMESTAMP` of the column) is read as UTC.
    if mysql_zero_date_type(dialect, db_type) {
        let t = db_type.trim().to_ascii_lowercase();
        let value = if !setup_runs && (t == "timestamp" || t.starts_with("timestamp(")) {
            format!(
                "TIMESTAMPADD(MICROSECOND, CAST(UNIX_TIMESTAMP({column}) * 1000000 AS SIGNED), \
                TIMESTAMP '1970-01-01 00:00:00')"
            )
        } else {
            column.to_string()
        };
        return Some(format!("IF({}, NULL, {value})", mysql_zero_date(column)));
    }
    None
}

/// A ClickHouse array (of arrays) of `DateTime` as `DateTime64(0)` elements; `None` for
/// another type.
fn clickhouse_array_datetime(db_type: &str, x: &str, depth: usize) -> Option<String> {
    let t = super::schema::clickhouse_value_type(db_type);
    if t.starts_with("array(") {
        let written = super::schema::clickhouse_unwrapped(db_type);
        let open = written.find('(')?;
        let inner = written[open + 1..].strip_suffix(')')?;
        let v = format!("pl_x{depth}");
        let element = clickhouse_array_datetime(inner, &v, depth + 1)?;
        return Some(format!("arrayMap({v} -> {element}, {x})"));
    }
    (t == "datetime" || t.starts_with("datetime(")).then(|| format!("toDateTime64({x}, 0)"))
}

/// A ClickHouse array (of arrays) of `DateTime64(7..9)` as the elements' nanoseconds since
/// the epoch, which the cast to the planned `List(Datetime("ns"))` reads exactly; `None`
/// for another type.
fn clickhouse_array_nanoseconds(db_type: &str, x: &str, depth: usize) -> Option<String> {
    let t = super::schema::clickhouse_value_type(db_type);
    if t.starts_with("array(") {
        let written = db_type.trim();
        let open = written.find('(')?;
        // `Nullable(Array(…))` can't be: the wrappers are only around the element
        let inner = written[open + 1..].strip_suffix(')')?;
        let v = format!("pl_x{depth}");
        let element = clickhouse_array_nanoseconds(inner, &v, depth + 1)?;
        return Some(format!("arrayMap({v} -> {element}, {x})"));
    }
    declared_dtype(DialectName::Clickhouse, db_type)
        .is_some_and(|d| matches!(d, DataType::Datetime(TimeUnit::Nanoseconds, _)))
        .then(|| format!("toUnixTimestamp64Nano({x})"))
}

/// An Oracle `NUMBER` as text, whatever the session's NLS settings (a live connection's
/// may write a decimal comma). `TM9` writes every digit; no calendar applies.
fn oracle_number_text(column: &str) -> String {
    format!("TO_CHAR({column}, 'TM9', 'NLS_NUMERIC_CHARACTERS=''.,''')")
}

/// DuckDB values that Arrow can't carry fail the read, naming the column: `TIME
/// '24:00:00'` and `TIMETZ '24:00:00+00'` (read as NULL), `±infinity` dates and
/// timestamps (read as their raw `i64::MAX`), and `HUGEINT` / `UHUGEINT` values of more
/// than the 38 digits of the `Decimal(38, 0)` they are read as (they don't round-trip, or
/// wrap past `i128`). Also inside lists, arrays, structs, maps and unions.
fn duckdb_value_check(db_type: &str, column: &str) -> Option<String> {
    let bad = duckdb_bad_value(db_type, column, 0)?;
    let msg = DialectName::Duckdb.quote_str(&format!(
        "column {column} holds a value Polars can't represent: "
    ));
    Some(format!(
        "CASE WHEN {bad} THEN error({msg} || CAST({column} AS VARCHAR)) ELSE {column} END"
    ))
}

/// SQL that is true where the DuckDB value `x` of type `t` is (or holds) a value Arrow
/// can't carry (see [`duckdb_value_check`]); `None` if no value of the type can be one.
/// Lists are searched with a list comprehension, which every DuckDB version since 0.8
/// reads (its lambda syntax changed in 1.3).
fn duckdb_bad_value(t: &str, x: &str, depth: usize) -> Option<String> {
    let t = t.trim();
    let starts = |p: &str| t.get(..p.len()).is_some_and(|s| s.eq_ignore_ascii_case(p));
    // `T[]` (a list), `T[n]` (an array)
    if t.ends_with(']')
        && let Some(open) = t.rfind('[')
    {
        let v = format!("pl_x{depth}");
        let bad = duckdb_bad_value(&t[..open], &v, depth + 1)?;
        return Some(format!("len([1 FOR {v} IN {x} IF {bad}]) > 0"));
    }
    // (a NULL check is no error: `NULL OR TRUE` is true)
    let any = |checks: Vec<String>| {
        (!checks.is_empty()).then(|| {
            checks
                .iter()
                .map(|c| format!("({c})"))
                .collect::<Vec<_>>()
                .join(" OR ")
        })
    };
    for (prefix, extract) in [("STRUCT(", "struct_extract"), ("UNION(", "union_extract")] {
        if starts(prefix) {
            let fields = t[prefix.len()..].strip_suffix(')')?;
            let checks = split_top_level(fields)
                .into_iter()
                .filter_map(|field| {
                    let (name, ft) = duckdb_field(field)?;
                    let value = format!("{extract}({x}, {})", DialectName::Duckdb.quote_str(&name));
                    duckdb_bad_value(ft, &value, depth)
                })
                .collect();
            return any(checks);
        }
    }
    if starts("MAP(") {
        let parts = split_top_level(t["MAP(".len()..].strip_suffix(')')?);
        let [k, v] = parts.as_slice() else {
            return None;
        };
        let checks = [(k, "map_keys"), (v, "map_values")]
            .into_iter()
            .filter_map(|(ft, f)| duckdb_bad_value(&format!("{ft}[]"), &format!("{f}({x})"), depth))
            .collect();
        return any(checks);
    }
    Some(match t.to_ascii_uppercase().as_str() {
        "TIME" => format!("{x} = TIME '24:00:00'"),
        "TIME_NS" => format!("{x} = CAST('24:00:00' AS TIME_NS)"),
        // read as the local time of day, its offset dropped
        "TIME WITH TIME ZONE" | "TIMETZ" => format!("CAST({x} AS TIME) = TIME '24:00:00'"),
        "DATE"
        | "TIMESTAMP"
        | "TIMESTAMP_S"
        | "TIMESTAMP_MS"
        | "TIMESTAMP_NS"
        | "TIMESTAMP WITH TIME ZONE" => format!("isinf({x})"),
        "HUGEINT" => format!("({x} > {MAX_DECIMAL_38} OR {x} < -{MAX_DECIMAL_38})"),
        "UHUGEINT" => format!("{x} > {MAX_DECIMAL_38}"),
        _ => return None,
    })
}

/// The largest value of a `Decimal(38, 0)`.
const MAX_DECIMAL_38: &str = "99999999999999999999999999999999999999";

/// The parts of a type's argument list, split at its top-level commas.
fn split_top_level(s: &str) -> Vec<&str> {
    // inside `"identifiers"` and `'enum labels'` (where `(`, `"` or `,` are text; a doubled quote
    // closes and reopens)
    let (mut parts, mut depth, mut quote, mut start) = (vec![], 0usize, None::<char>, 0);
    for (i, c) in s.char_indices() {
        match (c, quote) {
            ('"' | '\'', None) => quote = Some(c),
            (q, Some(open)) if q == open => quote = None,
            (_, Some(_)) => {},
            ('(' | '[', None) => depth += 1,
            (')' | ']', None) => depth = depth.saturating_sub(1),
            (',', None) if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            },
            _ => {},
        }
    }
    parts.push(s[start..].trim());
    parts
}

/// A DuckDB struct (or union) field, `name TYPE` or `"a name" TYPE`: its name and type.
fn duckdb_field(field: &str) -> Option<(String, &str)> {
    let field = field.trim();
    if let Some(rest) = field.strip_prefix('"') {
        // `""` is a quote in the name
        let mut end = 0;
        let bytes = rest.as_bytes();
        while end < bytes.len() {
            if bytes[end] == b'"' {
                if bytes.get(end + 1) == Some(&b'"') {
                    end += 2;
                    continue;
                }
                return Some((rest[..end].replace("\"\"", "\""), &rest[end + 1..]));
            }
            end += 1;
        }
        return None;
    }
    let (name, t) = field.split_once(char::is_whitespace)?;
    Some((name.to_string(), t))
}

/// A MySQL `DATE` / `DATETIME` / `TIMESTAMP`, which may hold zero dates; or a date column
/// that may be one ([`MYSQL_DATE`]).
fn mysql_zero_date_type(dialect: DialectName, db_type: &str) -> bool {
    let t = db_type.trim().to_ascii_lowercase();
    let base = t.split('(').next().unwrap_or(&t);
    matches!(dialect, DialectName::Mysql | DialectName::Singlestore)
        && (matches!(base, "date" | "datetime" | "timestamp") || db_type == MYSQL_DATE)
}

/// Whether a MySQL date column (quoted) holds a zero date; NULL for NULL.
fn mysql_zero_date(column: &str) -> String {
    format!("MONTH({column}) = 0 OR DAYOFMONTH({column}) = 0")
}

/// A pushed filter on MySQL date columns, evaluated as if their zero dates were NULL, as
/// they are read. Each leaf predicate (whatever isn't an AND, OR or NOT) that references a
/// date column `c` becomes `(leaf AND NOT zero(c)) OR (zero(c) AND leaf[c := NULL])`, nested
/// over the date columns the leaf itself references. `zero(c)` is never NULL, so this is
/// exact under three-valued logic: where it is false the leaf is unchanged, where true the
/// leaf sees NULL. The database folds the second branch to a constant (false, except for
/// `IS NULL`), keeping index ranges on `c` usable. The rewrite is per leaf, so the filter
/// grows linearly (wrapping the whole filter once per column doubled it each time).
///
/// `widen`: a test for NULL on a column of unknown type also keeps its zero dates, which makes the
/// filter a superset: for a user's filter only, not a partition's predicate (the partitions must
/// not overlap).
fn zero_dates_as_null(config: &DatabaseScanConfig, filter: SqlExpr, widen: bool) -> SqlExpr {
    match filter {
        SqlExpr::Binary(l, op @ (SqlBinOp::And | SqlBinOp::Or), r) => SqlExpr::Binary(
            Box::new(zero_dates_as_null(config, *l, widen)),
            op,
            Box::new(zero_dates_as_null(config, *r, widen)),
        ),
        SqlExpr::Not(x) => SqlExpr::Not(Box::new(zero_dates_as_null(config, *x, widen))),
        mut leaf => {
            for c in config.columns.iter() {
                if !references(&leaf, &c.name) {
                    continue;
                }
                let Some(zero) = zero_date_check(config, &c.name) else {
                    // A column of unknown type (`schema=`) may be a date column holding zero dates,
                    // which the read refuses. A test for NULL must not hide such a row (a zero date
                    // is not NULL): the filter keeps it too (and is a superset, see
                    // `Translator::translate`), and the read fails on it as it does without a
                    // filter.
                    if widen && zero_date_candidate(config, c) && tests_null(&leaf, &c.name) {
                        let q = config.dialect.quote_ident(&c.name);
                        leaf = SqlExpr::binary(
                            leaf,
                            SqlBinOp::Or,
                            SqlExpr::Raw(format!("IFNULL({}, FALSE)", mysql_zero_date(&q))),
                        );
                    }
                    continue;
                };
                let as_null = substitute_null(&leaf, &c.name);
                leaf = SqlExpr::binary(
                    SqlExpr::binary(leaf, SqlBinOp::And, SqlExpr::Not(Box::new(zero.clone()))),
                    SqlBinOp::Or,
                    SqlExpr::binary(zero, SqlBinOp::And, as_null),
                );
            }
            leaf
        },
    }
}

/// `TRUE` where a MySQL date column holds a zero date (`FALSE` for NULL), if the column
/// may hold zero dates.
fn zero_date_check(config: &DatabaseScanConfig, column: &str) -> Option<SqlExpr> {
    let dialect = config.dialect;
    let t = config.column(column)?.db_type.as_deref()?;
    mysql_zero_date_type(dialect, t).then(|| {
        SqlExpr::Raw(format!(
            "IFNULL({}, FALSE)",
            mysql_zero_date(&dialect.quote_ident(column))
        ))
    })
}

/// An `ORDER BY` column. A MySQL date column is ordered by its values as read, zero
/// dates as NULL: a date with a zero month or day (`2020-00-01`) sorts among the real
/// dates otherwise. The conversion is written on the relation's column, so it applies
/// whether or not the column is selected. (An index on the column no longer gives the
/// order then: the database sorts.)
///
/// Other columns are ordered by the relation's column too, not the select item of that
/// name, which may be converted (e.g. to text, which sorts otherwise): see
/// `SqlQuery::qualified`.
fn order_by_expr(config: &DatabaseScanConfig, column: &PlSmallStr) -> SqlExpr {
    if zero_date_check(config, column).is_none() {
        return SqlExpr::col(column.clone());
    }
    let dialect = config.dialect;
    let relation = relation_qualifier(config);
    let qualified = format!("{relation}.{}", dialect.quote_ident(column));
    SqlExpr::Raw(format!(
        "IF({}, NULL, {qualified})",
        mysql_zero_date(&qualified)
    ))
}

fn references(e: &SqlExpr, column: &str) -> bool {
    use SqlExpr as E;
    match e {
        E::Column(c) => c == column,
        E::Literal(_) | E::Raw(_) => false,
        E::Binary(l, _, r) | E::IsNotDistinctFrom(l, r, _) => {
            references(l, column) || references(r, column)
        },
        E::Not(x) | E::IsNull(x, _) | E::InList(x, _, _) => references(x, column),
        E::Between(x, lo, hi) => [x, lo, hi].iter().any(|x| references(x, column)),
        E::Like { expr, .. } => references(expr, column),
    }
}

/// A MySQL column of unknown type, read as a type a date may be read as (`schema=`): it may be a
/// date column holding zero dates, which the read refuses.
pub(crate) fn zero_date_candidate(config: &DatabaseScanConfig, c: &ColumnMeta) -> bool {
    matches!(
        config.dialect,
        DialectName::Mysql | DialectName::Singlestore
    ) && c.db_type.is_none()
        && matches!(
            c.dtype,
            DataType::String
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt32
                | DataType::UInt64
        )
}

/// Whether `e` tests `column` for NULL (`IS [NOT] NULL`, `<=>`).
pub(crate) fn tests_null(e: &SqlExpr, column: &str) -> bool {
    use SqlExpr as E;
    match e {
        E::IsNull(x, _) => references(x, column),
        E::IsNotDistinctFrom(l, r, _) => references(l, column) || references(r, column),
        E::Binary(l, _, r) => tests_null(l, column) || tests_null(r, column),
        E::Not(x) | E::InList(x, _, _) => tests_null(x, column),
        E::Between(x, lo, hi) => [x, lo, hi].iter().any(|x| tests_null(x, column)),
        E::Like { expr, .. } => tests_null(expr, column),
        E::Column(_) | E::Literal(_) | E::Raw(_) => false,
    }
}

/// `e` with `column` replaced by NULL.
fn substitute_null(e: &SqlExpr, column: &str) -> SqlExpr {
    use SqlExpr as E;
    let sub = |x: &SqlExpr| Box::new(substitute_null(x, column));
    match e {
        E::Column(c) if c == column => E::Literal(SqlLiteral::Null),
        E::Column(_) | E::Literal(_) | E::Raw(_) => e.clone(),
        E::Binary(l, op, r) => E::Binary(sub(l), *op, sub(r)),
        E::IsNotDistinctFrom(l, r, b) => E::IsNotDistinctFrom(sub(l), sub(r), *b),
        E::Not(x) => E::Not(sub(x)),
        E::IsNull(x, b) => E::IsNull(sub(x), *b),
        E::InList(x, list, b) => E::InList(sub(x), list.clone(), *b),
        E::Between(x, lo, hi) => E::Between(sub(x), sub(lo), sub(hi)),
        E::Like {
            expr,
            pattern,
            escape,
        } => E::Like {
            expr: sub(expr),
            pattern: pattern.clone(),
            escape: *escape,
        },
    }
}

/// The columns of a table as select items, with the conversions applied, if any column
/// needs one.
fn converted_select_list(
    dialect: DialectName,
    backend: &BackendKind,
    setup_runs: bool,
    own_connection: bool,
    info: &CatalogInfo,
) -> Option<Vec<SqlSelectItem>> {
    let mut any = false;
    // DuckDB tables have no declared types (they are typed by their values), but some
    // columns are still selected through a conversion (`text_conversion`)
    let types = if info.types.is_empty() && dialect == DialectName::Duckdb {
        info.compare_order
            .iter()
            .filter_map(|n| Some((n.clone(), info.compare_types.get(n)?.clone())))
            .collect()
    } else {
        info.types.clone()
    };
    let items = types
        .iter()
        .map(|(name, t)| {
            let quoted = dialect.quote_ident(name);
            let conversion = if info.types.is_empty() {
                // only what changes the type: the value checks are the read's
                dialect.text_conversion(t, &quoted)
            } else {
                column_conversion(dialect, backend, setup_runs, own_connection, t, &quoted)
            };
            match conversion {
                Some(sql) => {
                    any = true;
                    SqlSelectItem::Expr {
                        sql,
                        alias: name.clone(),
                    }
                },
                None => SqlSelectItem::Column(name.clone()),
            }
        })
        .collect::<Vec<_>>();
    any.then_some(items)
}

/// The SQL a filter compares a column through, where not the column itself: a SQLite column
/// read converted, whose values the database would compare as other values than Polars
/// reads. One read as reals or integers without that affinity may hold integers (`2**53 +
/// 1`, read as `9007199254740992.0`), integral reals or numbers' text (`'1'`, read as 1),
/// or reals cut to integers (`schema_overrides`: 1.5 read as 1); a
/// text one blobs (`X'7631'`, read as `'v1'`). The converted value is compared, as read (an
/// index on the column isn't used then); a text one as `CAST(x AS TEXT)`, which keeps its
/// collation (it holds no numbers: TEXT affinity, see `catalog::compares_as_text`).
pub(crate) fn compared_column_sql(config: &DatabaseScanConfig, column: &str) -> Option<String> {
    let c = config.column(column)?;
    // a DuckDB column read as text (`TIMETZ`) is compared as that text
    if config.dialect == DialectName::Duckdb
        && c.db_type.as_deref() == Some(DUCKDB_TEXT)
        && c.dtype == DataType::String
    {
        return Some(format!(
            "CAST({} AS VARCHAR)",
            config.dialect.quote_ident(column)
        ));
    }
    if config.dialect != DialectName::Sqlite {
        return None;
    }
    let db_type = c.db_type.as_deref()?;
    let quoted = config.dialect.quote_ident(column);
    if db_type == SQLITE_TEXT && c.dtype == DataType::String {
        return Some(format!("CAST({quoted} AS TEXT)"));
    }
    if ![
        SQLITE_REAL,
        SQLITE_REAL_CUT,
        SQLITE_UNTYPED_INTEGER,
        SQLITE_INTEGER_CUT,
    ]
    .contains(&db_type)
    {
        return None;
    }
    if db_type == SQLITE_REAL {
        // an integer is compared as it is, which SQLite compares to a float exactly: a row
        // is kept when its true value satisfies the filter (and the read then fails on it
        // if a real can't hold it); rounded, it could be dropped silently
        let x = &quoted;
        let ops = SqliteOps {
            portable: !config.options.connection.opened_by_plugin(),
        };
        let (is_int, is_text) = (ops.is_integer(x), ops.is_text(x));
        let (int_text, real_text) = (ops.text_is(x, "INTEGER"), ops.text_is(x, "REAL"));
        return Some(format!(
            "CASE WHEN {is_int} THEN {x} \
            WHEN {is_text} AND {int_text} \
            THEN CAST({x} AS INTEGER) \
            WHEN {is_text} AND {real_text} \
            THEN CAST({x} AS REAL) \
            WHEN {is_text} THEN {m} || {x} ELSE {x} END",
            m = ops.mark()
        ));
    }
    column_conversion(
        config.dialect,
        &config.backend.kind,
        config.session_setup_runs(),
        config.options.connection.opened_by_plugin(),
        db_type,
        &quoted,
    )
}

/// Whether the driver %-formats the statements of this scan (`params=`): the SQL the plugin
/// writes doubles its `%`.
fn percent_formatted(config: &DatabaseScanConfig) -> bool {
    crate::filter::percent_formatted(&config.options.connection, &config.backend.kind)
}

/// `sql`, written by the plugin, with its `%` doubled where the driver %-formats it.
fn escape_percent(config: &DatabaseScanConfig, sql: String) -> String {
    if percent_formatted(config) {
        sql.replace('%', "%%")
    } else {
        sql
    }
}

/// Select items as fixed SQL, for the discovery queries that don't go through the renderer.
fn fixed_select_list(dialect: DialectName, items: &[SqlSelectItem]) -> String {
    items
        .iter()
        .map(|item| match item {
            SqlSelectItem::Column(c) => dialect.quote_ident(c),
            SqlSelectItem::Expr { sql, alias } => {
                format!("{sql} AS {}", dialect.quote_ident(alias))
            },
            _ => "*".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The result schema of the relation, without reading rows, and the columns typed from a
/// sample of their values (not by the result's types).
fn discover_schema(
    session: &mut dyn DatabaseSession,
    options: &DatabaseScanOptions,
    dialect: DialectName,
    backend: &BackendKind,
    query: &QueryDiscovery,
    select_list: Option<&[SqlSelectItem]>,
    info: &CatalogInfo,
) -> PolarsResult<(Schema, PlHashSet<PlSmallStr>, Vec<PlSmallStr>)> {
    let is_query = matches!(options.relation, DatabaseRelation::Query { .. });
    let probe = match query.probe.as_ref().filter(|_| is_query) {
        Some(probe) => probe.clone(),
        None => format!(
            "SELECT {} FROM {} WHERE 1 = 0",
            select_list.map_or_else(|| "*".to_string(), |s| fixed_select_list(dialect, s)),
            quoted_relation(dialect, &options.relation)
        ),
    };
    let first_batch = |session: &mut dyn DatabaseSession, sql: &str| {
        if verbose() {
            eprintln!("[scan_database]: schema from: {sql}");
        }
        // ConnectorX builds no Arrow array of ClickHouse decimals (`Array(Decimal(..))`),
        // and says so only with an Arrow error: for a query (a table's are read as text,
        // see `column_conversion`) that selects one
        let connectorx_array = |e: PolarsError| {
            if matches!(backend, BackendKind::ConnectorX)
                && dialect == DialectName::Clickhouse
                && e.to_string()
                    .contains("column types must match schema types")
            {
                polars_err!(
                    InvalidOperation:
                    "ConnectorX can't read an array of decimals from ClickHouse (a column of \
                    this query's result): leave it out of the query, or pin another driver, \
                    e.g. driver=\"adbc\" or \"clickhouse_connect\" ({e})"
                )
            } else {
                e
            }
        };
        let mut batches = session.execute(sql, Some(1)).map_err(connectorx_array)?;
        match batches.next().transpose().map_err(connectorx_array)? {
            Some(first) => Ok(first),
            None => polars_bail!(
                ComputeError: "database returned no result for the schema query: {sql}"
            ),
        }
    };
    // A query is never run whole to find its schema: the read runs it, and running it
    // twice could write twice (`nextval`). One that can't be wrapped is probed wrapped
    // all the same if it doesn't parse (the database may take it); else it is run, for
    // its first batch, only if that can't change anything.
    let mut ran_query = false;
    let first = match &options.relation {
        DatabaseRelation::Query { sql } if !query.wrappable => {
            let probed = match query.parses {
                false => first_batch(session, &probe),
                true => Err(polars_err!(ComputeError: "the query can't be wrapped")),
            };
            match probed {
                Ok(first) => first,
                Err(_) if query.pure_read => {
                    ran_query = true;
                    first_batch(session, strip_statement_end(sql))?
                },
                Err(e) => polars_bail!(
                    InvalidOperation:
                    "the schema of this query can't be found without running it, which the \
                    read would then do again; pass schema=..., or write the query so that \
                    it can go inside a derived table: {e}"
                ),
            }
        },
        _ => first_batch(session, &probe)?,
    };

    let mut schema: Schema = first
        .schema()
        .iter()
        .map(|(name, dtype)| (name.clone(), normalize_dtype(dtype)))
        .collect();

    // Row-wise drivers can't type columns of an empty result, and SQLite has no column
    // types at all (ADBC calls every column of an empty result Int64). Type them from a
    // sample, which runs a query: not one known to be volatile (`nextval`), nor one that
    // may write where the session doesn't refuse it. SQLite's functions can't write.
    let sample_types = dialect == DialectName::Sqlite;
    let sampled = schema
        .iter()
        .filter(|(_, dt)| sample_types || dt.is_null())
        .map(|(name, _)| name.clone())
        .collect::<PlHashSet<_>>();
    let untyped = !sampled.is_empty();
    let mut mixed = Vec::new();
    let may_sample = !is_query
        || sample_types
        || !(query.volatile || (query.side_effects && !query.read_only_session));
    if untyped && !ran_query && may_sample && dialect.supports_limit() {
        let renderer = sql_renderer()?;
        // ConnectorX can't read a SQLite column whose values have several storage classes:
        // their classes are sampled instead (see `sqlite_class_dtype`).
        // A connection of the user's (a row driver) reads a column of a blob and other values as
        // the first value's type, and the rest as NULL: its classes are sampled too.
        let own_connection = !options.connection.opened_by_plugin();
        let mut classes = sample_types && matches!(backend, BackendKind::ConnectorX);
        // (a connection of the user's may have replaced `typeof`: it calls no function there)
        let ops = SqliteOps {
            portable: own_connection,
        };
        let class_items = || -> Vec<SqlSelectItem> {
            schema
                .iter_names()
                .map(|c| SqlSelectItem::Expr {
                    sql: ops.storage_class(&dialect.quote_ident(c)),
                    alias: c.clone(),
                })
                .collect()
        };
        let items = if classes {
            class_items()
        } else if sample_types {
            select_list.map_or_else(|| vec![SqlSelectItem::Star], <[_]>::to_vec)
        } else {
            // Only the untyped columns: the others may hold values a row driver can't
            // build a frame of (psycopg's JSON arrays), which the read converts.
            match select_list {
                Some(list) => list
                    .iter()
                    .filter(|item| match item {
                        SqlSelectItem::Column(c) | SqlSelectItem::Expr { alias: c, .. } => {
                            sampled.contains(c)
                        },
                        _ => false,
                    })
                    .cloned()
                    .collect(),
                None => schema
                    .iter_names()
                    .filter(|c| sampled.contains(*c))
                    .cloned()
                    .map(SqlSelectItem::Column)
                    .collect(),
            }
        };
        let render = |items: Vec<SqlSelectItem>| -> PolarsResult<String> {
            let mut sample_query = SqlQuery::select(items, from_clause(options));
            sample_query.limit = Some(TYPE_SAMPLE_ROWS);
            sample_query.escape_percent =
                crate::filter::percent_formatted(&options.connection, backend);
            renderer.render(&sample_query, dialect)
        };
        let sample_sql = render(items)?;
        if verbose() {
            eprintln!("[scan_database]: typing columns from a sample: {sample_sql}");
        }
        let sample = match session.query(&sample_sql) {
            Ok(sample) => sample,
            // a row driver can't build a frame of a column of values of several classes (a blob
            // next to numbers): their classes are sampled instead
            Err(e) if sample_types && !classes => {
                let sql = render(class_items())?;
                if verbose() {
                    eprintln!("[scan_database]: the sample failed ({e}); typing by classes: {sql}");
                }
                classes = true;
                session.query(&sql)?
            },
            Err(e) => return Err(e),
        };
        for (name, dtype) in schema.iter_mut() {
            if (dtype.is_null() || (sample_types && sample.height() > 0))
                && let Ok(col) = sample.column(name)
            {
                *dtype = if classes {
                    sqlite_class_dtype(col)
                } else {
                    normalize_dtype(col.dtype())
                };
            }
            // No values to type by (ADBC types an empty result's columns Int64): the
            // declared type's affinity tells what values the column holds.
            if sample_types
                && (sample.height() == 0 || dtype.is_null())
                && let Some(t) = info.compare_types.get(name)
            {
                *dtype = sqlite_affinity_dtype(t);
            }
        }
    }
    // A row driver of the user's reads a column of blobs and other values as its first value's
    // type, and the others as NULL: the classes of the Binary columns are sampled (not of all: a
    // volatile expression is evaluated once per test there).
    if sample_types
        && !options.connection.opened_by_plugin()
        && !matches!(backend, BackendKind::ConnectorX)
        && may_sample
        && dialect.supports_limit()
    {
        let ops = SqliteOps { portable: true };
        let items = schema
            .iter()
            .filter(|(_, dt)| matches!(dt, DataType::Binary))
            .map(|(name, _)| SqlSelectItem::Expr {
                sql: ops.storage_class(&dialect.quote_ident(name)),
                alias: name.clone(),
            })
            .collect::<Vec<_>>();
        if !items.is_empty() {
            let mut sample_query = SqlQuery::select(items, from_clause(options));
            sample_query.limit = Some(TYPE_SAMPLE_ROWS);
            sample_query.escape_percent =
                crate::filter::percent_formatted(&options.connection, backend);
            let sql = sql_renderer()?.render(&sample_query, dialect)?;
            if let Ok(classes) = session.query(&sql) {
                for col in classes.columns() {
                    if sqlite_blob_with_others(col) {
                        mixed.push(col.name().clone());
                    }
                }
            }
        }
    }
    // Still untyped (every sampled value is NULL, or no rows to sample): a dtype any value
    // can be cast to, not `Null`, which no value read later would fit.
    for (_, dtype) in schema.iter_mut() {
        if dtype.is_null() {
            *dtype = DataType::String;
        }
    }
    Ok((schema, sampled, mixed))
}

/// Whether sampled storage classes hold a blob and a value of another class.
fn sqlite_blob_with_others(classes: &Column) -> bool {
    let Ok(ca) = classes.str() else {
        return false;
    };
    let seen: PlHashSet<&str> = ca.iter().flatten().filter(|c| *c != "null").collect();
    seen.contains("blob") && seen.len() > 1
}

/// The dtype of a SQLite column by the storage classes (`typeof`) of its sampled values, as
/// the other drivers type them; `Null` without values.
fn sqlite_class_dtype(classes: &Column) -> DataType {
    let mut seen = PlHashSet::new();
    if let Ok(ca) = classes.str() {
        seen.extend(ca.iter().flatten().filter(|c| *c != "null"));
    }
    let only = |allowed: &[&str]| seen.iter().all(|c| allowed.contains(c));
    match () {
        _ if seen.is_empty() => DataType::Null,
        _ if only(&["integer"]) => DataType::Int64,
        _ if only(&["integer", "real"]) => DataType::Float64,
        // a blob next to other values is read as one, and fails on those (`SQLITE_BLOB`)
        _ if seen.contains("blob") => DataType::Binary,
        _ => DataType::String,
    }
}

/// The dtype a SQLite column of declared type `t` reads as, by its type affinity (the
/// SQLite rules). `NUMERIC` affinity (`DATE`, `DECIMAL`, `BOOLEAN`, …) holds integers,
/// floats or text alike: `String`, which they all cast to.
fn sqlite_affinity_dtype(t: &str) -> DataType {
    let t = t.to_ascii_uppercase();
    if t.contains("INT") {
        DataType::Int64
    } else if t.contains("CHAR") || t.contains("CLOB") || t.contains("TEXT") {
        DataType::String
    } else if t.contains("BLOB") {
        DataType::Binary
    } else if t.contains("REAL") || t.contains("FLOA") || t.contains("DOUB") {
        DataType::Float64
    } else {
        DataType::String
    }
}

/// A SQLite column of TEXT affinity that is read as a number (`schema_overrides=`/`schema=`):
/// SQLite compares and sorts its values as text (`'50' > '100'`), so integer cuts or an
/// `ORDER BY` on it don't follow the numbers read.
fn sqlite_text_read_as_number(config: &DatabaseScanConfig, column: &str) -> bool {
    config.sqlite_text_numbers.iter().any(|c| c == column)
}

/// Whether a SQLite declared type has TEXT affinity (`CHAR`, `CLOB`, `TEXT`, and no `INT`).
fn sqlite_text_affinity(t: &str) -> bool {
    let t = t.to_ascii_uppercase();
    !t.contains("INT") && (t.contains("CHAR") || t.contains("CLOB") || t.contains("TEXT"))
}

fn from_clause(options: &DatabaseScanOptions) -> SqlFrom {
    match &options.relation {
        DatabaseRelation::Table { parts, .. } => SqlFrom::Table(parts.clone()),
        DatabaseRelation::Query { sql } => SqlFrom::Subquery(strip_statement_end(sql).into()),
    }
}

/// A pushed conjunct of the scan predicate.
#[derive(Clone, Debug)]
pub struct PushedFilter {
    pub expr: SqlExpr,
    /// The database returns exactly the rows Polars keeps. Otherwise a superset.
    pub exact: bool,
    /// The columns it compares.
    pub columns: Vec<PlSmallStr>,
}

/// What Polars pushed into a scan (the resolver's projection, filters and limit).
#[derive(Clone, Debug, Default)]
pub struct ScanPushdown {
    /// Columns to read, in schema order. `None` reads every column.
    pub columns: Option<Vec<PlSmallStr>>,
    pub filters: Vec<PushedFilter>,
    /// Number of top-level conjuncts in the scan predicate, pushed or not.
    pub total_filters: usize,
    /// Rows needed from the start of the scan.
    pub limit: Option<u64>,
}

/// Resolves partitions and renders the SQL of every partition. Runs on every resolution, which
/// Polars does after predicate, projection and slice pushdown.
///
/// `target_partitions` is the distributed planner's target, which overrides automatic counts
/// (not a `partition_num` the user gave).
pub fn compile_scan(
    config: &DatabaseScanConfig,
    pushdown: ScanPushdown,
    target_partitions: Option<usize>,
) -> PolarsResult<CompiledScan> {
    let options = &config.options;
    let dialect = config.dialect;

    if let Some(reason) = &config.pushdown_disabled {
        return compile_unwrapped(config, reason.clone(), pushdown.total_filters);
    }

    let renderer = sql_renderer()?;
    let from = from_clause(options);

    let mut columns = pushdown
        .columns
        .clone()
        .unwrap_or_else(|| config.schema.iter_names().cloned().collect());
    for c in &options.order_by {
        polars_ensure!(
            config.schema.contains(c),
            ColumnNotFound: "order_by column '{c}' is not in the database source"
        );
        polars_ensure!(
            !sqlite_text_read_as_number(config, c),
            InvalidOperation:
            "order_by column '{c}' is a SQLite TEXT column read as a number: SQLite sorts its \
            values as text ('100' before '50'). Read it as a string, or declare it INTEGER"
        );
    }
    columns.dedup();
    check_connectorx_arrays(config, &columns)?;
    check_refused_columns(config, &columns)?;

    let (partitions, partitioning) = resolve_partitions(
        config,
        renderer.as_ref(),
        &from,
        &pushdown.filters,
        target_partitions,
    )?;
    if options.consistency == Consistency::Snapshot && partitions.len() > 1 {
        polars_bail!(
            InvalidOperation:
            "consistency='snapshot' is not supported for {dialect} with {} partitions yet; \
            pass partition_on=None to read with one consistent query",
            partitions.len()
        );
    }

    let select: Vec<SqlSelectItem> = if columns.is_empty() {
        vec![SqlSelectItem::One]
    } else {
        let mut items: Vec<_> = columns.iter().map(|c| select_item(config, c)).collect();
        items.extend(utf8_checks(config, &columns));
        items
    };
    let order_by: Vec<SqlOrderBy> = options
        .order_by
        .iter()
        .map(|c| SqlOrderBy {
            expr: order_by_expr(config, c),
            descending: false,
        })
        .collect();
    let limit = pushdown.limit.filter(|_| dialect.supports_limit());
    // the columns read through a conversion, under their own name
    let qualified: Vec<PlSmallStr> = select
        .iter()
        .filter_map(|item| match item {
            SqlSelectItem::Expr { alias, .. } if config.schema.contains(alias) => {
                Some(alias.clone())
            },
            _ => None,
        })
        .collect();

    let pushed: Vec<SqlExpr> = pushdown
        .filters
        .iter()
        .map(|f| zero_dates_as_null(config, f.expr.clone(), true))
        .collect();
    // The partitions differ in their predicate only: the shared filter (an `IN` list of
    // thousands of values) is rendered once, not by each partition.
    let escape_percent = percent_formatted(config);
    let query = SqlQuery {
        columns: select.clone(),
        from: from.clone(),
        filter: pushed.clone(),
        order_by: order_by.clone(),
        limit,
        qualified: qualified.clone(),
        escape_percent,
    };
    let count = SqlQuery {
        columns: vec![SqlSelectItem::CountStar],
        from: from.clone(),
        filter: pushed.clone(),
        order_by: vec![],
        limit: None,
        qualified: vec![],
        escape_percent,
    };
    // A zero date goes to the partition of the NULLs, as which it is read.
    let predicates: Vec<Option<SqlExpr>> = partitions
        .iter()
        .map(|part| {
            part.predicate
                .clone()
                .map(|p| zero_dates_as_null(config, p, false))
        })
        .collect();
    let compiled: Vec<CompiledPartition> = renderer
        .render_partitions(&query, &count, &predicates, dialect)?
        .into_iter()
        .map(|(sql, count_sql)| CompiledPartition { sql, count_sql })
        .collect();

    let exact_filters = pushdown.filters.iter().filter(|f| f.exact).count();
    let out = CompiledScan {
        renderer: renderer.id(),
        all_exact: pushdown.total_filters == pushdown.filters.len()
            && exact_filters == pushdown.filters.len(),
        columns,
        partitioning,
        pushed_filters: pushdown.filters.len(),
        filter_columns: pushdown
            .filters
            .iter()
            .flat_map(|f| f.columns.iter().cloned())
            .collect(),
        exact_filters,
        total_filters: pushdown.total_filters,
        limit,
        pushdown_disabled: None,
        repeatable: query_repeatable(config),
        partitions: compiled,
    };
    if verbose() {
        eprintln!(
            "[scan_database]: {} partition(s) ({}), sql[0]: {}",
            out.partitions.len(),
            out.partitioning,
            out.partitions[0].sql
        );
    }
    Ok(out)
}

/// A column as a select item, read through its conversion if it has one.
fn select_item(config: &DatabaseScanConfig, c: &PlSmallStr) -> SqlSelectItem {
    let dialect = config.dialect;
    // SQLite reads an unknown double-quoted name as a string literal: a column missing from
    // the source (`schema=`, or dropped since planning) would read as its own name in every
    // row. Qualified by the relation, it is an error.
    let ident = if dialect == DialectName::Sqlite {
        format!("{}.{}", relation_qualifier(config), dialect.quote_ident(c))
    } else {
        dialect.quote_ident(c)
    };
    let conversion = config
        .column(c)
        .and_then(|m| m.db_type.as_deref())
        .and_then(|t| {
            column_conversion(
                dialect,
                &config.backend.kind,
                config.session_setup_runs(),
                config.options.connection.opened_by_plugin(),
                t,
                &ident,
            )
        });
    match conversion {
        Some(sql) => SqlSelectItem::Expr {
            sql,
            alias: c.clone(),
        },
        None if dialect == DialectName::Sqlite => SqlSelectItem::Expr {
            sql: ident,
            alias: c.clone(),
        },
        None => SqlSelectItem::Column(c.clone()),
    }
}

/// The columns the plugin reads as epoch nanoseconds, a number, where the database type is
/// a temporal one (ClickHouse `DateTime64(7..9)` through `toUnixTimestamp64Nano`, see
/// `column_conversion`): the planned type, also an override of it, says the unit
/// (`cast_to_user_schema`). Empty without `schema_overrides` (the only case that needs it).
pub(crate) fn numeric_temporal_conversions(config: &DatabaseScanConfig) -> PlHashSet<PlSmallStr> {
    let mut out = PlHashSet::new();
    if config.options.schema_overrides.is_none()
        || config.dialect != DialectName::Clickhouse
        || matches!(config.backend.kind, BackendKind::Adbc { .. })
    {
        return out;
    }
    for c in config.columns.iter() {
        let nanoseconds = c.db_type.as_deref().is_some_and(|t| {
            matches!(
                declared_dtype(config.dialect, t),
                Some(DataType::Datetime(TimeUnit::Nanoseconds, _))
            )
        });
        if nanoseconds
            && matches!(
                select_item(config, &c.name),
                SqlSelectItem::Expr { sql, .. } if sql.starts_with("toUnixTimestamp64Nano(")
            )
        {
            out.insert(c.name.clone());
        }
    }
    out
}

/// The columns whose read may legitimately be another kind of value than the planned dtype
/// (`cast_columns` doesn't refuse a retyped column there): every column but DuckDB's, whose drivers
/// return what the type says (as text only where the plugin converts it, a marker `db_type`). Other
/// drivers read some numbers as text (ADBC's MySQL `YEAR`) and some instants as naive UTC time
/// (ClickHouse, Postgres), by their own conversions.
pub(crate) fn converted_columns(config: &DatabaseScanConfig) -> PlHashSet<PlSmallStr> {
    config
        .columns
        .iter()
        .filter(|c| {
            config.dialect != DialectName::Duckdb
                || c.db_type.as_deref().is_some_and(|t| t.starts_with('<'))
        })
        .map(|c| c.name.clone())
        .collect()
}

/// The MySQL `TIMESTAMP` columns typed as text (`schema_overrides`): their instants are read as UTC
/// wall-clock time, which is naive where the SQL converts them (see `column_conversion`), and
/// written with the offset where the bridge converts them (`+00:00`), as the other reads of the
/// column do.
pub(crate) fn mysql_timestamp_text_columns(config: &DatabaseScanConfig) -> Vec<PlSmallStr> {
    if config.dialect != DialectName::Mysql {
        return vec![];
    }
    config
        .columns
        .iter()
        .filter(|c| {
            c.dtype == DataType::String
                && c.db_type.as_deref().is_some_and(|t| {
                    let t = t.trim().to_ascii_lowercase();
                    t == "timestamp" || t.starts_with("timestamp(")
                })
        })
        .map(|c| c.name.clone())
        .collect()
}

/// The columns declared `String` under `schema=` whose read as a time of day can't be told from a
/// `TIMETZ` (DuckDB, Postgres and Redshift have one; the drivers return its local time without the
/// offset), mapped to the SQL that reads them as text. Python drivers say by the result's
/// description (see [`check_time_with_zone`]), the Arrow ones can't.
pub(crate) fn time_text_risk_columns(config: &DatabaseScanConfig) -> PlHashMap<PlSmallStr, String> {
    let has_timetz = matches!(
        config.dialect,
        DialectName::Duckdb | DialectName::Postgresql | DialectName::Redshift
    );
    if !config.user_schema
        || !has_timetz
        || !matches!(
            config.backend.kind,
            BackendKind::Adbc { .. } | BackendKind::ConnectorX
        )
    {
        return PlHashMap::new();
    }
    config
        .columns
        .iter()
        .filter(|c| c.dtype == DataType::String && c.db_type.is_none())
        .map(|c| {
            let quoted = config.dialect.quote_ident(&c.name);
            (
                c.name.clone(),
                format!("`{}`", config.dialect.cast_to_text(&quoted)),
            )
        })
        .collect()
}

/// The SQL Server columns read as text whose digits below the planned unit are cut (toward
/// the earlier instant): a `datetime2` / `datetimeoffset` of 7 fractional digits the user
/// typed `Datetime("us")` or `("ms")` (`schema_overrides`, `schema=`): the digits are the
/// ones the type has beyond the unit asked for, which isn't an error (and is not rounded: the
/// maximum, `9999-12-31 23:59:59.9999999`, would overflow). Filters on such a column aren't
/// pushed (it is retyped: not comparable).
pub(crate) fn truncating_columns(config: &DatabaseScanConfig) -> PlHashSet<PlSmallStr> {
    if config.dialect != DialectName::Mssql {
        return PlHashSet::new();
    }
    config
        .columns
        .iter()
        .filter(|c| {
            matches!(
                c.dtype,
                DataType::Datetime(TimeUnit::Microseconds | TimeUnit::Milliseconds, _)
            ) && c.db_type.as_deref().is_some_and(|t| {
                t == MSSQL_DATETIME
                    || matches!(
                        super::schema::mssql_nanosecond_type(t),
                        Some("datetime2" | "datetimeoffset")
                    )
            })
        })
        .map(|c| c.name.clone())
        .collect()
}

/// Refuses a `TIMETZ` column the driver's result description shows under `schema=` (declared
/// `String` or `Time`): it is read as the local time of day, its offset dropped.
pub(crate) fn check_time_with_zone(
    session: &dyn DatabaseSession,
    config: &DatabaseScanConfig,
) -> PolarsResult<()> {
    if !config.user_schema {
        return Ok(());
    }
    for name in session.time_with_zone_columns() {
        let Some(column) = config.column(&name) else {
            continue;
        };
        if column.db_type.is_some() || !matches!(column.dtype, DataType::String | DataType::Time) {
            continue;
        }
        let quoted = config.dialect.quote_ident(&name);
        polars_bail!(
            InvalidOperation:
            "column '{name}' is a time with a time zone (`TIMETZ`), which the driver returns as \
            the local time of day, without its offset: {} can't keep it. Read it as text in a \
            query (`{}`), declared String, or leave the column out",
            column.dtype,
            config.dialect.cast_to_text(&quoted)
        );
    }
    Ok(())
}

/// A batch's naive datetimes of `names` as UTC instants.
pub(crate) fn as_utc_instants(mut df: DataFrame, names: &[PlSmallStr]) -> PolarsResult<DataFrame> {
    for name in names {
        let Ok(col) = df.column(name) else {
            continue;
        };
        if let DataType::Datetime(tu, None) = col.dtype() {
            let aware = col.cast(&DataType::Datetime(*tu, Some(TimeZone::UTC)))?;
            df.replace(name, aware)?;
        }
    }
    Ok(df)
}

/// The quoted relation a scan's columns can be qualified by.
fn relation_qualifier(config: &DatabaseScanConfig) -> String {
    match &config.options.relation {
        relation @ DatabaseRelation::Table { .. } => quoted_relation(config.dialect, relation),
        DatabaseRelation::Query { .. } => config.dialect.quote_ident("_pl"),
    }
}

/// ClickHouse `String`s hold any bytes. ConnectorX silently stops reading at text that isn't
/// valid UTF-8, and clickhouse_connect hex-encodes it: such text fails the read instead, as
/// it does on ADBC; also in arrays (of arrays). The checks are extra select items, which the
/// cast to the schema drops.
fn utf8_checks(config: &DatabaseScanConfig, columns: &[PlSmallStr]) -> Vec<SqlSelectItem> {
    let dialect = config.dialect;
    if dialect != DialectName::Clickhouse || matches!(config.backend.kind, BackendKind::Adbc { .. })
    {
        return vec![];
    }
    // names that are none of the source's columns (nor of the query's, which are the same)
    let mut taken: PlHashSet<PlSmallStr> = config.schema.iter_names().cloned().collect();
    let mut fresh = |i: usize| {
        let mut name = PlSmallStr::from(format!("__pl_utf8_{i}"));
        while taken.contains(&name) {
            name = format!("_{name}").into();
        }
        taken.insert(name.clone());
        name
    };
    columns
        .iter()
        .filter_map(|c| {
            let mut dtype = config.schema.get(c)?;
            let mut nested = false;
            while let DataType::List(inner) = dtype {
                (dtype, nested) = (inner, true);
            }
            (*dtype == DataType::String).then_some((c, nested))
        })
        .enumerate()
        .map(|(i, (c, nested))| {
            let value = match select_item(config, c) {
                SqlSelectItem::Expr { sql, .. } => sql,
                _ => dialect.quote_ident(c),
            };
            let msg = dialect.quote_str(&format!(
                "column '{c}' holds text that is not valid UTF-8; read it as hex({c}) in a query"
            ));
            // not `LowCardinality`, whose functions also run on its dictionary's default
            let valid = |x: &str| format!("ifNull(isValidUTF8(CAST({x} AS Nullable(String))), 1)");
            let valid = if nested {
                format!("arrayAll(pl_s -> {}, arrayFlatten({value}))", valid("pl_s"))
            } else {
                valid(&value)
            };
            SqlSelectItem::Expr {
                sql: format!("throwIf(NOT {valid}, {msg})"),
                alias: fresh(i),
            }
        })
        .collect()
}

/// A scan without pushdown: the relation as given, one partition, everything else in Polars.
fn compile_unwrapped(
    config: &DatabaseScanConfig,
    reason: PlSmallStr,
    total_filters: usize,
) -> PolarsResult<CompiledScan> {
    let dialect = config.dialect;
    let order_by = &config.options.order_by;
    check_connectorx_arrays(
        config,
        &config.schema.iter_names().cloned().collect::<Vec<_>>(),
    )?;
    check_refused_columns(
        config,
        &config.schema.iter_names().cloned().collect::<Vec<_>>(),
    )?;
    let (sql, count_sql) = match &config.options.relation {
        DatabaseRelation::Query { sql } => {
            polars_ensure!(
                order_by.is_empty(),
                InvalidOperation:
                "order_by needs pushdown, which is off for this scan ({reason}); \
                put the ORDER BY in the query instead"
            );
            let sql = strip_statement_end(sql);
            // Columns read through a conversion (Oracle time zones) need the query wrapped.
            let mut items = config
                .schema
                .iter_names()
                .map(|c| select_item(config, c))
                .collect::<Vec<_>>();
            if sql_renderer()
                .and_then(|r| r.inspect_query(sql, dialect))
                .is_ok_and(|f| wrap_blocker(&f, dialect).is_none())
            {
                let names = config.schema.iter_names().cloned().collect::<Vec<_>>();
                items.extend(utf8_checks(config, &names));
            }
            let needs_wrapper = items
                .iter()
                .any(|i| matches!(i, SqlSelectItem::Expr { .. }));
            // A server that drops a derived table's ORDER BY would return the rows in another
            // order: the wrapper repeats the query's ORDER BY outside, on its output columns
            // (the conversions stay: a TIMESTAMP's instant, a FLOAT's digits). Where a key is
            // no output column, the query can't be both converted and ordered.
            let outer_order = if needs_wrapper && config.derived_order_dropped {
                let facts = sql_renderer().and_then(|r| r.inspect_query(sql, dialect));
                // A `SELECT *` has the planned columns; under `schema=` those are the user's, in
                // the user's order, so the database's `ORDER BY 1` isn't theirs first one.
                let names = config.schema.iter_names().cloned().collect::<Vec<_>>();
                let order = sql_renderer()
                    .and_then(|r| r.outer_order_by(sql, dialect, &names, !config.user_schema))
                    .ok()
                    .flatten();
                match facts {
                    Ok(f) if wrap_blocker(&f, dialect).is_none() && order.is_some() => order,
                    _ => polars_bail!(
                        InvalidOperation:
                        "the query's ORDER BY can't be kept: columns of this scan need a \
                        conversion in a wrapping query, whose server drops the ORDER BY of a \
                        derived table, and a key of the ORDER BY is no column the query \
                        selects (or the query can't be wrapped); pass order_by= to \
                        scan_database instead of the query's ORDER BY, sort in Polars, or \
                        CAST the columns in the query"
                    ),
                }
            } else {
                None
            };
            let read = if needs_wrapper {
                // the alias the select items are qualified by (SQLite)
                let mut read = format!(
                    "SELECT {} FROM ({}) {}",
                    escape_percent(config, fixed_select_list(dialect, &items)),
                    wrappable(sql),
                    relation_qualifier(config)
                );
                if let Some(order) = outer_order {
                    read.push_str(&format!(
                        " ORDER BY {}",
                        escape_percent(config, order.to_string())
                    ));
                }
                read
            } else {
                sql.to_string()
            };
            (
                read,
                format!("SELECT COUNT(*) FROM ({}) pl_q", wrappable(sql)),
            )
        },
        relation => {
            let from = quoted_relation(dialect, relation);
            // the select list applies the column conversions, as a pushed-down scan does
            let names = config.schema.iter_names().cloned().collect::<Vec<_>>();
            let mut items = names
                .iter()
                .map(|c| select_item(config, c))
                .collect::<Vec<_>>();
            items.extend(utf8_checks(config, &names));
            let list = if items
                .iter()
                .any(|i| matches!(i, SqlSelectItem::Expr { .. }))
            {
                escape_percent(config, fixed_select_list(dialect, &items))
            } else {
                "*".to_string()
            };
            let mut sql = format!("SELECT {list} FROM {from}");
            if !order_by.is_empty() {
                for c in order_by {
                    polars_ensure!(
                        config.schema.contains(c),
                        ColumnNotFound: "order_by column '{c}' is not in the database source"
                    );
                    polars_ensure!(
                        !sqlite_text_read_as_number(config, c),
                        InvalidOperation:
                        "order_by column '{c}' is a SQLite TEXT column read as a number: \
                        SQLite sorts its values as text ('100' before '50'). Read it as a \
                        string, or declare it INTEGER"
                    );
                }
                // the table's columns, not the select items that convert them
                let cols = order_by
                    .iter()
                    .map(|c| match order_by_expr(config, c) {
                        SqlExpr::Raw(sql) => sql,
                        _ => format!("{from}.{}", dialect.quote_ident(c)),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                sql.push_str(&format!(" ORDER BY {cols}"));
            }
            (sql, format!("SELECT COUNT(*) FROM {from}"))
        },
    };
    Ok(CompiledScan {
        renderer: "none".into(),
        partitions: vec![CompiledPartition { sql, count_sql }],
        all_exact: total_filters == 0,
        columns: config.schema.iter_names().cloned().collect(),
        partitioning: "1".into(),
        pushed_filters: 0,
        filter_columns: vec![],
        exact_filters: 0,
        total_filters,
        limit: None,
        pushdown_disabled: Some(reason),
        repeatable: query_repeatable(config),
    })
}

/// Whether a query returns the same rows, and changes nothing, each time it runs. Where a
/// function called in a `SELECT` may write (Postgres, MySQL), one sqlglot doesn't know may
/// be such a function: then only where the session refuses a write. (SQLite's functions
/// can't write; SQL Server's and ClickHouse's can't in a query; Oracle's only in an
/// autonomous transaction.)
fn repeatable(facts: &QueryFacts, dialect: DialectName, read_only_session: bool) -> bool {
    let functions_write = matches!(
        dialect,
        DialectName::Postgresql | DialectName::Mysql | DialectName::Singlestore
    );
    facts.repeatable && (!facts.side_effects || !functions_write || read_only_session)
}

/// Where a source runs volatile functions its own SQL doesn't show ([`source_volatility`]).
#[derive(Debug, Default)]
struct SourceVolatility {
    /// Why it may return other rows or values each time it runs, if it may.
    volatile: Option<PlSmallStr>,
    /// Its columns whose values may differ each time (a view's `random() AS r`).
    columns: Vec<PlSmallStr>,
    /// Any of its columns' values may differ each time.
    all_columns: bool,
    /// The source is a table the catalog says is no view: its declared types hold.
    plain_table: bool,
    /// A view it reads has a positional join (`PASTE`, `POSITIONAL`): no filter is pushed.
    positional: bool,
}

const UNKNOWN_FUNCTIONS: &str = "the functions or operators it calls, which the database \
    couldn't be asked about (a user's may stand in for a built-in)";

/// The views a scan reads (the table, or those its query reads, and theirs in turn), and
/// the rounds of SQL-defined functions it looks through, at most: one past this is taken to
/// be volatile.
const MAX_VIEW_LOOKUPS: usize = 32;
const MAX_FUNCTION_ROUNDS: usize = 32;

/// Whether reading the source runs a volatile function that its SQL doesn't show: one in
/// a view it reads (the database runs a view's query each time; SQLite and MySQL merge it
/// into the reading query, calling `random()` again for a filter), or one sqlglot doesn't
/// know (`myrand()`) that the catalog says is volatile, or can't tell about (see
/// [`catalog::volatile_functions`]). Each partition would read other rows, and a filter
/// on such a column keep others than it returns.
///
/// `names`: the source's columns. `names_in_order`: they are all of the source's, in its
/// order (found by discovery), and a view's columns are matched to its query's by
/// position (MySQL's definition leaves out a column list); otherwise (`schema=`, the
/// user's names, in any order, maybe some) by name, and a name its query doesn't have
/// may be any of them. A view that can't be inspected (its definition hidden, or not
/// parsed, or its lookup failed) counts as volatile; a dialect without a view lookup,
/// as no view.
fn source_volatility(
    session: &mut dyn DatabaseSession,
    dialect: DialectName,
    relation: &DatabaseRelation,
    facts: Option<&QueryFacts>,
    names: &[PlSmallStr],
    names_in_order: bool,
) -> SourceVolatility {
    let mut out = SourceVolatility::default();
    // the functions the parser knows, which a user's may have replaced
    let mut known: Vec<PlSmallStr> = facts.map(|f| f.known_functions.clone()).unwrap_or_default();
    // Postgres: the types the SQL names, which a user's cast to or from them runs for (and a user's
    // operator on them, with those of the columns of the relations it reads)
    let mut casts: Vec<PlSmallStr> = facts.map(|f| f.cast_types.clone()).unwrap_or_default();
    // (the relation, the qualifier of the view naming it, whether its columns are the
    // source's, whether it is read with `FINAL`)
    let mut pending: Vec<(Vec<PlSmallStr>, Vec<PlSmallStr>, bool, bool)> = match relation {
        DatabaseRelation::Table { parts, .. } => vec![(parts.to_vec(), vec![], true, false)],
        DatabaseRelation::Query { .. } => facts
            .map(|f| {
                f.tables
                    .iter()
                    .map(|t| (t.clone(), vec![], false, f.final_tables.contains(t)))
                    .collect()
            })
            .unwrap_or_default(),
    };
    // every relation looked up, and the names the SQL writes (`None`: every column)
    let mut looked_up: Vec<(Vec<PlSmallStr>, Vec<PlSmallStr>)> = vec![];
    let mut refs: Option<PlHashSet<PlSmallStr>> = Some(PlHashSet::new());
    let union_refs = |refs: &mut Option<PlHashSet<PlSmallStr>>, f: &QueryFacts| match (
        refs.as_mut(),
        &f.referenced_columns,
    ) {
        (Some(set), Some(names)) => set.extend(names.iter().cloned()),
        _ => *refs = None,
    };
    if let Some(f) = facts {
        union_refs(&mut refs, f);
    }
    // Postgres: columns cast to a date or time, whose text may name `now`
    let mut cast_columns: PlHashSet<PlSmallStr> = PlHashSet::new();
    if let Some(f) = facts {
        cast_columns.extend(f.temporal_cast_columns.iter().cloned());
    }
    let mut seen: PlHashSet<(Vec<PlSmallStr>, Vec<PlSmallStr>)> = PlHashSet::new();
    let renderer = sql_renderer().ok();
    // Databricks: the session's SQL path, read once (for the built-ins' place in it and for where a
    // user's function is looked up)
    let mut dbx_path: Option<Option<String>> = None;
    // where rows are merged (`GROUP BY`, `DISTINCT`) and a collation equates different texts, the
    // database returns any one of them
    let mut collapses = facts.is_some_and(|f| f.collapses_rows);
    let mut collated = facts.is_some_and(|f| f.collated);
    // the functions of a query resolve along the session's SQL path, where a user's function may
    // stand ahead of the built-ins (a view's own path is read with its text)
    if dialect == DialectName::Databricks
        && matches!(relation, DatabaseRelation::Query { .. })
        && !known.is_empty()
        && !dbx_path
            .get_or_insert_with(|| catalog::databricks_path(session))
            .as_deref()
            .is_some_and(catalog::databricks_builtins_first)
    {
        out.volatile = Some(
            "the session's SQL path (`SET PATH`), which puts a schema ahead of the \
            built-in functions, so a user's function may stand where the query calls a \
            built-in,"
                .into(),
        );
        out.all_columns = true;
    }
    // Functions defined in SQL (DuckDB macros, ClickHouse functions) are as volatile as
    // their bodies, which may call others, and read views.
    let mut checked: PlHashSet<PlSmallStr> = PlHashSet::new();
    let mut checked_casts: PlHashSet<PlSmallStr> = PlHashSet::new();
    // (the relations a Postgres lookup of operators and casts has seen)
    let mut checked_relations = 0;
    // (an unparsed Postgres query: any user's operator may run, its types unknown)
    let mut any_operator = facts.is_some_and(|f| f.any_operator);
    let mut objects: Vec<String> = vec![];
    // DuckDB macros of operators and built-ins' names run where the query doesn't name
    // them (`a + 1`, `trim(LEADING ..)`): a volatile one makes every source on the
    // connection volatile, however the query is written (see
    // [`catalog::shadowing_macros`]). They are inspected in the first round.
    let mut shadow = if dialect == DialectName::Duckdb {
        match catalog::shadowing_macros(session) {
            Some(macros) => macros,
            None => {
                // can't tell: as for the other lookups that fail
                out.volatile = Some(
                    "a macro the database couldn't list (one of an operator's or a built-in's                     name may be volatile)"
                        .into(),
                );
                out.all_columns = true;
                vec![]
            },
        }
    } else {
        vec![]
    };
    let mut round = 0;
    'inspect: loop {
        while let Some((parts, context, own_columns, final_read)) = pending.pop() {
            // Names are already dialect-normalized or catalog-resolved. Keep their
            // case and resolution context: Postgres v and "V" can be different views.
            let key = (context.clone(), parts.clone());
            if !seen.insert(key) {
                continue;
            }
            if seen.len() > MAX_VIEW_LOOKUPS {
                out.volatile = Some("one of the many views it reads".into());
                out.all_columns = true;
                break 'inspect;
            }
            looked_up.push((parts.clone(), context.clone()));
            let (sql, qualifier) = match catalog::view_definition(
                session, dialect, &parts, &context,
            ) {
                catalog::ViewLookup::View { sql, qualifier } => (sql, qualifier),
                lookup @ (catalog::ViewLookup::Table { .. }
                | catalog::ViewLookup::MergeOnRead { .. }) => {
                    let (generated, merging) = match lookup {
                        catalog::ViewLookup::Table { generated } => (generated, false),
                        catalog::ViewLookup::MergeOnRead { generated } => (generated, true),
                        _ => unreachable!(),
                    };
                    out.plain_table |= own_columns;
                    let name = parts.join(".");
                    if merging && !final_read {
                        out.volatile = Some(
                                format!(
                                    "'{name}', a ClickHouse table whose engine folds rows at \
                                    background merges (ReplacingMergeTree, CollapsingMergeTree, \
                                    SummingMergeTree, AggregatingMergeTree ...), so its rows differ \
                                    before and after a merge unless the query reads it `FINAL`,"
                                )
                                .into(),
                            );
                        out.all_columns = true;
                    }
                    // columns the database computes on each read: proven like a query
                    for (column, expr) in generated {
                        let proven = renderer
                            .as_ref()
                            .and_then(|r| r.inspect_query(&format!("SELECT {expr}"), dialect).ok())
                            .filter(|f| f.parses && f.repeatable);
                        match proven {
                            Some(f) => {
                                known.extend(f.known_functions);
                                casts.extend(f.cast_types);
                                cast_columns.extend(f.temporal_cast_columns);
                                pending.extend(
                                    f.tables.into_iter().map(|t| (t, vec![], false, false)),
                                );
                            },
                            None => {
                                out.volatile = Some(
                                    format!(
                                        "the generated column '{column}' of '{name}', whose \
                                            expression isn't proven to give the same values each \
                                            time,"
                                    )
                                    .into(),
                                );
                                out.columns.push(column.as_str().into());
                            },
                        }
                    }
                    continue;
                },
                catalog::ViewLookup::Ttl => {
                    out.volatile = Some(
                        format!(
                            "'{}', a ClickHouse table with a TTL, which drops rows or \
                            changes values in the background, so its rows differ before and \
                            after, even with `FINAL`,",
                            parts.join(".")
                        )
                        .into(),
                    );
                    out.all_columns = true;
                    continue;
                },
                catalog::ViewLookup::Unknown | catalog::ViewLookup::Failed => {
                    let name = parts.join(".");
                    // (a Databricks temporary view lists no definition; a foreign, external or
                    // system table, an unresolved synonym, a name the catalog doesn't show)
                    out.volatile = Some(
                        format!(
                            "'{name}', which the catalog doesn't show to be an ordinary \
                                table or a view that can be read (a temporary view, a \
                                foreign or external table, a system view, an unresolved name), \
                                so it can't be shown to repeat its rows,"
                        )
                        .into(),
                    );
                    out.all_columns = true;
                    continue;
                },
            };
            let name = parts.join(".");
            let view = renderer
                .as_ref()
                .and_then(|r| r.inspect_view(&sql, dialect).ok())
                .filter(|v| v.parses);
            let Some(view) = view else {
                out.volatile = Some(format!("the view '{name}', which can't be inspected,").into());
                out.all_columns = true;
                continue;
            };
            if !view.repeatable {
                // a volatile function, a `LIMIT`, rows picked in no set order
                out.volatile = Some(format!("the view '{name}'").into());
            }
            collapses |= view.collapses_rows;
            collated |= view.collated;
            out.positional |= view.positional_join || view.unsafe_settings;
            union_refs(&mut refs, &view);
            cast_columns.extend(view.temporal_cast_columns.iter().cloned());
            casts.extend(view.cast_types.iter().cloned());
            let any_volatile = view.all_columns_volatile || !view.volatile_columns.is_empty();
            match &view.output_columns {
                Some(outputs)
                    if own_columns
                        && !view.all_columns_volatile
                        && names_in_order
                        && outputs.len() == names.len() =>
                {
                    for v in &view.volatile_columns {
                        if let Some(i) = outputs.iter().position(|o| o.eq_ignore_ascii_case(v)) {
                            out.columns.push(names[i].clone());
                        }
                    }
                },
                // (not MySQL's: its definition leaves out a column list that may permute the
                // query's own aliases)
                Some(outputs)
                    if own_columns
                        && !view.all_columns_volatile
                        && !names_in_order
                        && dialect != DialectName::Mysql =>
                {
                    for n in names {
                        let volatile = match outputs.iter().find(|o| o.eq_ignore_ascii_case(n)) {
                            Some(o) => view
                                .volatile_columns
                                .iter()
                                .any(|v| v.eq_ignore_ascii_case(o)),
                            // not among the query's names (a MySQL view's column list)
                            None => any_volatile,
                        };
                        if volatile {
                            out.columns.push(n.clone());
                        }
                    }
                },
                // read through a query or another view: any of the source's columns may be one
                _ if any_volatile => out.all_columns = true,
                _ => {},
            }
            known.extend(view.known_functions);
            // its own unqualified names resolve where it was found
            let finals = view.final_tables;
            pending.extend(view.tables.into_iter().map(|t| {
                let f = finals.contains(&t);
                (t, qualifier.clone(), false, f)
            }));
        }
        // built-ins the parser knows, which a user's function may stand in for
        let mut functions: Vec<PlSmallStr> =
            known.drain(..).filter(|n| !checked.contains(n)).collect();
        functions.sort();
        functions.dedup();
        // (Postgres: the operators and casts a user defined, which run for syntax and for the types
        // of the columns, are looked up once at least)
        let postgres = matches!(dialect, DialectName::Postgresql | DialectName::Redshift);
        let cast_types: Vec<PlSmallStr> = casts
            .drain(..)
            .filter(|c| !checked_casts.contains(c))
            .collect();
        if functions.is_empty()
            && shadow.is_empty()
            && cast_types.is_empty()
            && !(postgres && (looked_up.len() > checked_relations || any_operator))
        {
            break;
        }
        if round == MAX_FUNCTION_ROUNDS {
            out.volatile = Some(UNKNOWN_FUNCTIONS.into());
            break;
        }
        round += 1;
        checked.extend(functions.iter().cloned());
        checked_relations = looked_up.len();
        let all_operators = std::mem::take(&mut any_operator);
        checked_casts.extend(cast_types.iter().cloned());
        let path = (dialect == DialectName::Databricks)
            .then(|| {
                dbx_path
                    .get_or_insert_with(|| catalog::databricks_path(session))
                    .clone()
            })
            .flatten();
        let Some(found) = catalog::user_functions(
            session,
            dialect,
            &functions,
            &cast_types,
            &looked_up.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>(),
            all_operators,
            path.as_deref(),
        ) else {
            // the database can't say whether a user's function stands in for a built-in
            out.volatile = Some(UNKNOWN_FUNCTIONS.into());
            break;
        };
        let mut bodies: Vec<(Option<String>, String)> =
            found.bodies.into_iter().map(|b| (None, b)).collect();
        bodies.extend(shadow.drain(..).map(|(n, body)| (Some(n), body)));
        let mut volatile = found.found;
        objects.extend(found.objects.iter().cloned());
        // the shadowing macros whose bodies are volatile
        let mut shadowing: Vec<String> = vec![];
        for (macro_name, body) in &bodies {
            // a table macro's body is a query, a scalar one's an expression
            let first = body
                .trim_start()
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next();
            let sql = match first.map(str::to_ascii_lowercase).as_deref() {
                Some("select" | "with" | "from") => body.clone(),
                _ => format!("SELECT {body}"),
            };
            match renderer
                .as_ref()
                .and_then(|r| r.inspect_query(&sql, dialect).ok())
                .filter(|f| f.parses)
            {
                Some(f) => {
                    volatile |= !f.repeatable;
                    if let (Some(n), false) = (macro_name, f.repeatable) {
                        shadowing.push(n.clone());
                    }
                    known.extend(f.known_functions);
                    casts.extend(f.cast_types);
                    // the views it reads (a view's columns aren't the source's then)
                    pending.extend(f.tables.into_iter().map(|t| (t, vec![], false, false)));
                },
                None => {
                    volatile = true;
                    shadowing.extend(macro_name.clone());
                },
            }
        }
        if volatile {
            out.volatile = Some(
                format!(
                    "{} (a user's function, operator or cast that may \
                    stand where the query calls a built-in, or a macro whose body isn't proven{}{})",
                    if objects.is_empty() {
                        format!(
                            "one of the functions {}",
                            functions
                                .iter()
                                .map(|f| f.to_string())
                                .chain(shadowing.iter().map(|m| format!("\"{m}\"")))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    } else {
                        format!("the user's {}", objects.join(", "))
                    },
                    if dialect == DialectName::Sqlite {
                        "; register a deterministic user function with `deterministic=True`"
                    } else {
                        ""
                    },
                    if shadowing.is_empty() {
                        ""
                    } else {
                        "; a macro of an operator's or a built-in's name, volatile by its body, \
                        runs wherever the query uses it, so every source on this connection \
                        counts as volatile"
                    }
                )
                .into(),
            );
            out.all_columns = true;
            break;
        }
    }
    // a `GROUP BY`, `DISTINCT` or set operation over a text whose collation equates different
    // spellings returns any one of them: proven only where the columns it names have a binary (or
    // deterministic) collation, which the catalog says
    if collapses && out.volatile.is_none() {
        let mut why: Option<String> = collated.then(|| "a COLLATE clause".to_string());
        if why.is_none() {
            for (parts, context) in &looked_up {
                match catalog::nonbinary_collated_columns(session, dialect, parts, context) {
                    Some(cols) => {
                        if let Some(c) = cols.iter().find(|c| {
                            refs.as_ref().is_none_or(|r| r.contains(c.as_str()))
                                || c.as_str() == "*"
                        }) {
                            why = Some(format!("column '{c}' of '{}'", parts.join(".")));
                        }
                    },
                    None => {
                        why = Some(format!(
                            "the collations of '{}', which couldn't be read",
                            parts.join(".")
                        ));
                    },
                }
                if why.is_some() {
                    break;
                }
            }
        }
        if let Some(why) = why {
            out.volatile = Some(
                format!(
                    "a GROUP BY, DISTINCT or set operation over text of a collation that \
                    may equate different texts (case- or accent-insensitive, nondeterministic: \
                    {why}), which returns any one of the spellings that compare equal,"
                )
                .into(),
            );
            out.all_columns = true;
        }
    }
    // a text column cast to a date or time may hold `'now'` (Postgres)
    if !cast_columns.is_empty() && out.volatile.is_none() {
        for (parts, context) in &looked_up {
            match catalog::text_columns(session, dialect, parts, context) {
                Some(cols) => {
                    if let Some(c) = cols.iter().find(|c| cast_columns.contains(c.as_str())) {
                        out.volatile = Some(
                            format!(
                                "the text column '{c}' of '{}' cast to a date or time, whose \
                                values may read as a time relative to now ('now', 'today', ...),",
                                parts.join(".")
                            )
                            .into(),
                        );
                    }
                },
                None => {
                    out.volatile = Some(
                        format!(
                            "the column types of '{}', which couldn't be read (a cast to a \
                            date or time may read a text 'now'),",
                            parts.join(".")
                        )
                        .into(),
                    );
                },
            }
            if out.volatile.is_some() {
                out.all_columns = true;
                break;
            }
        }
    }
    if verbose()
        && let Some(reason) = &out.volatile
    {
        eprintln!("[scan_database]: the source is volatile: {reason}");
    }
    out
}

/// Words of a time of day or a zone (Postgres' zone abbreviations, and its zone names that are one
/// word) that may stand around `now`, `today` ...: `'today EST'`, `'13:00 yesterday'`. The same
/// list as `_sqlglot.py`'s `_TIME_WORDS`.
const TIME_WORDS: &[&str] = &[
    "am",
    "pm",
    "at",
    "allballs",
    "z",
    "zulu",
    "ut",
    "utc",
    "uct",
    "gmt",
    "est",
    "edt",
    "cst",
    "cdt",
    "mst",
    "mdt",
    "pst",
    "pdt",
    "akst",
    "akdt",
    "hst",
    "ast",
    "adt",
    "nst",
    "ndt",
    "bst",
    "cet",
    "cest",
    "eet",
    "eest",
    "wet",
    "west",
    "msk",
    "ist",
    "jst",
    "kst",
    "hkt",
    "pkt",
    "wib",
    "wit",
    "wita",
    "aest",
    "aedt",
    "acst",
    "acdt",
    "awst",
    "nzst",
    "nzdt",
    "sast",
    "cat",
    "eat",
    "wat",
    "sst",
    "chst",
    "universal",
    "japan",
    "singapore",
    "hongkong",
    "israel",
    "cuba",
    "egypt",
    "eire",
    "iceland",
    "iran",
    "jamaica",
    "libya",
    "navajo",
    "poland",
    "portugal",
    "turkey",
    "gb",
    "nz",
    "prc",
    "roc",
    "rok",
];

/// Whether a token (lower case) is a time or a zone: digits with `:`, `.`, `/`, `+`, `-` (and `am`
/// / `pm` after them), a POSIX zone (`pst8pdt`, `utc+5`), a zone path (`europe/paris`), or a word
/// of [`TIME_WORDS`].
fn time_like(token: &str) -> bool {
    if TIME_WORDS.contains(&token) {
        return true;
    }
    let is_time_char = |c: char| c.is_ascii_digit() || matches!(c, ':' | '.' | '/' | '+' | '-');
    // digits and separators, then maybe `am`, `pm`, `a.m.`
    let digits_end = token
        .find(|c: char| !is_time_char(c))
        .unwrap_or(token.len());
    if digits_end > 0 {
        let rest = &token[digits_end..];
        if matches!(rest, "" | "am" | "pm" | "a.m." | "p.m." | "a.m" | "p.m") {
            return true;
        }
    }
    // letters, an optional sign, digits (`pst8pdt`, `utc+5`, `gmt0`)
    let letters = token
        .find(|c: char| !c.is_ascii_lowercase())
        .unwrap_or(token.len());
    if letters > 0 && letters < token.len() {
        let rest = token[letters..].trim_start_matches(['+', '-']);
        if rest.starts_with(|c: char| c.is_ascii_digit())
            && rest.chars().all(|c| c.is_ascii_alphanumeric() || c == ':')
        {
            return true;
        }
    }
    // a zone path: words joined by `/`
    let mut parts = token.split('/');
    parts
        .next()
        .is_some_and(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
        && token.contains('/')
        && parts.all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '-'))
        })
}

/// Whether a text is one a database may read as a time relative to the statement: a word of `now`,
/// `today`, `tomorrow`, `yesterday` (any case) with nothing else but time-like tokens around it, in
/// any order (`'yesterday 13:00'`, `'today EST'`, `'EST today'`, `'13:00 yesterday'`, `'today
/// Europe/Paris'`, `'today allballs'`); not `'Now Corp'`. The same rule as `_sqlglot.py`'s
/// `_relative_time_text`.
fn relative_time_text(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    let mut seen = false;
    for token in lower
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
    {
        if matches!(token, "now" | "today" | "tomorrow" | "yesterday") {
            seen = true;
            continue;
        }
        // `now+1`: a word with an offset
        let word_end = token
            .find(|c: char| !c.is_ascii_lowercase())
            .unwrap_or(token.len());
        let (word, offset) = token.split_at(word_end);
        if matches!(word, "now" | "today" | "tomorrow" | "yesterday")
            && !offset.is_empty()
            && offset
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, ':' | '.' | '/' | '+' | '-'))
        {
            seen = true;
        } else if !time_like(token) {
            return false;
        }
    }
    seen
}

/// Whether a query's bound parameters hold a text naming a time relative to the statement (the
/// databases read it, cast to a timestamp or a date, as the current time, whose value the SQL
/// doesn't show). Judged on the quoted strings of the parameters' `repr`.
fn params_time_relative(connection: &ConnectionSpec) -> bool {
    let Some(key) = connection.query_params_key.as_ref() else {
        return false;
    };
    let mut chars = key.chars();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' {
            let mut text = String::new();
            while let Some(d) = chars.next() {
                if d == '\\' {
                    chars.next();
                } else if d == c {
                    break;
                } else {
                    text.push(d);
                }
            }
            if relative_time_text(&text) {
                return true;
            }
        }
    }
    false
}

/// Whether the source is proven to return the same rows each time it runs. A query that can't be
/// inspected is not.
fn query_repeatable(config: &DatabaseScanConfig) -> bool {
    if config.volatile_source {
        return false;
    }
    match &config.options.relation {
        DatabaseRelation::Table { .. } => true,
        DatabaseRelation::Query { sql } => sql_renderer()
            .and_then(|r| r.inspect_query(sql, config.dialect))
            .is_ok_and(|f| {
                f.parses
                    && !params_time_relative(&config.options.connection)
                    && repeatable(
                        &f,
                        config.dialect,
                        read_only_session(
                            config.dialect,
                            &config.backend.kind,
                            config.options.connection.is_live(),
                        ),
                    )
            }),
    }
}

/// Turns the scan's `PartitionSpec` into partitions, querying bounds where needed.
fn resolve_partitions(
    config: &DatabaseScanConfig,
    renderer: &dyn SqlRenderer,
    from: &SqlFrom,
    filters: &[PushedFilter],
    target_partitions: Option<usize>,
) -> PolarsResult<(Vec<Partition>, PlSmallStr)> {
    let options = &config.options;
    let one = || Ok((vec![Partition { predicate: None }], PlSmallStr::from("1")));
    let max_connections = config.max_connections();

    // A live connection reads partitions one after another on one connection; splitting
    // only adds queries.
    if config.options.connection.is_live() && target_partitions.is_none() {
        if let PartitionSpec::Auto { .. } = options.partitioning {
            return one();
        }
    }

    let (column, n, range, why) = match &options.partitioning {
        PartitionSpec::None => return one(),
        PartitionSpec::Predicates(preds) => {
            let parts = partition::predicate_partitions(preds)?;
            polars_ensure!(
                options.order_by.is_empty() || parts.len() <= 1,
                InvalidOperation:
                "order_by doesn't give a global order across {} partition_predicates; \
                partition on the first order_by column instead",
                parts.len()
            );
            let desc = format!("{} (partition_predicates)", parts.len());
            return Ok((parts, desc.into()));
        },
        spec @ PartitionSpec::Column {
            column,
            num_partitions,
            range,
            ..
        } => {
            polars_ensure!(
                options.order_by.is_empty() || options.order_by[0] == *column,
                InvalidOperation:
                "order_by={:?} needs partitioning on '{}', not on '{column}'",
                options.order_by, options.order_by[0]
            );
            // The user's `partition_num` (e.g. a cap on the database's load) wins over the
            // distributed target.
            let n = num_partitions
                .or(target_partitions)
                .unwrap_or_else(|| partition::target_count(spec, max_connections));
            (column.clone(), n, range.clone(), "partition_on")
        },
        PartitionSpec::Auto { num_partitions } => {
            // A snapshot is one query until partitions can share one: never split
            // automatically, so it doesn't start failing once the table grows. Nor a
            // volatile source (a view of `random()`): each partition would read other rows.
            if matches!(options.relation, DatabaseRelation::Query { .. })
                || options.consistency == Consistency::Snapshot
                || config.volatile_source
            {
                return one();
            }
            // With order_by, a range on its first column also gives a global order.
            let column = match options.order_by.first() {
                Some(c) => Some(c.clone()),
                None => config
                    .columns
                    .iter()
                    .find(|c| c.is_key)
                    .map(|c| c.name.clone()),
            };
            let Some(column) = column else {
                if verbose() {
                    eprintln!(
                        "[scan_database]: one partition: no single-column key; \
                        pass partition_on= to read in parallel"
                    );
                }
                return one();
            };
            let n = match (*num_partitions, target_partitions, config.stats.bytes) {
                (Some(n), _, _) | (None, Some(n), _) => n,
                (None, None, Some(bytes)) if bytes < MIN_PARTITIONED_BYTES => return one(),
                (None, None, Some(bytes)) => {
                    (bytes.div_ceil(BYTES_PER_PARTITION) as usize).clamp(1, 4 * max_connections)
                },
                (None, None, None) => max_connections,
            };
            (column, n, None, "auto")
        },
    };

    // The column and the range are checked whatever the partition count, so a mistake
    // doesn't depend on the machine or the table size.
    let Some(dtype) = config.schema.get(&column) else {
        polars_bail!(ColumnNotFound: "partition column '{column}' is not in the database source");
    };
    polars_ensure!(
        !config.column(&column).is_some_and(|m| m.unstable),
        InvalidOperation:
        "this source can't be partitioned on '{column}': it is computed by an aggregate over \
        floats or a volatile function, whose values differ in their last bits (or more) \
        between runs, so a row near a cut would be read by two partitions or none; partition \
        on another column, or CAST the aggregate's argument to an integer or decimal type"
    );
    // The cuts compare against the column: a decimal read as a float (`schema_overrides`)
    // keeps decimal cuts, which MySQL orders right (#201).
    let declared = config
        .column(&column)
        .and_then(|m| m.db_type.as_deref())
        .and_then(|t| declared_dtype(config.dialect, t));
    let kind_dtype = match &declared {
        Some(d @ DataType::Decimal(..)) if dtype.is_float() => d,
        _ => dtype,
    };
    if sqlite_text_read_as_number(config, &column) {
        if why == "auto" {
            return one();
        }
        polars_bail!(
            InvalidOperation:
            "partition column '{column}' is a SQLite TEXT column read as {dtype}: SQLite \
            compares its values as text ('50' > '100'), so range cuts would overlap. Read it \
            as a string, partition on an INTEGER column, or use partition_predicates"
        );
    }
    let Some(mut kind) = partition::range_kind(kind_dtype) else {
        if why == "auto" {
            return one();
        }
        polars_bail!(
            InvalidOperation:
            "partition column '{column}' has dtype {dtype}; range partitioning needs an integer, decimal, \
            float or date column (use partition_predicates for other types)"
        );
    };

    // ConnectorX's bundled SQLite, and a SQLite built by MSVC (Python on Windows), parse a
    // float literal inexactly (exponents beyond about +-85 miss the stored value): cuts closer
    // than that error can be read out of order, so a row is lost or read twice.
    if matches!(kind, partition::RangeKind::Float)
        && config.dialect == DialectName::Sqlite
        && (cfg!(windows) || matches!(config.backend.kind, BackendKind::ConnectorX))
    {
        if why == "auto" {
            return one();
        }
        polars_bail!(
            InvalidOperation:
            "can't partition '{column}' by ranges: this SQLite reads float literals \
            inexactly, so a range on a float column could lose rows. Use another driver, \
            partition on an integer column, or use partition_predicates"
        );
    }

    if matches!(kind, partition::RangeKind::Int) && config.maybe_mysql_year(&column) {
        kind = partition::RangeKind::Year;
    }

    let range = match range {
        Some(range) => {
            let (lo, hi) = (
                partition::scalar_value(&range.0, kind)?,
                partition::scalar_value(&range.1, kind)?,
            );
            polars_ensure!(
                lo <= hi,
                InvalidOperation: "partition_range must be (low, high), got ({}, {})",
                range.0.value(), range.1.value()
            );
            Some((lo, hi))
        },
        None => None,
    };
    if n <= 1 {
        return one();
    }
    // MariaDB compares an indexed `FLOAT(M, D)` / `DOUBLE(M, D)` column against a cut as the
    // value rounded to D digits where a scan reads it unrounded: a row equal to a cut (stored
    // as 0.00999999977, cut 0.01) is lost to the index range and kept by a table scan, so
    // no ranges can be told to partition one without losing or repeating rows.
    if matches!(kind, partition::RangeKind::Float)
        && config.dialect == DialectName::Mysql
        && config
            .column(&column)
            .and_then(|m| m.db_type.as_deref())
            .is_some_and(declared_float_digits)
    {
        let backend = open_backend(&config.backend, None)?;
        let mut session = open_session_with_retries(
            backend.as_ref(),
            &config.options.connection,
            config.dialect,
        )?;
        if is_mariadb(session.as_mut()) {
            if why == "auto" {
                return one();
            }
            polars_bail!(
                InvalidOperation:
                "can't partition '{column}' by ranges: MariaDB compares a float column \
                declared with digits (FLOAT(M, D)) other than it reads it, so a range there \
                loses rows. Partition on another column, or use partition_predicates"
            );
        }
    }
    // Under `schema=` the planning looked nothing up: automatic partitions of a volatile
    // source (a view of `random()`) are one, as they would be otherwise, not a read that
    // fails (`check_unplanned_volatility`). The bounds query opens a session anyway.
    if why == "auto" && config.volatility_unchecked {
        let backend = open_backend(&config.backend, None)?;
        let mut session = open_session_with_retries(
            backend.as_ref(),
            &config.options.connection,
            config.dialect,
        )?;
        if unplanned_volatility(session.as_mut(), config)
            .volatile
            .is_some()
        {
            if verbose() {
                eprintln!("[scan_database]: one partition: the source is volatile");
            }
            return one();
        }
    }

    let (lo, hi, bounds_from) = match range {
        Some((lo, hi)) => (lo, hi, "partition_range"),
        None => {
            let (lo, hi) = query_bounds(config, renderer, from, &column, filters)?;
            match (lo, hi) {
                (Some(lo), Some(hi)) => (lo, hi, "MIN/MAX"),
                // Empty table, or only NULLs.
                _ => return one(),
            }
        },
    };

    // Databricks reads a date in the Julian-Gregorian gap (1582-10-05..14) back as 1582-10-15 but
    // compares it as written, so a cut at or before that day loses rows: no ranges there (the one
    // query reads them all).
    if config.dialect == DialectName::Databricks
        && matches!(kind, partition::RangeKind::Date)
        && lo <= crate::filter::DATABRICKS_GAP_END_DAYS as f64
    {
        if why == "auto" {
            return one();
        }
        polars_bail!(
            InvalidOperation:
            "can't partition '{column}' by ranges: Databricks compares the dates up to \
            1582-10-15 other than it reads them, so a range there loses rows; \
            pass partition_on=None"
        );
    }
    // A float column whose database type is unknown (`schema=`) may be a MySQL DECIMAL,
    // which MySQL 8 misorders against a float cut (#201): decimal cuts, which a DOUBLE
    // compares as the same doubles, where they fit `DECIMAL(65, 30)`.
    if matches!(kind, partition::RangeKind::Float)
        && matches!(
            config.dialect,
            DialectName::Mysql | DialectName::Singlestore
        )
        && config.column(&column).is_none_or(|c| c.db_type.is_none())
        && lo.abs() < 1e34
        && hi.abs() < 1e34
    {
        kind = partition::RangeKind::Decimal(30);
    }
    let nulls_first = partition::nulls_sort_first(config.dialect);
    let nans_first = partition::nans_sort_first(config.dialect);
    let parts = partition::range_partitions(
        &column,
        lo,
        hi,
        n,
        kind,
        nulls_first,
        nans_first,
        config.dialect,
    );
    let desc = format!(
        "{} (range on {}, bounds from {bounds_from})",
        parts.len(),
        config.dialect.quote_ident(&column)
    );
    Ok((parts, desc.into()))
}

/// `MIN` and `MAX` of a column, over the rows the exact filters keep.
///
/// Spark's `MAX` of a float column holding NaN is NaN (the greatest value), which no range can
/// be cut from: its bounds are those of the other values (NaN rows fall into the last partition,
/// where `>= cut` is true for them).
fn query_bounds(
    config: &DatabaseScanConfig,
    renderer: &dyn SqlRenderer,
    from: &SqlFrom,
    column: &PlSmallStr,
    filters: &[PushedFilter],
) -> PolarsResult<(Option<f64>, Option<f64>)> {
    let bounds = run_bounds(config, renderer, from, column, filters, false)?;
    if config.dialect == DialectName::Databricks
        && (bounds.0.is_some_and(f64::is_nan) || bounds.1.is_some_and(f64::is_nan))
    {
        if verbose() {
            eprintln!(
                "[scan_database]: partition bounds are NaN: bounds again without the NaN \
                rows of '{column}'"
            );
        }
        return run_bounds(config, renderer, from, column, filters, true);
    }
    Ok(bounds)
}

fn run_bounds(
    config: &DatabaseScanConfig,
    renderer: &dyn SqlRenderer,
    from: &SqlFrom,
    column: &PlSmallStr,
    filters: &[PushedFilter],
    without_nan: bool,
) -> PolarsResult<(Option<f64>, Option<f64>)> {
    let mut query = SqlQuery::select(
        vec![
            SqlSelectItem::Min(column.clone()),
            SqlSelectItem::Max(column.clone()),
        ],
        from.clone(),
    );
    query.escape_percent = percent_formatted(config);
    query.filter = filters
        .iter()
        .filter(|f| f.exact)
        .map(|f| f.expr.clone())
        .collect();
    // MySQL zero dates are read as NULL, and no driver reads them as a bound.
    if let Some(zero) = zero_date_check(config, column) {
        query.filter.push(SqlExpr::Not(Box::new(zero)));
    }
    // NaN (and infinity) is above every finite value
    if without_nan {
        query.filter.push(SqlExpr::binary(
            SqlExpr::col(column.clone()),
            SqlBinOp::LtEq,
            SqlExpr::lit(SqlLiteral::Float(f64::MAX)),
        ));
    }
    let sql = renderer.render(&query, config.dialect)?;
    if let Some(bounds) = config.bounds_cache.lock().unwrap().get(&sql) {
        return Ok(*bounds);
    }
    if verbose() {
        eprintln!("[scan_database]: partition bounds: {sql}");
    }
    let backend = open_backend(&config.backend, None)?;
    let mut session =
        open_session_with_retries(backend.as_ref(), &config.options.connection, config.dialect)?;
    let df = session.query(&sql)?;
    polars_ensure!(
        df.width() == 2 && df.height() == 1,
        ComputeError: "unexpected result shape for the partition bounds query: {:?}", df.shape()
    );
    let bounds = (
        partition::bound_value(&df.columns()[0]),
        partition::bound_value(&df.columns()[1]),
    );
    config.bounds_cache.lock().unwrap().insert(sql, bounds);
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    #[test]
    fn declared_float_digits_are_told_from_plain_floats() {
        for t in [
            "float(7,3)",
            "FLOAT(7, 3)",
            "double(20,10)",
            "float(7,3) unsigned",
        ] {
            assert!(super::declared_float_digits(t), "{t}");
        }
        for t in [
            "float",
            "double",
            "float(24)",
            "float unsigned",
            "decimal(7,3)",
            "int(11)",
        ] {
            assert!(!super::declared_float_digits(t), "{t}");
        }
    }

    #[test]
    fn utc_zone_is_decided_by_name() {
        for z in [
            "UTC",
            "utc",
            "+00:00",
            "Etc/UTC",
            "Coordinated Universal Time",
        ] {
            assert!(is_utc_zone(z), "{z}");
        }
        // London's winter (a DST zone), BST, and zones at offset 0 now
        for z in ["GMT", "BST", "Europe/London", "WET", "+01:00", "SYSTEM", ""] {
            assert!(!is_utc_zone(z), "{z}");
        }
    }

    use super::*;
    use crate::database::backend::BatchIter;

    #[test]
    fn keyword_guard() {
        assert_eq!(first_keyword("  -- x\n /* y */ (SELECT 1)"), "SELECT");
        assert!(check_read_only("DELETE FROM t", None, true, DialectName::Generic).is_err());
        assert!(
            check_read_only(
                "/* c */ insert into t values (1)",
                None,
                true,
                DialectName::Generic
            )
            .is_err()
        );
        assert!(check_read_only("SELECT * FROM t", None, false, DialectName::Generic).is_ok());
        let facts = QueryFacts {
            unproven: vec![],
            parses: true,
            read_only: false,
            ..Default::default()
        };
        let cte = "WITH d AS (DELETE FROM t) SELECT 1";
        assert!(check_read_only(cte, Some(&facts), true, DialectName::Generic).is_err());
        // a query that doesn't parse: only plainly a SELECT, unless the session refuses
        // writes itself
        for sql in [
            cte,
            "SELECT * INTO t2 FROM t",
            "SELECT 1; DELETE FROM t",
            "WITH d AS (SELECT 1) SELECT * FROM d",
        ] {
            assert!(
                check_read_only(sql, None, false, DialectName::Generic).is_err(),
                "{sql}"
            );
            assert!(
                check_read_only(sql, None, true, DialectName::Generic).is_ok(),
                "{sql}"
            );
        }
        assert!(
            check_read_only(
                "(SELECT a FROM t USING <);",
                None,
                false,
                DialectName::Generic
            )
            .is_ok()
        );
        // SQL Server runs a statement after a query with no `;`
        let exec = "SELECT a FROM c EXEC('DELETE FROM v') USING <";
        assert!(check_read_only(exec, None, false, DialectName::Mssql).is_err());
        assert!(check_read_only(exec, None, false, DialectName::Generic).is_ok());
        assert!(read_only_session(
            DialectName::Postgresql,
            &BackendKind::Python {
                module: "psycopg".into()
            },
            false
        ));
        assert!(!read_only_session(
            DialectName::Postgresql,
            &BackendKind::ConnectorX,
            false
        ));
        assert!(!read_only_session(
            DialectName::Mysql,
            &BackendKind::Adbc {
                driver: "mysql".into()
            },
            true
        ));
        assert!(!read_only_session(
            DialectName::Sqlite,
            &BackendKind::Adbc {
                driver: "sqlite".into()
            },
            false
        ));
    }

    #[test]
    fn duckdb_nested_values_are_checked() {
        let bad = |t: &str| duckdb_bad_value(t, "\"c\"", 0);
        assert_eq!(bad("INTEGER[]"), None);
        assert_eq!(bad("STRUCT(a INTEGER, b VARCHAR)"), None);
        assert_eq!(
            bad("TIMESTAMP[]").unwrap(),
            "len([1 FOR pl_x0 IN \"c\" IF isinf(pl_x0)]) > 0"
        );
        assert_eq!(
            bad("DATE[][]").unwrap(),
            "len([1 FOR pl_x0 IN \"c\" IF len([1 FOR pl_x1 IN pl_x0 IF isinf(pl_x1)]) > 0]) > 0"
        );
        assert_eq!(
            bad(r#"STRUCT("a ""b," TIME, n INTEGER, m MAP(VARCHAR, UHUGEINT))"#).unwrap(),
            format!(
                "(struct_extract(\"c\", 'a \"b,') = TIME '24:00:00') OR \
                ((len([1 FOR pl_x0 IN map_values(struct_extract(\"c\", 'm')) \
                IF pl_x0 > {MAX_DECIMAL_38}]) > 0))"
            )
        );
        assert!(bad("TIME WITH TIME ZONE").is_some());
        assert!(bad("HUGEINT").is_some());
        assert!(bad("STRUCT(t TIMESTAMP_NS)[2]").is_some());
    }

    #[test]
    fn wrapped_comments_end_with_a_newline() {
        assert_eq!(wrappable("SELECT 1 -- x"), "SELECT 1 -- x\n");
        assert_eq!(wrappable("SELECT 1 # x"), "SELECT 1 # x\n");
        assert_eq!(wrappable("SELECT 1;"), "SELECT 1");
    }

    fn conversion(dialect: DialectName, db_type: &str) -> Option<String> {
        column_conversion(
            dialect,
            &BackendKind::ConnectorX,
            true,
            true,
            db_type,
            "\"c\"",
        )
    }

    fn portable_conversion(db_type: &str) -> Option<String> {
        column_conversion(
            DialectName::Sqlite,
            &BackendKind::ConnectorX,
            true,
            false,
            db_type,
            "\"c\"",
        )
    }

    #[test]
    fn sqlite_override_cut_reads_integer_text() {
        // a same-dtype integer override cut reals, but left `'7'` as text, which SQLite
        // compares above every number
        let cut = conversion(DialectName::Sqlite, SQLITE_INTEGER_CUT).unwrap();
        assert!(cut.contains("typeof(\"c\") = 'real'"), "{cut}");
        assert!(
            cut.contains(
                "WHEN typeof(\"c\") = 'text' AND CAST(CAST(\"c\" AS INTEGER) AS TEXT) = \"c\" \
                THEN CAST(\"c\" AS INTEGER)"
            ),
            "{cut}"
        );
        // other text fails the read, as in a column of no override
        assert!(cut.contains("THEN char(1) || \"c\""), "{cut}");
        // on a connection the user made, with no function to redefine
        let cut = portable_conversion(SQLITE_INTEGER_CUT).unwrap();
        assert!(
            cut.contains(
                "WHEN (+(\"c\") > 9e999 AND +(\"c\") < X'') AND CAST(CAST(\"c\" AS INTEGER) \
                AS BLOB) = CAST(\"c\" AS BLOB) THEN CAST(\"c\" AS INTEGER)"
            ),
            "{cut}"
        );
        assert!(cut.contains("THEN '\u{1}' || \"c\""), "{cut}");
    }

    #[test]
    fn sqlite_conversions_on_a_users_connection_call_no_function() {
        // a live connection or a factory's may redefine any SQL function and collation
        for t in [
            SQLITE_TEXT,
            SQLITE_REAL,
            SQLITE_REAL_CUT,
            SQLITE_INTEGER,
            SQLITE_UNTYPED_INTEGER,
            SQLITE_INTEGER_CUT,
        ] {
            let sql = portable_conversion(t).unwrap();
            for f in ["typeof(", "char(", "substr(", "abs(", "length(", "instr("] {
                assert!(!sql.contains(f), "{t} calls {f}: {sql}");
            }
            assert!(!sql.contains("AS TEXT) ="), "{t} compares text: {sql}");
        }
    }

    #[test]
    fn mysql_dates_of_unknown_type_may_be_zero() {
        // a query's date column (on a driver without the type lookup), or one under
        // `schema=`, was read and compared as stored
        let zero = "IF(MONTH(\"c\") = 0 OR DAYOFMONTH(\"c\") = 0, NULL, \"c\")";
        assert_eq!(
            conversion(DialectName::Mysql, MYSQL_DATE).as_deref(),
            Some(zero)
        );
        assert!(mysql_zero_date_type(DialectName::Mysql, MYSQL_DATE));
        assert!(!mysql_zero_date_type(DialectName::Postgresql, MYSQL_DATE));
    }

    #[test]
    fn duckdb_hugeint_retyped_is_read_as_text() {
        assert_eq!(
            conversion(DialectName::Duckdb, DUCKDB_TEXT).as_deref(),
            Some("CAST(\"c\" AS VARCHAR)")
        );
    }

    /// Fails every statement, as a catalog lookup given the user's `params=` did.
    struct Failing;

    impl DatabaseSession for Failing {
        fn execute(&mut self, _: &str, _: Option<usize>) -> PolarsResult<BatchIter> {
            polars_bail!(ComputeError: "the query has 0 placeholders but 1 parameters were passed")
        }
    }

    #[test]
    fn failed_view_lookup_is_volatile() {
        // a failed lookup counted as no view: the view of `random()` went unchecked
        let relation = DatabaseRelation::Table {
            parts: vec!["v".into()],
            quoted: vec![false],
        };
        let names = ["id".into(), "r".into()];
        let found = source_volatility(
            &mut Failing,
            DialectName::Postgresql,
            &relation,
            None,
            &names,
            true,
        );
        assert!(found.volatile.is_some() && found.all_columns);
        // a dialect without a view lookup is not proven either
        let found = source_volatility(
            &mut Failing,
            DialectName::Trino,
            &relation,
            None,
            &names,
            true,
        );
        assert!(found.volatile.is_some());
    }

    /// Records every statement and answers with no rows.
    struct Recorder(Vec<String>);

    impl DatabaseSession for Recorder {
        fn execute(&mut self, sql: &str, _: Option<usize>) -> PolarsResult<BatchIter> {
            self.0.push(sql.to_string());
            let df = DataFrame::new(0, vec![Column::new_empty("a".into(), &DataType::String)])?;
            Ok(Box::new(std::iter::once(Ok(df))))
        }
    }

    #[test]
    fn databricks_sql_path_is_read_once_per_planning() {
        // the built-ins' place on the path and the lookup of a user's function both need it
        let relation = DatabaseRelation::Query {
            sql: "SELECT upper(x) FROM t".into(),
        };
        let facts = QueryFacts {
            parses: true,
            known_functions: vec!["upper".into()],
            ..Default::default()
        };
        let mut rec = Recorder(vec![]);
        source_volatility(
            &mut rec,
            DialectName::Databricks,
            &relation,
            Some(&facts),
            &[],
            true,
        );
        let reads = rec.0.iter().filter(|s| s.contains("current_path")).count();
        assert_eq!(reads, 1, "{:?}", rec.0);
    }

    #[test]
    fn relative_time_text_is_a_keyword_alone_or_with_a_time() {
        for t in [
            "now",
            " Today ",
            "yesterday 13:00",
            "today allballs",
            "TOMORROW 04:05 pm",
            "now+1",
            "today EST",
            "EST today",
            "13:00 yesterday",
            "04:05 tomorrow",
            "today Europe/Paris",
            "today PST8PDT",
            "Today at 13:00 UTC+5",
        ] {
            assert!(relative_time_text(t), "{t}");
        }
        for t in [
            "Now Corp",
            "nowhere",
            "2024-01-01",
            "tomorrowland",
            "today's special",
            "Now Corp",
            "Today Cafe",
            "",
        ] {
            assert!(!relative_time_text(t), "{t}");
        }
    }
}
