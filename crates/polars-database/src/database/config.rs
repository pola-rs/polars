use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use polars_core::prelude::{DataType, SchemaRef};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_str::PlSmallStr;
use serde::{Deserialize, Serialize};

use super::backend::{BackendChoice, BackendKind};
use super::connection::ConnectionSpec;
use super::dialect::{DialectName, StringSemantics};
use super::options::DatabaseScanOptions;

/// What schema discovery learned about a column.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnMeta {
    pub name: PlSmallStr,
    pub dtype: DataType,
    /// The database type name, where known.
    pub db_type: Option<PlSmallStr>,
    /// String comparison semantics (from the column's collation).
    pub strings: StringSemantics,
    /// Part of a single-column primary key or clustered key.
    pub is_key: bool,
    /// The database compares the column's values the way Polars sees them. False for
    /// columns read through a conversion (`CAST(uuid AS TEXT)`) or retyped by
    /// `schema_overrides`: predicates on them aren't pushed.
    pub comparable: bool,
}

/// Cheap catalog statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableStats {
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
    /// Postgres heap pages (`pg_class.relpages`).
    pub pages: Option<u64>,
}

/// The SQL of one partition, rendered at planning time.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompiledPartition {
    pub sql: String,
    /// `SELECT COUNT(*)` over the same rows, for row counts.
    pub count_sql: String,
}

/// The output of [`compile_scan`](super::compile_scan), one per resolution.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompiledScan {
    /// e.g. `"sqlglot 30.24.0"`.
    pub renderer: PlSmallStr,
    /// Indexed by the scan's source id.
    pub partitions: Vec<CompiledPartition>,
    /// The scan predicate is fully applied by the SQL (`FULL_FILTER`).
    pub all_exact: bool,
    /// Columns the SQL returns, in order. A superset of what the reader emits.
    pub columns: Vec<PlSmallStr>,
    /// For explain: e.g. `"16 (range on \"id\", bounds from MIN/MAX)"`.
    pub partitioning: PlSmallStr,
    pub pushed_filters: usize,
    /// The columns the pushed filters compare.
    #[serde(default)]
    pub filter_columns: Vec<PlSmallStr>,
    pub exact_filters: usize,
    pub total_filters: usize,
    pub limit: Option<u64>,
    /// Why the query could not be wrapped, if pushdown was turned off for it.
    pub pushdown_disabled: Option<PlSmallStr>,
    /// The SQL returns the same rows each time it runs (`QueryFacts::repeatable`; always
    /// for a table). Otherwise a `COUNT(*)` counts other rows than the read returns.
    pub repeatable: bool,
}

/// A database scan, as held by the resolver and its reader builders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatabaseScanConfig {
    pub options: Arc<DatabaseScanOptions>,
    pub dialect: DialectName,
    pub backend: BackendChoice,
    /// The planned schema. Every partition is cast to it.
    pub schema: SchemaRef,
    pub columns: Arc<[ColumnMeta]>,
    pub stats: TableStats,
    /// Set by each resolution ([`compile_scan`](super::compile_scan)).
    pub compiled: Option<Arc<CompiledScan>>,
    /// Per-worker connection budget, set by the distributed planner.
    pub connection_share: Option<usize>,
    /// Stable hash of `options` ([`DatabaseScanOptions::fingerprint`]).
    pub fingerprint: u64,
    /// Why the relation is read as given, without pushdown (`pushdown=False`, or a query that
    /// can't be wrapped).
    pub pushdown_disabled: Option<PlSmallStr>,
    /// The source runs a volatile function its SQL doesn't show (in a view it reads, or a
    /// function sqlglot doesn't know): it may return other rows each time it runs.
    #[serde(default)]
    pub volatile_source: bool,
    /// Planned under `schema=`, without I/O: the views the source reads and the functions
    /// sqlglot doesn't know weren't inspected. The read does that on its first session
    /// ([`check_unplanned_volatility`](super::planner::check_unplanned_volatility)).
    #[serde(default)]
    pub volatility_unchecked: bool,
    /// Partition bounds already queried, by the bounds SQL: optimizing the plan again
    /// (another `collect`, `explain`) doesn't query them again. The outer partitions are
    /// open-ended, so stale bounds only skew partition sizes, never lose rows.
    #[serde(skip)]
    pub bounds_cache: BoundsCache,
}

/// [`DatabaseScanConfig::session_setup_runs`], before the config exists.
pub(crate) fn session_setup_runs(connection: &ConnectionSpec, kind: &BackendKind) -> bool {
    !connection.is_live() && kind.has_sessions()
}

pub type BoundsCache = Arc<Mutex<PlHashMap<String, (Option<f64>, Option<f64>)>>>;

impl DatabaseScanConfig {
    pub fn column(&self, name: &str) -> Option<&ColumnMeta> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Whether a column may be a MySQL `YEAR`, which reads an integer 1-99 compared with it
    /// as a year (2001-2069 and 1970-1999); 0 and every other integer compare as themselves.
    /// An integer column whose type is unknown (`schema=`) may be one.
    pub fn maybe_mysql_year(&self, column: &str) -> bool {
        matches!(self.dialect, DialectName::Mysql | DialectName::Singlestore)
            && self
                .column(column)
                .is_some_and(|c| match c.db_type.as_deref() {
                    Some(t) => {
                        let t = t.trim().to_ascii_lowercase();
                        t == "year" || t.starts_with("year(")
                    },
                    None => c.dtype.is_integer(),
                })
    }

    /// Concurrent connections for a scan on this machine.
    pub fn max_connections(&self) -> usize {
        if self.options.connection.is_live() {
            return 1;
        }
        self.connection_share
            .or(self.options.fetch.max_connections)
            .unwrap_or(self.dialect.default_max_connections())
            .max(1)
    }

    /// Key-value lines for `explain()`. Never contains secrets.
    pub fn explain_properties(&self) -> Vec<(&'static str, String)> {
        let mut out = vec![("database", format!("{} via {}", self.dialect, self.backend))];
        out.push((
            "relation",
            match &self.options.relation {
                super::options::DatabaseRelation::Table { parts, .. } => parts
                    .iter()
                    .map(|p| self.dialect.quote_ident(p))
                    .collect::<Vec<_>>()
                    .join("."),
                super::options::DatabaseRelation::Query { .. } => "<query>".to_string(),
            },
        ));
        let Some(c) = &self.compiled else {
            out.push(("sql", "<rendered by the optimizer>".into()));
            return out;
        };
        out.push(("partitions", c.partitioning.to_string()));
        match &c.pushdown_disabled {
            Some(reason) => out.push(("pushdown", format!("off ({reason})"))),
            None => out.push((
                "pushed",
                format!(
                    "projection {}/{}, filters {}/{} ({} exact), limit: {}",
                    c.columns.len(),
                    self.schema.len(),
                    c.pushed_filters,
                    c.total_filters,
                    c.exact_filters,
                    c.limit.map_or_else(|| "no".to_string(), |l| l.to_string()),
                ),
            )),
        }
        out.push(("renderer", c.renderer.to_string()));
        if let Some(p) = c.partitions.first() {
            out.push(("sql[0]", p.sql.clone()));
        }
        out
    }

    /// Whether every session runs the dialect's session setup (`open_session`): not on a live
    /// connection, which is the user's session, nor on ConnectorX or arrow-odbc, which have no
    /// sessions.
    pub fn session_setup_runs(&self) -> bool {
        session_setup_runs(&self.options.connection, &self.backend.kind)
    }
}

impl PartialEq for DatabaseScanConfig {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
            && self.dialect == other.dialect
            && self.backend == other.backend
            && self.compiled == other.compiled
            && self.connection_share == other.connection_share
    }
}

impl Eq for DatabaseScanConfig {}

impl Hash for DatabaseScanConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
        self.dialect.hash(state);
        self.compiled.hash(state);
    }
}
