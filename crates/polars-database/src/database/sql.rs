//! Dialect-neutral SQL owned by Polars, and the interface that renders it.

use std::sync::{Arc, LazyLock, RwLock};

use polars_core::prelude::{AnyValue, DataType, TimeUnit};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_str::PlSmallStr;
use serde::{Deserialize, Serialize};

use super::dialect::DialectName;

/// A literal together with the SQL type it should be written as.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SqlLiteral {
    Null,
    Bool(bool),
    Int(i128),
    /// Finite floats only. NaN and infinities are never pushed.
    Float(f64),
    /// Exact decimal text, e.g. `"12.50"`.
    Decimal(String),
    String(String),
    /// `YYYY-MM-DD`
    Date(String),
    /// `YYYY-MM-DD HH:MM:SS[.ffffff]`, in UTC when `with_tz` is set.
    Timestamp {
        text: String,
        with_tz: bool,
    },
    /// `HH:MM:SS[.ffffff]`
    Time(String),
}

impl SqlLiteral {
    /// The literal for a Polars value compared against a column of `dtype`. `None` if the value
    /// can't be written exactly: NaN and infinities, sub-microsecond times, and values that
    /// don't fit the column type.
    pub fn from_any_value(av: &AnyValue<'_>, dtype: &DataType) -> Option<Self> {
        use polars_arrow::temporal_conversions as tc;

        if av.is_null() {
            return Some(Self::Null);
        }
        // `AnyValue::dtype` panics on categorical values
        if matches!(
            av,
            AnyValue::Categorical(..)
                | AnyValue::CategoricalOwned(..)
                | AnyValue::Enum(..)
                | AnyValue::EnumOwned(..)
        ) {
            return None;
        }
        if dtype.is_categorical() || dtype.is_enum() {
            return None;
        }
        let av_dtype = av.dtype();
        let av = if av_dtype == *dtype {
            av.clone()
        } else {
            // only lossless casts: a rounded literal (e.g. `1.501` written at a column's scale
            // 2) would change the comparison
            let cast = av.strict_cast(dtype)?;
            if cast.strict_cast(&av_dtype).as_ref() != Some(av) {
                return None;
            }
            cast
        };
        Some(match av {
            AnyValue::Boolean(b) => Self::Bool(b),
            AnyValue::Float32(f) => Self::float(f as f64)?,
            AnyValue::Float64(f) => Self::float(f)?,
            av if av.dtype().is_integer() => Self::Int(av.extract::<i128>()?),
            AnyValue::String(v) => Self::String(v.to_string()),
            AnyValue::StringOwned(v) => Self::String(v.to_string()),
            AnyValue::Date(days) => Self::Date(tc::date32_to_date_opt(days)?.to_string()),
            AnyValue::Datetime(v, tu, tz) => Self::Timestamp {
                text: timestamp_text(v, tu)?,
                with_tz: tz.is_some(),
            },
            AnyValue::DatetimeOwned(v, tu, tz) => Self::Timestamp {
                text: timestamp_text(v, tu)?,
                with_tz: tz.is_some(),
            },
            AnyValue::Time(ns) => {
                if ns % 1000 != 0 {
                    return None;
                }
                Self::Time(
                    tc::time64ns_to_time_opt(ns)?
                        .format("%H:%M:%S%.6f")
                        .to_string(),
                )
            },
            AnyValue::Decimal(v, _, scale) => Self::Decimal(decimal_text(v, scale)),
            _ => return None,
        })
    }

    fn float(f: f64) -> Option<Self> {
        f.is_finite().then_some(Self::Float(f))
    }

    /// Strings and text-like literals, for collation checks.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

/// `YYYY-MM-DD HH:MM:SS.ffffff`, in UTC for tz-aware values. `None` for sub-microsecond values.
fn timestamp_text(v: i64, tu: TimeUnit) -> Option<String> {
    use polars_arrow::temporal_conversions as tc;

    let dt = match tu {
        TimeUnit::Nanoseconds => {
            if v % 1000 != 0 {
                return None;
            }
            tc::timestamp_ns_to_datetime_opt(v)?
        },
        TimeUnit::Microseconds => tc::timestamp_us_to_datetime_opt(v)?,
        TimeUnit::Milliseconds => tc::timestamp_ms_to_datetime_opt(v)?,
    };
    Some(dt.format("%Y-%m-%d %H:%M:%S%.6f").to_string())
}

fn decimal_text(v: i128, scale: usize) -> String {
    let digits = v.unsigned_abs().to_string();
    let sign = if v < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let digits = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = digits.split_at(digits.len() - scale);
    format!("{sign}{int}.{frac}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqlBinOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

impl SqlBinOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::NotEq => "<>",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::And => "AND",
            Self::Or => "OR",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SqlExpr {
    Column(PlSmallStr),
    Literal(SqlLiteral),
    Binary(Box<SqlExpr>, SqlBinOp, Box<SqlExpr>),
    Not(Box<SqlExpr>),
    /// `expr IS NULL` (`false`) or `expr IS NOT NULL` (`true`).
    IsNull(Box<SqlExpr>, bool),
    /// `expr [NOT] IN (…)`
    InList(Box<SqlExpr>, Vec<SqlLiteral>, bool),
    /// `expr BETWEEN low AND high`
    Between(Box<SqlExpr>, Box<SqlExpr>, Box<SqlExpr>),
    /// `expr LIKE pattern ESCAPE escape`. The pattern is already escaped.
    Like {
        expr: Box<SqlExpr>,
        pattern: String,
        escape: char,
    },
    /// `a IS [NOT] DISTINCT FROM b` (`true` = NOT DISTINCT, i.e. null-safe equality).
    IsNotDistinctFrom(Box<SqlExpr>, Box<SqlExpr>, bool),
    /// A user-written SQL fragment (partition predicates). Inserted as-is, parenthesised.
    Raw(String),
}

impl SqlExpr {
    pub fn col(name: impl Into<PlSmallStr>) -> Self {
        Self::Column(name.into())
    }

    pub fn lit(v: SqlLiteral) -> Self {
        Self::Literal(v)
    }

    pub fn binary(l: SqlExpr, op: SqlBinOp, r: SqlExpr) -> Self {
        Self::Binary(Box::new(l), op, Box::new(r))
    }

    pub fn and_all(mut exprs: Vec<SqlExpr>) -> Option<Self> {
        let first = exprs.pop()?;
        Some(
            exprs
                .into_iter()
                .rev()
                .fold(first, |acc, e| Self::binary(e, SqlBinOp::And, acc)),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SqlSelectItem {
    Column(PlSmallStr),
    /// `*`
    Star,
    /// `COUNT(*)`
    CountStar,
    /// `1` (for scans that need rows but no columns)
    One,
    /// `MIN(col)`
    Min(PlSmallStr),
    /// `MAX(col)`
    Max(PlSmallStr),
    /// `<sql> AS "alias"`: fixed SQL from the dialect rules (never user input).
    Expr {
        sql: String,
        alias: PlSmallStr,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SqlFrom {
    /// `[catalog.][schema.]name`, unquoted parts.
    Table(Vec<PlSmallStr>),
    /// A user query, wrapped as a derived table.
    Subquery(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SqlOrderBy {
    pub expr: SqlExpr,
    pub descending: bool,
}

/// A dialect-neutral query: only what scan pushdown produces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SqlQuery {
    pub columns: Vec<SqlSelectItem>,
    pub from: SqlFrom,
    /// ANDed conjuncts.
    pub filter: Vec<SqlExpr>,
    pub order_by: Vec<SqlOrderBy>,
    pub limit: Option<u64>,
    /// Columns that `filter` and `order_by` reference through the relation (`"t"."c"`,
    /// `"_pl"."c"`): a select item reads them through a conversion under the same name,
    /// which `ORDER BY` (and ClickHouse's `WHERE`) resolves a bare name to.
    pub qualified: Vec<PlSmallStr>,
}

impl SqlQuery {
    pub fn select(columns: Vec<SqlSelectItem>, from: SqlFrom) -> Self {
        Self {
            columns,
            from,
            filter: vec![],
            order_by: vec![],
            limit: None,
            qualified: vec![],
        }
    }
}

/// Facts about a user query, for query-mode wrapping.
#[derive(Clone, Debug, Default)]
pub struct QueryFacts {
    pub parses: bool,
    pub single_select: bool,
    pub has_cte: bool,
    pub top_level_order_by_without_limit: bool,
    /// `None` if it can't be known without the database (e.g. `SELECT *`).
    pub output_columns: Option<Vec<PlSmallStr>>,
    /// One query that only reads: no DML or DDL anywhere (CTEs included), no `SELECT INTO`.
    pub read_only: bool,
    /// Returns the same rows each time it runs on the same data: no `LIMIT` / `OFFSET` /
    /// `FETCH` / `TOP` (which pick rows), no `TABLESAMPLE`, no random functions.
    pub repeatable: bool,
    /// A `LIMIT` / `OFFSET` / `FETCH` (or a lock clause) ends the query: it can't take
    /// another `LIMIT`.
    pub top_level_limit: bool,
    /// A CTE, a query inside the query (not just parenthesised or in a `UNION`), or a
    /// parenthesised one with a `LIMIT` of its own.
    pub nested_queries: bool,
    /// Running it may change something: it calls a volatile function (`nextval`), or one
    /// the parser doesn't know (a user's function may write).
    pub side_effects: bool,
    /// Such a function is in a set operation, a query without `FROM`, or a condition,
    /// where MariaDB runs it under a `LIMIT 0`.
    pub side_effects_outside_rows: bool,
    /// Output columns whose values come from volatile functions (`random() AS r`).
    pub volatile_columns: Vec<PlSmallStr>,
    /// A volatile function's values may reach any output column (through a subquery, a
    /// CTE or `*`).
    pub all_columns_volatile: bool,
    /// Lower-cased names of the functions the parser doesn't know: the database says
    /// whether they are volatile ([`volatile_functions`](super::catalog::volatile_functions)).
    pub unknown_functions: Vec<PlSmallStr>,
    /// Lower-cased names of the functions it knows, on a database where a user's function
    /// may replace one (SQLite, DuckDB): the database says whether one did.
    pub known_functions: Vec<PlSmallStr>,
    /// The tables (or views) it reads, `[catalog.][schema.]name` as written; not its CTEs.
    pub tables: Vec<Vec<PlSmallStr>>,
    /// The query without the semicolons and comments after its last token, if it parses.
    pub body: Option<PlSmallStr>,
}

/// Turns [`SqlQuery`] into SQL text for a dialect.
pub trait SqlRenderer: Send + Sync {
    /// e.g. `"sqlglot 30.24.0"`; recorded in the plan and shown by explain().
    fn id(&self) -> PlSmallStr;
    fn render(&self, query: &SqlQuery, dialect: DialectName) -> PolarsResult<String>;
    fn inspect_query(&self, sql: &str, dialect: DialectName) -> PolarsResult<QueryFacts>;
    /// The facts of a view's query, from its definition (its `CREATE VIEW` statement, or
    /// only its query): the output columns are the view's.
    fn inspect_view(&self, sql: &str, dialect: DialectName) -> PolarsResult<QueryFacts>;
}

type RendererFactory = Arc<dyn Fn() -> PolarsResult<Arc<dyn SqlRenderer>> + Send + Sync>;

static RENDERERS: LazyLock<RwLock<PlHashMap<PlSmallStr, RendererFactory>>> =
    LazyLock::new(Default::default);

/// Registers a renderer under a name, e.g. `"sqlglot"`.
pub fn register_sql_renderer(
    name: &str,
    factory: impl Fn() -> PolarsResult<Arc<dyn SqlRenderer>> + Send + Sync + 'static,
) {
    RENDERERS
        .write()
        .unwrap()
        .insert(name.into(), Arc::new(factory));
}

pub const DEFAULT_SQL_RENDERER: &str = "sqlglot";

/// The session's renderer: `POLARS_DATABASE_SQL_RENDERER`, or [`DEFAULT_SQL_RENDERER`].
pub fn sql_renderer() -> PolarsResult<Arc<dyn SqlRenderer>> {
    let name = std::env::var("POLARS_DATABASE_SQL_RENDERER")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_SQL_RENDERER.to_string());
    let factory = RENDERERS.read().unwrap().get(name.as_str()).cloned();
    match factory {
        Some(f) => f(),
        None => {
            let available = RENDERERS
                .read()
                .unwrap()
                .keys()
                .map(|k| k.to_string())
                .collect::<Vec<_>>();
            polars_bail!(
                ComputeError:
                "database SQL renderer '{name}' is not available (registered: {available:?})"
            )
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals() {
        assert_eq!(decimal_text(1250, 2), "12.50");
        assert_eq!(decimal_text(-5, 3), "-0.005");
        assert_eq!(decimal_text(7, 0), "7");
        assert_eq!(
            SqlLiteral::from_any_value(&AnyValue::Int64(300), &DataType::Int8),
            None
        );
        assert_eq!(
            SqlLiteral::from_any_value(&AnyValue::Int64(3), &DataType::Int8),
            Some(SqlLiteral::Int(3))
        );
        assert_eq!(
            SqlLiteral::from_any_value(&AnyValue::Float64(f64::NAN), &DataType::Float64),
            None
        );
        assert_eq!(
            SqlLiteral::from_any_value(
                &AnyValue::Datetime(1_500, TimeUnit::Nanoseconds, None),
                &DataType::Datetime(TimeUnit::Nanoseconds, None)
            ),
            None
        );
        assert_eq!(
            SqlLiteral::from_any_value(
                &AnyValue::Datetime(1_000_000, TimeUnit::Microseconds, None),
                &DataType::Datetime(TimeUnit::Microseconds, None)
            ),
            Some(SqlLiteral::Timestamp {
                text: "1970-01-01 00:00:01.000000".into(),
                with_tz: false
            })
        );
    }
}
