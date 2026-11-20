//! Pushed filters: filter documents (`filter_expr.v1`, see
//! `polars_io_ext_ffi::lazyframe_resolver_v1`) → dialect-neutral SQL.
//!
//! A pushed filter may make the database return more rows than Polars keeps, never fewer. Each
//! translated filter is classed as exact or superset; anything else is not pushed. Only exact
//! filters are reported as applied: Polars applies the others again.
use polars_core::prelude::*;
use polars_core::scalar::Scalar;
use serde_json::Value;

use crate::database::DatabaseScanConfig;
use crate::database::backend::BackendKind;
use crate::database::dialect::{DialectName, StringSemantics};
use crate::database::sql::{SqlBinOp, SqlExpr, SqlLiteral};

/// A comparison operator of the filter document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    /// `eq_missing`: NULL equals NULL, never NULL itself.
    EqMissing,
    /// `ne_missing`.
    NotEqMissing,
}

impl CmpOp {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "eq" => Self::Eq,
            "neq" => Self::NotEq,
            "lt" => Self::Lt,
            "lt_eq" => Self::LtEq,
            "gt" => Self::Gt,
            "gt_eq" => Self::GtEq,
            "eq_missing" => Self::EqMissing,
            "neq_missing" => Self::NotEqMissing,
            _ => return None,
        })
    }

    /// The operator with its operands swapped (`a < b` is `b > a`).
    fn swap_operands(self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::LtEq => Self::GtEq,
            Self::Gt => Self::Lt,
            Self::GtEq => Self::LtEq,
            op => op,
        }
    }

    fn is_ordering(self) -> bool {
        matches!(self, Self::Lt | Self::LtEq | Self::Gt | Self::GtEq)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Closed {
    Both,
    Left,
    Right,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrOp {
    StartsWith,
    EndsWith,
    ContainsLiteral,
}

/// A parsed filter document.
#[derive(Clone, Debug)]
pub enum Node {
    Column(PlSmallStr),
    Literal(Scalar),
    And(Box<Node>, Box<Node>),
    Or(Box<Node>, Box<Node>),
    Compare(Box<Node>, CmpOp, Box<Node>),
    Not(Box<Node>),
    /// `(expr, negated)`: `is_null`, or `is_not_null` when negated.
    IsNull(Box<Node>, bool),
    IsBetween(Box<Node>, Box<Node>, Box<Node>, Closed),
    IsIn {
        expr: Box<Node>,
        values: Series,
        nulls_equal: bool,
    },
    Str(StrOp, Box<Node>, Box<Node>),
    Cast(Box<Node>, DataType),
    /// Anything the document doesn't describe, or a malformed node.
    Unknown,
}

impl Node {
    /// The columns it names.
    pub fn columns(&self, out: &mut Vec<PlSmallStr>) {
        match self {
            Self::Column(c) => out.push(c.clone()),
            Self::Literal(_) | Self::Unknown => {},
            Self::And(a, b) | Self::Or(a, b) | Self::Compare(a, _, b) | Self::Str(_, a, b) => {
                a.columns(out);
                b.columns(out);
            },
            Self::Not(a) | Self::IsNull(a, _) | Self::Cast(a, _) | Self::IsIn { expr: a, .. } => {
                a.columns(out)
            },
            Self::IsBetween(a, b, c, _) => {
                a.columns(out);
                b.columns(out);
                c.columns(out);
            },
        }
    }

    /// Whether a part of it isn't described ([`Node::Unknown`]): its columns aren't known.
    pub fn has_unknown(&self) -> bool {
        match self {
            Self::Unknown => true,
            Self::Column(_) | Self::Literal(_) => false,
            Self::And(a, b) | Self::Or(a, b) | Self::Compare(a, _, b) | Self::Str(_, a, b) => {
                a.has_unknown() || b.has_unknown()
            },
            Self::Not(a) | Self::IsNull(a, _) | Self::Cast(a, _) | Self::IsIn { expr: a, .. } => {
                a.has_unknown()
            },
            Self::IsBetween(a, b, c, _) => a.has_unknown() || b.has_unknown() || c.has_unknown(),
        }
    }

    /// Parses a document. Parts that don't parse become [`Node::Unknown`], which is never pushed.
    ///
    /// `serde_json` stops at 128 levels of nesting, which a left-deep chain of 63 `or`s (two
    /// levels each) reaches. The limit is lifted, and replaced by [`MAX_DOC_DEPTH`], checked
    /// before parsing: the parser, the tree's recursive drop and the translation each need
    /// stack in proportion to the depth.
    pub fn parse(doc: &str) -> Self {
        if !within_depth(doc, MAX_DOC_DEPTH) {
            return Self::Unknown;
        }
        let mut de = serde_json::Deserializer::from_str(doc);
        de.disable_recursion_limit();
        let de = serde_stacker::Deserializer::new(&mut de);
        match <Value as serde::Deserialize>::deserialize(de) {
            Ok(v) => Self::from_value(&v),
            Err(_) => Self::Unknown,
        }
    }

    fn from_value(v: &Value) -> Self {
        Self::try_from_value(v).unwrap_or(Self::Unknown)
    }

    fn try_from_value(v: &Value) -> Option<Self> {
        let obj = v.as_object()?;
        if let Some(name) = obj.get("col") {
            return Some(Self::Column(name.as_str()?.into()));
        }
        if let Some(lit) = obj.get("lit") {
            let lit = lit.as_object()?;
            let dtype = parse_dtype(lit.get("dtype")?.as_str()?)?;
            let av = parse_value(&dtype, lit.get("value")?)?;
            return Some(Self::Literal(Scalar::new(dtype, av)));
        }
        let op = obj.get("op")?.as_str()?;
        let args: Vec<Node> = obj
            .get("args")?
            .as_array()?
            .iter()
            .map(Self::from_value)
            .collect();
        let mut args = args.into_iter();
        let mut arg = || args.next().map(Box::new);
        Some(match op {
            "and" => Self::And(arg()?, arg()?),
            "or" => Self::Or(arg()?, arg()?),
            "not" => Self::Not(arg()?),
            "is_null" => Self::IsNull(arg()?, false),
            "is_not_null" => Self::IsNull(arg()?, true),
            "is_between" => {
                let closed = match obj.get("closed")?.as_str()? {
                    "both" => Closed::Both,
                    "left" => Closed::Left,
                    "right" => Closed::Right,
                    "none" => Closed::None,
                    _ => return None,
                };
                Self::IsBetween(arg()?, arg()?, arg()?, closed)
            },
            "is_in" => {
                let values = obj.get("values")?.as_object()?;
                let dtype = parse_dtype(values.get("dtype")?.as_str()?)?;
                let avs = values
                    .get("values")?
                    .as_array()?
                    .iter()
                    .map(|v| parse_value(&dtype, v))
                    .collect::<Option<Vec<_>>>()?;
                let values =
                    Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &avs, &dtype, true)
                        .ok()?;
                Self::IsIn {
                    expr: arg()?,
                    values,
                    nulls_equal: obj.get("nulls_equal")?.as_bool()?,
                }
            },
            "starts_with" => Self::Str(StrOp::StartsWith, arg()?, arg()?),
            "ends_with" => Self::Str(StrOp::EndsWith, arg()?, arg()?),
            "contains_literal" => Self::Str(StrOp::ContainsLiteral, arg()?, arg()?),
            "cast" => Self::Cast(arg()?, parse_dtype(obj.get("dtype")?.as_str()?)?),
            op => Self::Compare(arg()?, CmpOp::parse(op)?, arg()?),
        })
    }
}

/// The deepest nesting of a filter document that is parsed: an `or` chain of about 500 terms
/// (an operator is two levels: its object and its `args`).
const MAX_DOC_DEPTH: usize = 1024;

/// Whether the JSON text nests at most `max` levels of objects and arrays.
fn within_depth(doc: &str, max: usize) -> bool {
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    for b in doc.bytes() {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {},
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > max {
                    return false;
                }
            },
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {},
        }
    }
    true
}

fn parse_time_unit(s: &str) -> Option<TimeUnit> {
    Some(match s.trim() {
        "ns" => TimeUnit::Nanoseconds,
        "us" => TimeUnit::Microseconds,
        "ms" => TimeUnit::Milliseconds,
        _ => return None,
    })
}

/// A dtype of the filter document.
pub fn parse_dtype(s: &str) -> Option<DataType> {
    Some(match s {
        "null" => DataType::Null,
        "bool" => DataType::Boolean,
        "str" => DataType::String,
        "i8" => DataType::Int8,
        "i16" => DataType::Int16,
        "i32" => DataType::Int32,
        "i64" => DataType::Int64,
        "i128" => DataType::Int128,
        "u8" => DataType::UInt8,
        "u16" => DataType::UInt16,
        "u32" => DataType::UInt32,
        "u64" => DataType::UInt64,
        "u128" => DataType::UInt128,
        "f32" => DataType::Float32,
        "f64" => DataType::Float64,
        "date" => DataType::Date,
        "time" => DataType::Time,
        s => {
            let (head, rest) = s.split_once('[')?;
            let args = rest.strip_suffix(']')?;
            match head {
                "datetime" => match args.split_once(',') {
                    Some((unit, tz)) => DataType::Datetime(
                        parse_time_unit(unit)?,
                        TimeZone::opt_try_new(Some(tz.trim())).ok()?,
                    ),
                    None => DataType::Datetime(parse_time_unit(args)?, None),
                },
                "duration" => DataType::Duration(parse_time_unit(args)?),
                "decimal" => {
                    let (p, s) = args.split_once(',')?;
                    DataType::Decimal(p.trim().parse().ok()?, s.trim().parse().ok()?)
                },
                _ => return None,
            }
        },
    })
}

/// A value of `dtype` in the filter document. `None` if it doesn't parse.
pub fn parse_value(dtype: &DataType, v: &Value) -> Option<AnyValue<'static>> {
    if v.is_null() {
        return Some(AnyValue::Null);
    }
    let text = || v.as_str();
    Some(match dtype {
        DataType::Null => AnyValue::Null,
        DataType::Boolean => AnyValue::Boolean(v.as_bool()?),
        DataType::String => AnyValue::StringOwned(text()?.into()),
        DataType::Int8 => AnyValue::Int8(text()?.parse().ok()?),
        DataType::Int16 => AnyValue::Int16(text()?.parse().ok()?),
        DataType::Int32 => AnyValue::Int32(text()?.parse().ok()?),
        DataType::Int64 => AnyValue::Int64(text()?.parse().ok()?),
        DataType::Int128 => AnyValue::Int128(text()?.parse().ok()?),
        DataType::UInt8 => AnyValue::UInt8(text()?.parse().ok()?),
        DataType::UInt16 => AnyValue::UInt16(text()?.parse().ok()?),
        DataType::UInt32 => AnyValue::UInt32(text()?.parse().ok()?),
        DataType::UInt64 => AnyValue::UInt64(text()?.parse().ok()?),
        DataType::UInt128 => AnyValue::UInt128(text()?.parse().ok()?),
        DataType::Float32 => AnyValue::Float32(text()?.parse().ok()?),
        DataType::Float64 => AnyValue::Float64(text()?.parse().ok()?),
        DataType::Date => AnyValue::Date(i32::try_from(v.as_i64()?).ok()?),
        DataType::Time => AnyValue::Time(text()?.parse().ok()?),
        DataType::Datetime(tu, tz) => AnyValue::DatetimeOwned(
            text()?.parse().ok()?,
            *tu,
            tz.clone().map(std::sync::Arc::new),
        ),
        DataType::Duration(tu) => AnyValue::Duration(text()?.parse().ok()?, *tu),
        DataType::Decimal(p, s) => {
            let v = text()?.parse().ok()?;
            if !decimal_fits(v, *p) {
                return None;
            }
            AnyValue::Decimal(v, *p, *s)
        },
        _ => return None,
    })
}

/// See [`Translator::binds_params`].
pub(crate) fn binds_params(connection: &crate::database::connection::ConnectionSpec) -> bool {
    // or given as the driver's `execute()` argument (`parameters`, psycopg's and
    // mysql.connector's `params`, PyMySQL's `args`)
    connection.has_query_params()
        || connection
            .execute_options
            .iter()
            .any(|(k, _)| matches!(k.as_str(), "parameters" | "params" | "args"))
}

/// See [`Translator::sqlalchemy`].
pub(crate) fn sqlalchemy(kind: &BackendKind) -> bool {
    match kind {
        BackendKind::Sqlalchemy => true,
        BackendKind::Python { module } => module == "sqlalchemy",
        _ => false,
    }
}

/// Whether an output column's name, written into the wrapper SQL, would be read as a
/// placeholder by the driver that binds `params=` (`%` for `%s`-style drivers, `:name` for
/// SQLAlchemy's `text()`). Where the plugin can write the name so that it isn't (see
/// [`percent_formatted`], [`percent_matched`]), it isn't one.
pub(crate) fn name_reads_as_placeholder(
    connection: &crate::database::connection::ConnectionSpec,
    kind: &BackendKind,
    name: &str,
) -> bool {
    if !binds_params(connection) {
        return false;
    }
    let percent = if percent_formatted(connection, kind) {
        false
    } else if percent_matched(connection, kind) {
        // only `%s` and `%(name)s`; `%%` isn't read as an escape, so no `%s` can be written
        name.contains("%s") || name.contains("%(")
    } else {
        name.contains('%')
    };
    percent || (sqlalchemy(kind) && name.contains(':'))
}

/// How a MySQL driver's Python module reads `%` in a statement run with `params=`.
fn mysql_driver(kind: &BackendKind) -> Option<&'static str> {
    match kind {
        BackendKind::Python { module } => match module.as_str() {
            "pymysql" | "MySQLdb" => Some("format"),
            // Connector/Python (the module of its connections is `mysql`)
            "mysql" | "mysql.connector" => Some("match"),
            _ => None,
        },
        _ => None,
    }
}

/// Whether the driver %-formats the statements it runs with `params=` (PyMySQL, mysqlclient:
/// Python's `%` operator), so that a `%` the plugin writes into the SQL (a column's name) is
/// doubled. Other drivers that format with `%` aren't covered: the name keeps pushdown off
/// there ([`name_reads_as_placeholder`]).
pub(crate) fn percent_formatted(
    connection: &crate::database::connection::ConnectionSpec,
    kind: &BackendKind,
) -> bool {
    binds_params(connection) && mysql_driver(kind) == Some("format")
}

/// Whether the driver replaces only the placeholders `%s` and `%(name)s` of the statement
/// it runs with `params=` (MySQL Connector/Python: a regular expression), leaving every
/// other `%`, `%%` too, as it is.
fn percent_matched(
    connection: &crate::database::connection::ConnectionSpec,
    kind: &BackendKind,
) -> bool {
    binds_params(connection) && mysql_driver(kind) == Some("match")
}

/// See [`Translator::mysql_without_setup`]. SingleStore has no session setup at all: its
/// `sql_mode` and time zone are unknown.
fn mysql_without_setup(dialect: DialectName, setup_runs: bool) -> bool {
    match dialect {
        DialectName::Mysql => !setup_runs,
        DialectName::Singlestore => true,
        _ => false,
    }
}

/// Whether a decimal's digits fit its precision.
fn decimal_fits(v: i128, precision: usize) -> bool {
    v.unsigned_abs()
        .checked_ilog10()
        .is_none_or(|digits| (digits as usize) < precision)
}

/// Whether an expression holds an `IN (…)` list.
fn contains_in_list(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::InList(..) => true,
        SqlExpr::Binary(l, _, r) => contains_in_list(l) || contains_in_list(r),
        SqlExpr::Not(e) => contains_in_list(e),
        _ => false,
    }
}

/// The operands of a chain of `and` (or `or`) nodes, left to right.
fn chain<'n>(node: &'n Node, or: bool, out: &mut Vec<&'n Node>) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        match n {
            Node::And(l, r) if !or => stack.extend([&**r, &**l]),
            Node::Or(l, r) if or => stack.extend([&**r, &**l]),
            n => out.push(n),
        }
    }
}

/// Chains up to this long are written as the document has them, left-deep: `((a OR b) OR c)`.
const MAX_LEFT_DEEP_CHAIN: usize = 32;

fn left_deep(parts: Vec<SqlExpr>, op: SqlBinOp) -> SqlExpr {
    let mut parts = parts.into_iter();
    let first = parts.next().expect("a chain has a part");
    parts.fold(first, |acc, e| SqlExpr::binary(acc, op, e))
}

/// The parts joined as a balanced tree (their order kept).
fn balanced(mut parts: Vec<SqlExpr>, op: SqlBinOp) -> SqlExpr {
    while parts.len() > 1 {
        let mut next = Vec::with_capacity(parts.len().div_ceil(2));
        let mut it = parts.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => SqlExpr::binary(a, op, b),
                None => a,
            });
        }
        parts = next;
    }
    parts.pop().expect("a chain has a part")
}

/// How a translated expression relates to the Polars predicate `P`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// `P(row)` implies the SQL is TRUE: usable as a pre-filter that Polars re-applies.
    Superset,
    /// The SQL is TRUE exactly when `P(row)` is true.
    Exact,
}

enum Operand<'a> {
    Column(&'a PlSmallStr, &'a DataType),
    Literal(&'a Scalar),
}

pub struct Translator<'a> {
    pub config: &'a DatabaseScanConfig,
}

impl Translator<'_> {
    fn dialect(&self) -> DialectName {
        self.config.dialect
    }

    fn strings(&self, column: &str) -> StringSemantics {
        let s = self
            .config
            .column(column)
            .map_or_else(|| self.dialect().default_string_semantics(), |c| c.strings);
        // Oracle compares byte-wise only under `NLS_COMP=BINARY`, which the session setup
        // sets. A live session (or a logon trigger, an instance default) may be linguistic.
        if self.dialect() == DialectName::Oracle && !self.config.session_setup_runs() {
            return StringSemantics {
                empty_is_null: s.empty_is_null,
                ..StringSemantics::UNKNOWN
            };
        }
        s
    }

    /// Whether the driver binds `params=` into the SQL: a `%` in a pushed literal would then
    /// be read as a placeholder by `%s`-style drivers, so such literals aren't pushed.
    fn binds_params(&self) -> bool {
        binds_params(&self.config.options.connection)
    }

    /// Whether queries run through SQLAlchemy: a SQLAlchemy URI, Engine or Connection.
    fn sqlalchemy(&self) -> bool {
        sqlalchemy(&self.config.backend.kind)
    }

    /// MySQL literals are written for the session setup: backslash escapes on
    /// (`NO_BACKSLASH_ESCAPES` removed) and a UTC time zone. Where it doesn't run, the
    /// server may read them otherwise, so such literals aren't pushed.
    fn mysql_without_setup(&self) -> bool {
        mysql_without_setup(self.dialect(), self.config.session_setup_runs())
    }

    /// A literal that can be written into the SQL (see `binds_params`, `mysql_without_setup`).
    /// NUL can only be written where the renderer escapes it, or where the query text goes
    /// to a driver that reads it whole (DuckDB's Python module): elsewhere it ends the query
    /// text for C drivers (ADBC, ODBC), and Postgres rejects it.
    fn writable(&self, lit: &SqlLiteral) -> bool {
        let Some(s) = lit.as_str() else {
            return true;
        };
        // SQLAlchemy binds `params=` through `text()`, which reads `:name` (and `\:`) in
        // string literals too
        let placeholder =
            self.binds_params() && (s.contains('%') || (self.sqlalchemy() && s.contains(':')));
        // the renderer (sqlglot) writes `\\` and control characters (NUL, `\n`, `\t`, `\r`, `\b`,
        // `\x1a`, and more in older versions) as backslash escapes
        let escape =
            self.mysql_without_setup() && s.chars().any(|c| c == '\\' || c.is_ascii_control());
        // Where string literals read backslash escapes, the renderer's spelling of a control
        // character (`\a`, `\v`, `\f` in sqlglot 28, an octal `\012` in 30 for Databricks, a raw
        // NUL) means a letter or another character; only tab, newline and carriage return are
        // written as the escape the database reads. Where they don't (Flight SQL servers may be
        // MySQL-family, Generic is unknown), a backslash is sent as is, and read as an escape.
        let control = s
            .chars()
            .any(|c| c.is_ascii_control() && !matches!(c, '\t' | '\n' | '\r'));
        let backslash_escapes = match self.dialect() {
            DialectName::Mysql
            | DialectName::Singlestore
            | DialectName::Snowflake
            | DialectName::Bigquery
            | DialectName::Databricks
            | DialectName::Redshift
            | DialectName::Clickhouse => control,
            DialectName::Flightsql | DialectName::Generic => control || s.contains('\\'),
            _ => false,
        };
        let nul = s.contains('\0')
            && !matches!(
                self.dialect(),
                DialectName::Mysql
                    | DialectName::Clickhouse
                    | DialectName::Snowflake
                    | DialectName::Bigquery
                    | DialectName::Databricks
            )
            && !(self.dialect() == DialectName::Duckdb
                && matches!(&self.config.backend.kind, BackendKind::Python { module } if module == "duckdb"));
        // ConnectorX's ClickHouse client reads every `?` as a bind placeholder; it parses
        // SQLite and ClickHouse SQL itself, reading `\'` as an escaped quote (`'a\''b'`
        // isn't closed), and a run of quotes (`''''''`) wrongly
        let connectorx = matches!(self.config.backend.kind, BackendKind::ConnectorX);
        let question = connectorx && s.contains('?') && self.dialect() == DialectName::Clickhouse;
        let quotes = connectorx
            && matches!(
                self.dialect(),
                DialectName::Sqlite | DialectName::Clickhouse
            )
            && (s.contains('\\') || s.contains("''"));
        !(placeholder || escape || backslash_escapes || nul || question || quotes)
    }

    /// Whether a literal compared with the column means the same value to the database.
    /// MySQL and MariaDB read an integer 1-99 compared with a `YEAR` column as a year
    /// (2001-2069 and 1970-1999); 0 and every other integer compare as themselves. A
    /// decimal literal wider than its precision (DuckDB `HUGEINT` is read as
    /// `Decimal(38, 0)` but holds 39 digits) isn't a valid value.
    fn literal_fits(&self, name: &str, av: &AnyValue<'_>) -> bool {
        // Spark widens a FLOAT column to DOUBLE for the comparison: only a literal that is
        // exactly an f32 (as a double) meets the stored value (`float(1.1) = 1.1` is false).
        if self.dialect() == DialectName::Databricks
            && self.spark_float(name) == Some("float")
            && let Some(v) = av.extract::<f64>()
            && !v.is_nan()
            && f64::from(v as f32) != v
        {
            return false;
        }
        if let AnyValue::Decimal(v, p, _) = av
            && !decimal_fits(*v, *p)
        {
            return false;
        }
        if self.config.maybe_mysql_year(name) {
            let y = av.extract::<i64>();
            // A declared `YEAR` holds 1901-2155 (and 0): an indexed MariaDB column compared
            // with 2200 matched nothing.
            let declared = self
                .config
                .column(name)
                .is_some_and(|c| c.db_type.is_some());
            return !y.is_some_and(|y| {
                (1..=99).contains(&y) || (declared && y != 0 && !(1901..=2155).contains(&y))
            });
        }
        // An indexed MySQL `DATETIME(fsp)` / `TIMESTAMP(fsp)` compared for equality rounds a
        // finer literal to `fsp` digits: `= '…00.5'` matched `…01`.
        if let AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _) = av
            && let Some(fsp) = self.mysql_fsp(name)
        {
            let per_second: i64 = match tu {
                TimeUnit::Nanoseconds => 1_000_000_000,
                TimeUnit::Microseconds => 1_000_000,
                TimeUnit::Milliseconds => 1_000,
            };
            let step = per_second / 10i64.pow(fsp.min(9)).min(per_second);
            return v.rem_euclid(step.max(1)) == 0;
        }
        // With the fraction digits unknown (`schema=`, a query's column) the column may hold
        // none: only whole seconds are compared as they are.
        if matches!(
            self.dialect(),
            DialectName::Mysql | DialectName::Singlestore
        ) && self.mysql_fsp(name).is_none()
        {
            let sub_second = match av {
                AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _) => {
                    let per_second: i64 = match tu {
                        TimeUnit::Nanoseconds => 1_000_000_000,
                        TimeUnit::Microseconds => 1_000_000,
                        TimeUnit::Milliseconds => 1_000,
                    };
                    v.rem_euclid(per_second) != 0
                },
                AnyValue::Time(v) => v.rem_euclid(1_000_000_000) != 0,
                _ => false,
            };
            return !sub_second;
        }
        true
    }

    /// The fractional-second digits of a MySQL `DATETIME` / `TIMESTAMP` column, if declared.
    fn mysql_fsp(&self, name: &str) -> Option<u32> {
        if !matches!(
            self.dialect(),
            DialectName::Mysql | DialectName::Singlestore
        ) {
            return None;
        }
        let t = self.config.column(name)?.db_type.as_deref()?;
        let t = t.trim().to_ascii_lowercase();
        let rest = t
            .strip_prefix("datetime")
            .or_else(|| t.strip_prefix("timestamp"))?;
        match rest.trim() {
            "" => Some(0),
            r => r.strip_prefix('(')?.strip_suffix(')')?.trim().parse().ok(),
        }
    }

    /// Whether a boolean column holds integers (or other numbers): in MySQL and SQLite always,
    /// elsewhere where its declared type is known and isn't a boolean.
    fn booleans_are_integers(&self, name: &str) -> bool {
        self.dialect().booleans_are_integers()
            || self
                .config
                .column(name)
                .and_then(|c| c.db_type.as_deref())
                .is_some_and(|t| {
                    !matches!(t.trim().to_ascii_lowercase().as_str(), "bool" | "boolean")
                })
    }

    /// `b` is `value`. MySQL and SQLite hold booleans as integers, any of which but 0 Polars
    /// reads as true, while `TRUE` is 1. So does a column of a known numeric type read as a
    /// boolean (a DuckDB `BIGINT` or `DOUBLE`): `id = TRUE` would be `id = 1`.
    fn boolean_is(&self, name: &PlSmallStr, value: bool) -> SqlExpr {
        let col = SqlExpr::col(name.clone());
        if self.booleans_are_integers(name) {
            let op = if value { SqlBinOp::NotEq } else { SqlBinOp::Eq };
            return SqlExpr::binary(col, op, SqlExpr::lit(SqlLiteral::Int(0)));
        }
        SqlExpr::binary(col, SqlBinOp::Eq, SqlExpr::lit(SqlLiteral::Bool(value)))
    }

    /// The column as a comparison's operand: through its conversion where the database
    /// would compare other values than Polars reads (see `planner::compared_column_sql`).
    fn compared(&self, name: &PlSmallStr) -> SqlExpr {
        match crate::database::planner::compared_column_sql(self.config, name) {
            Some(sql) => SqlExpr::Raw(sql),
            None => SqlExpr::col(name.clone()),
        }
    }

    /// Whether predicates on the column can be pushed at all (see `ColumnMeta::comparable`).
    fn comparable(&self, column: &str) -> bool {
        self.config.column(column).is_none_or(|c| c.comparable)
    }

    /// The columns [`translate`](Self::translate) compares in the database: not those of a
    /// conjunct it drops (a column it doesn't push, or an operand it can't translate).
    pub fn pushed_columns(&self, node: &Node, out: &mut Vec<PlSmallStr>) {
        match node {
            Node::And(..) => {
                let mut operands = vec![];
                chain(node, false, &mut operands);
                for side in operands {
                    if self.translate(side).is_some() {
                        self.pushed_columns(side, out);
                    }
                }
            },
            Node::Or(left, right) => {
                self.pushed_columns(left, out);
                self.pushed_columns(right, out);
            },
            Node::Not(inner) => self.pushed_columns(inner, out),
            _ => node.columns(out),
        }
    }

    /// A boolean expression as SQL, with its class. `None` when it can't be pushed.
    pub fn translate(&self, node: &Node) -> Option<(SqlExpr, Class)> {
        let (expr, class) = self.translate_node(node)?;
        // A test for NULL on a MySQL column of unknown type is widened to keep zero dates
        // (`zero_dates_as_null`): a superset.
        if class == Class::Exact
            && self.config.columns.iter().any(|c| {
                crate::database::planner::zero_date_candidate(self.config, c)
                    && crate::database::planner::tests_null(&expr, &c.name)
            })
        {
            return Some((expr, Class::Superset));
        }
        Some((expr, class))
    }

    fn translate_node(&self, node: &Node) -> Option<(SqlExpr, Class)> {
        match node {
            Node::And(..) | Node::Or(..) => {
                let or = matches!(node, Node::Or(..));
                let mut operands = vec![];
                chain(node, or, &mut operands);
                let op = if or { SqlBinOp::Or } else { SqlBinOp::And };
                // the database and the renderer recurse as deep as the tree: a long chain is
                // joined balanced, as deep as the logarithm of its length
                let join = if operands.len() > MAX_LEFT_DEEP_CHAIN {
                    balanced
                } else {
                    left_deep
                };
                let mut class = Class::Exact;
                let mut parts = Vec::with_capacity(operands.len());
                for operand in operands {
                    match self.translate(operand) {
                        Some((e, c)) => {
                            class = class.min(c);
                            parts.push(e);
                        },
                        // Dropping one side of an AND keeps every row the AND keeps.
                        None if !or => class = Class::Superset,
                        None => return None,
                    }
                }
                if parts.is_empty() {
                    return None;
                }
                Some((join(parts, op), class))
            },
            Node::Compare(left, op, right) => self.comparison(left, *op, right),
            Node::Column(name) => {
                // A boolean column used as the predicate.
                let dtype = self.config.schema.get(name)?;
                if !self.comparable(name) {
                    return None;
                }
                (dtype == &DataType::Boolean && self.dialect().supports_boolean())
                    .then(|| (self.boolean_is(name, true), Class::Exact))
            },
            Node::Not(inner) => {
                // The negation of a superset is a subset: only exact expressions negate.
                let (inner, class) = self.translate(inner)?;
                // ClickHouse's `x IN (…)` is 0, not NULL, for a NULL `x` under
                // `transform_null_in = 1` (a user or profile setting): `NOT` would keep the
                // NULL rows Polars drops. Positively, 0 drops them as NULL does.
                if self.dialect() == DialectName::Clickhouse && contains_in_list(&inner) {
                    return None;
                }
                (class == Class::Exact).then(|| (SqlExpr::Not(Box::new(inner)), Class::Exact))
            },
            Node::IsNull(e, negated) => {
                let (name, _) = self.column(e)?;
                Some((
                    SqlExpr::IsNull(Box::new(SqlExpr::col(name.clone())), *negated),
                    Class::Exact,
                ))
            },
            Node::IsBetween(e, lo, hi, closed) => {
                let (lo_op, hi_op) = match closed {
                    Closed::Both => (CmpOp::GtEq, CmpOp::LtEq),
                    Closed::Left => (CmpOp::GtEq, CmpOp::Lt),
                    Closed::Right => (CmpOp::Gt, CmpOp::LtEq),
                    Closed::None => (CmpOp::Gt, CmpOp::Lt),
                };
                let (l, lc) = self.comparison(e, lo_op, lo)?;
                let (r, rc) = self.comparison(e, hi_op, hi)?;
                Some((SqlExpr::binary(l, SqlBinOp::And, r), lc.min(rc)))
            },
            Node::IsIn {
                expr,
                values,
                nulls_equal,
            } => self.is_in(expr, values, *nulls_equal),
            Node::Str(op, e, pat) => self.string(*op, e, pat),
            Node::Literal(_) | Node::Cast(_, _) | Node::Unknown => None,
        }
    }

    fn string(&self, op: StrOp, e: &Node, pat: &Node) -> Option<(SqlExpr, Class)> {
        let (name, dtype) = self.column(e)?;
        let Node::Literal(lit) = pat else {
            return None;
        };
        // `LIKE` patterns hold `%`, which `%s`-style drivers read as a placeholder,
        // and `ESCAPE '\\'` relies on MySQL's session setup.
        if !dtype.is_string() || self.binds_params() || self.mysql_without_setup() {
            return None;
        }
        // SQLite's `LIKE` reads a value only up to its first NUL: `'a' || char(0) || 'b'
        // LIKE '%b'` is false, which no superset class recovers.
        if self.dialect() == DialectName::Sqlite {
            return None;
        }
        let pat = lit.value().extract_str()?;
        if !self.writable(&SqlLiteral::String(pat.into())) {
            return None;
        }
        // ClickHouse's `LIKE` misreads a pattern that holds a NUL: `s LIKE '%\0'` matches
        // every row.
        if self.dialect() == DialectName::Clickhouse && pat.contains('\0') {
            return None;
        }
        let (prefix, suffix) = match op {
            StrOp::StartsWith => ("", "%"),
            StrOp::EndsWith => ("%", ""),
            StrOp::ContainsLiteral => ("%", "%"),
        };
        let strings = self.strings(name);
        if strings.empty_is_null && pat.is_empty() {
            return None;
        }
        // Postgres rejects `LIKE` on a nondeterministic collation, which an unknown one
        // (a query's column) may be: compared byte-wise instead, as `COLLATE "C"`.
        let column = if self.dialect() == DialectName::Postgresql && !strings.equality_exact {
            SqlExpr::Raw(format!(
                "{} COLLATE \"C\"",
                self.dialect().quote_ident(name)
            ))
        } else {
            // (a DuckDB TIMETZ read as text: `LIKE` on the column itself is a binder error)
            self.compared(name)
        };
        let pattern = format!("{prefix}{}{suffix}", self.escape_like(pat));
        let class = if strings.equality_exact && !self.dialect().like_case_insensitive() {
            Class::Exact
        } else if self.dialect().like_matches_per_character()
            // A MySQL index range for `LIKE 'a%'` under a PAD SPACE collation starts at
            // `'a'` padded with blanks, past `'a\tb'`: the match is no superset.
            && !(matches!(op, StrOp::StartsWith)
                && matches!(self.dialect(), DialectName::Mysql | DialectName::Singlestore))
        {
            Class::Superset
        } else {
            // A collation may match `LIKE` by its collation elements: under Czech `ch`
            // is one letter, so `'chata' LIKE 'c%'` is false (Oracle linguistic sorts,
            // SQL Server), and the match is no superset of Polars'.
            return None;
        };
        Some((
            SqlExpr::Like {
                expr: Box::new(column),
                pattern,
                escape: '\\',
            },
            class,
        ))
    }

    /// Whether a date or timestamp literal is within what the dialect's types hold.
    fn temporal_in_range(&self, av: &AnyValue<'_>) -> bool {
        let (lo, hi) = self.dialect().temporal_literal_days();
        let days = match av {
            AnyValue::Date(d) => *d as i64,
            AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _) => {
                let per_day = match tu {
                    TimeUnit::Nanoseconds => 86_400_000_000_000,
                    TimeUnit::Microseconds => 86_400_000_000,
                    TimeUnit::Milliseconds => 86_400_000,
                };
                v.div_euclid(per_day)
            },
            _ => return true,
        };
        // A DATE written from a constant in the Julian-Gregorian gap (1582-10-05..14) reads back
        // as 1582-10-15 but is compared as the day written: `d = DATE '1582-10-15'` drops it,
        // and so do `>=` and `>` against the days up to it (a `<` or `<=` against a later day
        // keeps it). Timestamps in the gap compare right (checked live), and so do dates
        // from a later day on.
        if self.dialect() == DialectName::Databricks
            && matches!(av, AnyValue::Date(_))
            && days <= DATABRICKS_GAP_END_DAYS
        {
            return false;
        }
        // Oracle normalises a DATE or TIMESTAMP literal in the gap (1582-10-05..14) to 1582-10-15,
        // so the comparison runs against a day Polars never wrote: every operator then admits or
        // drops the 1582-10-15 row. Stored values read back right.
        if self.dialect() == DialectName::Oracle
            && (ORACLE_GAP_START_DAYS..DATABRICKS_GAP_END_DAYS).contains(&days)
        {
            return false;
        }
        (lo..=hi).contains(&days)
    }

    fn escape_like(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            let special = matches!(c, '%' | '_' | '\\')
                || (c == '[' && matches!(self.dialect(), DialectName::Mssql));
            if special {
                out.push('\\');
            }
            out.push(c);
        }
        out
    }

    fn is_in(&self, e: &Node, values: &Series, nulls_equal: bool) -> Option<(SqlExpr, Class)> {
        let (name, dtype) = self.column(e)?;
        // `IN (TRUE)` is `IN (1)` (see `boolean_is`)
        if dtype == &DataType::Boolean && self.booleans_are_integers(name) {
            return None;
        }

        if !literal_comparable(dtype, values.dtype()) {
            return None;
        }
        let mut has_null = false;
        let mut lits = Vec::with_capacity(values.len());
        for av in values.iter() {
            if av.is_null() {
                has_null = true;
                continue;
            }
            if !self.temporal_in_range(&av) || !self.literal_fits(name, &av) {
                return None;
            }
            let lit = SqlLiteral::from_any_value(&av, dtype)?;
            if !self.writable(&lit) {
                return None;
            }
            if let Some(s) = lit.as_str()
                && s.is_empty()
                && self.strings(name).empty_is_null
            {
                return None;
            }
            lits.push(lit);
        }
        if lits.is_empty() {
            return None;
        }
        let mut class = self.leaf_class(name, dtype, CmpOp::Eq)?;

        let col = || Box::new(self.compared(name));
        let bytes: usize = lits.iter().map(|l| l.estimated_bytes() + 2).sum();
        let mut expr = if lits.len() <= self.dialect().max_in_list()
            && bytes <= self.dialect().max_in_list_bytes()
        {
            SqlExpr::InList(col(), lits, false)
        } else {
            // Too long for one statement: a range that contains every value.
            self.leaf_class(name, dtype, CmpOp::Lt)?;
            let mut sorted = values.drop_nulls().sort(Default::default()).ok()?;
            sorted = sorted.rechunk();
            let lo = SqlLiteral::from_any_value(&sorted.get(0).ok()?, dtype)?;
            let hi = SqlLiteral::from_any_value(&sorted.get(sorted.len() - 1).ok()?, dtype)?;
            class = Class::Superset;
            SqlExpr::Between(
                col(),
                Box::new(SqlExpr::lit(lo)),
                Box::new(SqlExpr::lit(hi)),
            )
        };
        if nulls_equal {
            // With nulls_equal the result is never NULL, so the SQL must not be either.
            expr = if has_null {
                SqlExpr::binary(expr, SqlBinOp::Or, SqlExpr::IsNull(col(), false))
            } else {
                SqlExpr::binary(SqlExpr::IsNull(col(), true), SqlBinOp::And, expr)
            };
        }
        Some((expr, class))
    }

    fn column<'a>(&'a self, node: &'a Node) -> Option<(&'a PlSmallStr, &'a DataType)> {
        match self.operand(node)? {
            Operand::Column(name, dtype) => Some((name, dtype)),
            Operand::Literal(_) => None,
        }
    }

    fn operand<'a>(&'a self, node: &'a Node) -> Option<Operand<'a>> {
        match node {
            Node::Column(name) => {
                let (name, dtype) = self.config.schema.get_full(name).map(|(_, n, d)| (n, d))?;
                self.comparable(name)
                    .then_some(Operand::Column(name, dtype))
            },
            Node::Literal(lit) => Some(Operand::Literal(lit)),
            // A lossless widening of an integer column compares like the column itself.
            Node::Cast(expr, dtype) => match self.operand(expr)? {
                Operand::Column(name, col_dtype)
                    if col_dtype.is_integer()
                        && dtype.is_integer()
                        && dtype.is_signed_integer() >= col_dtype.is_signed_integer()
                        && widens(col_dtype, dtype) =>
                {
                    Some(Operand::Column(name, col_dtype))
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether a `Time` column may be a `TIME WITH TIME ZONE` in the database.
    fn maybe_time_with_zone(&self, name: &str) -> bool {
        match self.config.column(name).and_then(|c| c.db_type.as_deref()) {
            Some(t) => {
                let t = t.trim().to_ascii_lowercase();
                t.starts_with("timetz") || t.contains("with time zone")
            },
            None => matches!(
                self.dialect(),
                DialectName::Postgresql
                    | DialectName::Redshift
                    | DialectName::Duckdb
                    | DialectName::Trino
                    | DialectName::Teradata
            ),
        }
    }

    /// A Databricks column's declared float type (`double` or `float`), lower-case, where it
    /// is known.
    fn spark_float(&self, name: &str) -> Option<&'static str> {
        match self
            .config
            .column(name)?
            .db_type
            .as_deref()?
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "double" => Some("double"),
            "float" => Some("float"),
            _ => None,
        }
    }

    /// Whether the column's declared type is a date, time or timestamp one.
    fn temporal_db_type(&self, name: &str) -> bool {
        self.config
            .column(name)
            .and_then(|c| c.db_type.as_deref())
            .is_some_and(|t| {
                let t = t.trim().to_ascii_lowercase();
                t.starts_with("date") || t.starts_with("time")
            })
    }

    /// Whether a SQL Server date-time column's values are ones a microsecond literal
    /// compares exactly, as far as its declared type tells: not the legacy `datetime` and
    /// `smalldatetime`, nor fractional seconds beyond 6 digits (`datetime2`, `time` and
    /// `datetimeoffset` without digits have 7), nor a type that isn't known.
    fn mssql_temporal_matches(&self, name: &str) -> bool {
        let Some(t) = self.config.column(name).and_then(|c| c.db_type.as_deref()) else {
            return false;
        };
        let t = t.trim().to_ascii_lowercase();
        let (base, args) = t.split_once('(').unwrap_or((&t, ""));
        match base.trim() {
            "datetime" | "smalldatetime" => false,
            "datetime2" | "datetimeoffset" | "time" => args
                .trim_end_matches(')')
                .trim()
                .parse::<u32>()
                .is_ok_and(|scale| scale <= 6),
            _ => true,
        }
    }

    /// The class of `column <op> literal`, from the column's type and semantics.
    fn leaf_class(&self, name: &str, dtype: &DataType, op: CmpOp) -> Option<Class> {
        match dtype {
            // An integer read from a date or timestamp column (`schema=` with a type to go by)
            // is its epoch count, which the database compares as a number (MySQL: 20231231).
            dt if dt.is_integer() => (!self.temporal_db_type(name)).then_some(Class::Exact),
            DataType::Boolean => self.dialect().supports_boolean().then_some(Class::Exact),
            // SQL Server's legacy `datetime` holds 1/300 s ticks, and `datetime2(7)`,
            // `time(7)` and `datetimeoffset(7)` 100 ns, which every driver cuts to
            // microseconds: the microsecond literal doesn't match them exactly. Without
            // the column's type (a synonym, a failed lookup) it may be any of them.
            DataType::Datetime(_, _) | DataType::Time
                if self.dialect() == DialectName::Mssql && !self.mssql_temporal_matches(name) =>
            {
                None
            },
            // ClickHouse parses a literal without a zone in the server's time zone, which
            // Polars doesn't know.
            DataType::Datetime(_, None) if self.dialect() == DialectName::Clickhouse => None,
            // A MySQL `TIMESTAMP` literal is read in the session time zone, UTC only after
            // the session setup.
            DataType::Datetime(_, Some(_)) if self.mysql_without_setup() => None,
            // A naive literal is a `TIMESTAMP_NTZ`, which Databricks compares with a
            // `TIMESTAMP` (LTZ) column in the session's time zone: only a column known to
            // be an NTZ one is compared.
            DataType::Datetime(_, None)
                if self.dialect() == DialectName::Databricks
                    && !self
                        .config
                        .column(name)
                        .and_then(|c| c.db_type.as_deref())
                        .is_some_and(|t| t.trim().eq_ignore_ascii_case("timestamp_ntz")) =>
            {
                None
            },
            // An aware literal is a `TIMESTAMP_LTZ`, which Databricks converts a
            // `TIMESTAMP_NTZ` column's value to in the session's time zone, not UTC: a
            // column read as aware (`databricks.sql` once did) mustn't be compared so. Only
            // a column known to be an LTZ one is compared (the type lookup may have failed).
            DataType::Datetime(_, Some(_))
                if self.dialect() == DialectName::Databricks
                    && !self
                        .config
                        .column(name)
                        .and_then(|c| c.db_type.as_deref())
                        .is_some_and(|t| {
                            matches!(
                                t.trim().to_ascii_lowercase().as_str(),
                                "timestamp" | "timestamp_ltz"
                            )
                        }) =>
            {
                None
            },
            DataType::Date | DataType::Datetime(_, _) => self
                .dialect()
                .temporal_comparisons_exact()
                .then_some(Class::Exact),
            // A `TIME WITH TIME ZONE` column is read as its local time, but the database
            // compares it with a `TIME` literal in UTC (Postgres), or not at all (DuckDB's
            // ordering comparisons): where the column's type is unknown on a dialect that
            // has the type, it may be one.
            DataType::Time if self.maybe_time_with_zone(name) => None,
            DataType::Time => self.dialect().supports_time().then_some(Class::Exact),
            // An unconstrained `numeric` / `NUMBER` is read rounded to this scale, so the
            // database compares digits Polars doesn't see.
            dt if *dt == crate::database::schema::UNCONSTRAINED_DECIMAL => None,
            DataType::Decimal(_, _) => self
                .dialect()
                .decimal_comparisons_exact()
                .then_some(Class::Exact),
            // Oracle `BINARY_DOUBLE` keeps -0.0 apart from 0.0, which Polars treats as equal;
            // `NUMBER` and `FLOAT` hold decimals that the f64 Polars reads only approximates.
            // No Oracle float comparison is pushed.
            dt if dt.is_float() && self.dialect() == DialectName::Oracle => None,
            // MySQL's text protocol (PyMySQL, mysql.connector) and ADBC send a 4-byte `FLOAT`
            // rounded to 6 digits, while the database compares the stored value. A `FLOAT`
            // column is read exactly (as a `DOUBLE`, see `column_conversion`); a float whose
            // type is unknown (a query's on a row driver, `schema=`) may be one read rounded.
            dt if dt.is_float()
                && matches!(
                    self.dialect(),
                    DialectName::Mysql | DialectName::Singlestore
                )
                && self
                    .config
                    .column(name)
                    .is_some_and(|c| c.db_type.is_none()) =>
            {
                None
            },
            // An indexed MariaDB `DOUBLE(M, D)` rounds an equality literal to `D` digits.
            dt if dt.is_float()
                && !op.is_ordering()
                && matches!(
                    self.dialect(),
                    DialectName::Mysql | DialectName::Singlestore
                )
                && self
                    .config
                    .column(name)
                    .and_then(|c| c.db_type.as_deref())
                    .is_some_and(|t| t.contains('(')) =>
            {
                None
            },
            // Spark orders and compares floats as Polars does (NaN is the greatest and equals
            // itself, -0.0 = 0.0), for a column of known type: a DECIMAL or another type read
            // as a float is compared as it is stored.
            dt if dt.is_float() && self.dialect() == DialectName::Databricks => {
                matches!(self.spark_float(name), Some("double" | "float")).then_some(Class::Exact)
            },
            dt if dt.is_float() => self
                .dialect()
                .float_comparisons_exact()
                .then_some(Class::Exact),
            DataType::String => {
                let s = self.strings(name);
                match op {
                    op if op.is_ordering() => s.ordering_exact.then_some(Class::Exact),
                    CmpOp::Eq | CmpOp::EqMissing => Some(if s.equality_exact {
                        Class::Exact
                    } else {
                        Class::Superset
                    }),
                    // `<>` under a case-insensitive collation drops rows Polars keeps.
                    _ => s.equality_exact.then_some(Class::Exact),
                }
            },
            _ => None,
        }
    }

    fn comparison(&self, left: &Node, op: CmpOp, right: &Node) -> Option<(SqlExpr, Class)> {
        let (l, r) = (self.operand(left)?, self.operand(right)?);
        let (name, dtype, lit, op) = match (l, r) {
            (Operand::Column(n, d), Operand::Literal(lit)) => (n, d, lit, op),
            (Operand::Literal(lit), Operand::Column(n, d)) => (n, d, lit, op.swap_operands()),
            (Operand::Column(a, ad), Operand::Column(b, bd)) => {
                return self.column_comparison(a, ad, op, b, bd);
            },
            _ => return None,
        };

        let bin_op = match op {
            CmpOp::Eq | CmpOp::EqMissing => SqlBinOp::Eq,
            CmpOp::NotEq | CmpOp::NotEqMissing => SqlBinOp::NotEq,
            CmpOp::Lt => SqlBinOp::Lt,
            CmpOp::LtEq => SqlBinOp::LtEq,
            CmpOp::Gt => SqlBinOp::Gt,
            CmpOp::GtEq => SqlBinOp::GtEq,
        };
        let col = || self.compared(name);

        let av = lit.value();
        if av.is_null() {
            return match op {
                // `eq_missing(None)` is `is_null`.
                CmpOp::EqMissing => Some((SqlExpr::IsNull(Box::new(col()), false), Class::Exact)),
                CmpOp::NotEqMissing => Some((SqlExpr::IsNull(Box::new(col()), true), Class::Exact)),
                // A comparison with NULL keeps no rows in either system.
                _ => None,
            };
        }

        if !literal_comparable(dtype, lit.dtype()) || !self.literal_fits(name, av) {
            return None;
        }
        let lit = SqlLiteral::from_any_value(av, dtype)?;
        if !self.writable(&lit) || !self.temporal_in_range(av) {
            return None;
        }
        if let Some(s) = lit.as_str()
            && s.is_empty()
            && self.strings(name).empty_is_null
        {
            return None;
        }
        if matches!(lit, SqlLiteral::Bool(_)) && !self.dialect().supports_boolean() {
            return None;
        }
        let class = self.leaf_class(name, dtype, op)?;
        let mut expr = match lit {
            // `b = TRUE` / `b <> FALSE`, as `boolean_is` writes them
            SqlLiteral::Bool(v) if self.booleans_are_integers(name) => match bin_op {
                SqlBinOp::Eq => self.boolean_is(name, v),
                SqlBinOp::NotEq => self.boolean_is(name, !v),
                // integers order otherwise than booleans
                _ => return None,
            },
            lit => SqlExpr::binary(col(), bin_op, SqlExpr::lit(lit)),
        };
        // `eq_missing` / `ne_missing` are never NULL, so their SQL must not be either: under
        // a NOT, a NULL would drop rows that Polars keeps.
        if op == CmpOp::NotEqMissing {
            // NULL <> x is NULL; ne_missing(NULL, x) is true.
            expr = SqlExpr::binary(SqlExpr::IsNull(Box::new(col()), false), SqlBinOp::Or, expr);
        } else if op == CmpOp::EqMissing {
            // NULL = x is NULL; eq_missing(NULL, x) is false.
            expr = SqlExpr::binary(SqlExpr::IsNull(Box::new(col()), true), SqlBinOp::And, expr);
        }
        Some((expr, class))
    }

    /// `a <op> b` on two columns: only for types without collation or NaN concerns.
    fn column_comparison(
        &self,
        a: &PlSmallStr,
        ad: &DataType,
        op: CmpOp,
        b: &PlSmallStr,
        bd: &DataType,
    ) -> Option<(SqlExpr, Class)> {
        let date_exact =
            matches!(ad, DataType::Date) && self.dialect().temporal_comparisons_exact();
        if ad != bd || !(ad.is_integer() || date_exact) {
            return None;
        }
        let bin_op = match op {
            CmpOp::Eq => SqlBinOp::Eq,
            CmpOp::NotEq => SqlBinOp::NotEq,
            CmpOp::Lt => SqlBinOp::Lt,
            CmpOp::LtEq => SqlBinOp::LtEq,
            CmpOp::Gt => SqlBinOp::Gt,
            CmpOp::GtEq => SqlBinOp::GtEq,
            CmpOp::EqMissing | CmpOp::NotEqMissing => return None,
        };
        // each as Polars reads it (a SQLite number's text is read as that number)
        Some((
            SqlExpr::binary(self.compared(a), bin_op, self.compared(b)),
            Class::Exact,
        ))
    }
}

/// Whether a literal of dtype `lit` compared with a column of dtype `col` is what the database
/// compares once the literal is written as the column's type. Polars either compares them
/// as values of a type they share (integers, floats, decimals, which the literal's lossless
/// cast to the column's type meets) or refuses (a string and an integer, a naive and an
/// aware datetime, two time zones), and a refusal must not become a pushed filter that
/// returns rows. An integer compared with a float is Polars' float comparison (the
/// integer rounded), not the database's exact one.
fn literal_comparable(col: &DataType, lit: &DataType) -> bool {
    if col == lit {
        return true;
    }
    match (col, lit) {
        (c, l) if c.is_integer() && l.is_integer() => true,
        (c, l) if c.is_float() && (l.is_integer() || l.is_float()) => true,
        (DataType::Decimal(..), l) => l.is_integer() || matches!(l, DataType::Decimal(..)),
        (DataType::Datetime(_, a), DataType::Datetime(_, b)) => a == b,
        _ => false,
    }
}

/// 1582-10-15, the first day of the Gregorian calendar, in days since the epoch.
pub(crate) const DATABRICKS_GAP_END_DAYS: i64 = -141_427;
/// 1582-10-05, the first day of the Julian-Gregorian gap; the gap ends before `DATABRICKS_GAP_END_DAYS`.
const ORACLE_GAP_START_DAYS: i64 = -141_437;

/// The deepest an expression may nest in a statement: the renderer recurses through it (in
/// Python, whose stack is short), and so does the database.
const MAX_EXPR_DEPTH: usize = 100;

/// What one statement takes of the pushed filters: a count, a depth and a size the database
/// accepts (see [`DialectName::max_conjuncts`], [`DialectName::max_statement_bytes`]). The
/// resolver passes each translated filter, in order, through [`admit`](Self::admit); one that
/// doesn't fit is not pushed (Polars applies it, as it does for any filter it can't translate),
/// and later, smaller ones may still fit.
pub struct PushBudget {
    dialect: DialectName,
    conjuncts: usize,
    bytes: usize,
}

impl PushBudget {
    pub fn new(dialect: DialectName) -> Self {
        Self {
            dialect,
            conjuncts: 0,
            bytes: 0,
        }
    }

    pub fn admit(&mut self, pushed: Option<(SqlExpr, Class)>) -> Option<(SqlExpr, Class)> {
        let (expr, _) = pushed.as_ref()?;
        let bytes = expr.estimated_bytes();
        if self.conjuncts >= self.dialect.max_conjuncts()
            || self.bytes + bytes > self.dialect.max_statement_bytes()
            || expr.depth() > MAX_EXPR_DEPTH
        {
            return None;
        }
        self.conjuncts += 1;
        self.bytes += bytes;
        pushed
    }
}

/// Whether every value of `from` is representable in `to` (both integers).
fn widens(from: &DataType, to: &DataType) -> bool {
    let bits = |d: &DataType| -> Option<u32> {
        Some(match d {
            DataType::Int8 | DataType::UInt8 => 8,
            DataType::Int16 | DataType::UInt16 => 16,
            DataType::Int32 | DataType::UInt32 => 32,
            DataType::Int64 | DataType::UInt64 => 64,
            DataType::Int128 | DataType::UInt128 => 128,
            _ => return None,
        })
    };
    match (bits(from), bits(to)) {
        (Some(f), Some(t)) => {
            if from.is_signed_integer() == to.is_signed_integer() {
                t >= f
            } else {
                // unsigned -> signed needs one more bit.
                to.is_signed_integer() && t > f
            }
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_documents() {
        let n = Node::parse(
            r#"{"op": "and", "args": [
                {"op": "gt_eq", "args": [{"col": "a"}, {"lit": {"dtype": "i64", "value": "5"}}]},
                {"unknown": "x.map(f)"}
            ]}"#,
        );
        let Node::And(l, r) = n else { panic!() };
        assert!(matches!(*r, Node::Unknown));
        let Node::Compare(c, CmpOp::GtEq, lit) = *l else {
            panic!()
        };
        assert!(matches!(*c, Node::Column(ref n) if n == "a"));
        let Node::Literal(s) = *lit else { panic!() };
        assert_eq!(s.value(), &AnyValue::Int64(5));

        assert_eq!(
            parse_dtype("datetime[us, UTC]"),
            Some(DataType::Datetime(
                TimeUnit::Microseconds,
                Some(TimeZone::UTC)
            ))
        );
        assert_eq!(
            parse_dtype("decimal[10, 2]"),
            Some(DataType::Decimal(10, 2))
        );
        assert!(matches!(Node::parse("not json"), Node::Unknown));
        // DuckDB `HUGEINT` read as `Decimal(38, 0)`: 39 digits don't fit
        let max = r#"{"op": "eq", "args": [{"col": "h"},
            {"lit": {"dtype": "decimal[38, 0]", "value": "170141183460469231731687303715884105727"}}]}"#;
        assert!(
            matches!(Node::parse(max), Node::Compare(_, _, ref r) if matches!(**r, Node::Unknown))
        );
        assert!(mysql_without_setup(DialectName::Singlestore, true));
        assert!(mysql_without_setup(DialectName::Mysql, false));
        assert!(!mysql_without_setup(DialectName::Mysql, true));
        assert!(!mysql_without_setup(DialectName::Postgresql, false));
        assert!(decimal_fits(0, 1));
        assert!(decimal_fits(-99_999, 5));
        assert!(!decimal_fits(100_000, 5));
        assert!(decimal_fits(10i128.pow(38) - 1, 38));
        assert!(!decimal_fits(i128::MAX, 38));
        assert!(!decimal_fits(i128::MIN, 38));
        assert!(matches!(
            Node::parse(r#"{"op": "frobnicate", "args": [{"col": "a"}, {"col": "b"}]}"#),
            Node::Unknown
        ));
    }

    use std::sync::Arc;

    use crate::database::backend::BackendChoice;
    use crate::database::connection::{ConnectionSpec, ConnectionTarget};
    use crate::database::options::{DatabaseRelation, DatabaseScanOptions, PartitionSpec};

    fn spec() -> ConnectionSpec {
        ConnectionSpec::new(ConnectionTarget::Uri("x://h/db".into()))
    }

    /// A table scan of `columns` (`(name, dtype, db_type)`).
    fn config(
        dialect: DialectName,
        kind: BackendKind,
        columns: &[(&str, DataType, Option<&str>)],
    ) -> DatabaseScanConfig {
        let options = DatabaseScanOptions {
            connection: spec(),
            relation: DatabaseRelation::Table {
                parts: vec!["t".into()],
                quoted: vec![false],
            },
            dialect: Some(dialect),
            driver: None,
            partitioning: PartitionSpec::None,
            order_by: vec![],
            consistency: Default::default(),
            fetch: Default::default(),
            pushdown: true,
            schema_overrides: None,
        };
        let metas: Vec<_> = columns
            .iter()
            .map(|(n, d, t)| crate::database::config::ColumnMeta {
                name: (*n).into(),
                dtype: d.clone(),
                db_type: t.map(Into::into),
                strings: StringSemantics::UNKNOWN,
                is_key: false,
                comparable: true,
            })
            .collect();
        DatabaseScanConfig {
            options: Arc::new(options),
            dialect,
            backend: BackendChoice {
                kind,
                version: None,
                location: None,
                reason: "test".into(),
            },
            schema: Arc::new(
                columns
                    .iter()
                    .map(|(n, d, _)| Field::new((*n).into(), d.clone()))
                    .collect(),
            ),
            columns: metas.into(),
            stats: Default::default(),
            compiled: None,
            connection_share: None,
            fingerprint: 0,
            pushdown_disabled: None,
            volatile_source: false,
            volatility_unchecked: false,
            user_schema: false,
            derived_order_dropped: false,
            refused_columns: vec![],
            bounds_cache: Default::default(),
        }
    }

    fn adbc(driver: &str) -> BackendKind {
        BackendKind::Adbc {
            driver: driver.into(),
        }
    }

    fn class_of(config: &DatabaseScanConfig, doc: &str) -> Option<Class> {
        Translator { config }
            .translate(&Node::parse(doc))
            .map(|t| t.1)
    }

    fn cmp(op: &str, col: &str, dtype: &str, value: &str) -> String {
        // dates and booleans are JSON numbers and booleans, the rest strings
        let value = if matches!(dtype, "date" | "bool") {
            value.to_string()
        } else {
            format!("\"{value}\"")
        };
        format!(
            r#"{{"op": "{op}", "args": [{{"col": "{col}"}}, {{"lit": {{"dtype": "{dtype}", "value": {value}}}}}]}}"#
        )
    }

    #[test]
    fn databricks_float_comparisons() {
        // #406
        let c = |t, d| config(DialectName::Databricks, adbc("databricks"), &[("f", d, t)]);
        let eq = |v: &str| cmp("eq", "f", "f64", v);
        let double = c(Some("double"), DataType::Float64);
        assert_eq!(class_of(&double, &eq("1.1")), Some(Class::Exact));
        assert_eq!(
            class_of(&double, &cmp("gt", "f", "f64", "-0.5")),
            Some(Class::Exact)
        );
        // a FLOAT column widens to DOUBLE: only a literal that is exactly an f32 meets it
        let float = c(Some("float"), DataType::Float32);
        assert_eq!(class_of(&float, &eq("0.5")), Some(Class::Exact));
        assert_eq!(class_of(&float, &eq("1.1")), None);
        assert_eq!(
            class_of(&float, &eq(&format!("{}", f64::from(1.1f32)))),
            Some(Class::Exact)
        );
        // the type is unknown (a query's, `schema=`)
        assert_eq!(class_of(&c(None, DataType::Float64), &eq("0.5")), None);
        assert_eq!(
            class_of(&c(Some("decimal(10,2)"), DataType::Float64), &eq("0.5")),
            None
        );
    }

    #[test]
    fn databricks_aware_comparison_on_ntz_column() {
        // #422: a column typed as aware that is a `TIMESTAMP_NTZ` is not compared
        let utc = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        let naive = DataType::Datetime(TimeUnit::Microseconds, None);
        let c =
            |d: DataType, t| config(DialectName::Databricks, adbc("databricks"), &[("ts", d, t)]);
        let aware = cmp("eq", "ts", "datetime[us, UTC]", "1719835200000000");
        let plain = cmp("eq", "ts", "datetime[us]", "1719835200000000");
        // an NTZ column: naive is compared, aware (a `schema=` or an older read) is not
        let ntz = c(naive.clone(), Some("timestamp_ntz"));
        assert_eq!(class_of(&ntz, &plain), Some(Class::Exact));
        let ntz_aware = c(utc.clone(), Some("timestamp_ntz"));
        assert_eq!(class_of(&ntz_aware, &aware), None);
        // an LTZ column: aware is compared, naive (a `schema=`) is not
        let ltz = c(utc, Some("timestamp"));
        assert_eq!(class_of(&ltz, &aware), Some(Class::Exact));
        let ltz_naive = c(naive, Some("timestamp"));
        assert_eq!(class_of(&ltz_naive, &plain), None);
    }

    #[test]
    fn column_names_read_as_placeholders_under_params() {
        // #417
        let mut with = spec();
        with.execute_options.push(("params".into(), "x".into()));
        let psycopg = BackendKind::Python {
            module: "psycopg".into(),
        };
        assert!(name_reads_as_placeholder(&with, &psycopg, "rate%"));
        assert!(!name_reads_as_placeholder(&with, &psycopg, "a:b"));
        assert!(name_reads_as_placeholder(
            &with,
            &BackendKind::Sqlalchemy,
            "a:b"
        ));
        assert!(!name_reads_as_placeholder(&with, &psycopg, "plain"));
        assert!(!name_reads_as_placeholder(&spec(), &psycopg, "rate%"));
        // MySQL's drivers: the `%` is doubled where the plugin writes it (#425)
        let pymysql = BackendKind::Python {
            module: "pymysql".into(),
        };
        assert!(!name_reads_as_placeholder(&with, &pymysql, "rate%"));
        assert!(percent_formatted(&with, &pymysql));
        assert!(!percent_formatted(&spec(), &pymysql));
        assert!(!percent_formatted(&with, &psycopg));
        // Connector/Python replaces only `%s`: other names are written as they are
        let connector = BackendKind::Python {
            module: "mysql".into(),
        };
        assert!(!percent_formatted(&with, &connector));
        assert!(!name_reads_as_placeholder(&with, &connector, "a%b"));
        assert!(name_reads_as_placeholder(&with, &connector, "x%s"));
    }

    #[test]
    fn has_unknown_finds_undescribed_parts() {
        assert!(Node::parse(r#"{"unknown": "x"}"#).has_unknown());
        assert!(!Node::parse(&cmp("eq", "a", "i64", "1")).has_unknown());
        let and = format!(
            r#"{{"op": "and", "args": [{}, {{"unknown": "f"}}]}}"#,
            cmp("eq", "a", "i64", "1")
        );
        assert!(Node::parse(&and).has_unknown());
    }

    /// `(a == 0) | (b == 1) | …`, left-deep as Polars builds it.
    fn or_chain(terms: usize) -> String {
        let mut doc = cmp("eq", "a", "i64", "0");
        for i in 1..terms {
            doc = format!(
                r#"{{"op": "or", "args": [{doc}, {}]}}"#,
                cmp("eq", "b", "i64", &i.to_string())
            );
        }
        doc
    }

    fn translate_doc(config: &DatabaseScanConfig, doc: &str) -> Option<(SqlExpr, Class)> {
        Translator { config }.translate(&Node::parse(doc))
    }

    #[test]
    fn deep_documents_are_parsed_and_pushed() {
        // #431: `serde_json`'s 128-level limit made a left-deep OR of 63 terms `Unknown`
        let c = config(
            DialectName::Postgresql,
            adbc("postgresql"),
            &[("a", DataType::Int64, None), ("b", DataType::Int64, None)],
        );
        for terms in [62, 63, 200, 450] {
            let node = Node::parse(&or_chain(terms));
            assert!(!node.has_unknown(), "{terms} terms");
            let (expr, class) = translate_doc(&c, &or_chain(terms)).expect("pushed");
            assert_eq!(class, Class::Exact);
            // a long chain is balanced: as deep as the logarithm of its length
            assert!(expr.depth() <= terms.max(MAX_LEFT_DEEP_CHAIN) + 2);
            if terms > MAX_LEFT_DEEP_CHAIN {
                assert!(expr.depth() < 20, "{terms} terms: depth {}", expr.depth());
            }
        }
        // past the document limit: not pushed, no stack overflow
        assert!(matches!(Node::parse(&or_chain(600)), Node::Unknown));
        // the same, on a thread with a small stack (translation, drop)
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                let doc = or_chain(500);
                assert!(translate_doc(&c, &doc).is_some());
                // nested `not`s: depth in the tree itself
                let mut not = cmp("eq", "a", "i64", "0");
                for _ in 0..400 {
                    not = format!(r#"{{"op": "not", "args": [{not}]}}"#);
                }
                let node = Node::parse(&not);
                assert!(!node.has_unknown());
                let _ = Translator { config: &c }.translate(&node);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn and_or_chains_keep_their_class() {
        let c = config(
            DialectName::Postgresql,
            adbc("postgresql"),
            &[("a", DataType::Int64, None), ("s", DataType::String, None)],
        );
        let and = |terms: Vec<String>| {
            terms
                .into_iter()
                .reduce(|l, r| format!(r#"{{"op": "and", "args": [{l}, {r}]}}"#))
                .unwrap()
        };
        let or = |terms: Vec<String>| {
            terms
                .into_iter()
                .reduce(|l, r| format!(r#"{{"op": "or", "args": [{l}, {r}]}}"#))
                .unwrap()
        };
        let exact = cmp("gt", "a", "i64", "1");
        let unknown = r#"{"unknown": "f"}"#.to_string();
        // a dropped part of an AND leaves a superset; of an OR, nothing is pushed
        let doc = and(vec![exact.clone(), unknown.clone(), exact.clone()]);
        assert_eq!(class_of(&c, &doc), Some(Class::Superset));
        let doc = or(vec![exact.clone(), unknown.clone(), exact.clone()]);
        assert_eq!(class_of(&c, &doc), None);
        let doc = and(vec![unknown.clone(), unknown.clone()]);
        assert_eq!(class_of(&c, &doc), None);
        // long chains too
        let many = vec![exact.clone(); 100];
        assert_eq!(class_of(&c, &and(many.clone())), Some(Class::Exact));
        assert_eq!(class_of(&c, &or(many.clone())), Some(Class::Exact));
        let mut with_unknown = many;
        with_unknown[50] = unknown;
        assert_eq!(
            class_of(&c, &and(with_unknown.clone())),
            Some(Class::Superset)
        );
        assert_eq!(class_of(&c, &or(with_unknown)), None);
        // a string equality is a superset (unknown collation): so is its chain
        let s_eq = cmp("eq", "s", "str", "x");
        assert_eq!(class_of(&c, &or(vec![exact, s_eq])), Some(Class::Superset));
    }

    #[test]
    fn statement_budget() {
        let eq = |v: usize| cmp("neq", "id", "i64", &v.to_string());
        let cfg = |d, k: BackendKind| config(d, k, &[("id", DataType::Int64, None)]);
        // #445: SQLite refuses an expression tree deeper than 1000
        let sqlite = cfg(DialectName::Sqlite, adbc("sqlite"));
        let t = Translator { config: &sqlite };
        let mut budget = PushBudget::new(DialectName::Sqlite);
        let pushed = (0..1200)
            .filter(|i| budget.admit(t.translate(&Node::parse(&eq(*i)))).is_some())
            .count();
        assert_eq!(pushed, DialectName::Sqlite.max_conjuncts());
        assert!(pushed < 1000);
        let mut budget = PushBudget::new(DialectName::Postgresql);
        let pg = cfg(DialectName::Postgresql, adbc("postgresql"));
        let t = Translator { config: &pg };
        let pushed = (0..4000)
            .filter(|i| budget.admit(t.translate(&Node::parse(&eq(*i)))).is_some())
            .count();
        assert_eq!(pushed, DialectName::Postgresql.max_conjuncts());

        // #434: lists under the per-list cap, ORed (ClickHouse `max_query_size`)
        let ch = config(
            DialectName::Clickhouse,
            adbc("clickhouse"),
            &[("s", DataType::String, None), ("id", DataType::Int64, None)],
        );
        let list = |k: usize| {
            let values: Vec<String> = (0..1100)
                .map(|i| format!("\"v{k}-{i}-{}\"", "x".repeat(40)))
                .collect();
            format!(
                r#"{{"op": "is_in", "args": [{{"col": "s"}}], "nulls_equal": false,
                "values": {{"dtype": "str", "values": [{}]}}}}"#,
                values.join(",")
            )
        };
        let one = list(0);
        let t = Translator { config: &ch };
        let (expr, _) = t.translate(&Node::parse(&one)).unwrap();
        assert!(matches!(expr, SqlExpr::InList(..)));
        assert!(expr.estimated_bytes() <= DialectName::Clickhouse.max_statement_bytes());
        let or8 = (0..8)
            .map(list)
            .reduce(|l, r| format!(r#"{{"op": "or", "args": [{l}, {r}]}}"#))
            .unwrap();
        let pushed = t.translate(&Node::parse(&or8));
        assert!(pushed.is_some());
        let mut budget = PushBudget::new(DialectName::Clickhouse);
        assert!(budget.admit(pushed).is_none());
        // ... while the first under the budget go through, then no more
        let mut budget = PushBudget::new(DialectName::Clickhouse);
        let admitted = (0..8)
            .filter(|k| budget.admit(t.translate(&Node::parse(&list(*k)))).is_some())
            .count();
        assert!((1..8).contains(&admitted), "{admitted}");

        // MariaDB: a single literal over `max_allowed_packet`
        let my = config(
            DialectName::Mysql,
            adbc("mysql"),
            &[
                ("s", DataType::String, None),
                ("id", DataType::Int64, Some("bigint")),
            ],
        );
        let big = cmp("eq", "s", "str", &"x".repeat(17 << 20));
        let t = Translator { config: &my };
        let pushed = t.translate(&Node::parse(&big));
        let mut budget = PushBudget::new(DialectName::Mysql);
        assert!(budget.admit(pushed).is_none());
        // a filter that doesn't fit doesn't stop a small one after it
        assert!(
            budget
                .admit(t.translate(&Node::parse(&cmp("lt", "id", "i64", "3"))))
                .is_some()
        );
        // a single `IN` list never fills the statement (a longer one becomes a range)
        for d in [
            DialectName::Mysql,
            DialectName::Sqlite,
            DialectName::Clickhouse,
        ] {
            assert!(d.max_in_list_bytes() <= d.max_statement_bytes() / 2);
        }
    }

    #[test]
    fn literals_of_another_dtype_are_not_pushed() {
        // #432: Polars raises (or compares otherwise); the scan mustn't return rows
        let c = config(
            DialectName::Postgresql,
            adbc("postgresql"),
            &[
                ("id", DataType::Int64, None),
                ("s", DataType::String, None),
                ("f", DataType::Float64, None),
                (
                    "ntz",
                    DataType::Datetime(TimeUnit::Microseconds, None),
                    None,
                ),
                (
                    "ltz",
                    DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC)),
                    None,
                ),
                ("d", DataType::Date, None),
            ],
        );
        assert_eq!(class_of(&c, &cmp("eq", "id", "str", "1")), None);
        assert_eq!(class_of(&c, &cmp("eq", "s", "i64", "1")), None);
        assert_eq!(class_of(&c, &cmp("eq", "id", "f64", "1.0")), None);
        assert_eq!(class_of(&c, &cmp("eq", "id", "bool", "true")), None);
        assert_eq!(class_of(&c, &cmp("eq", "d", "datetime[us]", "0")), None);
        // naive against aware, aware against naive or another zone
        assert_eq!(
            class_of(&c, &cmp("eq", "ntz", "datetime[us, UTC]", "0")),
            None
        );
        assert_eq!(class_of(&c, &cmp("eq", "ltz", "datetime[us]", "0")), None);
        assert_eq!(
            class_of(&c, &cmp("eq", "ltz", "datetime[us, Europe/Berlin]", "0")),
            None
        );
        // what Polars compares as values: other integer widths, an integer with a float
        // column, another time unit
        assert_eq!(
            class_of(&c, &cmp("eq", "id", "i32", "1")),
            Some(Class::Exact)
        );
        assert_eq!(
            class_of(&c, &cmp("eq", "id", "u8", "1")),
            Some(Class::Exact)
        );
        assert_eq!(
            class_of(&c, &cmp("eq", "s", "str", "x")),
            Some(Class::Superset)
        );
        assert_eq!(
            class_of(&c, &cmp("eq", "ntz", "datetime[ms]", "1000")),
            Some(Class::Exact)
        );
        assert_eq!(
            class_of(&c, &cmp("eq", "ltz", "datetime[ns, UTC]", "1000000")),
            Some(Class::Exact)
        );
        assert_eq!(
            class_of(&c, &cmp("eq", "d", "date", "5")),
            Some(Class::Exact)
        );
        // `is_in` and `is_between` the same
        let is_in = |col: &str, dtype: &str, values: &str| {
            format!(
                r#"{{"op": "is_in", "args": [{{"col": "{col}"}}], "nulls_equal": false,
                "values": {{"dtype": "{dtype}", "values": [{values}]}}}}"#
            )
        };
        assert_eq!(class_of(&c, &is_in("id", "str", r#""1", "2""#)), None);
        assert_eq!(
            class_of(&c, &is_in("id", "i64", r#""1", "2""#)),
            Some(Class::Exact)
        );
        assert_eq!(class_of(&c, &is_in("s", "i64", r#""1""#)), None);
        let between = r#"{"op": "is_between", "closed": "both", "args": [{"col": "id"},
            {"lit": {"dtype": "str", "value": "1"}}, {"lit": {"dtype": "str", "value": "2"}}]}"#;
        assert_eq!(class_of(&c, between), None);
    }

    #[test]
    fn databricks_aware_literals_need_a_known_ltz_column() {
        // #428: a failed type lookup leaves `db_type` unknown
        let utc = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        let c = |t| {
            config(
                DialectName::Databricks,
                adbc("databricks"),
                &[("ts", utc.clone(), t)],
            )
        };
        let aware = cmp("eq", "ts", "datetime[us, UTC]", "1719835200000000");
        assert_eq!(class_of(&c(Some("timestamp")), &aware), Some(Class::Exact));
        assert_eq!(class_of(&c(Some("TIMESTAMP")), &aware), Some(Class::Exact));
        assert_eq!(class_of(&c(Some("timestamp_ntz")), &aware), None);
        assert_eq!(class_of(&c(None), &aware), None);
        // other dialects don't need the type
        let pg = config(
            DialectName::Postgresql,
            adbc("postgresql"),
            &[("ts", utc.clone(), None)],
        );
        assert_eq!(class_of(&pg, &aware), Some(Class::Exact));
    }

    #[test]
    fn databricks_dates_in_the_gregorian_gap() {
        // #446: DATE '1582-10-15' drops a row stored in the 1582-10-05..14 gap
        let c = |d| {
            config(
                d,
                adbc("databricks"),
                &[
                    ("d", DataType::Date, Some("date")),
                    (
                        "ts",
                        DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC)),
                        Some("timestamp"),
                    ),
                ],
            )
        };
        let db = c(DialectName::Databricks);
        // 1582-10-15 is -141427 days; 1582-10-16 is -141426
        for op in ["eq", "gt_eq", "lt", "gt"] {
            assert_eq!(
                class_of(&db, &cmp(op, "d", "date", "-141427")),
                None,
                "{op}"
            );
            assert_eq!(
                class_of(&db, &cmp(op, "d", "date", "-141430")),
                None,
                "{op}"
            );
            assert_eq!(
                class_of(&db, &cmp(op, "d", "date", "-141426")),
                Some(Class::Exact)
            );
        }
        // timestamps in the gap compare right (checked live)
        let us = |days: i64| (days * 86_400_000_000).to_string();
        let ts = |days: i64| cmp("eq", "ts", "datetime[us, UTC]", &us(days));
        assert_eq!(class_of(&db, &ts(-141_427)), Some(Class::Exact));
        assert_eq!(class_of(&db, &ts(-141_432)), Some(Class::Exact));
        // another dialect is untouched
        let pg = c(DialectName::Postgresql);
        assert_eq!(
            class_of(&pg, &cmp("eq", "d", "date", "-141427")),
            Some(Class::Exact)
        );
    }

    #[test]
    fn oracle_dates_in_the_gregorian_gap() {
        // Oracle normalises a DATE/TIMESTAMP literal in 1582-10-05..14 to 1582-10-15
        let cfg = config(
            DialectName::Oracle,
            adbc("oracle"),
            &[("d", DataType::Date, Some("DATE"))],
        );
        let t = Translator { config: &cfg };
        let us =
            |days: i64| AnyValue::Datetime(days * 86_400_000_000, TimeUnit::Microseconds, None);
        for days in [-141_437, -141_433, -141_428] {
            assert!(!t.temporal_in_range(&AnyValue::Date(days as i32)), "{days}");
            assert!(!t.temporal_in_range(&us(days)), "{days}");
        }
        for days in [-141_438, -141_427, -141_426, 0] {
            assert!(t.temporal_in_range(&AnyValue::Date(days as i32)), "{days}");
            assert!(t.temporal_in_range(&us(days)), "{days}");
        }
    }

    #[test]
    fn booleans_over_numeric_columns() {
        // #439: `id = TRUE` is `id = 1`, Polars reads every nonzero as true
        let c = |t| {
            config(
                DialectName::Duckdb,
                BackendKind::Python {
                    module: "duckdb".into(),
                },
                &[("b", DataType::Boolean, t)],
            )
        };
        let sql = |config: &DatabaseScanConfig, doc: &str| {
            let (e, class) = translate_doc(config, doc).unwrap();
            assert_eq!(class, Class::Exact);
            format!("{e:?}")
        };
        let col = r#"{"col": "b"}"#;
        let not = format!(r#"{{"op": "not", "args": [{col}]}}"#);
        for numeric in ["BIGINT", "DOUBLE", "INTEGER"] {
            let c = c(Some(numeric));
            let s = sql(&c, col);
            assert!(s.contains("NotEq") && s.contains("Int(0)"), "{s}");
            let s = sql(&c, &not);
            assert!(s.contains("Not(") && s.contains("NotEq"), "{s}");
        }
        for plain in [Some("BOOLEAN"), Some("bool"), None] {
            let s = sql(&c(plain), col);
            assert!(s.contains("Bool(true)"), "{s}");
        }
        let eq_true = cmp("eq", "b", "bool", "true");
        let s = sql(&c(Some("BIGINT")), &eq_true);
        assert!(s.contains("NotEq") && s.contains("Int(0)"), "{s}");
    }

    #[test]
    fn integers_over_temporal_columns_are_not_compared() {
        // #441: MySQL compares a DATE with a number as 20231231
        let c = |dialect, t| config(dialect, adbc("mysql"), &[("d", DataType::Int32, t)]);
        let gt = cmp("gt", "d", "i32", "19722");
        for t in ["date", "datetime(6)", "timestamp", "time"] {
            assert_eq!(
                class_of(&c(DialectName::Postgresql, Some(t)), &gt),
                None,
                "{t}"
            );
        }
        assert_eq!(
            class_of(&c(DialectName::Postgresql, Some("int")), &gt),
            Some(Class::Exact)
        );
        assert_eq!(
            class_of(&c(DialectName::Postgresql, None), &gt),
            Some(Class::Exact)
        );
    }
}
