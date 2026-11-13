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

    /// Parses a document. Parts that don't parse become [`Node::Unknown`], which is never pushed.
    pub fn parse(doc: &str) -> Self {
        match serde_json::from_str::<Value>(doc) {
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
        let connection = &self.config.options.connection;
        // or given as the driver's `execute()` argument (`parameters`, psycopg's and
        // mysql.connector's `params`, PyMySQL's `args`)
        connection.has_query_params()
            || connection
                .execute_options
                .iter()
                .any(|(k, _)| matches!(k.as_str(), "parameters" | "params" | "args"))
    }

    /// Whether queries run through SQLAlchemy: a SQLAlchemy URI, Engine or Connection.
    fn sqlalchemy(&self) -> bool {
        match &self.config.backend.kind {
            BackendKind::Sqlalchemy => true,
            BackendKind::Python { module } => module == "sqlalchemy",
            _ => false,
        }
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
        !(placeholder || escape || nul || question || quotes)
    }

    /// Whether a literal compared with the column means the same value to the database.
    /// MySQL and MariaDB read an integer 1-99 compared with a `YEAR` column as a year
    /// (2001-2069 and 1970-1999); 0 and every other integer compare as themselves. A
    /// decimal literal wider than its precision (DuckDB `HUGEINT` is read as
    /// `Decimal(38, 0)` but holds 39 digits) isn't a valid value.
    fn literal_fits(&self, name: &str, av: &AnyValue<'_>) -> bool {
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

    /// `b` is `value`. MySQL and SQLite hold booleans as integers, any of which but 0 Polars
    /// reads as true, while `TRUE` is 1.
    fn boolean_is(&self, name: &PlSmallStr, value: bool) -> SqlExpr {
        let col = SqlExpr::col(name.clone());
        if self.dialect().booleans_are_integers() {
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
            Node::And(left, right) => {
                for side in [left, right] {
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
        match node {
            Node::And(left, right) => match (self.translate(left), self.translate(right)) {
                (Some((l, lc)), Some((r, rc))) => {
                    Some((SqlExpr::binary(l, SqlBinOp::And, r), lc.min(rc)))
                },
                // Dropping one side of an AND keeps every row the AND keeps.
                (Some((e, _)), None) | (None, Some((e, _))) => Some((e, Class::Superset)),
                (None, None) => None,
            },
            Node::Or(left, right) => {
                let (l, lc) = self.translate(left)?;
                let (r, rc) = self.translate(right)?;
                Some((SqlExpr::binary(l, SqlBinOp::Or, r), lc.min(rc)))
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
            SqlExpr::col(name.clone())
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
        if dtype == &DataType::Boolean && self.dialect().booleans_are_integers() {
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
        // (a string's quotes and escapes, the separator; another literal's text)
        let bytes: usize = lits
            .iter()
            .map(|l| l.as_str().map_or(32, |s| s.len() + s.len() / 8 + 4))
            .sum();
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
                DialectName::Postgresql | DialectName::Redshift | DialectName::Duckdb
            ),
        }
    }

    /// The class of `column <op> literal`, from the column's type and semantics.
    fn leaf_class(&self, name: &str, dtype: &DataType, op: CmpOp) -> Option<Class> {
        match dtype {
            dt if dt.is_integer() => Some(Class::Exact),
            DataType::Boolean => self.dialect().supports_boolean().then_some(Class::Exact),
            // SQL Server's legacy `datetime` holds 1/300 s ticks, which the microsecond
            // literal doesn't match exactly.
            DataType::Datetime(_, _)
                if self.dialect() == DialectName::Mssql
                    && self
                        .config
                        .column(name)
                        .and_then(|c| c.db_type.as_deref())
                        .is_some_and(|t| {
                            matches!(
                                t.trim().to_ascii_lowercase().as_str(),
                                "datetime" | "smalldatetime"
                            )
                        }) =>
            {
                None
            },
            // ClickHouse parses a literal without a zone in the server's time zone, which
            // Polars doesn't know.
            DataType::Datetime(_, None) if self.dialect() == DialectName::Clickhouse => None,
            // A MySQL `TIMESTAMP` literal is read in the session time zone, UTC only after
            // the session setup.
            DataType::Datetime(_, Some(_)) if self.mysql_without_setup() => None,
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

        if !self.literal_fits(name, av) {
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
            SqlLiteral::Bool(v) if self.dialect().booleans_are_integers() => match bin_op {
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
}
