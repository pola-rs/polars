use std::fmt::Display;
use std::ops::Div;

use polars_core::export::regex;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::prelude::typed_lit;
use polars_plan::prelude::LiteralValue::Null;
use polars_time::Duration;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use regex::{Regex, RegexBuilder};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "dtype-decimal")]
use sqlparser::ast::ExactNumberInfo;
use sqlparser::ast::{
    ArrayAgg, ArrayElemTypeDef, BinaryOperator as SQLBinaryOperator, BinaryOperator, CastFormat,
    DataType as SQLDataType, DateTimeField, Expr as SQLExpr, Function as SQLFunction, Ident,
    Interval, JoinConstraint, ObjectName, OrderByExpr, Query as Subquery, SelectItem, TimezoneInfo,
    TrimWhereField, UnaryOperator, Value as SQLValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::functions::SQLFunctionVisitor;
use crate::SQLContext;

static DATE_LITERAL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
static TIME_LITERAL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();

#[inline]
#[cold]
#[must_use]
pub fn to_sql_interface_err(err: impl Display) -> PolarsError {
    PolarsError::SQLInterface(err.to_string().into())
}

fn timeunit_from_precision(prec: &Option<u64>) -> PolarsResult<TimeUnit> {
    Ok(match prec {
        None => TimeUnit::Microseconds,
        Some(n) if (1u64..=3u64).contains(n) => TimeUnit::Milliseconds,
        Some(n) if (4u64..=6u64).contains(n) => TimeUnit::Microseconds,
        Some(n) if (7u64..=9u64).contains(n) => TimeUnit::Nanoseconds,
        Some(n) => {
            polars_bail!(SQLSyntax: "invalid temporal type precision; expected 1-9, found {}", n)
        },
    })
}

pub(crate) fn map_sql_polars_datatype(data_type: &SQLDataType) -> PolarsResult<DataType> {
    Ok(match data_type {
        // ---------------------------------
        // array/list
        // ---------------------------------
        SQLDataType::Array(ArrayElemTypeDef::AngleBracket(inner_type))
        | SQLDataType::Array(ArrayElemTypeDef::SquareBracket(inner_type)) => {
            DataType::List(Box::new(map_sql_polars_datatype(inner_type)?))
        },

        // ---------------------------------
        // binary
        // ---------------------------------
        SQLDataType::Bytea
        | SQLDataType::Bytes(_)
        | SQLDataType::Binary(_)
        | SQLDataType::Blob(_)
        | SQLDataType::Varbinary(_) => DataType::Binary,

        // ---------------------------------
        // boolean
        // ---------------------------------
        SQLDataType::Boolean | SQLDataType::Bool => DataType::Boolean,

        // ---------------------------------
        // signed integer
        // ---------------------------------
        SQLDataType::Int(_) | SQLDataType::Integer(_) => DataType::Int32,
        SQLDataType::Int2(_) | SQLDataType::SmallInt(_) => DataType::Int16,
        SQLDataType::Int4(_) | SQLDataType::MediumInt(_) => DataType::Int32,
        SQLDataType::Int8(_) | SQLDataType::BigInt(_) => DataType::Int64,
        SQLDataType::TinyInt(_) => DataType::Int8,

        // ---------------------------------
        // unsigned integer: the following do not map to PostgreSQL types/syntax, but
        // are enabled for wider compatibility (eg: "CAST(col AS BIGINT UNSIGNED)").
        // ---------------------------------
        SQLDataType::UnsignedInt(_) | SQLDataType::UnsignedInteger(_) => DataType::UInt32,
        SQLDataType::UnsignedInt2(_) | SQLDataType::UnsignedSmallInt(_) => DataType::UInt16,
        SQLDataType::UnsignedInt4(_) | SQLDataType::UnsignedMediumInt(_) => DataType::UInt32,
        SQLDataType::UnsignedInt8(_) | SQLDataType::UnsignedBigInt(_) => DataType::UInt64,
        SQLDataType::UnsignedTinyInt(_) => DataType::UInt8, // see also: "custom" types below

        // ---------------------------------
        // float
        // ---------------------------------
        SQLDataType::Double | SQLDataType::DoublePrecision | SQLDataType::Float8 => {
            DataType::Float64
        },
        SQLDataType::Float(n_bytes) => match n_bytes {
            Some(n) if (1u64..=24u64).contains(n) => DataType::Float32,
            Some(n) if (25u64..=53u64).contains(n) => DataType::Float64,
            Some(n) => {
                polars_bail!(SQLSyntax: "unsupported `float` size; expected a value between 1 and 53, found {}", n)
            },
            None => DataType::Float64,
        },
        SQLDataType::Float4 | SQLDataType::Real => DataType::Float32,

        // ---------------------------------
        // decimal
        // ---------------------------------
        #[cfg(feature = "dtype-decimal")]
        SQLDataType::Dec(info) | SQLDataType::Decimal(info) | SQLDataType::Numeric(info) => {
            match *info {
                ExactNumberInfo::PrecisionAndScale(p, s) => {
                    DataType::Decimal(Some(p as usize), Some(s as usize))
                },
                ExactNumberInfo::Precision(p) => DataType::Decimal(Some(p as usize), Some(0)),
                ExactNumberInfo::None => DataType::Decimal(Some(38), Some(9)),
            }
        },

        // ---------------------------------
        // temporal
        // ---------------------------------
        SQLDataType::Date => DataType::Date,
        SQLDataType::Interval => DataType::Duration(TimeUnit::Microseconds),
        SQLDataType::Time(_, tz) => match tz {
            TimezoneInfo::None => DataType::Time,
            _ => {
                polars_bail!(SQLInterface: "`time` with timezone is not supported; found tz={}", tz)
            },
        },
        SQLDataType::Datetime(prec) => DataType::Datetime(timeunit_from_precision(prec)?, None),
        SQLDataType::Timestamp(prec, tz) => match tz {
            TimezoneInfo::None => DataType::Datetime(timeunit_from_precision(prec)?, None),
            _ => {
                polars_bail!(SQLInterface: "`timestamp` with timezone is not (yet) supported; found tz={}", tz)
            },
        },

        // ---------------------------------
        // string
        // ---------------------------------
        SQLDataType::Char(_)
        | SQLDataType::CharVarying(_)
        | SQLDataType::Character(_)
        | SQLDataType::CharacterVarying(_)
        | SQLDataType::Clob(_)
        | SQLDataType::String(_)
        | SQLDataType::Text
        | SQLDataType::Uuid
        | SQLDataType::Varchar(_) => DataType::String,

        // ---------------------------------
        // custom
        // ---------------------------------
        SQLDataType::Custom(ObjectName(idents), _) => match idents.as_slice() {
            [Ident { value, .. }] => match value.to_lowercase().as_str() {
                // these integer types are not supported by the PostgreSQL core distribution,
                // but they ARE available via `pguint` (https://github.com/petere/pguint), an
                // extension maintained by one of the PostgreSQL core developers.
                "uint1" => DataType::UInt8,
                "uint2" => DataType::UInt16,
                "uint4" | "uint" => DataType::UInt32,
                "uint8" => DataType::UInt64,
                // `pguint` also provides a 1 byte (8bit) integer type alias
                "int1" => DataType::Int8,
                _ => {
                    polars_bail!(SQLInterface: "datatype {:?} is not currently supported", value)
                },
            },
            _ => {
                polars_bail!(SQLInterface: "datatype {:?} is not currently supported", idents)
            },
        },
        _ => {
            polars_bail!(SQLInterface: "datatype {:?} is not currently supported", data_type)
        },
    })
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum SubqueryRestriction {
    // SingleValue,
    SingleColumn,
    // SingleRow,
    // Any
}

/// Recursively walks a SQL Expr to create a polars Expr
pub(crate) struct SQLExprVisitor<'a> {
    ctx: &'a mut SQLContext,
    active_schema: Option<&'a Schema>,
}

impl SQLExprVisitor<'_> {
    fn array_expr_to_series(&mut self, elements: &[SQLExpr]) -> PolarsResult<Series> {
        let array_elements = elements
            .iter()
            .map(|e| match e {
                SQLExpr::Value(v) => self.visit_any_value(v, None),
                SQLExpr::UnaryOp { op, expr } => match expr.as_ref() {
                    SQLExpr::Value(v) => self.visit_any_value(v, Some(op)),
                    _ => Err(polars_err!(SQLInterface: "expression {:?} is not yet supported", e)),
                },
                SQLExpr::Array(_) => {
                    // TODO: nested arrays (handle FnMut issues)
                    // let srs = self.array_expr_to_series(&[e.clone()])?;
                    // Ok(AnyValue::List(srs))
                    Err(polars_err!(SQLInterface: "nested array literals are not yet supported:\n{:?}", e))
                },
                _ => Err(polars_err!(SQLInterface: "expression {:?} is not yet supported", e)),
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        Series::from_any_values("", &array_elements, true)
    }

    fn visit_expr(&mut self, expr: &SQLExpr) -> PolarsResult<Expr> {
        match expr {
            SQLExpr::AllOp {
                left,
                compare_op,
                right,
            } => self.visit_all(left, compare_op, right),
            SQLExpr::AnyOp {
                left,
                compare_op,
                right,
            } => self.visit_any(left, compare_op, right),
            SQLExpr::Array(arr) => self.visit_array_expr(&arr.elem, true, None),
            SQLExpr::ArrayAgg(expr) => self.visit_arr_agg(expr),
            SQLExpr::Between {
                expr,
                negated,
                low,
                high,
            } => self.visit_between(expr, *negated, low, high),
            SQLExpr::BinaryOp { left, op, right } => self.visit_binary_op(left, op, right),
            SQLExpr::Cast {
                expr,
                data_type,
                format,
            } => self.visit_cast(expr, data_type, format, true),
            SQLExpr::Ceil { expr, .. } => Ok(self.visit_expr(expr)?.ceil()),
            SQLExpr::CompoundIdentifier(idents) => self.visit_compound_identifier(idents),
            SQLExpr::Extract { field, expr } => {
                parse_extract_date_part(self.visit_expr(expr)?, field)
            },
            SQLExpr::Floor { expr, .. } => Ok(self.visit_expr(expr)?.floor()),
            SQLExpr::Function(function) => self.visit_function(function),
            SQLExpr::Identifier(ident) => self.visit_identifier(ident),
            SQLExpr::InList {
                expr,
                list,
                negated,
            } => {
                let expr = self.visit_expr(expr)?;
                let elems = self.visit_array_expr(list, false, Some(&expr))?;
                let is_in = expr.is_in(elems);
                Ok(if *negated { is_in.not() } else { is_in })
            },
            SQLExpr::InSubquery {
                expr,
                subquery,
                negated,
            } => self.visit_in_subquery(expr, subquery, *negated),
            SQLExpr::Interval(interval) => self.visit_interval(interval),
            SQLExpr::IsDistinctFrom(e1, e2) => {
                Ok(self.visit_expr(e1)?.neq_missing(self.visit_expr(e2)?))
            },
            SQLExpr::IsFalse(expr) => Ok(self.visit_expr(expr)?.eq(lit(false))),
            SQLExpr::IsNotDistinctFrom(e1, e2) => {
                Ok(self.visit_expr(e1)?.eq_missing(self.visit_expr(e2)?))
            },
            SQLExpr::IsNotFalse(expr) => Ok(self.visit_expr(expr)?.eq(lit(false)).not()),
            SQLExpr::IsNotNull(expr) => Ok(self.visit_expr(expr)?.is_not_null()),
            SQLExpr::IsNotTrue(expr) => Ok(self.visit_expr(expr)?.eq(lit(true)).not()),
            SQLExpr::IsNull(expr) => Ok(self.visit_expr(expr)?.is_null()),
            SQLExpr::IsTrue(expr) => Ok(self.visit_expr(expr)?.eq(lit(true))),
            SQLExpr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => self.visit_like(*negated, expr, pattern, escape_char, false),
            SQLExpr::ILike {
                negated,
                expr,
                pattern,
                escape_char,
            } => self.visit_like(*negated, expr, pattern, escape_char, true),
            SQLExpr::Nested(expr) => self.visit_expr(expr),
            SQLExpr::Position { expr, r#in } => Ok(
                // note: SQL is 1-indexed, not 0-indexed
                (self
                    .visit_expr(r#in)?
                    .str()
                    .find(self.visit_expr(expr)?, true)
                    + typed_lit(1u32))
                .fill_null(typed_lit(0u32)),
            ),
            SQLExpr::RLike {
                // note: parses both RLIKE and REGEXP
                negated,
                expr,
                pattern,
                regexp: _,
            } => {
                let matches = self
                    .visit_expr(expr)?
                    .str()
                    .contains(self.visit_expr(pattern)?, true);
                Ok(if *negated { matches.not() } else { matches })
            },
            SQLExpr::Subquery(_) => polars_bail!(SQLInterface: "unexpected subquery"),
            SQLExpr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => self.visit_trim(expr, trim_where, trim_what, trim_characters),
            SQLExpr::TryCast {
                expr,
                data_type,
                format,
            } => self.visit_cast(expr, data_type, format, false),
            SQLExpr::UnaryOp { op, expr } => self.visit_unary_op(op, expr),
            SQLExpr::Value(value) => self.visit_literal(value),
            e @ SQLExpr::Case { .. } => self.visit_case_when_then(e),
            other => {
                polars_bail!(SQLInterface: "expression {:?} is not yet supported", other)
            },
        }
    }

    fn visit_subquery(
        &mut self,
        subquery: &Subquery,
        restriction: SubqueryRestriction,
    ) -> PolarsResult<Expr> {
        if subquery.with.is_some() {
            polars_bail!(SQLSyntax: "SQL subquery cannot be a CTE `with` clause");
        }

        let mut lf = self.ctx.execute_query_no_ctes(subquery)?;
        let schema = lf.schema_with_arenas(&mut self.ctx.lp_arena, &mut self.ctx.expr_arena)?;

        if restriction == SubqueryRestriction::SingleColumn {
            if schema.len() != 1 {
                polars_bail!(SQLSyntax: "SQL subquery returns more than one column");
            }
            let rand_string: String = thread_rng()
                .sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect();

            let schema_entry = schema.get_at_index(0);
            if let Some((old_name, _)) = schema_entry {
                let new_name = String::from(old_name.as_str()) + rand_string.as_str();
                lf = lf.rename([old_name.to_string()], [new_name.clone()]);
                return Ok(Expr::SubPlan(
                    SpecialEq::new(Arc::new(lf.logical_plan)),
                    vec![new_name],
                ));
            }
        };
        polars_bail!(SQLInterface: "subquery type not supported");
    }

    /// Visit a single SQL identifier.
    ///
    /// e.g. column
    fn visit_identifier(&self, ident: &Ident) -> PolarsResult<Expr> {
        Ok(col(&ident.value))
    }

    /// Visit a compound SQL identifier
    ///
    /// e.g. df.column or "df"."column"
    fn visit_compound_identifier(&mut self, idents: &[Ident]) -> PolarsResult<Expr> {
        match idents {
            [tbl_name, column_name] => {
                let mut lf = self
                    .ctx
                    .get_table_from_current_scope(&tbl_name.value)
                    .ok_or_else(|| {
                        polars_err!(
                            SQLInterface: "no table or alias named '{}' found",
                            tbl_name
                        )
                    })?;

                let schema =
                    lf.schema_with_arenas(&mut self.ctx.lp_arena, &mut self.ctx.expr_arena)?;
                if let Some((_, name, _)) = schema.get_full(&column_name.value) {
                    let resolved = &self.ctx.resolve_name(&tbl_name.value, &column_name.value);
                    Ok(if name != resolved {
                        col(resolved).alias(name)
                    } else {
                        col(name)
                    })
                } else {
                    polars_bail!(
                        SQLInterface: "no column named '{}' found in table '{}'",
                        column_name,
                        tbl_name
                    )
                }
            },
            _ => polars_bail!(
                SQLInterface: "invalid identifier {:?}",
                idents
            ),
        }
    }

    fn visit_interval(&self, interval: &Interval) -> PolarsResult<Expr> {
        if interval.last_field.is_some()
            || interval.leading_field.is_some()
            || interval.leading_precision.is_some()
            || interval.fractional_seconds_precision.is_some()
        {
            polars_bail!(SQLSyntax: "unsupported interval syntax: '{}'", interval)
        }
        let s = match &*interval.value {
            SQLExpr::UnaryOp { .. } => {
                polars_bail!(SQLSyntax: "unary ops are not valid on interval strings; found {}", interval.value)
            },
            SQLExpr::Value(SQLValue::SingleQuotedString(s)) => Some(s),
            _ => None,
        };
        match s {
            Some(s) if s.contains('-') => {
                polars_bail!(SQLInterface: "minus signs are not yet supported in interval strings; found '{}'", s)
            },
            Some(s) => Ok(lit(Duration::parse_interval(s))),
            None => polars_bail!(SQLSyntax: "invalid interval {:?}", interval),
        }
    }

    fn visit_like(
        &mut self,
        negated: bool,
        expr: &SQLExpr,
        pattern: &SQLExpr,
        escape_char: &Option<char>,
        case_insensitive: bool,
    ) -> PolarsResult<Expr> {
        if escape_char.is_some() {
            polars_bail!(SQLInterface: "ESCAPE char for LIKE/ILIKE is not yet supported; found '{}'", escape_char.unwrap());
        }
        let pat = match self.visit_expr(pattern) {
            Ok(Expr::Literal(LiteralValue::String(s))) => s,
            _ => {
                polars_bail!(SQLSyntax: "LIKE/ILIKE pattern must be a string literal; found {}", pattern)
            },
        };
        if pat.is_empty() || (!case_insensitive && pat.chars().all(|c| !matches!(c, '%' | '_'))) {
            // empty string or other exact literal match (eg: no wildcard chars)
            let op = if negated {
                BinaryOperator::NotEq
            } else {
                BinaryOperator::Eq
            };
            self.visit_binary_op(expr, &op, pattern)
        } else {
            // create regex from pattern containing SQL wildcard chars ('%' => '.*', '_' => '.')
            let mut rx = regex::escape(pat.as_str())
                .replace('%', ".*")
                .replace('_', ".");

            rx = format!("^{}{}$", if case_insensitive { "(?i)" } else { "" }, rx);

            let expr = self.visit_expr(expr)?;
            let matches = expr.str().contains(lit(rx), true);
            Ok(if negated { matches.not() } else { matches })
        }
    }

    /// Handle implicit temporal string comparisons.
    ///
    /// eg: "dt >= '2024-04-30'", or "dtm::date = '2077-10-10'"
    fn convert_temporal_strings(&mut self, left: &Expr, right: &Expr) -> Expr {
        if let (Some(name), Some(s), expr_dtype) = match (left, right) {
            // identify "col <op> string" expressions
            (Expr::Column(name), Expr::Literal(LiteralValue::String(s))) => {
                (Some(name.clone()), Some(s), None)
            },
            // identify "CAST(expr AS type) <op> string" and/or "expr::type <op> string" expressions
            (
                Expr::Cast {
                    expr, data_type, ..
                },
                Expr::Literal(LiteralValue::String(s)),
            ) => {
                if let Expr::Column(name) = &**expr {
                    (Some(name.clone()), Some(s), Some(data_type))
                } else {
                    (None, Some(s), Some(data_type))
                }
            },
            _ => (None, None, None),
        } {
            if expr_dtype.is_none() && self.active_schema.is_none() {
                right.clone()
            } else {
                let left_dtype = expr_dtype
                    .unwrap_or_else(|| self.active_schema.as_ref().unwrap().get(&name).unwrap());

                let dt_regex = DATE_LITERAL_RE
                    .get_or_init(|| RegexBuilder::new(r"^\d{4}-[01]\d-[0-3]\d").build().unwrap());
                let tm_regex = TIME_LITERAL_RE.get_or_init(|| {
                    RegexBuilder::new(r"^[012]\d:[0-5]\d:[0-5]\d")
                        .build()
                        .unwrap()
                });

                match left_dtype {
                    DataType::Time if tm_regex.is_match(s) => {
                        right.clone().strict_cast(left_dtype.clone())
                    },
                    DataType::Date if dt_regex.is_match(s) => {
                        right.clone().strict_cast(left_dtype.clone())
                    },
                    DataType::Datetime(_, _) if dt_regex.is_match(s) => {
                        if s.len() == 10 {
                            // handle upcast from ISO date string (10 chars) to datetime
                            lit(format!("{}T00:00:00", s)).strict_cast(left_dtype.clone())
                        } else {
                            lit(s.replacen(' ', "T", 1)).strict_cast(left_dtype.clone())
                        }
                    },
                    _ => right.clone(),
                }
            }
        } else {
            right.clone()
        }
    }

    /// Visit a SQL binary operator.
    ///
    /// e.g. "column + 1", "column1 <= column2"
    fn visit_binary_op(
        &mut self,
        left: &SQLExpr,
        op: &BinaryOperator,
        right: &SQLExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let mut right = self.visit_expr(right)?;
        right = self.convert_temporal_strings(&left, &right);

        Ok(match op {
            SQLBinaryOperator::And => left.and(right),
            SQLBinaryOperator::Divide => left / right,
            SQLBinaryOperator::DuckIntegerDivide => left.floor_div(right).cast(DataType::Int64),
            SQLBinaryOperator::Eq => left.eq(right),
            SQLBinaryOperator::Gt => left.gt(right),
            SQLBinaryOperator::GtEq => left.gt_eq(right),
            SQLBinaryOperator::Lt => left.lt(right),
            SQLBinaryOperator::LtEq => left.lt_eq(right),
            SQLBinaryOperator::Minus => left - right,
            SQLBinaryOperator::Modulo => left % right,
            SQLBinaryOperator::Multiply => left * right,
            SQLBinaryOperator::NotEq => left.eq(right).not(),
            SQLBinaryOperator::Or => left.or(right),
            SQLBinaryOperator::Plus => left + right,
            SQLBinaryOperator::Spaceship => left.eq_missing(right),
            SQLBinaryOperator::StringConcat => {
                left.cast(DataType::String) + right.cast(DataType::String)
            },
            SQLBinaryOperator::Xor => left.xor(right),
            // ----
            // Regular expression operators
            // ----
            SQLBinaryOperator::PGRegexMatch => match right {
                Expr::Literal(LiteralValue::String(_)) => left.str().contains(right, true),
                _ => polars_bail!(SQLSyntax: "invalid pattern for '~' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexNotMatch => match right {
                Expr::Literal(LiteralValue::String(_)) => left.str().contains(right, true).not(),
                _ => polars_bail!(SQLSyntax: "invalid pattern for '!~' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexIMatch => match right {
                Expr::Literal(LiteralValue::String(pat)) => {
                    left.str().contains(lit(format!("(?i){}", pat)), true)
                },
                _ => polars_bail!(SQLSyntax: "invalid pattern for '~*' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexNotIMatch => match right {
                Expr::Literal(LiteralValue::String(pat)) => {
                    left.str().contains(lit(format!("(?i){}", pat)), true).not()
                },
                _ => {
                    polars_bail!(SQLSyntax: "invalid pattern for '!~*' operator: {:?}", right)
                },
            },
            other => {
                polars_bail!(SQLInterface: "SQL operator {:?} is not yet supported", other)
            },
        })
    }

    /// Visit a SQL unary operator.
    ///
    /// e.g. +column or -column
    fn visit_unary_op(&mut self, op: &UnaryOperator, expr: &SQLExpr) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        Ok(match (op, expr.clone()) {
            // simplify the parse tree by special-casing common unary +/- ops
            (UnaryOperator::Plus, Expr::Literal(LiteralValue::Int(n))) => lit(n),
            (UnaryOperator::Plus, Expr::Literal(LiteralValue::Float(n))) => lit(n),
            (UnaryOperator::Minus, Expr::Literal(LiteralValue::Int(n))) => lit(-n),
            (UnaryOperator::Minus, Expr::Literal(LiteralValue::Float(n))) => lit(-n),
            // general case
            (UnaryOperator::Plus, _) => lit(0) + expr,
            (UnaryOperator::Minus, _) => lit(0) - expr,
            (UnaryOperator::Not, _) => expr.not(),
            other => polars_bail!(SQLInterface: "unary operator {:?} is not supported", other),
        })
    }

    /// Visit a SQL function.
    ///
    /// e.g. SUM(column) or COUNT(*)
    ///
    /// See [SQLFunctionVisitor] for more details
    fn visit_function(&mut self, function: &SQLFunction) -> PolarsResult<Expr> {
        let mut visitor = SQLFunctionVisitor {
            func: function,
            ctx: self.ctx,
        };
        visitor.visit_function()
    }

    /// Visit a SQL `ALL` expression.
    ///
    /// e.g. `a > ALL(y)`
    fn visit_all(
        &mut self,
        left: &SQLExpr,
        compare_op: &BinaryOperator,
        right: &SQLExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let right = self.visit_expr(right)?;

        match compare_op {
            BinaryOperator::Gt => Ok(left.gt(right.max())),
            BinaryOperator::Lt => Ok(left.lt(right.min())),
            BinaryOperator::GtEq => Ok(left.gt_eq(right.max())),
            BinaryOperator::LtEq => Ok(left.lt_eq(right.min())),
            BinaryOperator::Eq => polars_bail!(SQLSyntax: "ALL cannot be used with ="),
            BinaryOperator::NotEq => polars_bail!(SQLSyntax: "ALL cannot be used with !="),
            _ => polars_bail!(SQLInterface: "invalid comparison operator"),
        }
    }

    /// Visit a SQL `ANY` expression.
    ///
    /// e.g. `a != ANY(y)`
    fn visit_any(
        &mut self,
        left: &SQLExpr,
        compare_op: &BinaryOperator,
        right: &SQLExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let right = self.visit_expr(right)?;

        match compare_op {
            BinaryOperator::Gt => Ok(left.gt(right.min())),
            BinaryOperator::Lt => Ok(left.lt(right.max())),
            BinaryOperator::GtEq => Ok(left.gt_eq(right.min())),
            BinaryOperator::LtEq => Ok(left.lt_eq(right.max())),
            BinaryOperator::Eq => Ok(left.is_in(right)),
            BinaryOperator::NotEq => Ok(left.is_in(right).not()),
            _ => polars_bail!(SQLInterface: "invalid comparison operator"),
        }
    }

    /// Visit a SQL `ARRAY` list (including `IN` values).
    fn visit_array_expr(
        &mut self,
        elements: &[SQLExpr],
        result_as_element: bool,
        dtype_expr_match: Option<&Expr>,
    ) -> PolarsResult<Expr> {
        let mut elems = self.array_expr_to_series(elements)?;

        // handle implicit temporal strings, eg: "dt IN ('2024-04-30','2024-05-01')".
        // (not yet as versatile as the temporal string conversions in visit_binary_op)
        if let (Some(Expr::Column(name)), Some(schema)) =
            (dtype_expr_match, self.active_schema.as_ref())
        {
            if elems.dtype() == &DataType::String {
                if let Some(DataType::Date | DataType::Time | DataType::Datetime(_, _)) =
                    schema.get(name)
                {
                    elems = elems.strict_cast(&schema.get(name).unwrap().clone())?;
                }
            }
        }
        // if we are parsing the list as an element in a series, implode.
        // otherwise, return the series as-is.
        let res = if result_as_element {
            elems.implode()?.into_series()
        } else {
            elems
        };
        Ok(lit(res))
    }

    /// Visit a SQL `CAST` or `TRY_CAST` expression.
    ///
    /// e.g. `CAST(col AS INT)`, `col::int4`, or `TRY_CAST(col AS VARCHAR)`,
    fn visit_cast(
        &mut self,
        expr: &SQLExpr,
        data_type: &SQLDataType,
        format: &Option<CastFormat>,
        strict: bool,
    ) -> PolarsResult<Expr> {
        if format.is_some() {
            return Err(
                polars_err!(SQLInterface: "use of FORMAT is not currently supported in CAST"),
            );
        }
        let expr = self.visit_expr(expr)?;

        #[cfg(feature = "json")]
        if data_type == &SQLDataType::JSON {
            return Ok(expr.str().json_decode(None, None));
        }
        let polars_type = map_sql_polars_datatype(data_type)?;
        Ok(if strict {
            expr.strict_cast(polars_type)
        } else {
            expr.cast(polars_type)
        })
    }

    /// Visit a SQL literal.
    ///
    /// e.g. 1, 'foo', 1.0, NULL
    ///
    /// See [SQLValue] and [LiteralValue] for more details
    fn visit_literal(&self, value: &SQLValue) -> PolarsResult<Expr> {
        Ok(match value {
            SQLValue::Boolean(b) => lit(*b),
            SQLValue::DoubleQuotedString(s) => lit(s.clone()),
            #[cfg(feature = "binary_encoding")]
            SQLValue::HexStringLiteral(x) => {
                if x.len() % 2 != 0 {
                    polars_bail!(SQLSyntax: "hex string literal must have an even number of digits; found '{}'", x)
                };
                lit(hex::decode(x.clone()).unwrap())
            },
            SQLValue::Null => Expr::Literal(LiteralValue::Null),
            SQLValue::Number(s, _) => {
                // Check for existence of decimal separator dot
                if s.contains('.') {
                    s.parse::<f64>().map(lit).map_err(|_| ())
                } else {
                    s.parse::<i64>().map(lit).map_err(|_| ())
                }
                .map_err(|_| polars_err!(SQLInterface: "cannot parse literal: {:?}", s))?
            },
            SQLValue::SingleQuotedByteStringLiteral(b) => {
                // note: for PostgreSQL this represents a BIT string literal (eg: b'10101') not a BYTE string
                // literal (see https://www.postgresql.org/docs/current/datatype-bit.html), but sqlparser
                // patterned the token name after BigQuery (where b'str' really IS a byte string)
                bitstring_to_bytes_literal(b)?
            },
            SQLValue::SingleQuotedString(s) => lit(s.clone()),
            other => polars_bail!(SQLInterface: "SQL value {:?} is not supported", other),
        })
    }

    /// Visit a SQL literal (like [visit_literal]), but return AnyValue instead of Expr.
    fn visit_any_value(
        &self,
        value: &SQLValue,
        op: Option<&UnaryOperator>,
    ) -> PolarsResult<AnyValue> {
        Ok(match value {
            SQLValue::Boolean(b) => AnyValue::Boolean(*b),
            SQLValue::Null => AnyValue::Null,
            SQLValue::Number(s, _) => {
                let negate = match op {
                    Some(UnaryOperator::Minus) => true,
                    // no op should be taken as plus.
                    Some(UnaryOperator::Plus) | None => false,
                    Some(op) => {
                        polars_bail!(SQLInterface: "unary op {:?} not supported for numeric SQL value", op)
                    },
                };
                // Check for existence of decimal separator dot
                if s.contains('.') {
                    s.parse::<f64>()
                        .map(|n: f64| AnyValue::Float64(if negate { -n } else { n }))
                        .map_err(|_| ())
                } else {
                    s.parse::<i64>()
                        .map(|n: i64| AnyValue::Int64(if negate { -n } else { n }))
                        .map_err(|_| ())
                }
                .map_err(|_| polars_err!(SQLInterface: "cannot parse literal: {:?}", s))?
            },
            #[cfg(feature = "binary_encoding")]
            SQLValue::HexStringLiteral(x) => {
                if x.len() % 2 != 0 {
                    polars_bail!(SQLSyntax: "hex string literal must have an even number of digits; found '{}'", x)
                };
                AnyValue::BinaryOwned(hex::decode(x.clone()).unwrap())
            },
            SQLValue::SingleQuotedByteStringLiteral(b) => {
                // note: for PostgreSQL this represents a BIT literal (eg: b'10101') not BYTE
                let bytes_literal = bitstring_to_bytes_literal(b)?;
                match bytes_literal {
                    Expr::Literal(LiteralValue::Binary(v)) => AnyValue::BinaryOwned(v.to_vec()),
                    _ => {
                        polars_bail!(SQLInterface: "failed to parse bitstring literal: {:?}", b)
                    },
                }
            },
            SQLValue::SingleQuotedString(s) | SQLValue::DoubleQuotedString(s) => {
                AnyValue::StringOwned(s.into())
            },
            other => polars_bail!(SQLInterface: "SQL value {:?} is not yet supported", other),
        })
    }

    /// Visit a SQL `BETWEEN` expression.
    /// See [sqlparser::ast::Expr::Between] for more details
    fn visit_between(
        &mut self,
        expr: &SQLExpr,
        negated: bool,
        low: &SQLExpr,
        high: &SQLExpr,
    ) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        let low = self.visit_expr(low)?;
        let high = self.visit_expr(high)?;

        let low = self.convert_temporal_strings(&expr, &low);
        let high = self.convert_temporal_strings(&expr, &high);
        if negated {
            Ok(expr.clone().lt(low).or(expr.gt(high)))
        } else {
            Ok(expr.clone().gt_eq(low).and(expr.lt_eq(high)))
        }
    }

    /// Visit a SQL `TRIM` function.
    /// See [sqlparser::ast::Expr::Trim] for more details
    fn visit_trim(
        &mut self,
        expr: &SQLExpr,
        trim_where: &Option<TrimWhereField>,
        trim_what: &Option<Box<SQLExpr>>,
        trim_characters: &Option<Vec<SQLExpr>>,
    ) -> PolarsResult<Expr> {
        if trim_characters.is_some() {
            // TODO: allow compact snowflake/bigquery syntax?
            return Err(polars_err!(SQLSyntax: "unsupported TRIM syntax (custom chars)"));
        };
        let expr = self.visit_expr(expr)?;
        let trim_what = trim_what.as_ref().map(|e| self.visit_expr(e)).transpose()?;
        let trim_what = match trim_what {
            Some(Expr::Literal(LiteralValue::String(val))) => Some(val),
            None => None,
            _ => return self.err(&expr),
        };
        Ok(match (trim_where, trim_what) {
            (None | Some(TrimWhereField::Both), None) => expr.str().strip_chars(lit(Null)),
            (None | Some(TrimWhereField::Both), Some(val)) => expr.str().strip_chars(lit(val)),
            (Some(TrimWhereField::Leading), None) => expr.str().strip_chars_start(lit(Null)),
            (Some(TrimWhereField::Leading), Some(val)) => expr.str().strip_chars_start(lit(val)),
            (Some(TrimWhereField::Trailing), None) => expr.str().strip_chars_end(lit(Null)),
            (Some(TrimWhereField::Trailing), Some(val)) => expr.str().strip_chars_end(lit(val)),
        })
    }

    /// Visit a SQL `ARRAY_AGG` expression.
    fn visit_arr_agg(&mut self, expr: &ArrayAgg) -> PolarsResult<Expr> {
        let mut base = self.visit_expr(&expr.expr)?;
        if let Some(order_by) = expr.order_by.as_ref() {
            let (order_by, descending) = self.visit_order_by(order_by)?;
            base = base.sort_by(
                order_by,
                SortMultipleOptions::default().with_order_descending_multi(descending),
            );
        }
        if let Some(limit) = &expr.limit {
            let limit = match self.visit_expr(limit)? {
                Expr::Literal(LiteralValue::Int(n)) if n >= 0 => n as usize,
                _ => polars_bail!(SQLSyntax: "limit in ARRAY_AGG must be a positive integer"),
            };
            base = base.head(Some(limit));
        }
        if expr.distinct {
            base = base.unique_stable();
        }
        polars_ensure!(
            !expr.within_group,
            SQLInterface: "ARRAY_AGG WITHIN GROUP is not yet supported"
        );
        Ok(base.implode())
    }

    /// Visit a SQL subquery inside and `IN` expression.
    fn visit_in_subquery(
        &mut self,
        expr: &SQLExpr,
        subquery: &Subquery,
        negated: bool,
    ) -> PolarsResult<Expr> {
        let subquery_result = self.visit_subquery(subquery, SubqueryRestriction::SingleColumn)?;
        let expr = self.visit_expr(expr)?;
        if negated {
            Ok(expr.is_in(subquery_result).not())
        } else {
            Ok(expr.is_in(subquery_result))
        }
    }

    /// Visit a SQL `ORDER BY` expression.
    fn visit_order_by(&mut self, order_by: &[OrderByExpr]) -> PolarsResult<(Vec<Expr>, Vec<bool>)> {
        let mut expr = Vec::with_capacity(order_by.len());
        let mut descending = Vec::with_capacity(order_by.len());
        for order_by_expr in order_by {
            let e = self.visit_expr(&order_by_expr.expr)?;
            expr.push(e);
            let desc = order_by_expr.asc.unwrap_or(false);
            descending.push(desc);
        }

        Ok((expr, descending))
    }

    /// Visit `CASE` control flow expression.
    fn visit_case_when_then(&mut self, expr: &SQLExpr) -> PolarsResult<Expr> {
        if let SQLExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } = expr
        {
            polars_ensure!(
                conditions.len() == results.len(),
                SQLSyntax: "WHEN and THEN expressions must have the same length"
            );
            polars_ensure!(
                !conditions.is_empty(),
                SQLSyntax: "WHEN and THEN expressions must have at least one element"
            );

            let mut when_thens = conditions.iter().zip(results.iter());
            let first = when_thens.next();
            if first.is_none() {
                polars_bail!(SQLSyntax: "WHEN and THEN expressions must have at least one element");
            }
            let else_res = match else_result {
                Some(else_res) => self.visit_expr(else_res)?,
                None => polars_bail!(SQLSyntax: "ELSE expression is required"),
            };
            if let Some(operand_expr) = operand {
                let first_operand_expr = self.visit_expr(operand_expr)?;

                let first = first.unwrap();
                let first_cond = first_operand_expr.eq(self.visit_expr(first.0)?);
                let first_then = self.visit_expr(first.1)?;
                let expr = when(first_cond).then(first_then);
                let next = when_thens.next();

                let mut when_then = if let Some((cond, res)) = next {
                    let second_operand_expr = self.visit_expr(operand_expr)?;
                    let cond = second_operand_expr.eq(self.visit_expr(cond)?);
                    let res = self.visit_expr(res)?;
                    expr.when(cond).then(res)
                } else {
                    return Ok(expr.otherwise(else_res));
                };
                for (cond, res) in when_thens {
                    let new_operand_expr = self.visit_expr(operand_expr)?;
                    let cond = new_operand_expr.eq(self.visit_expr(cond)?);
                    let res = self.visit_expr(res)?;
                    when_then = when_then.when(cond).then(res);
                }
                return Ok(when_then.otherwise(else_res));
            }

            let first = first.unwrap();
            let first_cond = self.visit_expr(first.0)?;
            let first_then = self.visit_expr(first.1)?;
            let expr = when(first_cond).then(first_then);
            let next = when_thens.next();

            let mut when_then = if let Some((cond, res)) = next {
                let cond = self.visit_expr(cond)?;
                let res = self.visit_expr(res)?;
                expr.when(cond).then(res)
            } else {
                return Ok(expr.otherwise(else_res));
            };
            for (cond, res) in when_thens {
                let cond = self.visit_expr(cond)?;
                let res = self.visit_expr(res)?;
                when_then = when_then.when(cond).then(res);
            }
            Ok(when_then.otherwise(else_res))
        } else {
            unreachable!()
        }
    }

    fn err(&self, expr: &Expr) -> PolarsResult<Expr> {
        polars_bail!(SQLInterface: "expression {:?} is not yet supported", expr);
    }
}

fn collect_compound_identifiers(
    left: &[Ident],
    right: &[Ident],
    left_name: &str,
    right_name: &str,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    if left.len() == 2 && right.len() == 2 {
        let (tbl_a, col_a) = (&left[0].value, &left[1].value);
        let (tbl_b, col_b) = (&right[0].value, &right[1].value);

        // switch left/right operands if the caller has them in reverse
        if left_name == tbl_b || right_name == tbl_a {
            Ok((vec![col(col_b)], vec![col(col_a)]))
        } else {
            Ok((vec![col(col_a)], vec![col(col_b)]))
        }
    } else {
        polars_bail!(SQLInterface: "collect_compound_identifiers: Expected left.len() == 2 && right.len() == 2, but found left.len() == {:?}, right.len() == {:?}", left.len(), right.len());
    }
}

fn process_join_on(
    expression: &sqlparser::ast::Expr,
    left_name: &str,
    right_name: &str,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    if let SQLExpr::BinaryOp { left, op, right } = expression {
        match *op {
            BinaryOperator::Eq => {
                if let (SQLExpr::CompoundIdentifier(left), SQLExpr::CompoundIdentifier(right)) =
                    (left.as_ref(), right.as_ref())
                {
                    collect_compound_identifiers(left, right, left_name, right_name)
                } else {
                    polars_bail!(SQLInterface: "JOIN clauses support '=' constraints on identifiers; found lhs={:?}, rhs={:?}", left, right);
                }
            },
            BinaryOperator::And => {
                let (mut left_i, mut right_i) = process_join_on(left, left_name, right_name)?;
                let (mut left_j, mut right_j) = process_join_on(right, left_name, right_name)?;
                left_i.append(&mut left_j);
                right_i.append(&mut right_j);
                Ok((left_i, right_i))
            },
            _ => {
                polars_bail!(SQLInterface: "JOIN clauses support '=' constraints combined with 'AND'; found op = '{:?}'", op);
            },
        }
    } else if let SQLExpr::Nested(expr) = expression {
        process_join_on(expr, left_name, right_name)
    } else {
        polars_bail!(SQLInterface: "JOIN clauses support '=' constraints combined with 'AND'; found expression = {:?}", expression);
    }
}

pub(super) fn process_join_constraint(
    constraint: &JoinConstraint,
    left_name: &str,
    right_name: &str,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    if let JoinConstraint::On(SQLExpr::BinaryOp { left, op, right }) = constraint {
        if op == &BinaryOperator::And {
            let (mut left_on, mut right_on) = process_join_on(left, left_name, right_name)?;
            let (left_on_, right_on_) = process_join_on(right, left_name, right_name)?;
            left_on.extend(left_on_);
            right_on.extend(right_on_);
            return Ok((left_on, right_on));
        }
        if op != &BinaryOperator::Eq {
            polars_bail!(SQLInterface:
                "only equi-join constraints are supported; found '{:?}' op in\n{:?}", op, constraint)
        }
        match (left.as_ref(), right.as_ref()) {
            (SQLExpr::CompoundIdentifier(left), SQLExpr::CompoundIdentifier(right)) => {
                return collect_compound_identifiers(left, right, left_name, right_name);
            },
            (SQLExpr::Identifier(left), SQLExpr::Identifier(right)) => {
                return Ok((vec![col(&left.value)], vec![col(&right.value)]))
            },
            _ => {},
        }
    }
    if let JoinConstraint::Using(idents) = constraint {
        if !idents.is_empty() {
            let using: Vec<Expr> = idents.iter().map(|id| col(&id.value)).collect();
            return Ok((using.clone(), using.clone()));
        }
    }
    polars_bail!(SQLInterface: "unsupported SQL join constraint:\n{:?}", constraint);
}

/// parse a SQL expression to a polars expression
/// # Example
/// ```rust
/// # use polars_sql::{SQLContext, sql_expr};
/// # use polars_core::prelude::*;
/// # use polars_lazy::prelude::*;
/// # fn main() {
///
/// let mut ctx = SQLContext::new();
/// let df = df! {
///    "a" =>  [1, 2, 3],
/// }
/// .unwrap();
/// let expr = sql_expr("MAX(a)").unwrap();
/// df.lazy().select(vec![expr]).collect().unwrap();
/// # }
/// ```
pub fn sql_expr<S: AsRef<str>>(s: S) -> PolarsResult<Expr> {
    let mut ctx = SQLContext::new();

    let mut parser = Parser::new(&GenericDialect);
    parser = parser.with_options(ParserOptions {
        trailing_commas: true,
        ..Default::default()
    });

    let mut ast = parser
        .try_with_sql(s.as_ref())
        .map_err(to_sql_interface_err)?;
    let expr = ast.parse_select_item().map_err(to_sql_interface_err)?;

    Ok(match &expr {
        SelectItem::ExprWithAlias { expr, alias } => {
            let expr = parse_sql_expr(expr, &mut ctx, None)?;
            expr.alias(&alias.value)
        },
        SelectItem::UnnamedExpr(expr) => parse_sql_expr(expr, &mut ctx, None)?,
        _ => polars_bail!(SQLInterface: "unable to parse '{}' as Expr", s.as_ref()),
    })
}

pub(crate) fn parse_sql_expr(
    expr: &SQLExpr,
    ctx: &mut SQLContext,
    active_schema: Option<&Schema>,
) -> PolarsResult<Expr> {
    let mut visitor = SQLExprVisitor { ctx, active_schema };
    visitor.visit_expr(expr)
}

pub(crate) fn parse_sql_array(expr: &SQLExpr, ctx: &mut SQLContext) -> PolarsResult<Series> {
    match expr {
        SQLExpr::Array(arr) => {
            let mut visitor = SQLExprVisitor {
                ctx,
                active_schema: None,
            };
            visitor.array_expr_to_series(arr.elem.as_slice())
        },
        _ => polars_bail!(SQLSyntax: "Expected array expression, found {:?}", expr),
    }
}

pub(crate) fn parse_extract_date_part(expr: Expr, field: &DateTimeField) -> PolarsResult<Expr> {
    let field = match field {
        // handle 'DATE_PART' and all valid abbreviations/alternates
        DateTimeField::Custom(Ident { value, .. }) => {
            let value = value.to_ascii_lowercase();
            match value.as_str() {
                "millennium" | "millennia" => &DateTimeField::Millennium,
                "century" | "centuries" => &DateTimeField::Century,
                "decade" | "decades" => &DateTimeField::Decade,
                "isoyear" => &DateTimeField::Isoyear,
                "year" | "years" | "y" => &DateTimeField::Year,
                "quarter" | "quarters" => &DateTimeField::Quarter,
                "month" | "months" | "mon" | "mons" => &DateTimeField::Month,
                "dayofyear" | "doy" => &DateTimeField::DayOfYear,
                "dayofweek" | "dow" => &DateTimeField::DayOfWeek,
                "isoweek" | "week" | "weeks" => &DateTimeField::IsoWeek,
                "isodow" => &DateTimeField::Isodow,
                "day" | "days" | "d" => &DateTimeField::Day,
                "hour" | "hours" | "h" => &DateTimeField::Hour,
                "minute" | "minutes" | "mins" | "min" | "m" => &DateTimeField::Minute,
                "second" | "seconds" | "sec" | "secs" | "s" => &DateTimeField::Second,
                "millisecond" | "milliseconds" | "ms" => &DateTimeField::Millisecond,
                "microsecond" | "microseconds" | "us" => &DateTimeField::Microsecond,
                "nanosecond" | "nanoseconds" | "ns" => &DateTimeField::Nanosecond,
                #[cfg(feature = "timezones")]
                "timezone" => &DateTimeField::Timezone,
                "time" => &DateTimeField::Time,
                "epoch" => &DateTimeField::Epoch,
                _ => {
                    polars_bail!(SQLSyntax: "EXTRACT/DATE_PART does not support '{}' part", value)
                },
            }
        },
        _ => field,
    };
    Ok(match field {
        DateTimeField::Millennium => expr.dt().millennium(),
        DateTimeField::Century => expr.dt().century(),
        DateTimeField::Decade => expr.dt().year() / typed_lit(10i32),
        DateTimeField::Isoyear => expr.dt().iso_year(),
        DateTimeField::Year => expr.dt().year(),
        DateTimeField::Quarter => expr.dt().quarter(),
        DateTimeField::Month => expr.dt().month(),
        DateTimeField::Week(weekday) => {
            if weekday.is_some() {
                polars_bail!(SQLSyntax: "EXTRACT/DATE_PART does not support '{}' part", field)
            }
            expr.dt().week()
        },
        DateTimeField::IsoWeek => expr.dt().week(),
        DateTimeField::DayOfYear | DateTimeField::Doy => expr.dt().ordinal_day(),
        DateTimeField::DayOfWeek | DateTimeField::Dow => {
            let w = expr.dt().weekday();
            when(w.clone().eq(typed_lit(7i8)))
                .then(typed_lit(0i8))
                .otherwise(w)
        },
        DateTimeField::Isodow => expr.dt().weekday(),
        DateTimeField::Day => expr.dt().day(),
        DateTimeField::Hour => expr.dt().hour(),
        DateTimeField::Minute => expr.dt().minute(),
        DateTimeField::Second => expr.dt().second(),
        DateTimeField::Millisecond | DateTimeField::Milliseconds => {
            (expr.clone().dt().second() * typed_lit(1_000f64))
                + expr.dt().nanosecond().div(typed_lit(1_000_000f64))
        },
        DateTimeField::Microsecond | DateTimeField::Microseconds => {
            (expr.clone().dt().second() * typed_lit(1_000_000f64))
                + expr.dt().nanosecond().div(typed_lit(1_000f64))
        },
        DateTimeField::Nanosecond | DateTimeField::Nanoseconds => {
            (expr.clone().dt().second() * typed_lit(1_000_000_000f64)) + expr.dt().nanosecond()
        },
        DateTimeField::Time => expr.dt().time(),
        #[cfg(feature = "timezones")]
        DateTimeField::Timezone => expr.dt().base_utc_offset().dt().total_seconds(),
        DateTimeField::Epoch => {
            expr.clone()
                .dt()
                .timestamp(TimeUnit::Nanoseconds)
                .div(typed_lit(1_000_000_000i64))
                + expr.dt().nanosecond().div(typed_lit(1_000_000_000f64))
        },
        _ => {
            polars_bail!(SQLSyntax: "EXTRACT/DATE_PART does not support '{}' part", field)
        },
    })
}

fn bitstring_to_bytes_literal(b: &String) -> PolarsResult<Expr> {
    let n_bits = b.len();
    if !b.chars().all(|c| c == '0' || c == '1') || n_bits > 64 {
        polars_bail!(
            SQLSyntax:
            "bit string literal should contain only 0s and 1s and have length <= 64; found '{}' with length {}", b, n_bits
        )
    }
    let s = b.as_str();
    Ok(lit(match n_bits {
        0 => b"".to_vec(),
        1..=8 => u8::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
        9..=16 => u16::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
        17..=32 => u32::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
        _ => u64::from_str_radix(s, 2).unwrap().to_be_bytes().to_vec(),
    }))
}
