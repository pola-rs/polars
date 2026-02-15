//! Expressions that are supported by the Polars SQL interface.
//!
//! This is useful for syntax highlighting
//!
//! This module defines:
//! - all Polars SQL keywords [`all_keywords`]
//! - all Polars SQL functions [`all_functions`]

use std::fmt::Display;
use std::ops::Div;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::plans::DynLiteralValue;
use polars_plan::prelude::typed_lit;
use polars_time::Duration;
use polars_utils::unique_column_name;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sqlparser::ast::{
    AccessExpr, BinaryOperator as SQLBinaryOperator, CastFormat, CastKind, DataType as SQLDataType,
    DateTimeField, Expr as SQLExpr, Function as SQLFunction, Ident, Interval, Query as Subquery,
    SelectItem, Subscript, TimezoneInfo, TrimWhereField, TypedString, UnaryOperator,
    Value as SQLValue, ValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::SQLContext;
use crate::functions::SQLFunctionVisitor;
use crate::types::{
    bitstring_to_bytes_literal, is_iso_date, is_iso_datetime, is_iso_time, map_sql_dtype_to_polars,
};

#[inline]
#[cold]
#[must_use]
/// Convert a Display-able error to PolarsError::SQLInterface
pub fn to_sql_interface_err(err: impl Display) -> PolarsError {
    PolarsError::SQLInterface(err.to_string().into())
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
/// Categorises the type of (allowed) subquery constraint
pub enum SubqueryRestriction {
    /// Subquery must return a single column
    SingleColumn,
    // SingleRow,
    // SingleValue,
    // Any
}

/// Recursively walks a SQL Expr to create a polars Expr
pub(crate) struct SQLExprVisitor<'a> {
    ctx: &'a mut SQLContext,
    active_schema: Option<&'a Schema>,
}

impl SQLExprVisitor<'_> {
    fn array_expr_to_series(&mut self, elements: &[SQLExpr]) -> PolarsResult<Series> {
        let mut array_elements = Vec::with_capacity(elements.len());
        for e in elements {
            let val = match e {
                SQLExpr::Value(ValueWithSpan { value: v, .. }) => self.visit_any_value(v, None),
                SQLExpr::UnaryOp { op, expr } => match expr.as_ref() {
                    SQLExpr::Value(ValueWithSpan { value: v, .. }) => {
                        self.visit_any_value(v, Some(op))
                    },
                    _ => Err(polars_err!(SQLInterface: "array element {:?} is not supported", e)),
                },
                SQLExpr::Array(values) => {
                    let srs = self.array_expr_to_series(&values.elem)?;
                    Ok(AnyValue::List(srs))
                },
                _ => Err(polars_err!(SQLInterface: "array element {:?} is not supported", e)),
            }?
            .into_static();
            array_elements.push(val);
        }
        Series::from_any_values(PlSmallStr::EMPTY, &array_elements, true)
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
                is_some: _,
            } => self.visit_any(left, compare_op, right),
            SQLExpr::Array(arr) => self.visit_array_expr(&arr.elem, true, None),
            SQLExpr::Between {
                expr,
                negated,
                low,
                high,
            } => self.visit_between(expr, *negated, low, high),
            SQLExpr::BinaryOp { left, op, right } => self.visit_binary_op(left, op, right),
            SQLExpr::Cast {
                kind,
                expr,
                data_type,
                format,
            } => self.visit_cast(expr, data_type, format, kind),
            SQLExpr::Ceil { expr, .. } => Ok(self.visit_expr(expr)?.ceil()),
            SQLExpr::CompoundFieldAccess { root, access_chain } => {
                // simple subscript access (eg: "array_col[1]")
                if access_chain.len() == 1 {
                    match &access_chain[0] {
                        AccessExpr::Subscript(subscript) => {
                            return self.visit_subscript(root, subscript);
                        },
                        AccessExpr::Dot(_) => {
                            polars_bail!(SQLSyntax: "dot-notation field access is currently unsupported: {:?}", access_chain[0])
                        },
                    }
                }
                // chained dot/bracket notation (eg: "struct_col.field[2].foo[0].bar")
                polars_bail!(SQLSyntax: "complex field access chains are currently unsupported: {:?}", access_chain[0])
            },
            SQLExpr::CompoundIdentifier(idents) => self.visit_compound_identifier(idents),
            SQLExpr::Extract {
                field,
                syntax: _,
                expr,
            } => parse_extract_date_part(self.visit_expr(expr)?, field),
            SQLExpr::Floor { expr, .. } => Ok(self.visit_expr(expr)?.floor()),
            SQLExpr::Function(function) => self.visit_function(function),
            SQLExpr::Identifier(ident) => self.visit_identifier(ident),
            SQLExpr::InList {
                expr,
                list,
                negated,
            } => {
                let expr = self.visit_expr(expr)?;
                let elems = self.visit_array_expr(list, true, Some(&expr))?;
                let is_in = expr.is_in(elems, false);
                Ok(if *negated { is_in.not() } else { is_in })
            },
            SQLExpr::InSubquery {
                expr,
                subquery,
                negated,
            } => self.visit_in_subquery(expr, subquery, *negated),
            SQLExpr::Interval(interval) => Ok(lit(interval_to_duration(interval, true)?)),
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
                any,
                expr,
                pattern,
                escape_char,
            } => {
                if *any {
                    polars_bail!(SQLSyntax: "LIKE ANY is not a supported syntax")
                }
                let escape_str = escape_char.as_ref().and_then(|v| match v {
                    SQLValue::SingleQuotedString(s) => Some(s.clone()),
                    _ => None,
                });
                self.visit_like(*negated, expr, pattern, &escape_str, false)
            },
            SQLExpr::ILike {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            } => {
                if *any {
                    polars_bail!(SQLSyntax: "ILIKE ANY is not a supported syntax")
                }
                let escape_str = escape_char.as_ref().and_then(|v| match v {
                    SQLValue::SingleQuotedString(s) => Some(s.clone()),
                    _ => None,
                });
                self.visit_like(*negated, expr, pattern, &escape_str, true)
            },
            SQLExpr::Nested(expr) => self.visit_expr(expr),
            SQLExpr::Position { expr, r#in } => Ok(
                // note: SQL is 1-indexed
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
            SQLExpr::Substring {
                expr,
                substring_from,
                substring_for,
                ..
            } => self.visit_substring(expr, substring_from.as_deref(), substring_for.as_deref()),
            SQLExpr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => self.visit_trim(expr, trim_where, trim_what, trim_characters),
            SQLExpr::TypedString(TypedString {
                data_type,
                value:
                    ValueWithSpan {
                        value: SQLValue::SingleQuotedString(v),
                        ..
                    },
                uses_odbc_syntax: _,
            }) => match data_type {
                SQLDataType::Date => {
                    if is_iso_date(v) {
                        Ok(lit(v.as_str()).cast(DataType::Date))
                    } else {
                        polars_bail!(SQLSyntax: "invalid DATE literal '{}'", v)
                    }
                },
                SQLDataType::Time(None, TimezoneInfo::None) => {
                    if is_iso_time(v) {
                        Ok(lit(v.as_str()).str().to_time(StrptimeOptions {
                            strict: true,
                            ..Default::default()
                        }))
                    } else {
                        polars_bail!(SQLSyntax: "invalid TIME literal '{}'", v)
                    }
                },
                SQLDataType::Timestamp(None, TimezoneInfo::None) | SQLDataType::Datetime(None) => {
                    if is_iso_datetime(v) {
                        Ok(lit(v.as_str()).str().to_datetime(
                            None,
                            None,
                            StrptimeOptions {
                                strict: true,
                                ..Default::default()
                            },
                            lit("latest"),
                        ))
                    } else {
                        let fn_name = match data_type {
                            SQLDataType::Timestamp(_, _) => "TIMESTAMP",
                            SQLDataType::Datetime(_) => "DATETIME",
                            _ => unreachable!(),
                        };
                        polars_bail!(SQLSyntax: "invalid {} literal '{}'", fn_name, v)
                    }
                },
                _ => {
                    polars_bail!(SQLInterface: "typed literal should be one of DATE, DATETIME, TIME, or TIMESTAMP (found {})", data_type)
                },
            },
            SQLExpr::UnaryOp { op, expr } => self.visit_unary_op(op, expr),
            SQLExpr::Value(ValueWithSpan { value, .. }) => self.visit_literal(value),
            SQLExpr::Wildcard(_) => Ok(all().as_expr()),
            e @ SQLExpr::Case { .. } => self.visit_case_when_then(e),
            other => {
                polars_bail!(SQLInterface: "expression {:?} is not currently supported", other)
            },
        }
    }

    fn visit_subquery(
        &mut self,
        subquery: &Subquery,
        restriction: SubqueryRestriction,
    ) -> PolarsResult<Expr> {
        if subquery.with.is_some() {
            polars_bail!(SQLSyntax: "SQL subquery cannot be a CTE 'WITH' clause");
        }
        // note: we have to execute subqueries in an isolated scope to prevent
        // propagating any context/arena mutation into the rest of the query
        let lf = self
            .ctx
            .execute_isolated(|ctx| ctx.execute_query_no_ctes(subquery))?;

        if restriction == SubqueryRestriction::SingleColumn {
            let new_name = unique_column_name();
            return Ok(Expr::SubPlan(
                SpecialEq::new(Arc::new(lf.logical_plan)),
                // TODO: pass the implode depending on expr.
                vec![(
                    new_name.clone(),
                    first().as_expr().implode().alias(new_name.clone()),
                )],
            ));
        };
        polars_bail!(SQLInterface: "subquery type not supported");
    }

    /// Visit a single SQL identifier.
    ///
    /// e.g. column
    fn visit_identifier(&self, ident: &Ident) -> PolarsResult<Expr> {
        Ok(col(ident.value.as_str()))
    }

    /// Visit a compound SQL identifier
    ///
    /// e.g. tbl.column, struct.field, tbl.struct.field (inc. nested struct fields)
    fn visit_compound_identifier(&mut self, idents: &[Ident]) -> PolarsResult<Expr> {
        Ok(resolve_compound_identifier(self.ctx, idents, self.active_schema)?[0].clone())
    }

    fn visit_like(
        &mut self,
        negated: bool,
        expr: &SQLExpr,
        pattern: &SQLExpr,
        escape_char: &Option<String>,
        case_insensitive: bool,
    ) -> PolarsResult<Expr> {
        if escape_char.is_some() {
            polars_bail!(SQLInterface: "ESCAPE char for LIKE/ILIKE is not currently supported; found '{}'", escape_char.clone().unwrap());
        }
        let pat = match self.visit_expr(pattern) {
            Ok(Expr::Literal(lv)) if lv.extract_str().is_some() => {
                PlSmallStr::from_str(lv.extract_str().unwrap())
            },
            _ => {
                polars_bail!(SQLSyntax: "LIKE/ILIKE pattern must be a string literal; found {}", pattern)
            },
        };
        if pat.is_empty() || (!case_insensitive && pat.chars().all(|c| !matches!(c, '%' | '_'))) {
            // empty string or other exact literal match (eg: no wildcard chars)
            let op = if negated {
                SQLBinaryOperator::NotEq
            } else {
                SQLBinaryOperator::Eq
            };
            self.visit_binary_op(expr, &op, pattern)
        } else {
            // create regex from pattern containing SQL wildcard chars ('%' => '.*', '_' => '.')
            let mut rx = regex::escape(pat.as_str())
                .replace('%', ".*")
                .replace('_', ".");

            rx = format!(
                "^{}{}$",
                if case_insensitive { "(?is)" } else { "(?s)" },
                rx
            );

            let expr = self.visit_expr(expr)?;
            let matches = expr.str().contains(lit(rx), true);
            Ok(if negated { matches.not() } else { matches })
        }
    }

    fn visit_subscript(&mut self, expr: &SQLExpr, subscript: &Subscript) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        Ok(match subscript {
            Subscript::Index { index } => {
                let idx = adjust_one_indexed_param(self.visit_expr(index)?, true);
                expr.list().get(idx, true)
            },
            Subscript::Slice { .. } => {
                polars_bail!(SQLSyntax: "array slice syntax is not currently supported")
            },
        })
    }

    /// Handle implicit temporal string comparisons.
    ///
    /// eg: clauses such as -
    ///   "dt >= '2024-04-30'"
    ///   "dt = '2077-10-10'::date"
    ///   "dtm::date = '2077-10-10'
    fn convert_temporal_strings(&mut self, left: &Expr, right: &Expr) -> Expr {
        if let (Some(name), Some(s), expr_dtype) = match (left, right) {
            // identify "col <op> string" expressions
            (Expr::Column(name), Expr::Literal(lv)) if lv.extract_str().is_some() => {
                (Some(name.clone()), Some(lv.extract_str().unwrap()), None)
            },
            // identify "CAST(expr AS type) <op> string" and/or "expr::type <op> string" expressions
            (Expr::Cast { expr, dtype, .. }, Expr::Literal(lv)) if lv.extract_str().is_some() => {
                let s = lv.extract_str().unwrap();
                match &**expr {
                    Expr::Column(name) => (Some(name.clone()), Some(s), Some(dtype)),
                    _ => (None, Some(s), Some(dtype)),
                }
            },
            _ => (None, None, None),
        } {
            if expr_dtype.is_none() && self.active_schema.is_none() {
                right.clone()
            } else {
                let left_dtype = expr_dtype.map_or_else(
                    || {
                        self.active_schema
                            .as_ref()
                            .and_then(|schema| schema.get(&name))
                    },
                    |dt| dt.as_literal(),
                );
                match left_dtype {
                    Some(DataType::Time) if is_iso_time(s) => {
                        right.clone().str().to_time(StrptimeOptions {
                            strict: true,
                            ..Default::default()
                        })
                    },
                    Some(DataType::Date) if is_iso_date(s) => {
                        right.clone().str().to_date(StrptimeOptions {
                            strict: true,
                            ..Default::default()
                        })
                    },
                    Some(DataType::Datetime(tu, tz)) if is_iso_datetime(s) || is_iso_date(s) => {
                        if s.len() == 10 {
                            // handle upcast from ISO date string (10 chars) to datetime
                            lit(format!("{s}T00:00:00"))
                        } else {
                            lit(s.replacen(' ', "T", 1))
                        }
                        .str()
                        .to_datetime(
                            Some(*tu),
                            tz.clone(),
                            StrptimeOptions {
                                strict: true,
                                ..Default::default()
                            },
                            lit("latest"),
                        )
                    },
                    _ => right.clone(),
                }
            }
        } else {
            right.clone()
        }
    }

    fn struct_field_access_expr(
        &mut self,
        expr: &Expr,
        path: &str,
        infer_index: bool,
    ) -> PolarsResult<Expr> {
        let path_elems = if path.starts_with('{') && path.ends_with('}') {
            path.trim_matches(|c| c == '{' || c == '}')
        } else {
            path
        }
        .split(',');

        let mut expr = expr.clone();
        for p in path_elems {
            let p = p.trim();
            expr = if infer_index {
                match p.parse::<i64>() {
                    Ok(idx) => expr.list().get(lit(idx), true),
                    Err(_) => expr.struct_().field_by_name(p),
                }
            } else {
                expr.struct_().field_by_name(p)
            }
        }
        Ok(expr)
    }

    /// Visit a SQL binary operator.
    ///
    /// e.g. "column + 1", "column1 <= column2"
    fn visit_binary_op(
        &mut self,
        left: &SQLExpr,
        op: &SQLBinaryOperator,
        right: &SQLExpr,
    ) -> PolarsResult<Expr> {
        // check for (unsupported) scalar subquery comparisons
        if matches!(left, SQLExpr::Subquery(_)) || matches!(right, SQLExpr::Subquery(_)) {
            let (suggestion, str_op) = match op {
                SQLBinaryOperator::NotEq => ("; use 'NOT IN' instead", "!=".to_string()),
                SQLBinaryOperator::Eq => ("; use 'IN' instead", format!("{op}")),
                _ => ("", format!("{op}")),
            };
            polars_bail!(
                SQLSyntax: "subquery comparisons with '{str_op}' are not supported{suggestion}"
            );
        }

        // need special handling for interval offsets and comparisons
        let (lhs, mut rhs) = match (left, op, right) {
            (_, SQLBinaryOperator::Minus, SQLExpr::Interval(v)) => {
                let duration = interval_to_duration(v, false)?;
                return Ok(self
                    .visit_expr(left)?
                    .dt()
                    .offset_by(lit(format!("-{duration}"))));
            },
            (_, SQLBinaryOperator::Plus, SQLExpr::Interval(v)) => {
                let duration = interval_to_duration(v, false)?;
                return Ok(self
                    .visit_expr(left)?
                    .dt()
                    .offset_by(lit(format!("{duration}"))));
            },
            (SQLExpr::Interval(v1), _, SQLExpr::Interval(v2)) => {
                // shortcut interval comparison evaluation (-> bool)
                let d1 = interval_to_duration(v1, false)?;
                let d2 = interval_to_duration(v2, false)?;
                let res = match op {
                    SQLBinaryOperator::Gt => Ok(lit(d1 > d2)),
                    SQLBinaryOperator::Lt => Ok(lit(d1 < d2)),
                    SQLBinaryOperator::GtEq => Ok(lit(d1 >= d2)),
                    SQLBinaryOperator::LtEq => Ok(lit(d1 <= d2)),
                    SQLBinaryOperator::NotEq => Ok(lit(d1 != d2)),
                    SQLBinaryOperator::Eq | SQLBinaryOperator::Spaceship => Ok(lit(d1 == d2)),
                    _ => polars_bail!(SQLInterface: "invalid interval comparison operator"),
                };
                if res.is_ok() {
                    return res;
                }
                (self.visit_expr(left)?, self.visit_expr(right)?)
            },
            _ => (self.visit_expr(left)?, self.visit_expr(right)?),
        };
        rhs = self.convert_temporal_strings(&lhs, &rhs);

        Ok(match op {
            // ----
            // Bitwise operators
            // ----
            SQLBinaryOperator::BitwiseAnd => lhs.and(rhs),  // "x & y"
            SQLBinaryOperator::BitwiseOr => lhs.or(rhs),  // "x | y"
            SQLBinaryOperator::Xor => lhs.xor(rhs),  // "x XOR y"

            // ----
            // General operators
            // ----
            SQLBinaryOperator::And => lhs.and(rhs),  // "x AND y"
            SQLBinaryOperator::Divide => lhs / rhs,  // "x / y"
            SQLBinaryOperator::DuckIntegerDivide => lhs.floor_div(rhs).cast(DataType::Int64),  // "x // y"
            SQLBinaryOperator::Eq => lhs.eq(rhs),  // "x = y"
            SQLBinaryOperator::Gt => lhs.gt(rhs),  // "x > y"
            SQLBinaryOperator::GtEq => lhs.gt_eq(rhs),  // "x >= y"
            SQLBinaryOperator::Lt => lhs.lt(rhs),  // "x < y"
            SQLBinaryOperator::LtEq => lhs.lt_eq(rhs),  // "x <= y"
            SQLBinaryOperator::Minus => lhs - rhs,  // "x - y"
            SQLBinaryOperator::Modulo => lhs % rhs,  // "x % y"
            SQLBinaryOperator::Multiply => lhs * rhs,  // "x * y"
            SQLBinaryOperator::NotEq => lhs.eq(rhs).not(),  // "x != y"
            SQLBinaryOperator::Or => lhs.or(rhs),  // "x OR y"
            SQLBinaryOperator::Plus => lhs + rhs,  // "x + y"
            SQLBinaryOperator::Spaceship => lhs.eq_missing(rhs),  // "x <=> y"
            SQLBinaryOperator::StringConcat => {  // "x || y"
                lhs.cast(DataType::String) + rhs.cast(DataType::String)
            },
            SQLBinaryOperator::PGStartsWith => lhs.str().starts_with(rhs),  // "x ^@ y"
            // ----
            // Regular expression operators
            // ----
            SQLBinaryOperator::PGRegexMatch => match rhs {  // "x ~ y"
                Expr::Literal(ref lv) if lv.extract_str().is_some() => lhs.str().contains(rhs, true),
                _ => polars_bail!(SQLSyntax: "invalid pattern for '~' operator: {:?}", rhs),
            },
            SQLBinaryOperator::PGRegexNotMatch => match rhs {  // "x !~ y"
                Expr::Literal(ref lv) if lv.extract_str().is_some() => lhs.str().contains(rhs, true).not(),
                _ => polars_bail!(SQLSyntax: "invalid pattern for '!~' operator: {:?}", rhs),
            },
            SQLBinaryOperator::PGRegexIMatch => match rhs {  // "x ~* y"
                Expr::Literal(ref lv) if lv.extract_str().is_some() => {
                    let pat = lv.extract_str().unwrap();
                    lhs.str().contains(lit(format!("(?i){pat}")), true)
                },
                _ => polars_bail!(SQLSyntax: "invalid pattern for '~*' operator: {:?}", rhs),
            },
            SQLBinaryOperator::PGRegexNotIMatch => match rhs {  // "x !~* y"
                Expr::Literal(ref lv) if lv.extract_str().is_some() => {
                    let pat = lv.extract_str().unwrap();
                    lhs.str().contains(lit(format!("(?i){pat}")), true).not()
                },
                _ => {
                    polars_bail!(SQLSyntax: "invalid pattern for '!~*' operator: {:?}", rhs)
                },
            },
            // ----
            // LIKE/ILIKE operators
            // ----
            SQLBinaryOperator::PGLikeMatch  // "x ~~ y"
            | SQLBinaryOperator::PGNotLikeMatch  // "x !~~ y"
            | SQLBinaryOperator::PGILikeMatch  // "x ~~* y"
            | SQLBinaryOperator::PGNotILikeMatch => {  // "x !~~* y"
                let expr = if matches!(
                    op,
                    SQLBinaryOperator::PGLikeMatch | SQLBinaryOperator::PGNotLikeMatch
                ) {
                    SQLExpr::Like {
                        negated: matches!(op, SQLBinaryOperator::PGNotLikeMatch),
                        any: false,
                        expr: Box::new(left.clone()),
                        pattern: Box::new(right.clone()),
                        escape_char: None,
                    }
                } else {
                    SQLExpr::ILike {
                        negated: matches!(op, SQLBinaryOperator::PGNotILikeMatch),
                        any: false,
                        expr: Box::new(left.clone()),
                        pattern: Box::new(right.clone()),
                        escape_char: None,
                    }
                };
                self.visit_expr(&expr)?
            },
            // ----
            // JSON/Struct field access operators
            // ----
            SQLBinaryOperator::Arrow | SQLBinaryOperator::LongArrow => match rhs {  // "x -> y", "x ->> y"
                Expr::Literal(lv) if lv.extract_str().is_some() => {
                    let path = lv.extract_str().unwrap();
                    let mut expr = self.struct_field_access_expr(&lhs, path, false)?;
                    if let SQLBinaryOperator::LongArrow = op {
                        expr = expr.cast(DataType::String);
                    }
                    expr
                },
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(idx))) => {
                    let mut expr = self.struct_field_access_expr(&lhs, &idx.to_string(), true)?;
                    if let SQLBinaryOperator::LongArrow = op {
                        expr = expr.cast(DataType::String);
                    }
                    expr
                },
                _ => {
                    polars_bail!(SQLSyntax: "invalid json/struct path-extract definition: {:?}", right)
                },
            },
            SQLBinaryOperator::HashArrow | SQLBinaryOperator::HashLongArrow => {  // "x #> y", "x #>> y"
                match rhs {
                    Expr::Literal(lv) if lv.extract_str().is_some() => {
                        let path = lv.extract_str().unwrap();
                        let mut expr = self.struct_field_access_expr(&lhs, path, true)?;
                        if let SQLBinaryOperator::HashLongArrow = op {
                            expr = expr.cast(DataType::String);
                        }
                        expr
                    },
                    _ => {
                        polars_bail!(SQLSyntax: "invalid json/struct path-extract definition: {:?}", rhs)
                    }
                }
            },
            other => {
                polars_bail!(SQLInterface: "operator {:?} is not currently supported", other)
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
            (UnaryOperator::Plus, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n)))) => {
                lit(n)
            },
            (UnaryOperator::Plus, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(n)))) => {
                lit(n)
            },
            (UnaryOperator::Minus, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n)))) => {
                lit(-n)
            },
            (UnaryOperator::Minus, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(n)))) => {
                lit(-n)
            },
            // general case
            (UnaryOperator::Plus, _) => lit(0) + expr,
            (UnaryOperator::Minus, _) => lit(0) - expr,
            (UnaryOperator::Not, _) => match &expr {
                Expr::Column(name)
                    if self
                        .active_schema
                        .and_then(|schema| schema.get(name))
                        .is_some_and(|dtype| matches!(dtype, DataType::Boolean)) =>
                {
                    // if already boolean, can operate bitwise
                    expr.not()
                },
                // otherwise SQL "NOT" expects logical, not bitwise, behaviour (eg: on integers)
                _ => expr.strict_cast(DataType::Boolean).not(),
            },
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
            active_schema: self.active_schema,
        };
        visitor.visit_function()
    }

    /// Visit a SQL `ALL` expression.
    ///
    /// e.g. `a > ALL(y)`
    fn visit_all(
        &mut self,
        left: &SQLExpr,
        compare_op: &SQLBinaryOperator,
        right: &SQLExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let right = self.visit_expr(right)?;

        match compare_op {
            SQLBinaryOperator::Gt => Ok(left.gt(right.max())),
            SQLBinaryOperator::Lt => Ok(left.lt(right.min())),
            SQLBinaryOperator::GtEq => Ok(left.gt_eq(right.max())),
            SQLBinaryOperator::LtEq => Ok(left.lt_eq(right.min())),
            SQLBinaryOperator::Eq => polars_bail!(SQLSyntax: "ALL cannot be used with ="),
            SQLBinaryOperator::NotEq => polars_bail!(SQLSyntax: "ALL cannot be used with !="),
            _ => polars_bail!(SQLInterface: "invalid comparison operator"),
        }
    }

    /// Visit a SQL `ANY` expression.
    ///
    /// e.g. `a != ANY(y)`
    fn visit_any(
        &mut self,
        left: &SQLExpr,
        compare_op: &SQLBinaryOperator,
        right: &SQLExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let right = self.visit_expr(right)?;

        match compare_op {
            SQLBinaryOperator::Gt => Ok(left.gt(right.min())),
            SQLBinaryOperator::Lt => Ok(left.lt(right.max())),
            SQLBinaryOperator::GtEq => Ok(left.gt_eq(right.min())),
            SQLBinaryOperator::LtEq => Ok(left.lt_eq(right.max())),
            SQLBinaryOperator::Eq => Ok(left.is_in(right, false)),
            SQLBinaryOperator::NotEq => Ok(left.is_in(right, false).not()),
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
                if let Some(dtype) = schema.get(name) {
                    if matches!(
                        dtype,
                        DataType::Date | DataType::Time | DataType::Datetime(_, _)
                    ) {
                        elems = elems.strict_cast(dtype)?;
                    }
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
        dtype: &SQLDataType,
        format: &Option<CastFormat>,
        cast_kind: &CastKind,
    ) -> PolarsResult<Expr> {
        if format.is_some() {
            return Err(
                polars_err!(SQLInterface: "use of FORMAT is not currently supported in CAST"),
            );
        }
        let expr = self.visit_expr(expr)?;

        #[cfg(feature = "json")]
        if dtype == &SQLDataType::JSON {
            // @BROKEN: we cannot handle this.
            return Ok(expr.str().json_decode(DataType::Struct(Vec::new())));
        }
        let polars_type = map_sql_dtype_to_polars(dtype)?;
        Ok(match cast_kind {
            CastKind::Cast | CastKind::DoubleColon => expr.strict_cast(polars_type),
            CastKind::TryCast | CastKind::SafeCast => expr.cast(polars_type),
        })
    }

    /// Visit a SQL literal.
    ///
    /// e.g. 1, 'foo', 1.0, NULL
    ///
    /// See [SQLValue] and [LiteralValue] for more details
    fn visit_literal(&self, value: &SQLValue) -> PolarsResult<Expr> {
        // note: double-quoted strings will be parsed as identifiers, not literals
        Ok(match value {
            SQLValue::Boolean(b) => lit(*b),
            SQLValue::DollarQuotedString(s) => lit(s.value.clone()),
            #[cfg(feature = "binary_encoding")]
            SQLValue::HexStringLiteral(x) => {
                if x.len() % 2 != 0 {
                    polars_bail!(SQLSyntax: "hex string literal must have an even number of digits; found '{}'", x)
                };
                lit(hex::decode(x.clone()).unwrap())
            },
            SQLValue::Null => Expr::Literal(LiteralValue::untyped_null()),
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
                // literal (see https://www.postgresql.org/docs/current/datatype-bit.html), but sqlparser-rs
                // patterned the token name after BigQuery (where b'str' really IS a byte string)
                bitstring_to_bytes_literal(b)?
            },
            SQLValue::SingleQuotedString(s) => lit(s.clone()),
            other => {
                polars_bail!(SQLInterface: "value {:?} is not a supported literal type", other)
            },
        })
    }

    /// Visit a SQL literal (like [visit_literal]), but return AnyValue instead of Expr.
    fn visit_any_value(
        &self,
        value: &SQLValue,
        op: Option<&UnaryOperator>,
    ) -> PolarsResult<AnyValue<'_>> {
        Ok(match value {
            SQLValue::Boolean(b) => AnyValue::Boolean(*b),
            SQLValue::DollarQuotedString(s) => AnyValue::StringOwned(s.clone().value.into()),
            #[cfg(feature = "binary_encoding")]
            SQLValue::HexStringLiteral(x) => {
                if x.len() % 2 != 0 {
                    polars_bail!(SQLSyntax: "hex string literal must have an even number of digits; found '{}'", x)
                };
                AnyValue::BinaryOwned(hex::decode(x.clone()).unwrap())
            },
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
            SQLValue::SingleQuotedByteStringLiteral(b) => {
                // note: for PostgreSQL this represents a BIT literal (eg: b'10101') not BYTE
                let bytes_literal = bitstring_to_bytes_literal(b)?;
                match bytes_literal {
                    Expr::Literal(lv) if lv.extract_binary().is_some() => {
                        AnyValue::BinaryOwned(lv.extract_binary().unwrap().to_vec())
                    },
                    _ => {
                        polars_bail!(SQLInterface: "failed to parse bitstring literal: {:?}", b)
                    },
                }
            },
            SQLValue::SingleQuotedString(s) => AnyValue::StringOwned(s.as_str().into()),
            other => polars_bail!(SQLInterface: "value {:?} is not currently supported", other),
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
        Ok(if negated {
            expr.clone().lt(low).or(expr.gt(high))
        } else {
            expr.clone().gt_eq(low).and(expr.lt_eq(high))
        })
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
            Some(Expr::Literal(lv)) if lv.extract_str().is_some() => {
                Some(PlSmallStr::from_str(lv.extract_str().unwrap()))
            },
            None => None,
            _ => return self.err(&expr),
        };
        Ok(match (trim_where, trim_what) {
            (None | Some(TrimWhereField::Both), None) => {
                expr.str().strip_chars(lit(LiteralValue::untyped_null()))
            },
            (None | Some(TrimWhereField::Both), Some(val)) => expr.str().strip_chars(lit(val)),
            (Some(TrimWhereField::Leading), None) => expr
                .str()
                .strip_chars_start(lit(LiteralValue::untyped_null())),
            (Some(TrimWhereField::Leading), Some(val)) => expr.str().strip_chars_start(lit(val)),
            (Some(TrimWhereField::Trailing), None) => expr
                .str()
                .strip_chars_end(lit(LiteralValue::untyped_null())),
            (Some(TrimWhereField::Trailing), Some(val)) => expr.str().strip_chars_end(lit(val)),
        })
    }

    fn visit_substring(
        &mut self,
        expr: &SQLExpr,
        substring_from: Option<&SQLExpr>,
        substring_for: Option<&SQLExpr>,
    ) -> PolarsResult<Expr> {
        let e = self.visit_expr(expr)?;

        match (substring_from, substring_for) {
            // SUBSTRING(expr FROM start FOR length)
            (Some(from_expr), Some(for_expr)) => {
                let start = self.visit_expr(from_expr)?;
                let length = self.visit_expr(for_expr)?;

                // note: SQL is 1-indexed, so we need to adjust the offsets accordingly
                Ok(match (start.clone(), length.clone()) {
                    (Expr::Literal(lv), _) | (_, Expr::Literal(lv)) if lv.is_null() => lit(lv),
                    (_, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n)))) if n < 0 => {
                        polars_bail!(SQLSyntax: "SUBSTR does not support negative length ({})", n)
                    },
                    (Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))), _) if n > 0 => {
                        e.str().slice(lit(n - 1), length)
                    },
                    (Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))), _) => e
                        .str()
                        .slice(lit(0), (length + lit(n - 1)).clip_min(lit(0))),
                    (Expr::Literal(_), _) => {
                        polars_bail!(SQLSyntax: "invalid 'start' for SUBSTRING")
                    },
                    (_, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(_)))) => {
                        polars_bail!(SQLSyntax: "invalid 'length' for SUBSTRING")
                    },
                    _ => {
                        let adjusted_start = start - lit(1);
                        when(adjusted_start.clone().lt(lit(0)))
                            .then(e.clone().str().slice(
                                lit(0),
                                (length.clone() + adjusted_start.clone()).clip_min(lit(0)),
                            ))
                            .otherwise(e.str().slice(adjusted_start, length))
                    },
                })
            },
            // SUBSTRING(expr FROM start)
            (Some(from_expr), None) => {
                let start = self.visit_expr(from_expr)?;

                Ok(match start {
                    Expr::Literal(lv) if lv.is_null() => lit(lv),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n <= 0 => e,
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                        e.str().slice(lit(n - 1), lit(LiteralValue::untyped_null()))
                    },
                    Expr::Literal(_) => {
                        polars_bail!(SQLSyntax: "invalid 'start' for SUBSTRING")
                    },
                    _ => e
                        .str()
                        .slice(start - lit(1), lit(LiteralValue::untyped_null())),
                })
            },
            // SUBSTRING(expr) - not valid, but handle gracefully
            (None, _) => {
                polars_bail!(SQLSyntax: "SUBSTR expects 2-3 arguments (found 1)")
            },
        }
    }

    /// Visit a SQL subquery inside an `IN` expression.
    fn visit_in_subquery(
        &mut self,
        expr: &SQLExpr,
        subquery: &Subquery,
        negated: bool,
    ) -> PolarsResult<Expr> {
        let subquery_result = self.visit_subquery(subquery, SubqueryRestriction::SingleColumn)?;
        let expr = self.visit_expr(expr)?;
        Ok(if negated {
            expr.is_in(subquery_result, false).not()
        } else {
            expr.is_in(subquery_result, false)
        })
    }

    /// Visit `CASE` control flow expression.
    fn visit_case_when_then(&mut self, expr: &SQLExpr) -> PolarsResult<Expr> {
        if let SQLExpr::Case {
            case_token: _,
            end_token: _,
            operand,
            conditions,
            else_result,
        } = expr
        {
            polars_ensure!(
                !conditions.is_empty(),
                SQLSyntax: "WHEN and THEN expressions must have at least one element"
            );

            let mut when_thens = conditions.iter();
            let first = when_thens.next();
            if first.is_none() {
                polars_bail!(SQLSyntax: "WHEN and THEN expressions must have at least one element");
            }
            let else_res = match else_result {
                Some(else_res) => self.visit_expr(else_res)?,
                None => lit(LiteralValue::untyped_null()), // ELSE clause is optional; when omitted, it is implicitly NULL
            };
            if let Some(operand_expr) = operand {
                let first_operand_expr = self.visit_expr(operand_expr)?;

                let first = first.unwrap();
                let first_cond = first_operand_expr.eq(self.visit_expr(&first.condition)?);
                let first_then = self.visit_expr(&first.result)?;
                let expr = when(first_cond).then(first_then);
                let next = when_thens.next();

                let mut when_then = if let Some(case_when) = next {
                    let second_operand_expr = self.visit_expr(operand_expr)?;
                    let cond = second_operand_expr.eq(self.visit_expr(&case_when.condition)?);
                    let res = self.visit_expr(&case_when.result)?;
                    expr.when(cond).then(res)
                } else {
                    return Ok(expr.otherwise(else_res));
                };
                for case_when in when_thens {
                    let new_operand_expr = self.visit_expr(operand_expr)?;
                    let cond = new_operand_expr.eq(self.visit_expr(&case_when.condition)?);
                    let res = self.visit_expr(&case_when.result)?;
                    when_then = when_then.when(cond).then(res);
                }
                return Ok(when_then.otherwise(else_res));
            }

            let first = first.unwrap();
            let first_cond = self.visit_expr(&first.condition)?;
            let first_then = self.visit_expr(&first.result)?;
            let expr = when(first_cond).then(first_then);
            let next = when_thens.next();

            let mut when_then = if let Some(case_when) = next {
                let cond = self.visit_expr(&case_when.condition)?;
                let res = self.visit_expr(&case_when.result)?;
                expr.when(cond).then(res)
            } else {
                return Ok(expr.otherwise(else_res));
            };
            for case_when in when_thens {
                let cond = self.visit_expr(&case_when.condition)?;
                let res = self.visit_expr(&case_when.result)?;
                when_then = when_then.when(cond).then(res);
            }
            Ok(when_then.otherwise(else_res))
        } else {
            unreachable!()
        }
    }

    fn err(&self, expr: &Expr) -> PolarsResult<Expr> {
        polars_bail!(SQLInterface: "expression {:?} is not currently supported", expr);
    }
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
            expr.alias(alias.value.as_str())
        },
        SelectItem::UnnamedExpr(expr) => parse_sql_expr(expr, &mut ctx, None)?,
        _ => polars_bail!(SQLInterface: "unable to parse '{}' as Expr", s.as_ref()),
    })
}

pub(crate) fn interval_to_duration(interval: &Interval, fixed: bool) -> PolarsResult<Duration> {
    if interval.last_field.is_some()
        || interval.leading_field.is_some()
        || interval.leading_precision.is_some()
        || interval.fractional_seconds_precision.is_some()
    {
        polars_bail!(SQLSyntax: "unsupported interval syntax ('{}')", interval)
    }
    let s = match &*interval.value {
        SQLExpr::UnaryOp { .. } => {
            polars_bail!(SQLSyntax: "unary ops are not valid on interval strings; found {}", interval.value)
        },
        SQLExpr::Value(ValueWithSpan {
            value: SQLValue::SingleQuotedString(s),
            ..
        }) => Some(s),
        _ => None,
    };
    match s {
        Some(s) if s.contains('-') => {
            polars_bail!(SQLInterface: "minus signs are not yet supported in interval strings; found '{}'", s)
        },
        Some(s) => {
            // years, quarters, and months do not have a fixed duration; these
            // interval parts can only be used with respect to a reference point
            let duration = Duration::parse_interval(s);
            if fixed && duration.months() != 0 {
                polars_bail!(SQLSyntax: "fixed-duration interval cannot contain years, quarters, or months; found {}", s)
            };
            Ok(duration)
        },
        None => polars_bail!(SQLSyntax: "invalid interval {:?}", interval),
    }
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
                "century" | "centuries" | "c" => &DateTimeField::Century,
                "decade" | "decades" => &DateTimeField::Decade,
                "isoyear" => &DateTimeField::Isoyear,
                "year" | "years" | "y" => &DateTimeField::Year,
                "quarter" | "quarters" => &DateTimeField::Quarter,
                "month" | "months" | "mon" | "mons" => &DateTimeField::Month,
                "dayofyear" | "doy" => &DateTimeField::DayOfYear,
                "dayofweek" | "dow" => &DateTimeField::DayOfWeek,
                "isoweek" | "week" | "weeks" => &DateTimeField::IsoWeek,
                "isodow" => &DateTimeField::Isodow,
                "day" | "days" | "dayofmonth" | "d" => &DateTimeField::Day,
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
        DateTimeField::Year | DateTimeField::Years => expr.dt().year(),
        DateTimeField::Quarter => expr.dt().quarter(),
        DateTimeField::Month | DateTimeField::Months => expr.dt().month(),
        DateTimeField::Week(weekday) => {
            if weekday.is_some() {
                polars_bail!(SQLSyntax: "EXTRACT/DATE_PART does not support '{}' part", field)
            }
            expr.dt().week()
        },
        DateTimeField::IsoWeek | DateTimeField::Weeks => expr.dt().week(),
        DateTimeField::DayOfYear | DateTimeField::Doy => expr.dt().ordinal_day(),
        DateTimeField::DayOfWeek | DateTimeField::Dow => {
            let w = expr.dt().weekday();
            when(w.clone().eq(typed_lit(7i8)))
                .then(typed_lit(0i8))
                .otherwise(w)
        },
        DateTimeField::Isodow => expr.dt().weekday(),
        DateTimeField::Day | DateTimeField::Days => expr.dt().day(),
        DateTimeField::Hour | DateTimeField::Hours => expr.dt().hour(),
        DateTimeField::Minute | DateTimeField::Minutes => expr.dt().minute(),
        DateTimeField::Second | DateTimeField::Seconds => expr.dt().second(),
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
        DateTimeField::Timezone => expr.dt().base_utc_offset().dt().total_seconds(false),
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

/// Allow an expression that represents a 1-indexed parameter to
/// be adjusted from 1-indexed (SQL) to 0-indexed (Rust/Polars)
pub(crate) fn adjust_one_indexed_param(idx: Expr, null_if_zero: bool) -> Expr {
    match idx {
        Expr::Literal(sc) if sc.is_null() => lit(LiteralValue::untyped_null()),
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(0))) => {
            if null_if_zero {
                lit(LiteralValue::untyped_null())
            } else {
                idx
            }
        },
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n < 0 => idx,
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => lit(n - 1),
        // TODO: when 'saturating_sub' is available, should be able
        //  to streamline the when/then/otherwise block below -
        _ => when(idx.clone().gt(lit(0)))
            .then(idx.clone() - lit(1))
            .otherwise(if null_if_zero {
                when(idx.clone().eq(lit(0)))
                    .then(lit(LiteralValue::untyped_null()))
                    .otherwise(idx.clone())
            } else {
                idx.clone()
            }),
    }
}

fn resolve_column<'a>(
    ctx: &'a mut SQLContext,
    ident_root: &'a Ident,
    name: &'a str,
    dtype: &'a DataType,
) -> PolarsResult<(Expr, Option<&'a DataType>)> {
    let resolved = ctx.resolve_name(&ident_root.value, name);
    let resolved = resolved.as_str();
    Ok((
        if name != resolved {
            col(resolved).alias(name)
        } else {
            col(name)
        },
        Some(dtype),
    ))
}

pub(crate) fn resolve_compound_identifier(
    ctx: &mut SQLContext,
    idents: &[Ident],
    active_schema: Option<&Schema>,
) -> PolarsResult<Vec<Expr>> {
    // inference priority: table > struct > column
    let ident_root = &idents[0];
    let mut remaining_idents = idents.iter().skip(1);
    let mut lf = ctx.get_table_from_current_scope(&ident_root.value);

    // get schema from table (or the active/default schema)
    let schema = if let Some(ref mut lf) = lf {
        lf.schema_with_arenas(&mut ctx.lp_arena, &mut ctx.expr_arena)?
    } else {
        Arc::new(active_schema.cloned().unwrap_or_default())
    };

    // handle simple/unqualified column reference with no schema
    if lf.is_none() && schema.is_empty() {
        let (mut column, mut dtype): (Expr, Option<&DataType>) =
            (col(ident_root.value.as_str()), None);

        // traverse the remaining struct field path (if any)
        for ident in remaining_idents {
            let name = ident.value.as_str();
            match dtype {
                Some(DataType::Struct(fields)) if name == "*" => {
                    return Ok(fields
                        .iter()
                        .map(|fld| column.clone().struct_().field_by_name(&fld.name))
                        .collect());
                },
                Some(DataType::Struct(fields)) => {
                    dtype = fields
                        .iter()
                        .find(|fld| fld.name == name)
                        .map(|fld| &fld.dtype);
                },
                Some(dtype) if name == "*" => {
                    polars_bail!(SQLSyntax: "cannot expand '*' on non-Struct dtype; found {:?}", dtype)
                },
                _ => dtype = None,
            }
            column = column.struct_().field_by_name(name);
        }
        return Ok(vec![column]);
    }

    let name = &remaining_idents.next().unwrap().value;

    // handle "table.*" wildcard expansion
    if lf.is_some() && name == "*" {
        return schema
            .iter_names_and_dtypes()
            .map(|(name, dtype)| resolve_column(ctx, ident_root, name, dtype).map(|(expr, _)| expr))
            .collect();
    }

    // resolve column/struct reference
    let col_dtype: PolarsResult<(Expr, Option<&DataType>)> =
        match (lf.is_none(), schema.get(&ident_root.value)) {
            // root is a column/struct in schema (no table)
            (true, Some(dtype)) => {
                remaining_idents = idents.iter().skip(1);
                Ok((col(ident_root.value.as_str()), Some(dtype)))
            },
            // root is not in schema and no table found
            (true, None) => {
                polars_bail!(
                    SQLInterface: "no table or struct column named '{}' found",
                    ident_root
                )
            },
            // root is a table, resolve column from table schema
            (false, _) => {
                if let Some((_, col_name, dtype)) = schema.get_full(name) {
                    resolve_column(ctx, ident_root, col_name, dtype)
                } else {
                    polars_bail!(
                        SQLInterface: "no column named '{}' found in table '{}'",
                        name, ident_root
                    )
                }
            },
        };

    // additional ident levels index into struct fields (eg: "df.col.field.nested_field")
    let (mut column, mut dtype) = col_dtype?;
    for ident in remaining_idents {
        let name = ident.value.as_str();
        match dtype {
            Some(DataType::Struct(fields)) if name == "*" => {
                return Ok(fields
                    .iter()
                    .map(|fld| column.clone().struct_().field_by_name(&fld.name))
                    .collect());
            },
            Some(DataType::Struct(fields)) => {
                dtype = fields
                    .iter()
                    .find(|fld| fld.name == name)
                    .map(|fld| &fld.dtype);
            },
            Some(dtype) if name == "*" => {
                polars_bail!(SQLSyntax: "cannot expand '*' on non-Struct dtype; found {:?}", dtype)
            },
            _ => {
                dtype = None;
            },
        }
        column = column.struct_().field_by_name(name);
    }
    Ok(vec![column])
}
