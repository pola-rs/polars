use polars_arrow::error::to_compute_err;
use polars_core::prelude::*;
use polars_lazy::dsl::Expr;
use polars_lazy::prelude::*;
use polars_plan::prelude::{col, lit, when};
use sqlparser::ast::{
    ArrayAgg, BinaryOperator as SQLBinaryOperator, BinaryOperator, DataType as SQLDataType,
    Expr as SqlExpr, Function as SQLFunction, JoinConstraint, OrderByExpr, TrimWhereField,
    UnaryOperator, Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::functions::SqlFunctionVisitor;
use crate::SQLContext;

pub(crate) fn map_sql_polars_datatype(data_type: &SQLDataType) -> PolarsResult<DataType> {
    Ok(match data_type {
        SQLDataType::Array(Some(inner_type)) => {
            DataType::List(Box::new(map_sql_polars_datatype(inner_type)?))
        }
        SQLDataType::BigInt(_) => DataType::Int64,
        SQLDataType::Boolean => DataType::Boolean,
        SQLDataType::Char(_)
        | SQLDataType::Varchar(_)
        | SQLDataType::Uuid
        | SQLDataType::Clob(_)
        | SQLDataType::Text
        | SQLDataType::String => DataType::Utf8,
        SQLDataType::Date => DataType::Date,
        SQLDataType::Double => DataType::Float64,
        SQLDataType::Float(_) => DataType::Float32,
        SQLDataType::Int(_) => DataType::Int32,
        SQLDataType::Interval => DataType::Duration(TimeUnit::Milliseconds),
        SQLDataType::Real => DataType::Float32,
        SQLDataType::SmallInt(_) => DataType::Int16,
        SQLDataType::Time { .. } => DataType::Time,
        SQLDataType::Timestamp { .. } => DataType::Datetime(TimeUnit::Milliseconds, None),
        SQLDataType::TinyInt(_) => DataType::Int8,
        SQLDataType::UnsignedBigInt(_) => DataType::UInt64,
        SQLDataType::UnsignedInt(_) => DataType::UInt32,
        SQLDataType::UnsignedSmallInt(_) => DataType::UInt16,
        SQLDataType::UnsignedTinyInt(_) => DataType::UInt8,

        _ => polars_bail!(ComputeError: "SQL datatype {:?} is not yet supported", data_type),
    })
}

/// Recursively walks a SQL Expr to create a polars Expr
pub(crate) struct SqlExprVisitor<'a> {
    ctx: &'a SQLContext,
}

impl SqlExprVisitor<'_> {
    fn visit_expr(&self, expr: &SqlExpr) -> PolarsResult<Expr> {
        match expr {
            SqlExpr::AllOp(_) => Ok(self.visit_expr(expr)?.all()),
            SqlExpr::AnyOp(expr) => Ok(self.visit_expr(expr)?.any()),
            SqlExpr::ArrayAgg(expr) => self.visit_arr_agg(expr),
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => self.visit_between(expr, *negated, low, high),
            SqlExpr::BinaryOp { left, op, right } => self.visit_binary_op(left, op, right),
            SqlExpr::Cast { expr, data_type } => self.visit_cast(expr, data_type),
            SqlExpr::Ceil { expr, .. } => Ok(self.visit_expr(expr)?.ceil()),
            SqlExpr::CompoundIdentifier(idents) => self.visit_compound_identifier(idents),
            SqlExpr::Floor { expr, .. } => Ok(self.visit_expr(expr)?.floor()),
            SqlExpr::Function(function) => self.visit_function(function),
            SqlExpr::Identifier(ident) => self.visit_identifier(ident),
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => self.visit_is_in(expr, list, *negated),
            SqlExpr::IsDistinctFrom(e1, e2) => {
                Ok(self.visit_expr(e1)?.neq_missing(self.visit_expr(e2)?))
            }
            SqlExpr::IsNotDistinctFrom(e1, e2) => {
                Ok(self.visit_expr(e1)?.eq_missing(self.visit_expr(e2)?))
            }
            SqlExpr::IsFalse(expr) => Ok(self.visit_expr(expr)?.eq(lit(false))),
            SqlExpr::IsNotFalse(expr) => Ok(self.visit_expr(expr)?.eq(lit(false)).not()),
            SqlExpr::IsNotNull(expr) => Ok(self.visit_expr(expr)?.is_not_null()),
            SqlExpr::IsNotTrue(expr) => Ok(self.visit_expr(expr)?.eq(lit(true)).not()),
            SqlExpr::IsNull(expr) => Ok(self.visit_expr(expr)?.is_null()),
            SqlExpr::IsTrue(expr) => Ok(self.visit_expr(expr)?.eq(lit(true))),
            SqlExpr::Nested(expr) => self.visit_expr(expr),
            SqlExpr::Trim {
                expr,
                trim_where,
                trim_what,
            } => self.visit_trim(expr, trim_where, trim_what),
            SqlExpr::UnaryOp { op, expr } => self.visit_unary_op(op, expr),
            SqlExpr::Value(value) => self.visit_literal(value),
            e @ SqlExpr::Case { .. } => self.visit_when_then(e),
            other => {
                polars_bail!(InvalidOperation: "SQL expression {:?} is not yet supported", other)
            }
        }
    }

    /// Visit a compound identifier
    ///
    /// e.g. df.column or "df"."column"
    fn visit_compound_identifier(&self, idents: &[sqlparser::ast::Ident]) -> PolarsResult<Expr> {
        match idents {
            [tbl_name, column_name] => {
                let lf = self.ctx.table_map.get(&tbl_name.value).ok_or_else(|| {
                    polars_err!(
                        ComputeError: "no table named '{}' found",
                        tbl_name
                    )
                })?;

                let schema = lf.schema()?;
                if let Some((_, name, _)) = schema.get_full(&column_name.value) {
                    Ok(col(name))
                } else {
                    polars_bail!(
                        ColumnNotFound: "no column named '{}' found in table '{}'",
                        column_name,
                        tbl_name
                    )
                }
            }
            _ => polars_bail!(
                ComputeError: "Invalid identifier {:?}",
                idents
            ),
        }
    }

    fn visit_unary_op(&self, op: &UnaryOperator, expr: &SqlExpr) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        Ok(match op {
            UnaryOperator::Plus => lit(0) + expr,
            UnaryOperator::Minus => lit(0) - expr,
            UnaryOperator::Not => expr.not(),
            other => polars_bail!(InvalidOperation: "Unary operator {:?} is not supported", other),
        })
    }

    /// Visit a single identifier
    ///
    /// e.g. column
    fn visit_identifier(&self, ident: &sqlparser::ast::Ident) -> PolarsResult<Expr> {
        Ok(col(&ident.value))
    }

    /// Visit a binary operation
    ///
    /// e.g. column + 1 or column1 / column2
    fn visit_binary_op(
        &self,
        left: &SqlExpr,
        op: &BinaryOperator,
        right: &SqlExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let right = self.visit_expr(right)?;
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
                left.cast(DataType::Utf8) + right.cast(DataType::Utf8)
            }
            SQLBinaryOperator::Xor => left.xor(right),
            // ----
            // Regular expression operators
            // ----
            SQLBinaryOperator::PGRegexMatch => match right {
                Expr::Literal(LiteralValue::Utf8(_)) => left.str().contains(right, true),
                _ => polars_bail!(ComputeError: "Invalid pattern for '~' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexNotMatch => match right {
                Expr::Literal(LiteralValue::Utf8(_)) => left.str().contains(right, true).not(),
                _ => polars_bail!(ComputeError: "Invalid pattern for '!~' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexIMatch => match right {
                Expr::Literal(LiteralValue::Utf8(pat)) => {
                    left.str().contains(lit(format!("(?i){}", pat)), true)
                }
                _ => polars_bail!(ComputeError: "Invalid pattern for '~*' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexNotIMatch => match right {
                Expr::Literal(LiteralValue::Utf8(pat)) => {
                    left.str().contains(lit(format!("(?i){}", pat)), true).not()
                }
                _ => polars_bail!(ComputeError: "Invalid pattern for '!~*' operator: {:?}", right),
            },
            other => polars_bail!(ComputeError: "SQL operator {:?} is not yet supported", other),
        })
    }

    /// Visit a SQL function
    ///
    /// e.g. SUM(column) or COUNT(*)
    ///
    /// See [SqlFunctionVisitor] for more details
    fn visit_function(&self, function: &SQLFunction) -> PolarsResult<Expr> {
        let visitor = SqlFunctionVisitor {
            func: function,
            ctx: self.ctx,
        };
        visitor.visit_function()
    }

    /// Visit a SQL CAST
    ///
    /// e.g. `CAST(column AS INT)` or `column::INT`
    fn visit_cast(&self, expr: &SqlExpr, data_type: &SQLDataType) -> PolarsResult<Expr> {
        let polars_type = map_sql_polars_datatype(data_type)?;
        let expr = self.visit_expr(expr)?;

        Ok(expr.cast(polars_type))
    }

    /// Visit a SQL literal
    ///
    /// e.g. 1, 'foo', 1.0, NULL
    ///
    /// See [SqlValue] and [LiteralValue] for more details
    fn visit_literal(&self, value: &SqlValue) -> PolarsResult<Expr> {
        Ok(match value {
            SqlValue::Boolean(b) => lit(*b),
            SqlValue::DoubleQuotedString(s) => lit(s.clone()),
            SqlValue::HexStringLiteral(s) => lit(s.clone()),
            SqlValue::NationalStringLiteral(s) => lit(s.clone()),
            SqlValue::Null => Expr::Literal(LiteralValue::Null),
            SqlValue::Number(s, _) => {
                // Check for existence of decimal separator dot
                if s.contains('.') {
                    s.parse::<f64>().map(lit).map_err(|_| ())
                } else {
                    s.parse::<i64>().map(lit).map_err(|_| ())
                }
                .map_err(|_| polars_err!(ComputeError: "cannot parse literal: {:?}", s))?
            }
            SqlValue::SingleQuotedString(s) => lit(s.clone()),
            other => polars_bail!(ComputeError: "SQL value {:?} is not yet supported", other),
        })
    }

    // similar to visit_literal, but returns an AnyValue instead of Expr
    fn visit_anyvalue(&self, value: &SqlValue) -> PolarsResult<AnyValue> {
        Ok(match value {
            SqlValue::Boolean(b) => AnyValue::Boolean(*b),
            SqlValue::Null => AnyValue::Null,
            SqlValue::Number(s, _) => {
                // Check for existence of decimal separator dot
                if s.contains('.') {
                    s.parse::<f64>().map(AnyValue::Float64).map_err(|_| ())
                } else {
                    s.parse::<i64>().map(AnyValue::Int64).map_err(|_| ())
                }
                .map_err(|_| polars_err!(ComputeError: "cannot parse literal: {:?}"))?
            }
            SqlValue::SingleQuotedString(s)
            | SqlValue::NationalStringLiteral(s)
            | SqlValue::HexStringLiteral(s)
            | SqlValue::DoubleQuotedString(s) => AnyValue::Utf8Owned(s.into()),
            other => polars_bail!(ComputeError: "SQL value {:?} is not yet supported", other),
        })
    }

    /// Visit a SQL `BETWEEN` expression
    /// See [sqlparser::ast::Expr::Between] for more details
    fn visit_between(
        &self,
        expr: &SqlExpr,
        negated: bool,
        low: &SqlExpr,
        high: &SqlExpr,
    ) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        let low = self.visit_expr(low)?;
        let high = self.visit_expr(high)?;

        if negated {
            Ok(expr.clone().lt(low).or(expr.gt(high)))
        } else {
            Ok(expr.clone().gt_eq(low).and(expr.lt_eq(high)))
        }
    }

    /// Visit a SQL 'TRIM' function
    /// See [sqlparser::ast::Expr::Trim] for more details
    fn visit_trim(
        &self,
        expr: &SqlExpr,
        trim_where: &Option<TrimWhereField>,
        trim_what: &Option<Box<SqlExpr>>,
    ) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        let trim_what = trim_what.as_ref().map(|e| self.visit_expr(e)).transpose()?;
        let trim_what = match trim_what {
            Some(Expr::Literal(LiteralValue::Utf8(val))) => Some(val),
            None => None,
            _ => return self.err(&expr),
        };

        Ok(match (trim_where, trim_what) {
            (None | Some(TrimWhereField::Both), None) => expr.str().strip(None),
            (None | Some(TrimWhereField::Both), Some(val)) => expr.str().strip(Some(val)),
            (Some(TrimWhereField::Leading), None) => expr.str().lstrip(None),
            (Some(TrimWhereField::Leading), Some(val)) => expr.str().lstrip(Some(val)),
            (Some(TrimWhereField::Trailing), None) => expr.str().rstrip(None),
            (Some(TrimWhereField::Trailing), Some(val)) => expr.str().rstrip(Some(val)),
        })
    }

    /// Visit a SQL `ARRAY_AGG` expression
    fn visit_arr_agg(&self, expr: &ArrayAgg) -> PolarsResult<Expr> {
        let mut base = self.visit_expr(&expr.expr)?;

        if let Some(order_by) = expr.order_by.as_ref() {
            let (order_by, descending) = self.visit_order_by(order_by)?;
            base = base.sort_by(order_by, descending);
        }

        if let Some(limit) = &expr.limit {
            let limit = match self.visit_expr(limit)? {
                Expr::Literal(LiteralValue::UInt32(n)) => n as usize,
                Expr::Literal(LiteralValue::UInt64(n)) => n as usize,
                Expr::Literal(LiteralValue::Int32(n)) => n as usize,
                Expr::Literal(LiteralValue::Int64(n)) => n as usize,
                _ => polars_bail!(ComputeError: "limit in ARRAY_AGG must be a positive integer"),
            };
            base = base.head(Some(limit));
        }

        if expr.distinct {
            base = base.unique_stable();
        }

        polars_ensure!(
            !expr.within_group,
            ComputeError: "ARRAY_AGG WITHIN GROUP is not yet supported"
        );
        Ok(base.implode())
    }

    /// Visit a SQL `IN` expression
    fn visit_is_in(&self, expr: &SqlExpr, list: &[SqlExpr], negated: bool) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        let list = list
            .iter()
            .map(|e| {
                if let SqlExpr::Value(v) = e {
                    let av = self.visit_anyvalue(v)?;
                    Ok(av)
                } else {
                    Err(polars_err!(ComputeError: "SQL expression {:?} is not yet supported", e))
                }
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let s = Series::from_any_values("", &list, true)?;

        if negated {
            Ok(expr.is_in(lit(s)).not())
        } else {
            Ok(expr.is_in(lit(s)))
        }
    }

    fn visit_order_by(&self, order_by: &[OrderByExpr]) -> PolarsResult<(Vec<Expr>, Vec<bool>)> {
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

    fn visit_when_then(&self, expr: &SqlExpr) -> PolarsResult<Expr> {
        if let SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } = expr
        {
            if operand.is_some() {
                polars_bail!(ComputeError: "CASE operand is not yet supported");
            }

            polars_ensure!(
                conditions.len() == results.len(),
                ComputeError: "WHEN and THEN expressions must have the same length"
            );

            polars_ensure!(
                !conditions.is_empty(),
                ComputeError: "WHEN and THEN expressions must have at least one element"
            );

            let mut when_thens = conditions.iter().zip(results.iter());
            let first = when_thens.next();

            if first.is_none() {
                polars_bail!(ComputeError: "WHEN and THEN expressions must have at least one element");
            }

            let else_res = match else_result {
                Some(else_res) => self.visit_expr(else_res)?,
                None => polars_bail!(ComputeError: "ELSE expression is required"),
            };

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
        polars_bail!(ComputeError: "SQL expression {:?} is not yet supported", expr);
    }
}

pub(crate) fn parse_sql_expr(expr: &SqlExpr, ctx: &SQLContext) -> PolarsResult<Expr> {
    let visitor = SqlExprVisitor { ctx };
    visitor.visit_expr(expr)
}

pub(super) fn process_join_constraint(
    constraint: &JoinConstraint,
    left_name: &str,
    right_name: &str,
) -> PolarsResult<(Expr, Expr)> {
    if let JoinConstraint::On(SqlExpr::BinaryOp { left, op, right }) = constraint {
        match (left.as_ref(), right.as_ref()) {
            (SqlExpr::CompoundIdentifier(left), SqlExpr::CompoundIdentifier(right)) => {
                if left.len() == 2 && right.len() == 2 {
                    let tbl_a = &left[0].value;
                    let col_a = &left[1].value;
                    let tbl_b = &right[0].value;
                    let col_b = &right[1].value;

                    if let BinaryOperator::Eq = op {
                        if left_name == tbl_a && right_name == tbl_b {
                            return Ok((col(col_a), col(col_b)));
                        } else if left_name == tbl_b && right_name == tbl_a {
                            return Ok((col(col_b), col(col_a)));
                        }
                    }
                }
            }
            (SqlExpr::Identifier(left), SqlExpr::Identifier(right)) => {
                return Ok((col(&left.value), col(&right.value)))
            }
            _ => {}
        }
    }
    if let JoinConstraint::Using(idents) = constraint {
        if !idents.is_empty() {
            let cols = &idents[0].value;
            return Ok((col(cols), col(cols)));
        }
    }
    polars_bail!(InvalidOperation: "SQL join constraint {:?} is not yet supported", constraint);
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
    let ctx = SQLContext::new();

    let mut parser = Parser::new(&GenericDialect);
    parser = parser.with_options(ParserOptions {
        trailing_commas: true,
    });

    let mut ast = parser.try_with_sql(s.as_ref()).map_err(to_compute_err)?;

    let expr = ast.parse_expr().map_err(to_compute_err)?;

    parse_sql_expr(&expr, &ctx)
}
