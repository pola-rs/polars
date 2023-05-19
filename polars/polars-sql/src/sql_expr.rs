use polars_core::prelude::*;
use polars_lazy::dsl::Expr;
use polars_lazy::prelude::*;
use sqlparser::ast::{
    ArrayAgg, BinaryOperator as SQLBinaryOperator, BinaryOperator, DataType as SQLDataType,
    Expr as SqlExpr, Function as SQLFunction, JoinConstraint, OrderByExpr, TrimWhereField,
    UnaryOperator, Value as SqlValue,
};

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
            other => polars_bail!(ComputeError: "SQL expression {:?} is not yet supported", other),
        }
    }

    /// Visit a compound identifier
    ///
    /// e.g. df.column or "df"."column"
    fn visit_compound_identifier(&self, idents: &[sqlparser::ast::Ident]) -> PolarsResult<Expr> {
        polars_ensure!(
            idents.len() == 2,
            ComputeError: "compound identifier {:?} is not yet supported", idents,
        );
        let tbl_name = &idents[0].value;
        let refers_main_table =
            { self.ctx.table_map.len() == 1 && self.ctx.table_map.contains_key(tbl_name) };
        polars_ensure!(
            refers_main_table, ComputeError:
            "compound identifier {:?} is not yet supported if multiple tables are registered",
            idents
        );
        Ok(col(&idents[1].value))
    }

    fn visit_unary_op(&self, op: &UnaryOperator, expr: &SqlExpr) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        Ok(match op {
            UnaryOperator::Plus => lit(0) + expr,
            UnaryOperator::Minus => lit(0) - expr,
            UnaryOperator::Not => expr.not(),
            other => polars_bail!(ComputeError: "Unary operator {:?} is not supported", other),
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
            SQLBinaryOperator::StringConcat => {
                left.cast(DataType::Utf8) + right.cast(DataType::Utf8)
            }
            SQLBinaryOperator::Xor => left.xor(right),
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
                .map_err(|_| polars_err!(ComputeError: "cannot parse literal: {:?}"))?
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
            base = base.sort_by(vec![order_by], vec![descending]);
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

    fn visit_order_by(&self, order_by: &OrderByExpr) -> PolarsResult<(Expr, bool)> {
        let expr = self.visit_expr(&order_by.expr)?;
        let descending = order_by.asc.unwrap_or(false);
        Ok((expr, descending))
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
    polars_bail!(ComputeError: "SQL join constraint {:?} is not yet supported", constraint);
}
