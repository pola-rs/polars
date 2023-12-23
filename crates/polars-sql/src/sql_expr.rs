use polars_core::prelude::*;
use polars_error::to_compute_err;
use polars_lazy::dsl::Expr;
use polars_lazy::prelude::*;
use polars_plan::prelude::LiteralValue::Null;
use polars_plan::prelude::{col, lit, when};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use sqlparser::ast::{
    ArrayAgg, ArrayElemTypeDef, BinaryOperator as SQLBinaryOperator, BinaryOperator, CastFormat,
    DataType as SQLDataType, Expr as SqlExpr, Function as SQLFunction, Ident, JoinConstraint,
    OrderByExpr, Query as Subquery, SelectItem, TrimWhereField, UnaryOperator, Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::functions::SqlFunctionVisitor;
use crate::SQLContext;

pub(crate) fn map_sql_polars_datatype(data_type: &SQLDataType) -> PolarsResult<DataType> {
    Ok(match data_type {
        SQLDataType::Array(ArrayElemTypeDef::AngleBracket(inner_type))
        | SQLDataType::Array(ArrayElemTypeDef::SquareBracket(inner_type)) => {
            DataType::List(Box::new(map_sql_polars_datatype(inner_type)?))
        },
        SQLDataType::BigInt(_) => DataType::Int64,
        SQLDataType::Binary(_) | SQLDataType::Blob(_) | SQLDataType::Varbinary(_) => {
            DataType::Binary
        },
        SQLDataType::Boolean => DataType::Boolean,
        SQLDataType::Char(_)
        | SQLDataType::CharVarying(_)
        | SQLDataType::Character(_)
        | SQLDataType::CharacterVarying(_)
        | SQLDataType::Clob(_)
        | SQLDataType::String(_)
        | SQLDataType::Text
        | SQLDataType::Uuid
        | SQLDataType::Varchar(_) => DataType::String,
        SQLDataType::Date => DataType::Date,
        SQLDataType::Double | SQLDataType::DoublePrecision => DataType::Float64,
        SQLDataType::Float(_) => DataType::Float32,
        SQLDataType::Int(_) | SQLDataType::Integer(_) => DataType::Int32,
        SQLDataType::Interval => DataType::Duration(TimeUnit::Milliseconds),
        SQLDataType::Real => DataType::Float32,
        SQLDataType::SmallInt(_) => DataType::Int16,
        SQLDataType::Time { .. } => DataType::Time,
        SQLDataType::Timestamp { .. } => DataType::Datetime(TimeUnit::Milliseconds, None),
        SQLDataType::TinyInt(_) => DataType::Int8,
        SQLDataType::UnsignedBigInt(_) => DataType::UInt64,
        SQLDataType::UnsignedInt(_) | SQLDataType::UnsignedInteger(_) => DataType::UInt32,
        SQLDataType::UnsignedSmallInt(_) => DataType::UInt16,
        SQLDataType::UnsignedTinyInt(_) => DataType::UInt8,

        _ => polars_bail!(ComputeError: "SQL datatype {:?} is not yet supported", data_type),
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
pub(crate) struct SqlExprVisitor<'a> {
    ctx: &'a mut SQLContext,
}

impl SqlExprVisitor<'_> {
    fn visit_expr(&mut self, expr: &SqlExpr) -> PolarsResult<Expr> {
        match expr {
            SqlExpr::AllOp {
                left,
                compare_op,
                right,
            } => self.visit_all(left, compare_op, right),
            SqlExpr::AnyOp {
                left,
                compare_op,
                right,
            } => self.visit_any(left, compare_op, right),
            SqlExpr::ArrayAgg(expr) => self.visit_arr_agg(expr),
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => self.visit_between(expr, *negated, low, high),
            SqlExpr::BinaryOp { left, op, right } => self.visit_binary_op(left, op, right),
            SqlExpr::Cast {
                expr,
                data_type,
                format,
            } => self.visit_cast(expr, data_type, format),
            SqlExpr::Ceil { expr, .. } => Ok(self.visit_expr(expr)?.ceil()),
            SqlExpr::CompoundIdentifier(idents) => self.visit_compound_identifier(idents),
            SqlExpr::Floor { expr, .. } => Ok(self.visit_expr(expr)?.floor()),
            SqlExpr::Function(function) => self.visit_function(function),
            SqlExpr::Identifier(ident) => self.visit_identifier(ident),
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => self.visit_in_list(expr, list, *negated),
            SqlExpr::InSubquery {
                expr,
                subquery,
                negated,
            } => self.visit_in_subquery(expr, subquery, *negated),
            SqlExpr::Subquery(_) => polars_bail!(InvalidOperation: "Unexpected SQL Subquery"),
            SqlExpr::IsDistinctFrom(e1, e2) => {
                Ok(self.visit_expr(e1)?.neq_missing(self.visit_expr(e2)?))
            },
            SqlExpr::IsNotDistinctFrom(e1, e2) => {
                Ok(self.visit_expr(e1)?.eq_missing(self.visit_expr(e2)?))
            },
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
                trim_characters,
            } => self.visit_trim(expr, trim_where, trim_what, trim_characters),
            SqlExpr::UnaryOp { op, expr } => self.visit_unary_op(op, expr),
            SqlExpr::Value(value) => self.visit_literal(value),
            e @ SqlExpr::Case { .. } => self.visit_when_then(e),
            other => {
                polars_bail!(InvalidOperation: "SQL expression {:?} is not yet supported", other)
            },
        }
    }

    fn visit_subquery(
        &mut self,
        subquery: &Subquery,
        restriction: SubqueryRestriction,
    ) -> PolarsResult<Expr> {
        if subquery.with.is_some() {
            polars_bail!(InvalidOperation: "SQL subquery cannot be given CTEs");
        }

        let mut lf = self.ctx.execute_query_no_ctes(subquery)?;

        let schema = lf.schema()?;
        if restriction == SubqueryRestriction::SingleColumn {
            if schema.len() != 1 {
                polars_bail!(InvalidOperation: "SQL subquery will return more than one column");
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

        polars_bail!(InvalidOperation: "SQL subquery type not supported");
    }

    /// Visit a compound identifier
    ///
    /// e.g. df.column or "df"."column"
    fn visit_compound_identifier(&self, idents: &[Ident]) -> PolarsResult<Expr> {
        match idents {
            [tbl_name, column_name] => {
                let lf = self
                    .ctx
                    .get_table_from_current_scope(&tbl_name.value)
                    .ok_or_else(|| {
                        polars_err!(
                            ComputeError: "no table or alias named '{}' found",
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
            },
            _ => polars_bail!(
                ComputeError: "Invalid identifier {:?}",
                idents
            ),
        }
    }

    fn visit_unary_op(&mut self, op: &UnaryOperator, expr: &SqlExpr) -> PolarsResult<Expr> {
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
    fn visit_identifier(&self, ident: &Ident) -> PolarsResult<Expr> {
        Ok(col(&ident.value))
    }

    /// Visit a binary operation
    ///
    /// e.g. column + 1 or column1 / column2
    fn visit_binary_op(
        &mut self,
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
                left.cast(DataType::String) + right.cast(DataType::String)
            },
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
                },
                _ => polars_bail!(ComputeError: "Invalid pattern for '~*' operator: {:?}", right),
            },
            SQLBinaryOperator::PGRegexNotIMatch => match right {
                Expr::Literal(LiteralValue::Utf8(pat)) => {
                    left.str().contains(lit(format!("(?i){}", pat)), true).not()
                },
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
    fn visit_function(&mut self, function: &SQLFunction) -> PolarsResult<Expr> {
        let mut visitor = SqlFunctionVisitor {
            func: function,
            ctx: self.ctx,
        };
        visitor.visit_function()
    }

    /// Visit a SQL ALL
    ///
    /// e.g. `a > ALL(y)`
    fn visit_all(
        &mut self,
        left: &SqlExpr,
        compare_op: &BinaryOperator,
        right: &SqlExpr,
    ) -> PolarsResult<Expr> {
        let left = self.visit_expr(left)?;
        let right = self.visit_expr(right)?;

        match compare_op {
            BinaryOperator::Gt => Ok(left.gt(right.max())),
            BinaryOperator::Lt => Ok(left.lt(right.min())),
            BinaryOperator::GtEq => Ok(left.gt_eq(right.max())),
            BinaryOperator::LtEq => Ok(left.lt_eq(right.min())),
            BinaryOperator::Eq => polars_bail!(ComputeError: "ALL cannot be used with ="),
            BinaryOperator::NotEq => polars_bail!(ComputeError: "ALL cannot be used with !="),
            _ => polars_bail!(ComputeError: "Invalid comparison operator"),
        }
    }

    /// Visit a SQL ANY
    ///
    /// e.g. `a != ANY(y)`
    fn visit_any(
        &mut self,
        left: &SqlExpr,
        compare_op: &BinaryOperator,
        right: &SqlExpr,
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
            _ => polars_bail!(ComputeError: "Invalid comparison operator"),
        }
    }

    /// Visit a SQL CAST
    ///
    /// e.g. `CAST(column AS INT)` or `column::INT`
    fn visit_cast(
        &mut self,
        expr: &SqlExpr,
        data_type: &SQLDataType,
        format: &Option<CastFormat>,
    ) -> PolarsResult<Expr> {
        if format.is_some() {
            return Err(polars_err!(ComputeError: "unsupported use of FORMAT in CAST expression"));
        }
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
            },
            SqlValue::SingleQuotedString(s) => lit(s.clone()),
            other => polars_bail!(ComputeError: "SQL value {:?} is not yet supported", other),
        })
    }

    /// Visit a SQL literal (like [visit_literal]), but return AnyValue instead of Expr
    fn visit_anyvalue(
        &self,
        value: &SqlValue,
        op: Option<&UnaryOperator>,
    ) -> PolarsResult<AnyValue> {
        Ok(match value {
            SqlValue::Boolean(b) => AnyValue::Boolean(*b),
            SqlValue::Null => AnyValue::Null,
            SqlValue::Number(s, _) => {
                let negate = match op {
                    Some(UnaryOperator::Minus) => true,
                    // no op should be taken as plus.
                    Some(UnaryOperator::Plus) | None => false,
                    Some(op) => {
                        polars_bail!(ComputeError: "Unary op {:?} not supported for numeric SQL value", op)
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
                .map_err(|_| polars_err!(ComputeError: "cannot parse literal: {s:?}"))?
            },
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
        &mut self,
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
        &mut self,
        expr: &SqlExpr,
        trim_where: &Option<TrimWhereField>,
        trim_what: &Option<Box<SqlExpr>>,
        trim_characters: &Option<Vec<SqlExpr>>,
    ) -> PolarsResult<Expr> {
        if trim_characters.is_some() {
            // TODO: allow compact snowflake/bigquery syntax?
            return Err(polars_err!(ComputeError: "unsupported TRIM syntax"));
        };
        let expr = self.visit_expr(expr)?;
        let trim_what = trim_what.as_ref().map(|e| self.visit_expr(e)).transpose()?;
        let trim_what = match trim_what {
            Some(Expr::Literal(LiteralValue::Utf8(val))) => Some(val),
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

    /// Visit a SQL `ARRAY_AGG` expression
    fn visit_arr_agg(&mut self, expr: &ArrayAgg) -> PolarsResult<Expr> {
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
    fn visit_in_list(
        &mut self,
        expr: &SqlExpr,
        list: &[SqlExpr],
        negated: bool,
    ) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;
        let list = list
            .iter()
            .map(|e| {
                if let SqlExpr::Value(v) = e {
                    let av = self.visit_anyvalue(v, None)?;
                    Ok(av)
                } else if let SqlExpr::UnaryOp {op, expr} = e {
                    match expr.as_ref() {
                        SqlExpr::Value(v) => {
                            let av = self.visit_anyvalue(v, Some(op))?;
                            Ok(av)
                        },
                        _ => Err(polars_err!(ComputeError: "SQL expression {:?} is not yet supported", e))
                    }
                }else{
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

    fn visit_in_subquery(
        &mut self,
        expr: &SqlExpr,
        subquery: &Subquery,
        negated: bool,
    ) -> PolarsResult<Expr> {
        let expr = self.visit_expr(expr)?;

        let subquery_result = self.visit_subquery(subquery, SubqueryRestriction::SingleColumn)?;

        if negated {
            Ok(expr.is_in(subquery_result).not())
        } else {
            Ok(expr.is_in(subquery_result))
        }
    }

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

    fn visit_when_then(&mut self, expr: &SqlExpr) -> PolarsResult<Expr> {
        if let SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } = expr
        {
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
        polars_bail!(ComputeError: "SQL expression {:?} is not yet supported", expr);
    }
}

pub(crate) fn parse_sql_expr(expr: &SqlExpr, ctx: &mut SQLContext) -> PolarsResult<Expr> {
    let mut visitor = SqlExprVisitor { ctx };
    visitor.visit_expr(expr)
}

pub(super) fn process_join(
    left_tbl: LazyFrame,
    right_tbl: LazyFrame,
    constraint: &JoinConstraint,
    tbl_name: &str,
    join_tbl_name: &str,
    join_type: JoinType,
) -> PolarsResult<LazyFrame> {
    let (left_on, right_on) = process_join_constraint(constraint, tbl_name, join_tbl_name)?;

    Ok(left_tbl
        .join_builder()
        .with(right_tbl)
        .left_on(left_on)
        .right_on(right_on)
        .how(join_type)
        .finish())
}

pub(super) fn process_join_constraint(
    constraint: &JoinConstraint,
    left_name: &str,
    right_name: &str,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    if let JoinConstraint::On(SqlExpr::BinaryOp { left, op, right }) = constraint {
        if op != &BinaryOperator::Eq {
            polars_bail!(InvalidOperation:
                "SQL interface (currently) only supports basic equi-join \
                 constraints; found '{:?}' op in\n{:?}", op, constraint)
        }
        match (left.as_ref(), right.as_ref()) {
            (SqlExpr::CompoundIdentifier(left), SqlExpr::CompoundIdentifier(right)) => {
                if left.len() == 2 && right.len() == 2 {
                    let (tbl_a, col_a) = (&left[0].value, &left[1].value);
                    let (tbl_b, col_b) = (&right[0].value, &right[1].value);

                    if left_name == tbl_a && right_name == tbl_b {
                        return Ok((vec![col(col_a)], vec![col(col_b)]));
                    } else if left_name == tbl_b && right_name == tbl_a {
                        return Ok((vec![col(col_b)], vec![col(col_a)]));
                    }
                }
            },
            (SqlExpr::Identifier(left), SqlExpr::Identifier(right)) => {
                return Ok((vec![col(&left.value)], vec![col(&right.value)]))
            },
            _ => {},
        }
    }
    if let JoinConstraint::Using(idents) = constraint {
        if !idents.is_empty() {
            let mut using = Vec::with_capacity(idents.len());
            using.extend(idents.iter().map(|id| col(&id.value)));
            return Ok((using.clone(), using.clone()));
        }
    }
    polars_bail!(InvalidOperation: "Unsupported SQL join constraint:\n{:?}", constraint);
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

    let mut ast = parser.try_with_sql(s.as_ref()).map_err(to_compute_err)?;
    let expr = ast.parse_select_item().map_err(to_compute_err)?;

    Ok(match &expr {
        SelectItem::ExprWithAlias { expr, alias } => {
            let expr = parse_sql_expr(expr, &mut ctx)?;
            expr.alias(&alias.value)
        },
        SelectItem::UnnamedExpr(expr) => parse_sql_expr(expr, &mut ctx)?,
        _ => polars_bail!(InvalidOperation: "Unable to parse '{}' as Expr", s.as_ref()),
    })
}
