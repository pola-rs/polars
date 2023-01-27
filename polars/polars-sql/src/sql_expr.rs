use polars_core::prelude::*;
use polars_lazy::prelude::*;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, BinaryOperator, DataType as SQLDataType, Expr as SqlExpr,
    Function as SQLFunction, JoinConstraint, TrimWhereField, Value as SqlValue, WindowSpec,
};

use crate::context::TABLES;

fn map_sql_polars_datatype(data_type: &SQLDataType) -> PolarsResult<DataType> {
    Ok(match data_type {
        SQLDataType::Char(_)
        | SQLDataType::Varchar(_)
        | SQLDataType::Uuid
        | SQLDataType::Clob(_)
        | SQLDataType::Text
        | SQLDataType::String => DataType::Utf8,
        SQLDataType::Float(_) => DataType::Float32,
        SQLDataType::Real => DataType::Float32,
        SQLDataType::Double => DataType::Float64,
        SQLDataType::TinyInt(_) => DataType::Int8,
        SQLDataType::UnsignedTinyInt(_) => DataType::UInt8,
        SQLDataType::SmallInt(_) => DataType::Int16,
        SQLDataType::UnsignedSmallInt(_) => DataType::UInt16,
        SQLDataType::Int(_) => DataType::Int32,
        SQLDataType::UnsignedInt(_) => DataType::UInt32,
        SQLDataType::BigInt(_) => DataType::Int64,
        SQLDataType::UnsignedBigInt(_) => DataType::UInt64,

        SQLDataType::Boolean => DataType::Boolean,
        SQLDataType::Date => DataType::Date,
        SQLDataType::Time => DataType::Time,
        SQLDataType::Timestamp => DataType::Datetime(TimeUnit::Milliseconds, None),
        SQLDataType::Interval => DataType::Duration(TimeUnit::Milliseconds),
        SQLDataType::Array(inner_type) => {
            DataType::List(Box::new(map_sql_polars_datatype(inner_type)?))
        }
        _ => {
            return Err(PolarsError::ComputeError(
                format!(
                    "SQL Datatype {:?} was not supported in polars-sql yet!",
                    data_type
                )
                .into(),
            ))
        }
    })
}

fn cast_(expr: Expr, data_type: &SQLDataType) -> PolarsResult<Expr> {
    let polars_type = map_sql_polars_datatype(data_type)?;
    Ok(expr.cast(polars_type))
}

fn binary_op_(left: Expr, right: Expr, op: &SQLBinaryOperator) -> PolarsResult<Expr> {
    Ok(match op {
        SQLBinaryOperator::Plus => left + right,
        SQLBinaryOperator::Minus => left - right,
        SQLBinaryOperator::Multiply => left * right,
        SQLBinaryOperator::Divide => left / right,
        SQLBinaryOperator::Modulo => left % right,
        SQLBinaryOperator::StringConcat => left.cast(DataType::Utf8) + right.cast(DataType::Utf8),
        SQLBinaryOperator::Gt => left.gt(right),
        SQLBinaryOperator::Lt => left.lt(right),
        SQLBinaryOperator::GtEq => left.gt_eq(right),
        SQLBinaryOperator::LtEq => left.lt_eq(right),
        SQLBinaryOperator::Eq => left.eq(right),
        SQLBinaryOperator::NotEq => left.eq(right).not(),
        SQLBinaryOperator::And => left.and(right),
        SQLBinaryOperator::Or => left.or(right),
        SQLBinaryOperator::Xor => left.xor(right),
        _ => {
            return Err(PolarsError::ComputeError(
                format!("SQL Operator {:?} was not supported in polars-sql yet!", op).into(),
            ))
        }
    })
}

fn literal_expr(value: &SqlValue) -> PolarsResult<Expr> {
    Ok(match value {
        SqlValue::Number(s, _) => {
            // Check for existence of decimal separator dot
            if s.contains('.') {
                s.parse::<f64>().map(lit).map_err(|_| {
                    PolarsError::ComputeError(format!("Can't parse literal {:?}", s).into())
                })
            } else {
                s.parse::<i64>().map(lit).map_err(|_| {
                    PolarsError::ComputeError(format!("Can't parse literal {:?}", s).into())
                })
            }?
        }
        SqlValue::SingleQuotedString(s) => lit(s.clone()),
        SqlValue::NationalStringLiteral(s) => lit(s.clone()),
        SqlValue::HexStringLiteral(s) => lit(s.clone()),
        SqlValue::DoubleQuotedString(s) => lit(s.clone()),
        SqlValue::Boolean(b) => lit(*b),
        SqlValue::Null => Expr::Literal(LiteralValue::Null),
        _ => {
            return Err(PolarsError::ComputeError(
                format!(
                    "Parsing SQL Value {:?} was not supported in polars-sql yet!",
                    value
                )
                .into(),
            ))
        }
    })
}

pub(crate) fn parse_sql_expr(expr: &SqlExpr) -> PolarsResult<Expr> {
    let err = || {
        Err(PolarsError::ComputeError(
            format!(
                "Expression: {:?} was not supported in polars-sql yet. Please open a feature request.",
                expr
            )
            .into(),
        ))
    };

    Ok(match expr {
        SqlExpr::CompoundIdentifier(idents) => {
            if idents.len() != 2 {
                return err();
            }
            let tbl_name = &idents[0].value;
            let refers_main_table = TABLES.with(|cell| {
                let tables = cell.borrow();
                tables.len() == 1 && tables.contains(tbl_name)
            });

            if refers_main_table {
                col(&idents[1].value)
            } else {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Compounded identifier: {:?} is not yet supported  if multiple tables are registered",
                        expr
                    )
                        .into(),
                ));
            }
        }
        SqlExpr::Identifier(e) => col(&e.value),
        SqlExpr::BinaryOp { left, op, right } => {
            let left = parse_sql_expr(left)?;
            let right = parse_sql_expr(right)?;
            binary_op_(left, right, op)?
        }
        SqlExpr::Function(sql_function) => parse_sql_function(sql_function)?,
        SqlExpr::Cast { expr, data_type } => cast_(parse_sql_expr(expr)?, data_type)?,
        SqlExpr::Nested(expr) => parse_sql_expr(expr)?,
        SqlExpr::Value(value) => literal_expr(value)?,
        SqlExpr::IsNull(expr) => parse_sql_expr(expr)?.is_null(),
        SqlExpr::IsNotNull(expr) => parse_sql_expr(expr)?.is_not_null(),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let expr = parse_sql_expr(expr)?;
            let low = parse_sql_expr(low)?;
            let high = parse_sql_expr(high)?;

            if *negated {
                expr.clone().lt(low).or(expr.gt(high))
            } else {
                expr.clone().gt(low).and(expr.lt(high))
            }
        }
        SqlExpr::Trim {
            expr: sql_expr,
            trim_where,
        } => {
            let expr = parse_sql_expr(sql_expr)?;
            match trim_where {
                None => return Ok(expr.str().strip(None)),
                Some((TrimWhereField::Both, sql_expr)) => {
                    let lit = parse_sql_expr(sql_expr)?;
                    if let Expr::Literal(LiteralValue::Utf8(val)) = lit {
                        return Ok(expr.str().strip(Some(val)));
                    }
                }
                Some((TrimWhereField::Leading, sql_expr)) => {
                    let lit = parse_sql_expr(sql_expr)?;
                    if let Expr::Literal(LiteralValue::Utf8(val)) = lit {
                        return Ok(expr.str().lstrip(Some(val)));
                    }
                }
                Some((TrimWhereField::Trailing, sql_expr)) => {
                    let lit = parse_sql_expr(sql_expr)?;
                    if let Expr::Literal(LiteralValue::Utf8(val)) = lit {
                        return Ok(expr.str().rstrip(Some(val)));
                    }
                }
            }
            return err();
        }
        _ => return err(),
    })
}

fn apply_window_spec(expr: Expr, window_spec: &Option<WindowSpec>) -> PolarsResult<Expr> {
    Ok(match &window_spec {
        Some(window_spec) => {
            // Process for simple window specification, partition by first
            let partition_by = window_spec
                .partition_by
                .iter()
                .map(parse_sql_expr)
                .collect::<PolarsResult<Vec<_>>>()?;
            expr.over(partition_by)
            // Order by and Row range may not be supported at the moment
        }
        None => expr,
    })
}

fn parse_sql_function(sql_function: &SQLFunction) -> PolarsResult<Expr> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr};
    // Function name mostly do not have name space, so it mostly take the first args
    let function_name = sql_function.name.0[0].value.to_lowercase();
    let args: Vec<_> = sql_function
        .args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Named { arg, .. } => arg,
            FunctionArg::Unnamed(arg) => arg,
        })
        .collect();

    // single arg
    if let [FunctionArgExpr::Expr(sql_expr)] = args.as_slice() {
        let e = apply_window_spec(parse_sql_expr(sql_expr)?, &sql_function.over)?;
        Ok(match (function_name.as_str(), sql_function.distinct) {
            ("sum", false) => e.sum(),
            ("first", false) => e.first(),
            ("last", false) => e.last(),
            ("avg", false) => e.mean(),
            ("max", false) => e.max(),
            ("min", false) => e.min(),
            ("stddev" | "stddev_samp", false) => e.std(1),
            ("variance" | "var_samp", false) => e.var(1),
            ("array_agg", false) => e.list(),
            // Special case for wildcard args to count function.
            ("count", false) if matches!(args.as_slice(), [FunctionArgExpr::Wildcard]) => {
                lit(1i32).count()
            }
            ("count", false) => e.count(),
            ("count", true) => e.n_unique(),
            _ => {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Function {:?} with args {:?} was not supported in polars-sql yet!",
                        function_name, args
                    )
                    .into(),
                ))
            }
        })
    } else {
        Err(PolarsError::ComputeError(
            format!(
                "Function {:?} with args {:?} was not supported in polars-sql yet!",
                function_name, args
            )
            .into(),
        ))
    }
}

pub(super) fn process_join_constraint(
    constraint: &JoinConstraint,
    left_name: &str,
    right_name: &str,
) -> PolarsResult<(Expr, Expr)> {
    if let JoinConstraint::On(expr) = constraint {
        if let SqlExpr::BinaryOp { left, op, right } = expr {
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
    }
    Err(PolarsError::ComputeError(
        format!(
            "Join constraint {:?} not yet supported in polars-sql",
            constraint
        )
        .into(),
    ))
}
