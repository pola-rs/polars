use crate::utils::expr_to_root_column_name;
use crate::{
    dsl::{AggExpr, Expr, Operator},
    logical_plan::{LiteralValue, LogicalPlan},
};
use datafusion::datasource::MemTable;
use datafusion::logical_plan::ToDFSchema;
use datafusion::{
    logical_plan::{
        self as lpmod, col, when, Expr as DExpr, LogicalPlan as DLogicalPlan, Operator as DOperator,
    },
    physical_plan::aggregates::AggregateFunction,
    scalar::ScalarValue,
};
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod};
use polars_core::prelude::*;

pub fn to_datafusion_lit(lit: LiteralValue) -> Result<ScalarValue> {
    use LiteralValue::*;
    let sv = match lit {
        Boolean(v) => ScalarValue::Boolean(Some(v)),
        // should this be large utf8?
        Utf8(v) => ScalarValue::Utf8(Some(v)),
        UInt32(v) => ScalarValue::UInt32(Some(v)),
        Int32(v) => ScalarValue::Int32(Some(v)),
        Int64(v) => ScalarValue::Int64(Some(v)),
        Float32(v) => ScalarValue::Float32(Some(v)),
        Float64(v) => ScalarValue::Float64(Some(v)),
        #[cfg(all(feature = "temporal", feature = "dtype-date64"))]
        DateTime(v) => ScalarValue::Date64(Some(v.timestamp_millis())),
        lit => {
            return Err(PolarsError::Other(
                format!("Literal conversion for literal {:?} not yet supported", lit).into(),
            ))
        }
    };
    Ok(sv)
}

pub fn to_datafusion_op(op: Operator) -> DOperator {
    use Operator::*;

    match op {
        Eq => DOperator::Eq,
        NotEq => DOperator::NotEq,
        Lt => DOperator::Lt,
        LtEq => DOperator::LtEq,
        Gt => DOperator::Gt,
        GtEq => DOperator::GtEq,
        Plus => DOperator::Plus,
        Minus => DOperator::Minus,
        Multiply => DOperator::Multiply,
        Divide => DOperator::Divide,
        Modulus => DOperator::Modulus,
        And => DOperator::And,
        Or => DOperator::Or,
    }
}

pub fn to_datafusion_expr(expr: Expr) -> Result<DExpr> {
    use Expr::*;
    let root_name = expr_to_root_column_name(&expr)?;

    let expr = match expr {
        Alias(e, name) => DExpr::Alias(Box::new(to_datafusion_expr(*e)?), name.to_string()),
        Column(name) => DExpr::Column(name.to_string()),
        Literal(lv) => DExpr::Literal(to_datafusion_lit(lv)?),
        BinaryExpr { left, op, right } => DExpr::BinaryExpr {
            left: Box::new(to_datafusion_expr(*left)?),
            op: to_datafusion_op(op),
            right: Box::new(to_datafusion_expr(*right)?),
        },
        Not(e) => DExpr::Not(Box::new(to_datafusion_expr(*e)?)),
        IsNull(e) => DExpr::IsNull(Box::new(to_datafusion_expr(*e)?)),
        IsNotNull(e) => DExpr::IsNotNull(Box::new(to_datafusion_expr(*e)?)),
        Cast { expr, data_type } => DExpr::Cast {
            expr: Box::new(to_datafusion_expr(*expr)?),
            data_type: data_type.to_arrow(),
        },
        Sort { expr, reverse } => DExpr::Sort {
            expr: Box::new(to_datafusion_expr(*expr)?),
            asc: !reverse,
            nulls_first: true,
        },
        // an aggregation in polars has a different output name than one in DF, so
        // we add an extra alias.
        Agg(ae) => {
            let (agg_expr, agg_method) =
                match ae {
                    AggExpr::Min(e) => (
                        DExpr::AggregateFunction {
                            fun: AggregateFunction::Min,
                            args: vec![to_datafusion_expr(*e)?],
                            distinct: false,
                        },
                        GroupByMethod::Min,
                    ),
                    AggExpr::Max(e) => (
                        DExpr::AggregateFunction {
                            fun: AggregateFunction::Max,
                            args: vec![to_datafusion_expr(*e)?],
                            distinct: false,
                        },
                        GroupByMethod::Max,
                    ),
                    AggExpr::Sum(e) => (
                        DExpr::AggregateFunction {
                            fun: AggregateFunction::Sum,
                            args: vec![to_datafusion_expr(*e)?],
                            distinct: false,
                        },
                        GroupByMethod::Sum,
                    ),
                    AggExpr::Count(e) => (
                        DExpr::AggregateFunction {
                            fun: AggregateFunction::Count,
                            args: vec![to_datafusion_expr(*e)?],
                            distinct: false,
                        },
                        GroupByMethod::Count,
                    ),
                    AggExpr::Mean(e) => (
                        DExpr::AggregateFunction {
                            fun: AggregateFunction::Avg,
                            args: vec![to_datafusion_expr(*e)?],
                            distinct: false,
                        },
                        GroupByMethod::Mean,
                    ),
                    _ => return Err(PolarsError::Other(
                        "this aggregation is not yet supported in polars to datafusion conversion"
                            .into(),
                    )),
                };
            let out_name = fmt_groupby_column(&root_name, agg_method);
            DExpr::Alias(Box::new(agg_expr), out_name)
        }
        Ternary {
            predicate,
            truthy,
            falsy,
        } => when(
            to_datafusion_expr(*predicate)?,
            to_datafusion_expr(*truthy)?,
        )
        .otherwise(to_datafusion_expr(*falsy)?)
        .unwrap(),
        _ => {
            return Err(PolarsError::Other(
                format!(
                    "expr {:?} not yet supported in polars to datafustion conversion",
                    expr
                )
                .into(),
            ))
        }
    };
    Ok(expr)
}

pub fn to_datafusion_lp(lp: LogicalPlan) -> Result<DLogicalPlan> {
    use LogicalPlan::*;

    let out = match lp {
        Selection { input, predicate } => DLogicalPlan::Filter {
            input: Arc::new(to_datafusion_lp(*input)?),
            predicate: to_datafusion_expr(predicate)?,
        },
        Projection {
            input,
            expr,
            schema,
        } => DLogicalPlan::Projection {
            expr: expr
                .into_iter()
                .map(to_datafusion_expr)
                .collect::<Result<_>>()?,
            input: Arc::new(to_datafusion_lp(*input)?),
            schema: Arc::new(schema.to_arrow().to_dfschema().unwrap()),
        },
        Aggregate {
            input,
            keys,
            aggs,
            schema,
            apply,
        } => {
            if apply.is_some() {
                return Err(PolarsError::Other(
                    "Custom functions not yet supported in Polars to DataFusion conversion".into(),
                ));
            }
            DLogicalPlan::Aggregate {
                input: Arc::new(to_datafusion_lp(*input)?),
                group_expr: keys
                    .iter()
                    .map(|e| to_datafusion_expr(e.clone()))
                    .collect::<Result<_>>()?,
                aggr_expr: aggs
                    .into_iter()
                    .map(to_datafusion_expr)
                    .collect::<Result<_>>()?,
                schema: Arc::new(schema.to_arrow().to_dfschema().unwrap()),
            }
        }
        Sort {
            input,
            by_column,
            reverse,
        } => DLogicalPlan::Sort {
            input: Arc::new(to_datafusion_lp(*input)?),
            expr: vec![col(&by_column).sort(!reverse, true)],
        },
        Join {
            input_left,
            input_right,
            schema,
            how,
            left_on,
            right_on,
            ..
        } => {
            let join_type = match how {
                JoinType::Inner => lpmod::JoinType::Inner,
                JoinType::Left => lpmod::JoinType::Left,
                JoinType::Outer => {
                    return Err(PolarsError::Other(
                        "outer join not yet supported by DataFusion backend".into(),
                    ))
                }
            };
            DLogicalPlan::Join {
                left: Arc::new(to_datafusion_lp(*input_left)?),
                right: Arc::new(to_datafusion_lp(*input_right)?),
                schema: Arc::new(schema.to_arrow().to_dfschema().unwrap()),
                on: left_on
                    .into_iter()
                    .zip(right_on.into_iter())
                    .map(|(l, r)| match (l, r) {
                        (Expr::Column(l), Expr::Column(r)) => Ok((l.to_string(), r.to_string())),
                        _ => Err(PolarsError::Other("can only join on columns".into())),
                    })
                    .collect::<Result<Vec<_>>>()?,
                join_type,
            }
        }
        Slice { input, offset, len } if offset == 0 => DLogicalPlan::Limit {
            n: len,
            input: Arc::new(to_datafusion_lp(*input)?),
        },
        DataFrameScan { df, schema, .. } => {
            let rbs = df.as_record_batches()?;
            let projected_schema = Arc::new(schema.to_arrow().to_dfschema().unwrap());
            let schema = Arc::new(schema.to_arrow());
            let provider = MemTable::try_new(schema, vec![rbs]).unwrap();
            let ptr = Arc::as_ptr(&df) as usize;
            let table_name = format!("{}", ptr);

            DLogicalPlan::TableScan {
                table_name,
                source: Arc::new(provider),
                projection: None,
                filters: vec![],
                projected_schema,
                limit: None,
            }
        }
        _ => todo!(),
    };
    Ok(out)
}
