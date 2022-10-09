use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_plan::logical_plan::ArenaExprIter;
use polars_plan::prelude::{AAggExpr, AExpr};
use polars_utils::arena::{Arena, Node};
use polars_utils::IdxSize;

use crate::executors::sinks::groupby::aggregates::first::FirstAgg;
use crate::executors::sinks::groupby::aggregates::last::LastAgg;
use crate::executors::sinks::groupby::aggregates::{AggregateFn, SumAgg};
use crate::expressions::PhysicalPipedExpr;

pub fn can_convert_to_hash_agg(mut node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut can_run_partitioned = true;
    if expr_arena
        .iter(node)
        .map(|(_, ae)| {
            match ae {
                AExpr::Agg(_)
                | AExpr::Cast { .. }
                | AExpr::Literal(_)
                | AExpr::Column(_)
                | AExpr::BinaryExpr { .. }
                | AExpr::Ternary { .. }
                | AExpr::Alias(_, _) => {}
                _ => {
                    can_run_partitioned = false;
                }
            }
            ae
        })
        .filter(|ae| matches!(ae, AExpr::Agg(_)))
        .count()
        == 1
        && can_run_partitioned
    {
        // last expression must be agg or agg.alias
        if let AExpr::Alias(input, _) = expr_arena.get(node) {
            node = *input
        }
        match expr_arena.get(node) {
            AExpr::Agg(agg_fn) => {
                matches!(
                    agg_fn,
                    AAggExpr::Sum(_) | AAggExpr::First(_) | AAggExpr::Last(_)
                )
            }
            _ => false,
        }
    } else {
        false
    }
}

pub fn convert_to_hash_agg<F>(
    node: Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    to_physical: &F,
) -> (Arc<dyn PhysicalPipedExpr>, Box<dyn AggregateFn>)
where
    F: Fn(Node, &Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    match expr_arena.get(node) {
        AExpr::Alias(input, _) => convert_to_hash_agg(*input, expr_arena, schema, to_physical),
        AExpr::Agg(agg) => match agg {
            AAggExpr::Sum(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let agg_fn = match phys_expr.field(schema).unwrap().dtype.to_physical() {
                    DataType::Boolean => Box::new(SumAgg::<IdxSize>::new()) as Box<dyn AggregateFn>,
                    DataType::UInt8 => Box::new(SumAgg::<u8>::new()) as Box<dyn AggregateFn>,
                    DataType::UInt16 => Box::new(SumAgg::<u16>::new()) as Box<dyn AggregateFn>,
                    DataType::UInt32 => Box::new(SumAgg::<u32>::new()) as Box<dyn AggregateFn>,
                    DataType::UInt64 => Box::new(SumAgg::<u64>::new()) as Box<dyn AggregateFn>,
                    DataType::Int8 => Box::new(SumAgg::<i8>::new()) as Box<dyn AggregateFn>,
                    DataType::Int16 => Box::new(SumAgg::<i16>::new()) as Box<dyn AggregateFn>,
                    DataType::Int32 => Box::new(SumAgg::<i32>::new()) as Box<dyn AggregateFn>,
                    DataType::Int64 => Box::new(SumAgg::<i64>::new()) as Box<dyn AggregateFn>,
                    DataType::Float32 => Box::new(SumAgg::<f32>::new()) as Box<dyn AggregateFn>,
                    DataType::Float64 => Box::new(SumAgg::<f64>::new()) as Box<dyn AggregateFn>,
                    _ => unreachable!(),
                };
                (phys_expr, agg_fn)
            }
            AAggExpr::First(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let dtype = phys_expr.field(schema).unwrap().dtype;
                (
                    phys_expr,
                    Box::new(FirstAgg::new(dtype)) as Box<dyn AggregateFn>,
                )
            }
            AAggExpr::Last(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let dtype = phys_expr.field(schema).unwrap().dtype;
                (
                    phys_expr,
                    Box::new(LastAgg::new(dtype)) as Box<dyn AggregateFn>,
                )
            }
            _ => todo!(),
        },
        _ => todo!(),
    }
}
