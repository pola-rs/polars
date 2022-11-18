use std::any::Any;
use std::sync::Arc;

use polars_core::datatypes::Field;
use polars_core::error::PolarsResult;
use polars_core::prelude::{DataType, Series, IDX_DTYPE};
use polars_core::schema::Schema;
use polars_plan::logical_plan::ArenaExprIter;
use polars_plan::prelude::{AAggExpr, AExpr};
use polars_utils::arena::{Arena, Node};
use polars_utils::IdxSize;

use crate::executors::sinks::groupby::aggregates::count::CountAgg;
use crate::executors::sinks::groupby::aggregates::first::FirstAgg;
use crate::executors::sinks::groupby::aggregates::last::LastAgg;
use crate::executors::sinks::groupby::aggregates::mean::MeanAgg;
use crate::executors::sinks::groupby::aggregates::{AggregateFunction, SumAgg};
use crate::expressions::PhysicalPipedExpr;
use crate::operators::DataChunk;

struct Count {}

impl PhysicalPipedExpr for Count {
    fn evaluate(&self, _chunk: &DataChunk, _lazy_state: &dyn Any) -> PolarsResult<Series> {
        Ok(Series::new_empty("", &IDX_DTYPE))
    }

    fn field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        todo!()
    }
}

pub fn can_convert_to_hash_agg(mut node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut can_run_partitioned = true;
    if expr_arena
        .iter(node)
        .map(|(_, ae)| {
            match ae {
                AExpr::Agg(_)
                | AExpr::Count
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
        .filter(|ae| matches!(ae, AExpr::Agg(_) | AExpr::Count))
        .count()
        == 1
        && can_run_partitioned
    {
        // last expression must be agg or agg.alias
        if let AExpr::Alias(input, _) = expr_arena.get(node) {
            node = *input
        }
        match expr_arena.get(node) {
            AExpr::Count => true,
            AExpr::Agg(agg_fn) => {
                matches!(
                    agg_fn,
                    AAggExpr::Sum(_)
                        | AAggExpr::First(_)
                        | AAggExpr::Last(_)
                        | AAggExpr::Mean(_)
                        | AAggExpr::Count(_)
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
) -> (Arc<dyn PhysicalPipedExpr>, AggregateFunction)
where
    F: Fn(Node, &Arena<AExpr>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    match expr_arena.get(node) {
        AExpr::Alias(input, _) => convert_to_hash_agg(*input, expr_arena, schema, to_physical),
        AExpr::Count | AExpr::Agg(AAggExpr::Count(_)) => (
            Arc::new(Count {}),
            AggregateFunction::Count(CountAgg::new()),
        ),
        AExpr::Agg(agg) => match agg {
            AAggExpr::Sum(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let agg_fn = match phys_expr.field(schema).unwrap().dtype.to_physical() {
                    // Boolean is aggregated as the IDX type.
                    DataType::Boolean => {
                        if std::mem::size_of::<IdxSize>() == 4 {
                            AggregateFunction::SumU32(SumAgg::<u32>::new())
                        } else {
                            AggregateFunction::SumU64(SumAgg::<u64>::new())
                        }
                    }
                    // these are aggregated as i64 to prevent overflow
                    DataType::Int8 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::Int16 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::UInt8 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::UInt16 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    //  these stay true to there types
                    DataType::UInt32 => AggregateFunction::SumU32(SumAgg::<u32>::new()),
                    DataType::UInt64 => AggregateFunction::SumU64(SumAgg::<u64>::new()),
                    DataType::Int32 => AggregateFunction::SumI32(SumAgg::<i32>::new()),
                    DataType::Int64 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::Float32 => AggregateFunction::SumF32(SumAgg::<f32>::new()),
                    DataType::Float64 => AggregateFunction::SumF64(SumAgg::<f64>::new()),
                    _ => unreachable!(),
                };
                (phys_expr, agg_fn)
            }
            AAggExpr::Mean(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let agg_fn = match phys_expr.field(schema).unwrap().dtype.to_physical() {
                    dt if dt.is_integer() => AggregateFunction::MeanF64(MeanAgg::<f64>::new()),
                    // Boolean is aggregated as the IDX type.
                    DataType::Boolean => AggregateFunction::MeanF64(MeanAgg::<f64>::new()),
                    DataType::Float32 => AggregateFunction::MeanF32(MeanAgg::<f32>::new()),
                    DataType::Float64 => AggregateFunction::MeanF64(MeanAgg::<f64>::new()),
                    _ => unreachable!(),
                };
                (phys_expr, agg_fn)
            }
            AAggExpr::First(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let dtype = phys_expr.field(schema).unwrap().dtype;
                (phys_expr, AggregateFunction::First(FirstAgg::new(dtype)))
            }
            AAggExpr::Last(input) => {
                let phys_expr = to_physical(*input, expr_arena).unwrap();
                let dtype = phys_expr.field(schema).unwrap().dtype;
                (phys_expr, AggregateFunction::Last(LastAgg::new(dtype)))
            }
            agg => panic!("{:?} not yet implemented.", agg),
        },
        _ => todo!(),
    }
}
