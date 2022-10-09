use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_plan::prelude::{AAggExpr, AExpr};
use polars_utils::arena::{Arena, Node};

use crate::executors::sinks::groupby::aggregates::{AggregateFn, SumAgg};

pub fn can_convert_to_hash_agg(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            AAggExpr::Sum(input) => {
                matches!(expr_arena.get(*input), AExpr::Column(_))
            }
            _ => false,
        },
        _ => false,
    }
}

pub fn convert_to_hash_agg(
    node: Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> Option<(usize, Box<dyn AggregateFn>)> {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            AAggExpr::Sum(input) => {
                if let AExpr::Column(name) = expr_arena.get(*input) {
                    let (index, _, dtype) = schema.get_full(name).unwrap();
                    let agg = match dtype.to_physical() {
                        DataType::Int64 => {
                            Some(Box::new(SumAgg::<i64>::new()) as Box<dyn AggregateFn>)
                        }
                        DataType::Int32 => {
                            Some(Box::new(SumAgg::<i32>::new()) as Box<dyn AggregateFn>)
                        }
                        DataType::Float32 => {
                            Some(Box::new(SumAgg::<f32>::new()) as Box<dyn AggregateFn>)
                        }
                        DataType::Float64 => {
                            Some(Box::new(SumAgg::<f64>::new()) as Box<dyn AggregateFn>)
                        }
                        _ => None,
                    }?;

                    Some((index, agg))
                } else {
                    None
                }
            }
            _ => None,
        },
        _ => None,
    }
}
