use polars_core::error::feature_gated;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::extrema::*;
use super::sum::SumReduce;
use super::*;
use crate::reduce::len::LenReduce;
use crate::reduce::mean::MeanReduce;

/// Converts a node into a reduction + its associated selector expression.
pub fn into_reduction(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<(Box<dyn Reduction>, Node)> {
    let e = expr_arena.get(node);
    let field = e.to_field(schema, Context::Default, expr_arena)?;
    let out = match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(node) => (
                Box::new(SumReduce::new(field.dtype.clone())) as Box<dyn Reduction>,
                *node,
            ),
            IRAggExpr::Min {
                propagate_nans,
                input,
            } => {
                if *propagate_nans && field.dtype.is_float() {
                    feature_gated!("propagate_nans", {
                        let out: Box<dyn Reduction> = match field.dtype {
                            DataType::Float32 => Box::new(MinNanReduce::<Float32Type>::new()),
                            DataType::Float64 => Box::new(MinNanReduce::<Float64Type>::new()),
                            _ => unreachable!(),
                        };
                        (out, *input)
                    })
                } else {
                    (
                        Box::new(MinReduce::new(field.dtype.clone())) as Box<dyn Reduction>,
                        *input,
                    )
                }
            },
            IRAggExpr::Max {
                propagate_nans,
                input,
            } => {
                if *propagate_nans && field.dtype.is_float() {
                    feature_gated!("propagate_nans", {
                        let out: Box<dyn Reduction> = match field.dtype {
                            DataType::Float32 => Box::new(MaxNanReduce::<Float32Type>::new()),
                            DataType::Float64 => Box::new(MaxNanReduce::<Float64Type>::new()),
                            _ => unreachable!(),
                        };
                        (out, *input)
                    })
                } else {
                    (Box::new(MaxReduce::new(field.dtype.clone())) as _, *input)
                }
            },
            IRAggExpr::Mean(input) => {
                let out: Box<dyn Reduction> = Box::new(MeanReduce::new(field.dtype.clone()));
                (out, *input)
            },
            _ => unreachable!(),
        },
        AExpr::Len => {
            // Compute length on the first column, or if none exist we'll never
            // be called and correctly return 0 as length anyway.
            let out: Box<dyn Reduction> = Box::new(LenReduce::new());
            let expr = if let Some(first_column) = schema.iter_names().next() {
                expr_arena.add(AExpr::Column(first_column.as_str().into()))
            } else {
                expr_arena.add(AExpr::Literal(LiteralValue::Null))
            };
            (out, expr)
        },
        _ => unreachable!(),
    };
    Ok(out)
}
