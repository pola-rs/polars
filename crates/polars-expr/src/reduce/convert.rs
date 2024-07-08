use polars_core::datatypes::{Field, Float64Type};
use polars_core::prelude::{DataType, Float32Type};
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::*;
use super::sum::SumReduce;
use super::extrema::*;

pub fn can_convert_into_reduction(
    node: Node,
    expr_arena: Arena<AExpr>,
) -> bool {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Min {..}
            | IRAggExpr::Max {..}
            | IRAggExpr::Sum(_) => true,
            _ => false

        },
        _ => false
    }
}

pub fn into_reduction(
    node: Node,
    expr_arena: Arena<AExpr>,
    field: &Field,
) -> Option<(Box<dyn Reduction>, Node)> {
    let out = match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(node) => {
                (
                    Box::new(SumReduce::new(field.dtype.clone())) as Box<dyn Reduction>,
                    *node
                )
            },
            IRAggExpr::Min {
                propagate_nans,
                input
            } => {
                if *propagate_nans && field.dtype.is_float() {

                    let out: Box<dyn Reduction> = match field.dtype {
                        DataType::Float32 => Box::new(MinNanReduce::<Float32Type>::new()),
                        DataType::Float64 => Box::new(MinNanReduce::<Float64Type>::new()),
                        _ => unreachable!()
                    };
                    (
                        out,
                        *input
                    )
                } else {
                    (
                        Box::new(MinReduce::new(field.dtype.clone())) as Box<dyn Reduction>,
                        *input
                    )
                }
            },
            IRAggExpr::Max {
                propagate_nans,
                input
            } => {
                if *propagate_nans && field.dtype.is_float() {
                    let out: Box<dyn Reduction> = match field.dtype {
                        DataType::Float32 => Box::new(MaxNanReduce::<Float32Type>::new()),
                        DataType::Float64 => Box::new(MaxNanReduce::<Float64Type>::new()),
                        _ => unreachable!()
                    };
                    (
                        out,
                        *input
                    )

                } else {
                    (
                        Box::new(MaxReduce::new(field.dtype.clone())) as _,
                        *input
                    )
                }
            },
            _ => return None,
        },
        _ => {
            return None
        },
    };
    Some(out)
}
