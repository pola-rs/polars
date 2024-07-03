use polars_core::datatypes::{Field, Float64Type};
use polars_core::prelude::{DataType, Float32Type};
use polars_utils::arena::{Arena, Node};

use super::*;
use crate::prelude::{AExpr, IRAggExpr};
use super::sum::SumReduce;
use super::extrema::*;


pub fn into_reduction(
    node: Node,
    expr_arena: Arena<AExpr>,
    field: &Field,
) -> (Box<dyn Reduction>, Node) {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(node) => {
                (
                    Box::new(SumReduce::new(field.dtype.clone())),
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
                        Box::new(MinReduce::new(field.dtype.clone())),
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
                        Box::new(MaxReduce::new(field.dtype.clone())),
                        *input
                    )
                }
            },
            IRAggExpr::First(_) => {
                todo!()
            }
            _ => todo!(),
        },
        _ => {
            todo!()
        },
    }
}
