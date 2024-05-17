use polars_core::datatypes::Field;
use polars_utils::arena::{Arena, Node};
use crate::prelude::{AExpr, IRAggExpr};
use crate::reduce::sum::SumReduce;
use super::*;


pub fn into_reduction(node: Node, expr_arena: Arena<AExpr>, field: &Field) {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(node) => {
                SumReduce::new()
                }
            }
        }
    }
}