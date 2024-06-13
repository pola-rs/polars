use polars_core::datatypes::Field;
use polars_utils::arena::{Arena, Node};

use super::*;
use crate::prelude::{AExpr, IRAggExpr};
use crate::reduce::sum::SumReduce;


struct ReductionImpl {
    reduce: Box<dyn Reduction>,
    prepare: Node
}

impl ReductionImpl {
    fn new(reduce: Box<dyn Reduction>, prepare: Node) -> Self {
        ReductionImpl {
            reduce,
            prepare
        }

    }

}

pub fn into_reduction(
    node: Node,
    expr_arena: Arena<AExpr>,
    field: &Field,
) -> ReductionImpl {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(node) => {
                ReductionImpl::new(
                    Box::new(SumReduce::new(field.dtype.clone())),
                    *node
                )
            },
            _ => todo!(),
        },
        _ => {
            todo!()
        },
    }
}
