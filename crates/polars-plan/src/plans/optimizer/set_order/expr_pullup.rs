use polars_utils::arena::Arena;

use crate::plans::AExpr;
use crate::plans::set_order::expr_pushdown::{
    ExprOutputOrder, FrameOrderObserved, ObservableOrderingsResolver,
};

pub fn is_output_ordered(aexpr: &AExpr, arena: &Arena<AExpr>, frame_ordered: bool) -> bool {
    use ExprOutputOrder as O;

    match ObservableOrderingsResolver::new(if frame_ordered {
        O::Independent
    } else {
        O::None
    })
    .resolve_observable_orderings(aexpr, arena)
    {
        Ok(O::None) => false,
        Ok(O::Independent) => true,

        Ok(O::Frame | O::Both) | Err(FrameOrderObserved) => {
            // It is a logic error to hit this branch, as that would mean that frame ordering was
            // introduced into the expression tree from a non-column node.
            //
            // In release mode just conservatively indicate ordered output.
            if cfg!(debug_assertions) {
                unreachable!()
            } else {
                true
            }
        },
    }
}
