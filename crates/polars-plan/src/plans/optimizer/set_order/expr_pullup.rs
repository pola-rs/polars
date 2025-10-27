use polars_utils::arena::Arena;

use crate::plans::AExpr;
use crate::plans::set_order::expr_pushdown::{
    ColumnOrderObserved, ObservableOrders, ObservableOrdersResolver,
};

/// Returns whether the output of this `AExpr` contains any observable ordering.
pub fn is_output_ordered(
    aexpr: &AExpr,
    arena: &Arena<AExpr>,
    // Whether the input DataFrame is ordered
    frame_ordered: bool,
) -> bool {
    use ObservableOrders as O;

    match ObservableOrdersResolver::new(
        if frame_ordered {
            O::Independent
        } else {
            O::None
        },
        arena,
    )
    .resolve_observable_orders(aexpr)
    {
        Ok(O::None) => false,
        Ok(O::Independent) => true,

        Ok(O::Column | O::Both) | Err(ColumnOrderObserved) => {
            // It is a logic error to hit this branch, as that would mean that column ordering was
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
