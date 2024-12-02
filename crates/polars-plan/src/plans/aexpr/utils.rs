use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::*;

/// Checks if the top-level expression node is elementwise. If this is the case, then `stack` will
/// be extended further with any nested expression nodes.
pub fn is_elementwise(stack: &mut UnitVec<Node>, ae: &AExpr, expr_arena: &Arena<AExpr>) -> bool {
    use AExpr::*;

    if !ae.is_elementwise_top_level() {
        return false;
    }

    match ae {
        // Literals that aren't being projected are allowed to be non-scalar, so we don't add them
        // for inspection. (e.g. `is_in(<literal>)`).
        #[cfg(feature = "is_in")]
        Function {
            function: FunctionExpr::Boolean(BooleanFunction::IsIn),
            input,
            ..
        } => (|| {
            if let Some(rhs) = input.get(1) {
                assert_eq!(input.len(), 2); // A.is_in(B)
                let rhs = rhs.node();

                if matches!(expr_arena.get(rhs), AExpr::Literal { .. }) {
                    stack.extend([input[0].node()]);
                    return;
                }
            };

            ae.nodes(stack);
        })(),
        _ => ae.nodes(stack),
    }

    true
}

/// Recursive variant of `is_elementwise`
pub fn is_elementwise_rec<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>) -> bool {
    let mut stack = unitvec![];

    loop {
        if !is_elementwise(&mut stack, ae, expr_arena) {
            return false;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}

/// Recursive variant of `is_elementwise` that also forbids casting to categoricals. This function
/// is used to determine if an expression evaluation can be vertically parallelized.
pub fn is_elementwise_rec_no_cat_cast<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>) -> bool {
    let mut stack = unitvec![];

    loop {
        if !is_elementwise(&mut stack, ae, expr_arena) {
            return false;
        }

        #[cfg(feature = "dtype-categorical")]
        {
            if let AExpr::Cast {
                dtype: DataType::Categorical(..),
                ..
            } = ae
            {
                return false;
            }
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}

/// Check whether filters can be pushed past this expression.
///
/// A query, `with_columns(C).filter(P)` can be re-ordered as `filter(P).with_columns(C)`, iff
/// both P and C permit filter pushdown.
///
/// If filter pushdown is permitted, `stack` is extended with any input expression nodes that this
/// expression may have.
///
/// Note that this  function is not recursive - the caller should repeatedly
/// call this function with the `stack` to perform a recursive check.
pub(crate) fn permits_filter_pushdown(
    stack: &mut UnitVec<Node>,
    ae: &AExpr,
    expr_arena: &Arena<AExpr>,
) -> bool {
    // This is a subset of an `is_elementwise` check that also blocks exprs that raise errors
    // depending on the data. The idea is that, although the success value of these functions
    // are elementwise, their error behavior is non-elementwise. Their error behavior is essentially
    // performing an aggregation `ANY(evaluation_result_was_error)`, and if this is the case then
    // the query result should be an error.
    match ae {
        // Rows that go OOB on get/gather may be filtered out in earlier operations,
        // so we don't push these down.
        AExpr::Function {
            function: FunctionExpr::ListExpr(ListFunction::Get(false)),
            ..
        } => false,
        #[cfg(feature = "list_gather")]
        AExpr::Function {
            function: FunctionExpr::ListExpr(ListFunction::Gather(false)),
            ..
        } => false,
        #[cfg(feature = "dtype-array")]
        AExpr::Function {
            function: FunctionExpr::ArrayExpr(ArrayFunction::Get(false)),
            ..
        } => false,
        // TODO: There are a lot more functions that should be caught here.
        ae => is_elementwise(stack, ae, expr_arena),
    }
}

pub fn permits_filter_pushdown_rec<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>) -> bool {
    let mut stack = unitvec![];

    loop {
        if !permits_filter_pushdown(&mut stack, ae, expr_arena) {
            return false;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
}
