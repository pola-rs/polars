use std::hash::{Hash, Hasher};

use polars_utils::arena::{Arena, Node};

use crate::prelude::AExpr;

impl Hash for AExpr {
    // This hashes the variant, not the whole expression
    // IMPORTANT: This is also used for equality in some cases with blake3.
    // Make sure that all attributes that are important for equality are hashed. Nodes don't have
    // to be hashed.
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        match self {
            AExpr::Column(name) => name.hash(state),
            AExpr::StructField(name) => name.hash(state),
            AExpr::Literal(lv) => lv.hash(state),
            AExpr::Function {
                options,
                function,
                input: _,
            } => {
                options.hash(state);
                function.hash(state)
            },
            AExpr::AnonymousFunction {
                options,
                fmt_str,
                function,
                input: _,
            } => {
                fmt_str.hash(state);
                options.hash(state);
                function.hash(state);
            },
            AExpr::Agg(agg) => agg.hash(state),
            AExpr::SortBy { sort_options, .. } => sort_options.hash(state),
            AExpr::Cast {
                options,
                dtype,
                expr: _,
            } => {
                options.hash(state);
                dtype.hash(state);
            },
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling {
                function: _,
                index_column: _,
                period,
                offset,
                closed_window,
            } => {
                period.hash(state);
                offset.hash(state);
                closed_window.hash(state);
            },
            AExpr::Over {
                mapping,
                order_by,
                function: _,
                partition_by: _,
            } => {
                mapping.hash(state);
                if let Some(o) = order_by {
                    o.1.hash(state);
                }
            },
            AExpr::BinaryExpr {
                op,
                left: _,
                right: _,
            } => op.hash(state),
            AExpr::Element => {},
            AExpr::Explode { expr: _, options } => options.hash(state),
            AExpr::Sort { expr: _, options } => options.hash(state),
            AExpr::Gather {
                expr: _,
                idx: _,
                returns_scalar,
                null_on_oob: _,
            } => returns_scalar.hash(state),
            AExpr::Filter { input: _, by: _ } => {},
            AExpr::Ternary {
                predicate: _,
                truthy: _,
                falsy: _,
            } => {},
            AExpr::AnonymousAgg {
                input: _,
                fmt_str,
                function,
            } => {
                function.hash(state);
                fmt_str.hash(state);
            },
            AExpr::Eval {
                expr: _,
                evaluation: _,
                variant,
            } => variant.hash(state),
            AExpr::StructEval {
                expr: _,
                evaluation: _,
            } => {},
            AExpr::Slice {
                input: _,
                offset: _,
                length: _,
            } => {},
            AExpr::Len => {},
        }
    }
}

pub(crate) fn traverse_and_hash_aexpr<H: Hasher>(
    node: Node,
    expr_arena: &Arena<AExpr>,
    state: &mut H,
) {
    let mut scratch = vec![node];

    while let Some(node) = scratch.pop() {
        let ae = expr_arena.get(node);
        ae.hash(state);
        ae.children_rev(&mut scratch);
    }
}
