use std::hash::{Hash, Hasher};

use polars_utils::arena::{Arena, Node};

use crate::plans::ArenaExprIter;
use crate::prelude::AExpr;

impl Hash for AExpr {
    // This hashes the variant, not the whole expression
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        match self {
            AExpr::Column(name) => name.hash(state),
            AExpr::Literal(lv) => lv.hash(state),
            AExpr::Function {
                options, function, ..
            } => {
                options.hash(state);
                function.hash(state)
            },
            AExpr::AnonymousFunction { options, .. } => {
                options.hash(state);
            },
            AExpr::Agg(agg) => agg.hash(state),
            AExpr::SortBy { sort_options, .. } => sort_options.hash(state),
            AExpr::Cast {
                options: strict, ..
            } => strict.hash(state),
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
            AExpr::Over { mapping, .. } => mapping.hash(state),
            AExpr::BinaryExpr { op, .. } => op.hash(state),
            AExpr::Element => {},
            AExpr::Explode { expr: _, options } => options.hash(state),
            AExpr::Sort { expr: _, options } => options.hash(state),
            AExpr::Gather {
                expr: _,
                idx: _,
                returns_scalar,
            } => returns_scalar.hash(state),
            AExpr::Filter { input: _, by: _ } => {},
            AExpr::Ternary {
                predicate: _,
                truthy: _,
                falsy: _,
            } => {},
            AExpr::AnonymousStreamingAgg {
                input: _,
                fmt_str,
                function: _,
            } => {
                fmt_str.hash(state);
            },
            AExpr::Eval {
                expr: _,
                evaluation: _,
                variant,
            } => variant.hash(state),
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
    for (_, ae) in expr_arena.iter(node) {
        ae.hash(state);
    }
}
