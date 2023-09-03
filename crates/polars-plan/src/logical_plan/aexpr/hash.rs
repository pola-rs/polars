use std::hash::{Hash, Hasher};

use crate::prelude::AExpr;

impl Hash for AExpr {
    // This hashes the variant, not the whole expression
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        match self {
            AExpr::Column(name) => name.hash(state),
            AExpr::Alias(_, name) => name.hash(state),
            AExpr::Nth(v) => v.hash(state),
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
            AExpr::SortBy { descending, .. } => descending.hash(state),
            AExpr::Cast { strict, .. } => strict.hash(state),
            AExpr::Window { options, .. } => options.hash(state),
            AExpr::BinaryExpr { op, .. } => op.hash(state),
            _ => {},
        }
    }
}
