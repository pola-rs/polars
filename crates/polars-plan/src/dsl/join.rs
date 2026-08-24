#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Expr;

#[derive(Debug, strum_macros::IntoStaticStr, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum JoinCondition {
    Equi { on: Vec<(Expr, Expr)> },
    NonEqui { predicates: Vec<Expr> },
}

impl Default for JoinCondition {
    fn default() -> Self {
        JoinCondition::Equi { on: vec![] }
    }
}
