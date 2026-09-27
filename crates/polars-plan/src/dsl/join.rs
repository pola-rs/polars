#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Expr;

#[derive(Debug, strum_macros::IntoStaticStr, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum JoinCondition {
    Equi {
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
    },
    NonEqui {
        predicates: Vec<Expr>,
    },
}

impl Default for JoinCondition {
    fn default() -> Self {
        JoinCondition::Equi {
            left_on: vec![],
            right_on: vec![],
        }
    }
}
