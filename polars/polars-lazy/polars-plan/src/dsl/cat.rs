use serde::{Deserialize, Serialize};

use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct CategoricalNameSpace(pub(crate) Expr);

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalOrdering {
    /// Use the physical categories for sorting
    Physical,
    /// Use the string value for sorting
    Lexical,
}

impl CategoricalNameSpace {
    pub fn set_ordering(self, ordering: CategoricalOrdering) -> Expr {
        self.0
            .map_private(CategoricalFunction::SetOrdering(ordering).into())
    }
}
