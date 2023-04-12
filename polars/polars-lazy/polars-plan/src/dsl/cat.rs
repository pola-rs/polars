use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct CategoricalNameSpace(pub(crate) Expr);

#[derive(Copy, Clone, Debug)]
pub enum CategoricalOrdering {
    /// Use the physical categories for sorting
    Physical,
    /// Use the string value for sorting
    Lexical,
}

impl CategoricalNameSpace {
    pub fn set_ordering(self, ordering: CategoricalOrdering) -> Expr {
        let lexical = match ordering {
            CategoricalOrdering::Lexical => true,
            CategoricalOrdering::Physical => false,
        };
        self.0
            .map_private(CategoricalFunction::SetOrdering { lexical }.into())
    }
}
