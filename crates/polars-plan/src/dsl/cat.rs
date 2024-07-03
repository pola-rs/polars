use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct CategoricalNameSpace(pub(crate) Expr);

impl CategoricalNameSpace {
    pub fn get_categories(self) -> Expr {
        self.0
            .apply_private(CategoricalFunction::GetCategories.into())
    }
}
