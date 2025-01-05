use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct CategoricalNameSpace(pub(crate) Expr);

impl CategoricalNameSpace {
    pub fn get_categories(self) -> Expr {
        self.0
            .apply_private(CategoricalFunction::GetCategories.into())
    }

    #[cfg(feature = "strings")]
    pub fn len_bytes(self) -> Expr {
        self.0
            .map_private(FunctionExpr::Categorical(CategoricalFunction::LenBytes))
    }

    #[cfg(feature = "strings")]
    pub fn len_chars(self) -> Expr {
        self.0
            .map_private(FunctionExpr::Categorical(CategoricalFunction::LenChars))
    }

    #[cfg(feature = "strings")]
    pub fn starts_with(self, prefix: String) -> Expr {
        self.0
            .map_private(FunctionExpr::Categorical(CategoricalFunction::StartsWith(
                prefix,
            )))
    }

    #[cfg(feature = "strings")]
    pub fn ends_with(self, suffix: String) -> Expr {
        self.0
            .map_private(FunctionExpr::Categorical(CategoricalFunction::EndsWith(
                suffix,
            )))
    }
}
