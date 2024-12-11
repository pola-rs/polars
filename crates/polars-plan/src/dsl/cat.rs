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
    /// Check if a string value starts with the `sub` string.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::Categorical(CategoricalFunction::StartsWith),
            &[sub],
            false,
            None,
        )
    }

    #[cfg(feature = "strings")]
    /// Check if a string value ends with the `sub` string.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::Categorical(CategoricalFunction::EndsWith),
            &[sub],
            false,
            None,
        )
    }
}
