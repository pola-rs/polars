use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct CategoricalNameSpace(pub(crate) Expr);

impl CategoricalNameSpace {
    pub fn get_categories(self) -> Expr {
        self.0.map_unary(CategoricalFunction::GetCategories)
    }

    #[cfg(feature = "strings")]
    pub fn len_bytes(self) -> Expr {
        self.0.map_unary(CategoricalFunction::LenBytes)
    }

    #[cfg(feature = "strings")]
    pub fn len_chars(self) -> Expr {
        self.0.map_unary(CategoricalFunction::LenChars)
    }

    #[cfg(feature = "strings")]
    pub fn starts_with(self, prefix: String) -> Expr {
        self.0.map_unary(CategoricalFunction::StartsWith(prefix))
    }

    #[cfg(feature = "strings")]
    pub fn ends_with(self, suffix: String) -> Expr {
        self.0.map_unary(CategoricalFunction::EndsWith(suffix))
    }

    #[cfg(feature = "strings")]
    pub fn slice(self, offset: i64, length: Option<usize>) -> Expr {
        self.0.map_unary(CategoricalFunction::Slice(offset, length))
    }
}
