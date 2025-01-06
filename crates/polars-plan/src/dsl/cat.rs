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

    /// Check if a string value contains a literal substring.
    #[cfg(all(feature = "strings", feature = "regex"))]
    pub fn contains(self, pat: &str, literal: bool, strict: bool) -> Expr {
        self.0
            .map_private(FunctionExpr::Categorical(CategoricalFunction::Contains {
                pat: pat.into(),
                literal,
                strict: strict && !literal, // if literal, strict = false
            }))
    }

    /// Uses aho-corasick to find many patterns.
    ///
    /// # Arguments
    /// - `patterns`: an expression that evaluates to a String column
    /// - `ascii_case_insensitive`: Enable ASCII-aware case insensitive matching.
    ///   When this option is enabled, searching will be performed without respect to case for
    ///   ASCII letters (a-z and A-Z) only.
    #[cfg(feature = "find_many")]
    pub fn contains_any(self, patterns: Expr, ascii_case_insensitive: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::Categorical(CategoricalFunction::ContainsMany {
                ascii_case_insensitive,
            }),
            &[patterns],
            false,
            None,
        )
    }
}
