use super::function_expr::BinaryFunction;
use super::*;
/// Specialized expressions for [`Series`] of [`DataType::Utf8`].
pub struct BinaryNameSpace(pub(crate) Expr);

impl BinaryNameSpace {
    /// Check if a binary value contains a literal binary.
    pub fn contains_literal(self, pat: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::BinaryExpr(BinaryFunction::Contains),
            &[pat],
            false,
            true,
        )
    }

    /// Check if a binary value ends with the given sequence.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::BinaryExpr(BinaryFunction::EndsWith),
            &[sub],
            false,
            true,
        )
    }

    /// Check if a binary value starts with the given sequence.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::BinaryExpr(BinaryFunction::StartsWith),
            &[sub],
            false,
            true,
        )
    }
}
