use super::function_expr::BinaryFunction;
use super::*;
/// Specialized expressions for [`Series`] of [`DataType::Utf8`].
pub struct BinaryNameSpace(pub(crate) Expr);

impl BinaryNameSpace {
    /// Check if a binary value contains a literal binary.
    pub fn contains_literal<S: AsRef<[u8]>>(self, pat: S) -> Expr {
        let pat = pat.as_ref().into();
        self.0
            .map_private(BinaryFunction::Contains { pat, literal: true }.into())
    }

    /// Check if a binary value ends with the given sequence.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::BinaryExpr(BinaryFunction::EndsWith),
            &[sub],
            true,
        )
    }

    /// Check if a binary value starts with the given sequence.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::BinaryExpr(BinaryFunction::StartsWith),
            &[sub],
            true,
        )
    }
}
