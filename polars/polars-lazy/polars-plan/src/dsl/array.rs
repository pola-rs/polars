use crate::dsl::function_expr::{ArrayFunction, FunctionExpr};
use crate::prelude::*;

/// Specialized expressions for [`Series`] of [`DataType::List`].
pub struct ArrayNameSpace(pub Expr);

impl ArrayNameSpace {
    /// Compute the maximum of the items in every subarray.
    pub fn max(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Max))
    }

    /// Compute the minimum of the items in every subarray.
    pub fn min(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Min))
    }

    /// Compute the sum of the items in every subarray.
    pub fn sum(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Sum))
    }
}
