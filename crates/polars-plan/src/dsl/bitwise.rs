use std::sync::Arc;

use super::{AggExpr, BitwiseAggFunction, BitwiseFunction, Expr, FunctionExpr};

impl Expr {
    /// Evaluate the number of set bits.
    pub fn bitwise_count_ones(self) -> Self {
        self.apply_private(FunctionExpr::Bitwise(BitwiseFunction::CountOnes))
    }

    /// Evaluate the number of unset bits.
    pub fn bitwise_count_zeros(self) -> Self {
        self.apply_private(FunctionExpr::Bitwise(BitwiseFunction::CountZeros))
    }

    /// Evaluate the number most-significant set bits before seeing an unset bit.
    pub fn bitwise_leading_ones(self) -> Self {
        self.apply_private(FunctionExpr::Bitwise(BitwiseFunction::LeadingOnes))
    }

    /// Evaluate the number most-significant unset bits before seeing an set bit.
    pub fn bitwise_leading_zeros(self) -> Self {
        self.apply_private(FunctionExpr::Bitwise(BitwiseFunction::LeadingZeros))
    }

    /// Evaluate the number least-significant set bits before seeing an unset bit.
    pub fn bitwise_trailing_ones(self) -> Self {
        self.apply_private(FunctionExpr::Bitwise(BitwiseFunction::TrailingOnes))
    }

    /// Evaluate the number least-significant unset bits before seeing an set bit.
    pub fn bitwise_trailing_zeros(self) -> Self {
        self.apply_private(FunctionExpr::Bitwise(BitwiseFunction::TrailingZeros))
    }

    /// Perform an aggregation of bitwise ANDs
    pub fn bitwise_and(self) -> Self {
        Expr::Agg(AggExpr::Bitwise(Arc::new(self), BitwiseAggFunction::And))
    }

    /// Perform an aggregation of bitwise ORs
    pub fn bitwise_or(self) -> Self {
        Expr::Agg(AggExpr::Bitwise(Arc::new(self), BitwiseAggFunction::Or))
    }

    /// Perform an aggregation of bitwise XORs
    pub fn bitwise_xor(self) -> Self {
        Expr::Agg(AggExpr::Bitwise(Arc::new(self), BitwiseAggFunction::Xor))
    }
}
