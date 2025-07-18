use super::{BitwiseFunction, Expr, FunctionExpr};

impl Expr {
    /// Evaluate the number of set bits.
    pub fn bitwise_count_ones(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::CountOnes))
    }

    /// Evaluate the number of unset bits.
    pub fn bitwise_count_zeros(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::CountZeros))
    }

    /// Evaluate the number most-significant set bits before seeing an unset bit.
    pub fn bitwise_leading_ones(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::LeadingOnes))
    }

    /// Evaluate the number most-significant unset bits before seeing an set bit.
    pub fn bitwise_leading_zeros(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::LeadingZeros))
    }

    /// Evaluate the number least-significant set bits before seeing an unset bit.
    pub fn bitwise_trailing_ones(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::TrailingOnes))
    }

    /// Evaluate the number least-significant unset bits before seeing an set bit.
    pub fn bitwise_trailing_zeros(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::TrailingZeros))
    }

    /// Perform an aggregation of bitwise ANDs
    pub fn bitwise_and(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::And))
    }

    /// Perform an aggregation of bitwise ORs
    pub fn bitwise_or(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::Or))
    }

    /// Perform an aggregation of bitwise XORs
    pub fn bitwise_xor(self) -> Self {
        self.map_unary(FunctionExpr::Bitwise(BitwiseFunction::Xor))
    }
}
