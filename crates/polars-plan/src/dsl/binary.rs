use super::*;
/// Specialized expressions for [`Series`] of [`DataType::String`].
pub struct BinaryNameSpace(pub(crate) Expr);

impl BinaryNameSpace {
    /// Check if a binary value contains a literal binary.
    pub fn contains_literal(self, pat: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::Contains), pat)
    }

    /// Check if a binary value ends with the given sequence.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::EndsWith), sub)
    }

    /// Check if a binary value starts with the given sequence.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::StartsWith), sub)
    }

    /// Return the size (number of bytes) in each element.
    pub fn size_bytes(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Size))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_decode(self, strict: bool) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::HexDecode(strict)))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_encode(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::HexEncode))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn base64_decode(self, strict: bool) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Base64Decode(
                strict,
            )))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn base64_encode(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Base64Encode))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn reinterpret(self, to_type: impl Into<DataTypeExpr>, is_little_endian: bool) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Reinterpret(
                to_type.into(),
                is_little_endian,
            )))
    }

    pub fn slice(self, offset: Expr, length: Expr) -> Expr {
        self.0.map_ternary(
            FunctionExpr::BinaryExpr(BinaryFunction::Slice),
            offset,
            length,
        )
    }

    pub fn head(self, n: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::Head), n)
    }

    pub fn tail(self, n: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::Tail), n)
    }
}
