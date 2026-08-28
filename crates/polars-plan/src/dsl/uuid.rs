use super::*;

pub struct UuidNameSpace(pub(crate) Expr);

impl UuidNameSpace {
    /// Generate one random UUIDv4 for each input value.
    pub fn generate_v4(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::UuidExpr(UuidFunction::GenerateV4))
    }

    /// Generate one time-ordered UUIDv7 for each input value.
    pub fn generate_v7(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::UuidExpr(UuidFunction::GenerateV7))
    }

    /// Extract the UUID version field.
    pub fn version(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::UuidExpr(UuidFunction::Version))
    }

    /// Extract the UTC timestamp encoded in UUIDv7 values.
    ///
    /// If `strict` is false, non-v7 values become null instead of raising.
    #[cfg(feature = "dtype-datetime")]
    pub fn timestamp(self, strict: bool) -> Expr {
        self.0
            .map_unary(FunctionExpr::UuidExpr(UuidFunction::Timestamp { strict }))
    }
}
