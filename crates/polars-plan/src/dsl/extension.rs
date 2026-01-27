use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct ExtensionNameSpace(pub(crate) Expr);

impl ExtensionNameSpace {
    pub fn to(self, dtype: impl Into<DataTypeExpr>) -> Expr {
        self.0.map_unary(ExtensionFunction::To(dtype.into()))
    }

    pub fn storage(self) -> Expr {
        self.0.map_unary(ExtensionFunction::Storage)
    }
}
