use super::*;
use crate::dsl::function_expr::StructFunction;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub(crate) Expr);

impl StructNameSpace {
    pub fn field_by_index(self, index: i64) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::FieldByIndex(
                index,
            )))
            .with_function_options(|mut options| {
                options.allow_rename = true;
                options
            })
    }

    /// Retrieve one of the fields of this [`StructChunked`] as a new Series.
    pub fn field_by_name(self, name: &str) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::FieldByName(
                Arc::from(name),
            )))
            .with_function_options(|mut options| {
                options.allow_rename = true;
                options
            })
    }

    /// Add prefix to the field names of the [`StructChunked`].
    pub fn prefix(self, prefix: &str) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::Prefix(Arc::from(
                prefix,
            ))))
    }

    /// Rename the fields of the [`StructChunked`].
    pub fn rename_fields(self, names: Vec<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::RenameFields(
                Arc::from(names),
            )))
    }

    /// Add suffix to the field names of the [`StructChunked`].
    pub fn suffix(self, suffix: &str) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::Suffix(Arc::from(
                suffix,
            ))))
    }
}
