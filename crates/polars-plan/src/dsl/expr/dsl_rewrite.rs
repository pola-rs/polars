use polars_core::prelude::*;

use crate::dsl::Expr;

/// A type of expression that can be rewritten into another expressions.
/// Usable in Plugins, so that they can write expressions based on metadata / type information.
pub trait DslRewrite: Send + Sync {
    /// What expression we resolve this into.
    /// Returns a template where `Expr::RewriteInput(i)` refers to input `i`.
    ///
    /// Polars does not validate the output type of the returned expression.
    /// plugin authors are responsible for testing that their rewrite produces the intended dtype.
    fn rewrite(&self, inputs: &[Field], input_schema: &Schema) -> PolarsResult<Expr>;

    /// Name used for display / error messages / hashing.
    fn name(&self) -> PlSmallStr;
}
