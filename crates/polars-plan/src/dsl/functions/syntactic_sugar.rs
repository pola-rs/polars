use polars_core::chunked_array::cast::CastOptions;

use super::*;

/// Sum all the values in the column named `name`. Shorthand for `col(name).sum()`.
pub fn sum(name: &str) -> Expr {
    col(name).sum()
}

/// Find the minimum of all the values in the column named `name`. Shorthand for `col(name).min()`.
pub fn min(name: &str) -> Expr {
    col(name).min()
}

/// Find the maximum of all the values in the column named `name`. Shorthand for `col(name).max()`.
pub fn max(name: &str) -> Expr {
    col(name).max()
}

/// Find the mean of all the values in the column named `name`. Shorthand for `col(name).mean()`.
pub fn mean(name: &str) -> Expr {
    col(name).mean()
}

/// Find the mean of all the values in the column named `name`. Alias for [`mean`].
pub fn avg(name: &str) -> Expr {
    col(name).mean()
}

/// Find the median of all the values in the column named `name`. Shorthand for `col(name).median()`.
pub fn median(name: &str) -> Expr {
    col(name).median()
}

/// Find a specific quantile of all the values in the column named `name`.
pub fn quantile(name: &str, quantile: Expr, interpol: QuantileInterpolOptions) -> Expr {
    col(name).quantile(quantile, interpol)
}

/// Negates a boolean column.
pub fn not(expr: Expr) -> Expr {
    expr.not()
}

/// A column which is `true` wherever `expr` is null, `false` elsewhere.
pub fn is_null(expr: Expr) -> Expr {
    expr.is_null()
}

/// A column which is `false` wherever `expr` is null, `true` elsewhere.
pub fn is_not_null(expr: Expr) -> Expr {
    expr.is_not_null()
}

/// Casts the column given by `Expr` to a different type.
///
/// Follows the rules of Rust casting, with the exception that integers and floats can be cast to `DataType::Date` and
/// `DataType::DateTime(_, _)`. A column consisting entirely of of `Null` can be cast to any type, regardless of the
/// nominal type of the column.
pub fn cast(expr: Expr, data_type: DataType) -> Expr {
    Expr::Cast {
        expr: Arc::new(expr),
        data_type,
        options: CastOptions::NonStrict,
    }
}
