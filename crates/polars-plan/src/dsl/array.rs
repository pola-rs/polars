use polars_core::prelude::SortOptions;

use crate::dsl::function_expr::{ArrayFunction, FunctionExpr};
use crate::prelude::*;

/// Specialized expressions for [`Series`][Series] of [`DataType::List`][DataType::List].
///
/// [Series]: polars_core::prelude::Series
/// [DataType::List]: polars_core::prelude::DataType::List
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

    /// Keep only the unique values in every sub-array.
    pub fn unique(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Unique(false)))
    }

    /// Keep only the unique values in every sub-array.
    pub fn unique_stable(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Unique(true)))
    }

    /// Cast the Array column to List column with the same inner data type.
    pub fn to_list(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::ToList))
    }

    #[cfg(feature = "array_any_all")]
    /// Evaluate whether all boolean values are true for every subarray.
    pub fn all(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::All))
    }

    #[cfg(feature = "array_any_all")]
    /// Evaluate whether any boolean value is true for every subarray
    pub fn any(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Any))
    }

    pub fn sort(self, options: SortOptions) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Sort(options)))
    }

    pub fn reverse(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Reverse))
    }

    pub fn arg_min(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::ArgMin))
    }

    pub fn arg_max(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::ArgMax))
    }

    /// Get items in every sub-array by index.
    pub fn get(self, index: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ArrayExpr(ArrayFunction::Get),
            &[index],
            false,
            false,
        )
    }

    /// Join all string items in a sub-array and place a separator between them.
    /// # Error
    /// Raise if inner type of array is not `DataType::String`.
    pub fn join(self, separator: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ArrayExpr(ArrayFunction::Join),
            &[separator],
            false,
            false,
        )
    }

    #[cfg(feature = "is_in")]
    /// Check if the sub-array contains specific element
    pub fn contains<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();

        self.0.map_many_private(
            FunctionExpr::ArrayExpr(ArrayFunction::Contains),
            &[other],
            false,
            false,
        )
    }
}
