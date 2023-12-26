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

    /// Evaluate whether all boolean values are true for every subarray.
    pub fn all(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::All))
    }

    /// Evaluate whether any boolean value is true for every subarray
    pub fn any(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ArrayExpr(ArrayFunction::Any))
    }
}
