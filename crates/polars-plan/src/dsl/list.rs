use polars_core::prelude::*;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;

use crate::prelude::function_expr::ListFunction;
use crate::prelude::*;

/// Specialized expressions for [`Series`] of [`DataType::List`].
pub struct ListNameSpace(pub Expr);

impl ListNameSpace {
    #[cfg(feature = "list_any_all")]
    pub fn any(self) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::Any))
    }

    #[cfg(feature = "list_any_all")]
    pub fn all(self) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::All))
    }

    #[cfg(feature = "list_drop_nulls")]
    pub fn drop_nulls(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::DropNulls))
    }

    #[cfg(feature = "list_sample")]
    pub fn sample_n(
        self,
        n: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::Sample {
                is_fraction: false,
                with_replacement,
                shuffle,
                seed,
            }),
            n,
        )
    }

    #[cfg(feature = "list_sample")]
    pub fn sample_fraction(
        self,
        fraction: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::Sample {
                is_fraction: true,
                with_replacement,
                shuffle,
                seed,
            }),
            fraction,
        )
    }

    /// Return the number of elements in each list.
    ///
    /// Null values are treated like regular elements in this context.
    pub fn len(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Length))
    }

    /// Compute the maximum of the items in every sublist.
    pub fn max(self) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::Max))
    }

    /// Compute the minimum of the items in every sublist.
    pub fn min(self) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::Min))
    }

    /// Compute the sum the items in every sublist.
    pub fn sum(self) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::Sum))
    }

    /// Compute the mean of every sublist and return a `Series` of dtype `Float64`
    pub fn mean(self) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::Mean))
    }

    pub fn median(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Median))
    }

    pub fn std(self, ddof: u8) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Std(ddof)))
    }

    pub fn var(self, ddof: u8) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Var(ddof)))
    }

    /// Sort every sublist.
    pub fn sort(self, options: SortOptions) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Sort(options)))
    }

    /// Reverse every sublist
    pub fn reverse(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Reverse))
    }

    /// Keep only the unique values in every sublist.
    pub fn unique(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Unique(false)))
    }

    /// Keep only the unique values in every sublist.
    pub fn unique_stable(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::Unique(true)))
    }

    pub fn n_unique(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::NUnique))
    }

    /// Get items in every sublist by index.
    pub fn get(self, index: Expr, null_on_oob: bool) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::Get(null_on_oob)),
            index,
        )
    }

    /// Get items in every sublist by multiple indexes.
    ///
    /// # Arguments
    /// - `null_on_oob`: Return a null when an index is out of bounds.
    ///   This behavior is more expensive than defaulting to returning an `Error`.
    #[cfg(feature = "list_gather")]
    pub fn gather(self, index: Expr, null_on_oob: bool) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::Gather(null_on_oob)),
            index,
        )
    }

    #[cfg(feature = "list_gather")]
    pub fn gather_every(self, n: Expr, offset: Expr) -> Expr {
        self.0
            .map_ternary(FunctionExpr::ListExpr(ListFunction::GatherEvery), n, offset)
    }

    /// Get first item of every sublist.
    pub fn first(self) -> Expr {
        self.get(lit(0i64), true)
    }

    /// Get last item of every sublist.
    pub fn last(self) -> Expr {
        self.get(lit(-1i64), true)
    }

    /// Join all string items in a sublist and place a separator between them.
    /// # Error
    /// This errors if inner type of list `!= DataType::String`.
    pub fn join(self, separator: Expr, ignore_nulls: bool) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::Join(ignore_nulls)),
            separator,
        )
    }

    /// Return the index of the minimal value of every sublist
    pub fn arg_min(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::ArgMin))
    }

    /// Return the index of the maximum value of every sublist
    pub fn arg_max(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::ArgMax))
    }

    /// Diff every sublist.
    #[cfg(feature = "diff")]
    pub fn diff(self, n: i64, null_behavior: NullBehavior) -> Expr {
        self.0.map_unary(FunctionExpr::ListExpr(ListFunction::Diff {
            n,
            null_behavior,
        }))
    }

    /// Shift every sublist.
    pub fn shift(self, periods: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::ListExpr(ListFunction::Shift), periods)
    }

    /// Slice every sublist.
    pub fn slice(self, offset: Expr, length: Expr) -> Expr {
        self.0
            .map_ternary(FunctionExpr::ListExpr(ListFunction::Slice), offset, length)
    }

    /// Get the head of every sublist
    pub fn head(self, n: Expr) -> Expr {
        self.slice(lit(0), n)
    }

    /// Get the tail of every sublist
    pub fn tail(self, n: Expr) -> Expr {
        self.slice(lit(0i64) - n.clone().cast(DataType::Int64), n)
    }

    #[cfg(feature = "dtype-array")]
    /// Convert a List column into an Array column with the same inner data type.
    pub fn to_array(self, width: usize) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::ToArray(width)))
    }

    #[cfg(feature = "list_to_struct")]
    #[allow(clippy::wrong_self_convention)]
    /// Convert this `List` to a `Series` of type `Struct`. The width will be determined according to
    /// `ListToStructWidthStrategy` and the names of the fields determined by the given `name_generator`.
    ///
    /// # Schema
    ///
    /// A polars `LazyFrame` needs to know the schema at all time. The caller therefore must provide
    /// an `upper_bound` of struct fields that will be set.
    /// If this is incorrectly downstream operation may fail. For instance an `all().sum()` expression
    /// will look in the current schema to determine which columns to select.
    pub fn to_struct(self, args: ListToStructArgs) -> Expr {
        self.0
            .map_unary(FunctionExpr::ListExpr(ListFunction::ToStruct(args)))
    }

    #[cfg(feature = "is_in")]
    /// Check if the list array contain an element
    pub fn contains<E: Into<Expr>>(self, other: E, nulls_equal: bool) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::Contains { nulls_equal }),
            other.into(),
        )
    }

    #[cfg(feature = "list_count")]
    /// Count how often the value produced by ``element`` occurs.
    pub fn count_matches<E: Into<Expr>>(self, element: E) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::CountMatches),
            element.into(),
        )
    }

    #[cfg(feature = "list_sets")]
    fn set_operation(self, other: Expr, set_operation: SetOperation) -> Expr {
        self.0.map_binary(
            FunctionExpr::ListExpr(ListFunction::SetOperation(set_operation)),
            other,
        )
    }

    /// Return the SET UNION between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn union<E: Into<Expr>>(self, other: E) -> Expr {
        self.set_operation(other.into(), SetOperation::Union)
    }

    /// Return the SET DIFFERENCE between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn set_difference<E: Into<Expr>>(self, other: E) -> Expr {
        self.set_operation(other.into(), SetOperation::Difference)
    }

    /// Return the SET INTERSECTION between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn set_intersection<E: Into<Expr>>(self, other: E) -> Expr {
        self.set_operation(other.into(), SetOperation::Intersection)
    }

    /// Return the SET SYMMETRIC DIFFERENCE between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn set_symmetric_difference<E: Into<Expr>>(self, other: E) -> Expr {
        self.set_operation(other.into(), SetOperation::SymmetricDifference)
    }
}
