#![allow(ambiguous_glob_reexports)]
//! Domain specific language for the Lazy API.
#[cfg(feature = "rolling_window")]
use polars_core::utils::ensure_sorted_arg;
#[cfg(feature = "dtype-categorical")]
pub mod cat;
#[cfg(feature = "dtype-categorical")]
pub use cat::*;
mod arithmetic;
mod arity;
#[cfg(feature = "dtype-array")]
mod array;
pub mod binary;
pub mod consts;
#[cfg(feature = "temporal")]
pub mod dt;
mod expr;
mod expr_dyn_fn;
mod from;
pub(crate) mod function_expr;
pub mod functions;
mod list;
#[cfg(feature = "meta")]
mod meta;
mod name;
mod options;
#[cfg(feature = "python")]
pub mod python_udf;
#[cfg(feature = "random")]
mod random;
mod selector;
#[cfg(feature = "strings")]
pub mod string;
#[cfg(feature = "dtype-struct")]
mod struct_;
pub mod udf;
use std::fmt::Debug;
use std::sync::Arc;

pub use arity::*;
#[cfg(feature = "dtype-array")]
pub use array::*;
use arrow::legacy::prelude::QuantileInterpolOptions;
pub use expr::*;
pub use function_expr::schema::FieldsMapper;
pub use function_expr::*;
pub use functions::*;
pub use list::*;
#[cfg(feature = "meta")]
pub use meta::*;
pub use options::*;
use polars_core::prelude::*;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_core::series::IsSorted;
use polars_core::utils::try_get_supertype;
#[cfg(feature = "rolling_window")]
use polars_time::prelude::SeriesOpsTime;
pub(crate) use selector::Selector;
#[cfg(feature = "dtype-struct")]
pub use struct_::*;
pub use udf::UserDefinedFunction;

use crate::constants::MAP_LIST_NAME;
pub use crate::logical_plan::lit;
use crate::prelude::*;
use crate::utils::has_expr;
#[cfg(feature = "is_in")]
use crate::utils::has_leaf_literal;

impl Expr {
    /// Modify the Options passed to the `Function` node.
    pub(crate) fn with_function_options<F>(self, func: F) -> Expr
    where
        F: Fn(FunctionOptions) -> FunctionOptions,
    {
        match self {
            Self::AnonymousFunction {
                input,
                function,
                output_type,
                mut options,
            } => {
                options = func(options);
                Self::AnonymousFunction {
                    input,
                    function,
                    output_type,
                    options,
                }
            },
            Self::Function {
                input,
                function,
                mut options,
            } => {
                options = func(options);
                Self::Function {
                    input,
                    function,
                    options,
                }
            },
            _ => {
                panic!("implementation error")
            },
        }
    }

    /// Overwrite the function name used for formatting.
    /// (this is not intended to be used).
    #[doc(hidden)]
    pub fn with_fmt(self, name: &'static str) -> Expr {
        self.with_function_options(|mut options| {
            options.fmt_str = name;
            options
        })
    }

    /// Compare `Expr` with other `Expr` on equality.
    pub fn eq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::Eq, other.into())
    }

    /// Compare `Expr` with other `Expr` on equality where `None == None`.
    pub fn eq_missing<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::EqValidity, other.into())
    }

    /// Compare `Expr` with other `Expr` on non-equality.
    pub fn neq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::NotEq, other.into())
    }

    /// Compare `Expr` with other `Expr` on non-equality where `None == None`.
    pub fn neq_missing<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::NotEqValidity, other.into())
    }

    /// Check if `Expr` < `Expr`.
    pub fn lt<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::Lt, other.into())
    }

    /// Check if `Expr` > `Expr`.
    pub fn gt<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::Gt, other.into())
    }

    /// Check if `Expr` >= `Expr`.
    pub fn gt_eq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::GtEq, other.into())
    }

    /// Check if `Expr` <= `Expr`.
    pub fn lt_eq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::LtEq, other.into())
    }

    /// Negate `Expr`.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        self.map_private(BooleanFunction::Not.into())
    }

    /// Rename Column.
    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), Arc::from(name))
    }

    /// Run is_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_null(self) -> Self {
        self.map_private(BooleanFunction::IsNull.into())
    }

    /// Run is_not_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_null(self) -> Self {
        self.map_private(BooleanFunction::IsNotNull.into())
    }

    /// Drop null values.
    pub fn drop_nulls(self) -> Self {
        self.apply_private(FunctionExpr::DropNulls)
    }

    /// Drop NaN values.
    pub fn drop_nans(self) -> Self {
        self.apply_private(FunctionExpr::DropNans)
    }

    /// Reduce groups to minimal value.
    pub fn min(self) -> Self {
        AggExpr::Min {
            input: Box::new(self),
            propagate_nans: false,
        }
        .into()
    }

    /// Reduce groups to maximum value.
    pub fn max(self) -> Self {
        AggExpr::Max {
            input: Box::new(self),
            propagate_nans: false,
        }
        .into()
    }

    /// Reduce groups to minimal value.
    pub fn nan_min(self) -> Self {
        AggExpr::Min {
            input: Box::new(self),
            propagate_nans: true,
        }
        .into()
    }

    /// Reduce groups to maximum value.
    pub fn nan_max(self) -> Self {
        AggExpr::Max {
            input: Box::new(self),
            propagate_nans: true,
        }
        .into()
    }

    /// Reduce groups to the mean value.
    pub fn mean(self) -> Self {
        AggExpr::Mean(Box::new(self)).into()
    }

    /// Reduce groups to the median value.
    pub fn median(self) -> Self {
        AggExpr::Median(Box::new(self)).into()
    }

    /// Reduce groups to the sum of all the values.
    pub fn sum(self) -> Self {
        AggExpr::Sum(Box::new(self)).into()
    }

    /// Get the number of unique values in the groups.
    pub fn n_unique(self) -> Self {
        AggExpr::NUnique(Box::new(self)).into()
    }

    /// Get the first value in the group.
    pub fn first(self) -> Self {
        AggExpr::First(Box::new(self)).into()
    }

    /// Get the last value in the group.
    pub fn last(self) -> Self {
        AggExpr::Last(Box::new(self)).into()
    }

    /// Aggregate the group to a Series.
    pub fn implode(self) -> Self {
        AggExpr::Implode(Box::new(self)).into()
    }

    /// Compute the quantile per group.
    pub fn quantile(self, quantile: Expr, interpol: QuantileInterpolOptions) -> Self {
        AggExpr::Quantile {
            expr: Box::new(self),
            quantile: Box::new(quantile),
            interpol,
        }
        .into()
    }

    /// Get the group indexes of the group by operation.
    pub fn agg_groups(self) -> Self {
        AggExpr::AggGroups(Box::new(self)).into()
    }

    /// Alias for `explode`.
    pub fn flatten(self) -> Self {
        self.explode()
    }

    /// Explode the utf8/ list column.
    pub fn explode(self) -> Self {
        Expr::Explode(Box::new(self))
    }

    /// Slice the Series.
    /// `offset` may be negative.
    pub fn slice<E: Into<Expr>, F: Into<Expr>>(self, offset: E, length: F) -> Self {
        Expr::Slice {
            input: Box::new(self),
            offset: Box::new(offset.into()),
            length: Box::new(length.into()),
        }
    }

    /// Append expressions. This is done by adding the chunks of `other` to this [`Series`].
    pub fn append<E: Into<Expr>>(self, other: E, upcast: bool) -> Self {
        let output_type = if upcast {
            GetOutput::super_type()
        } else {
            GetOutput::same_type()
        };

        apply_binary(
            self,
            other.into(),
            move |mut a, mut b| {
                if upcast {
                    let dtype = try_get_supertype(a.dtype(), b.dtype())?;
                    a = a.cast(&dtype)?;
                    b = b.cast(&dtype)?;
                }
                a.append(&b)?;
                Ok(Some(a))
            },
            output_type,
        )
    }

    /// Get the first `n` elements of the Expr result.
    pub fn head(self, length: Option<usize>) -> Self {
        self.slice(lit(0), lit(length.unwrap_or(10) as u64))
    }

    /// Get the last `n` elements of the Expr result.
    pub fn tail(self, length: Option<usize>) -> Self {
        let len = length.unwrap_or(10);
        self.slice(lit(-(len as i64)), lit(len as u64))
    }

    /// Get unique values of this expression.
    pub fn unique(self) -> Self {
        self.apply_private(FunctionExpr::Unique(false))
    }

    /// Get unique values of this expression, while maintaining order.
    /// This requires more work than [`Expr::unique`].
    pub fn unique_stable(self) -> Self {
        self.apply_private(FunctionExpr::Unique(true))
    }

    /// Get the first index of unique values of this expression.
    pub fn arg_unique(self) -> Self {
        self.apply_private(FunctionExpr::ArgUnique)
    }

    /// Get the index value that has the minimum value.
    pub fn arg_min(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            returns_scalar: true,
            fmt_str: "arg_min",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| {
                Ok(Some(Series::new(
                    s.name(),
                    &[s.arg_min().map(|idx| idx as u32)],
                )))
            },
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index value that has the maximum value.
    pub fn arg_max(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            returns_scalar: true,
            fmt_str: "arg_max",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| {
                Ok(Some(Series::new(
                    s.name(),
                    &[s.arg_max().map(|idx| idx as u32)],
                )))
            },
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index values that would sort this expression.
    pub fn arg_sort(self, sort_options: SortOptions) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            fmt_str: "arg_sort",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| Ok(Some(s.arg_sort(sort_options).into_series())),
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    #[cfg(feature = "search_sorted")]
    /// Find indices where elements should be inserted to maintain order.
    pub fn search_sorted<E: Into<Expr>>(self, element: E, side: SearchSortedSide) -> Expr {
        let element = element.into();
        Expr::Function {
            input: vec![self, element],
            function: FunctionExpr::SearchSorted(side),
            options: FunctionOptions {
                collect_groups: ApplyOptions::GroupWise,
                returns_scalar: true,
                fmt_str: "search_sorted",
                cast_to_supertypes: true,
                ..Default::default()
            },
        }
    }

    /// Cast expression to another data type.
    /// Throws an error if conversion had overflows.
    pub fn strict_cast(self, data_type: DataType) -> Self {
        Expr::Cast {
            expr: Box::new(self),
            data_type,
            strict: true,
        }
    }

    /// Cast expression to another data type.
    pub fn cast(self, data_type: DataType) -> Self {
        Expr::Cast {
            expr: Box::new(self),
            data_type,
            strict: false,
        }
    }

    /// Take the values by idx.
    pub fn take<E: Into<Expr>>(self, idx: E) -> Self {
        Expr::Take {
            expr: Box::new(self),
            idx: Box::new(idx.into()),
            returns_scalar: false,
        }
    }

    /// Take the values by a single index.
    pub fn get<E: Into<Expr>>(self, idx: E) -> Self {
        Expr::Take {
            expr: Box::new(self),
            idx: Box::new(idx.into()),
            returns_scalar: true,
        }
    }

    /// Sort in increasing order. See [the eager implementation](Series::sort).
    pub fn sort(self, descending: bool) -> Self {
        Expr::Sort {
            expr: Box::new(self),
            options: SortOptions {
                descending,
                ..Default::default()
            },
        }
    }

    /// Sort with given options.
    pub fn sort_with(self, options: SortOptions) -> Self {
        Expr::Sort {
            expr: Box::new(self),
            options,
        }
    }

    /// Returns the `k` largest elements.
    ///
    /// This has time complexity `O(n + k log(n))`.
    #[cfg(feature = "top_k")]
    pub fn top_k(self, k: Expr) -> Self {
        self.apply_many_private(FunctionExpr::TopK(false), &[k], false, false)
    }

    /// Returns the `k` smallest elements.
    ///
    /// This has time complexity `O(n + k log(n))`.
    #[cfg(feature = "top_k")]
    pub fn bottom_k(self, k: Expr) -> Self {
        self.apply_many_private(FunctionExpr::TopK(true), &[k], false, false)
    }

    /// Reverse column
    pub fn reverse(self) -> Self {
        self.apply_private(FunctionExpr::Reverse)
    }

    /// Apply a function/closure once the logical plan get executed.
    ///
    /// This function is very similar to [`Expr::apply`], but differs in how it handles aggregations.
    ///
    ///  * `map` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
    ///  * `apply` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
    ///
    /// It is the responsibility of the caller that the schema is correct by giving
    /// the correct output_type. If None given the output type of the input expr is used.
    pub fn map<F>(self, function: F, output_type: GetOutput) -> Self
    where
        F: Fn(Series) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                fmt_str: "map",
                ..Default::default()
            },
        }
    }

    fn map_private(self, function_expr: FunctionExpr) -> Self {
        Expr::Function {
            input: vec![self],
            function: function_expr,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                ..Default::default()
            },
        }
    }

    /// Apply a function/closure once the logical plan get executed with many arguments.
    ///
    /// See the [`Expr::map`] function for the differences between [`map`](Expr::map) and [`apply`](Expr::apply).
    pub fn map_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::AnonymousFunction {
            input,
            function: SpecialEq::new(Arc::new(function)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                fmt_str: "",
                ..Default::default()
            },
        }
    }

    /// Apply a function/closure once the logical plan get executed.
    ///
    /// This function is very similar to [apply](Expr::apply), but differs in how it handles aggregations.
    ///
    ///  * `map` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
    ///  * `apply` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
    ///  * `map_list` should be used when the function expects a list aggregated series.
    pub fn map_list<F>(self, function: F, output_type: GetOutput) -> Self
    where
        F: Fn(Series) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyList,
                fmt_str: MAP_LIST_NAME,
                ..Default::default()
            },
        }
    }

    /// A function that cannot be expressed with `map` or `apply` and requires extra settings.
    pub fn function_with_options<F>(
        self,
        function: F,
        output_type: GetOutput,
        options: FunctionOptions,
    ) -> Self
    where
        F: Fn(Series) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options,
        }
    }

    /// Apply a function/closure over the groups. This should only be used in a group_by aggregation.
    ///
    /// It is the responsibility of the caller that the schema is correct by giving
    /// the correct output_type. If None given the output type of the input expr is used.
    ///
    /// This difference with [map](Self::map) is that `apply` will create a separate `Series` per group.
    ///
    /// * `map` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
    /// * `apply` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
    pub fn apply<F>(self, function: F, output_type: GetOutput) -> Self
    where
        F: Fn(Series) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::GroupWise,
                fmt_str: "",
                ..Default::default()
            },
        }
    }

    fn apply_private(self, function_expr: FunctionExpr) -> Self {
        Expr::Function {
            input: vec![self],
            function: function_expr,
            options: FunctionOptions {
                collect_groups: ApplyOptions::GroupWise,
                ..Default::default()
            },
        }
    }

    /// Apply a function/closure over the groups with many arguments. This should only be used in a group_by aggregation.
    ///
    /// See the [`Expr::apply`] function for the differences between [`map`](Expr::map) and [`apply`](Expr::apply).
    pub fn apply_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::AnonymousFunction {
            input,
            function: SpecialEq::new(Arc::new(function)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::GroupWise,
                fmt_str: "",
                ..Default::default()
            },
        }
    }

    pub fn apply_many_private(
        self,
        function_expr: FunctionExpr,
        arguments: &[Expr],
        auto_explode: bool,
        cast_to_supertypes: bool,
    ) -> Self {
        let mut input = Vec::with_capacity(arguments.len() + 1);
        input.push(self);
        input.extend_from_slice(arguments);

        Expr::Function {
            input,
            function: function_expr,
            options: FunctionOptions {
                collect_groups: ApplyOptions::GroupWise,
                returns_scalar: auto_explode,
                cast_to_supertypes,
                ..Default::default()
            },
        }
    }

    pub fn map_many_private(
        self,
        function_expr: FunctionExpr,
        arguments: &[Expr],
        returns_scalar: bool,
        cast_to_supertypes: bool,
    ) -> Self {
        let mut input = Vec::with_capacity(arguments.len() + 1);
        input.push(self);
        input.extend_from_slice(arguments);

        Expr::Function {
            input,
            function: function_expr,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                returns_scalar,
                cast_to_supertypes,
                ..Default::default()
            },
        }
    }

    /// Get mask of finite values if dtype is Float.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_finite(self) -> Self {
        self.map_private(BooleanFunction::IsFinite.into())
    }

    /// Get mask of infinite values if dtype is Float.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_infinite(self) -> Self {
        self.map_private(BooleanFunction::IsInfinite.into())
    }

    /// Get mask of NaN values if dtype is Float.
    pub fn is_nan(self) -> Self {
        self.map_private(BooleanFunction::IsNan.into())
    }

    /// Get inverse mask of NaN values if dtype is Float.
    pub fn is_not_nan(self) -> Self {
        self.map_private(BooleanFunction::IsNotNan.into())
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::shift).
    pub fn shift(self, n: Expr) -> Self {
        self.apply_many_private(FunctionExpr::Shift, &[n], false, false)
    }

    /// Shift the values in the array by some period and fill the resulting empty values.
    pub fn shift_and_fill<E: Into<Expr>>(self, n: E, fill_value: E) -> Self {
        self.apply_many_private(
            FunctionExpr::ShiftAndFill,
            &[n.into(), fill_value.into()],
            false,
            false,
        )
    }

    /// Cumulatively count values from 0 to len.
    #[cfg(feature = "cum_agg")]
    pub fn cumcount(self, reverse: bool) -> Self {
        self.apply_private(FunctionExpr::Cumcount { reverse })
    }

    /// Get an array with the cumulative sum computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cumsum(self, reverse: bool) -> Self {
        self.apply_private(FunctionExpr::Cumsum { reverse })
    }

    /// Get an array with the cumulative product computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cumprod(self, reverse: bool) -> Self {
        self.apply_private(FunctionExpr::Cumprod { reverse })
    }

    /// Get an array with the cumulative min computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cummin(self, reverse: bool) -> Self {
        self.apply_private(FunctionExpr::Cummin { reverse })
    }

    /// Get an array with the cumulative max computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cummax(self, reverse: bool) -> Self {
        self.apply_private(FunctionExpr::Cummax { reverse })
    }

    /// Get the product aggregation of an expression.
    pub fn product(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            returns_scalar: true,
            fmt_str: "product",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| Ok(Some(s.product())),
            GetOutput::map_dtype(|dt| {
                use DataType::*;
                match dt {
                    Float32 => Float32,
                    Float64 => Float64,
                    UInt64 => UInt64,
                    _ => Int64,
                }
            }),
            options,
        )
    }

    /// Fill missing value with next non-null.
    pub fn backward_fill(self, limit: FillNullLimit) -> Self {
        self.apply_private(FunctionExpr::BackwardFill { limit })
    }

    /// Fill missing value with previous non-null.
    pub fn forward_fill(self, limit: FillNullLimit) -> Self {
        self.apply_private(FunctionExpr::ForwardFill { limit })
    }

    /// Round underlying floating point array to given decimal numbers.
    #[cfg(feature = "round_series")]
    pub fn round(self, decimals: u32) -> Self {
        self.map_private(FunctionExpr::Round { decimals })
    }

    /// Round to a number of significant figures.
    #[cfg(feature = "round_series")]
    pub fn round_sig_figs(self, digits: i32) -> Self {
        self.map_private(FunctionExpr::RoundSF { digits })
    }

    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    pub fn floor(self) -> Self {
        self.map_private(FunctionExpr::Floor)
    }

    /// Constant Pi
    #[cfg(feature = "round_series")]
    pub fn pi() -> Self {
        lit(std::f64::consts::PI)
    }

    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    pub fn ceil(self) -> Self {
        self.map_private(FunctionExpr::Ceil)
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    pub fn clip(self, min: Expr, max: Expr) -> Self {
        self.map_many_private(
            FunctionExpr::Clip {
                has_min: true,
                has_max: true,
            },
            &[min, max],
            false,
            false,
        )
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    pub fn clip_max(self, max: Expr) -> Self {
        self.map_many_private(
            FunctionExpr::Clip {
                has_min: false,
                has_max: true,
            },
            &[max],
            false,
            false,
        )
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    pub fn clip_min(self, min: Expr) -> Self {
        self.map_many_private(
            FunctionExpr::Clip {
                has_min: true,
                has_max: false,
            },
            &[min],
            false,
            false,
        )
    }

    /// Convert all values to their absolute/positive value.
    #[cfg(feature = "abs")]
    pub fn abs(self) -> Self {
        self.map_private(FunctionExpr::Abs)
    }

    /// Apply window function over a subgroup.
    /// This is similar to a group_by + aggregation + self join.
    /// Or similar to [window functions in Postgres](https://www.postgresql.org/docs/9.1/tutorial-window.html).
    ///
    /// # Example
    ///
    /// ``` rust
    /// #[macro_use] extern crate polars_core;
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example() -> PolarsResult<()> {
    ///     let df = df! {
    ///             "groups" => &[1, 1, 2, 2, 1, 2, 3, 3, 1],
    ///             "values" => &[1, 2, 3, 4, 5, 6, 7, 8, 8]
    ///         }?;
    ///
    ///     let out = df
    ///      .lazy()
    ///      .select(&[
    ///          col("groups"),
    ///          sum("values").over([col("groups")]),
    ///      ])
    ///      .collect()?;
    ///     println!("{}", &out);
    ///     Ok(())
    /// }
    ///
    /// ```
    ///
    /// Outputs:
    ///
    /// ``` text
    /// ╭────────┬────────╮
    /// │ groups ┆ values │
    /// │ ---    ┆ ---    │
    /// │ i32    ┆ i32    │
    /// ╞════════╪════════╡
    /// │ 1      ┆ 16     │
    /// │ 1      ┆ 16     │
    /// │ 2      ┆ 13     │
    /// │ 2      ┆ 13     │
    /// │ …      ┆ …      │
    /// │ 1      ┆ 16     │
    /// │ 2      ┆ 13     │
    /// │ 3      ┆ 15     │
    /// │ 3      ┆ 15     │
    /// │ 1      ┆ 16     │
    /// ╰────────┴────────╯
    /// ```
    pub fn over<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, partition_by: E) -> Self {
        self.over_with_options(partition_by, Default::default())
    }

    pub fn over_with_options<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
        self,
        partition_by: E,
        options: WindowMapping,
    ) -> Self {
        let partition_by = partition_by
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect();
        Expr::Window {
            function: Box::new(self),
            partition_by,
            options: options.into(),
        }
    }

    #[cfg(feature = "dynamic_group_by")]
    pub fn rolling(self, options: RollingGroupOptions) -> Self {
        Expr::Window {
            function: Box::new(self),
            partition_by: vec![],
            options: WindowType::Rolling(options),
        }
    }

    fn fill_null_impl(self, fill_value: Expr) -> Self {
        let input = vec![self, fill_value];

        Expr::Function {
            input,
            // super type will be replaced by type coercion
            function: FunctionExpr::FillNull {
                // will be set by `type_coercion`.
                super_type: DataType::Unknown,
            },
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                cast_to_supertypes: true,
                ..Default::default()
            },
        }
    }

    /// Replace the null values by a value.
    pub fn fill_null<E: Into<Expr>>(self, fill_value: E) -> Self {
        self.fill_null_impl(fill_value.into())
    }

    /// Replace the floating point `NaN` values by a value.
    pub fn fill_nan<E: Into<Expr>>(self, fill_value: E) -> Self {
        // we take the not branch so that self is truthy value of `when -> then -> otherwise`
        // and that ensure we keep the name of `self`

        when(self.clone().is_not_nan())
            .then(self)
            .otherwise(fill_value.into())
    }
    /// Count the values of the Series
    /// or
    /// Get counts of the group by operation.
    pub fn count(self) -> Self {
        AggExpr::Count(Box::new(self)).into()
    }

    /// Standard deviation of the values of the Series.
    pub fn std(self, ddof: u8) -> Self {
        AggExpr::Std(Box::new(self), ddof).into()
    }

    /// Variance of the values of the Series.
    pub fn var(self, ddof: u8) -> Self {
        AggExpr::Var(Box::new(self), ddof).into()
    }

    /// Get a mask of duplicated values.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_unique")]
    pub fn is_duplicated(self) -> Self {
        self.apply_private(BooleanFunction::IsDuplicated.into())
    }

    /// Get a mask of unique values.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_unique")]
    pub fn is_unique(self) -> Self {
        self.apply_private(BooleanFunction::IsUnique.into())
    }

    /// Get the approximate count of unique values.
    #[cfg(feature = "approx_unique")]
    pub fn approx_n_unique(self) -> Self {
        self.apply_private(FunctionExpr::ApproxNUnique)
            .with_function_options(|mut options| {
                options.returns_scalar = true;
                options
            })
    }

    /// "and" operation.
    pub fn and<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::And, expr.into())
    }

    /// "xor" operation.
    pub fn xor<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::Xor, expr.into())
    }

    /// "or" operation.
    pub fn or<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::Or, expr.into())
    }

    /// Filter a single column.
    ///
    /// Should be used in aggregation context. If you want to filter on a
    /// DataFrame level, use `LazyFrame::filter`.
    pub fn filter<E: Into<Expr>>(self, predicate: E) -> Self {
        if has_expr(&self, |e| matches!(e, Expr::Wildcard)) {
            panic!("filter '*' not allowed, use LazyFrame::filter")
        };
        Expr::Filter {
            input: Box::new(self),
            by: Box::new(predicate.into()),
        }
    }

    /// Check if the values of the left expression are in the lists of the right expr.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_in")]
    pub fn is_in<E: Into<Expr>>(self, other: E) -> Self {
        let other = other.into();
        let has_literal = has_leaf_literal(&other);

        // lit(true).is_in() returns a scalar.
        let returns_scalar = all_leaf_literal(&self);

        let arguments = &[other];
        // we don't have to apply on groups, so this is faster
        if has_literal {
            self.map_many_private(
                BooleanFunction::IsIn.into(),
                arguments,
                returns_scalar,
                true,
            )
        } else {
            self.apply_many_private(
                BooleanFunction::IsIn.into(),
                arguments,
                returns_scalar,
                true,
            )
        }
    }

    /// Sort this column by the ordering of another column.
    /// Can also be used in a group_by context to sort the groups.
    pub fn sort_by<E: AsRef<[IE]>, IE: Into<Expr> + Clone, R: AsRef<[bool]>>(
        self,
        by: E,
        descending: R,
    ) -> Expr {
        let by = by.as_ref().iter().map(|e| e.clone().into()).collect();
        let descending = descending.as_ref().to_vec();
        Expr::SortBy {
            expr: Box::new(self),
            by,
            descending,
        }
    }

    #[cfg(feature = "repeat_by")]
    fn repeat_by_impl(self, by: Expr) -> Expr {
        self.apply_many_private(FunctionExpr::RepeatBy, &[by], false, false)
    }

    #[cfg(feature = "repeat_by")]
    /// Repeat the column `n` times, where `n` is determined by the values in `by`.
    /// This yields an `Expr` of dtype `List`.
    pub fn repeat_by<E: Into<Expr>>(self, by: E) -> Expr {
        self.repeat_by_impl(by.into())
    }

    #[cfg(feature = "is_first_distinct")]
    #[allow(clippy::wrong_self_convention)]
    /// Get a mask of the first unique value.
    pub fn is_first_distinct(self) -> Expr {
        self.apply_private(BooleanFunction::IsFirstDistinct.into())
    }

    #[cfg(feature = "is_last_distinct")]
    #[allow(clippy::wrong_self_convention)]
    /// Get a mask of the last unique value.
    pub fn is_last_distinct(self) -> Expr {
        self.apply_private(BooleanFunction::IsLastDistinct.into())
    }

    fn dot_impl(self, other: Expr) -> Expr {
        (self * other).sum()
    }

    /// Compute the dot/inner product between two expressions.
    pub fn dot<E: Into<Expr>>(self, other: E) -> Expr {
        self.dot_impl(other.into())
    }

    #[cfg(feature = "mode")]
    /// Compute the mode(s) of this column. This is the most occurring value.
    pub fn mode(self) -> Expr {
        self.apply_private(FunctionExpr::Mode)
    }

    /// Exclude a column from a wildcard/regex selection.
    ///
    /// You may also use regexes in the exclude as long as they start with `^` and end with `$`/
    pub fn exclude(self, columns: impl IntoVec<String>) -> Expr {
        let v = columns
            .into_vec()
            .into_iter()
            .map(|s| Excluded::Name(Arc::from(s)))
            .collect();
        Expr::Exclude(Box::new(self), v)
    }

    pub fn exclude_dtype<D: AsRef<[DataType]>>(self, dtypes: D) -> Expr {
        let v = dtypes
            .as_ref()
            .iter()
            .map(|dt| Excluded::Dtype(dt.clone()))
            .collect();
        Expr::Exclude(Box::new(self), v)
    }

    #[cfg(feature = "interpolate")]
    /// Fill null values using interpolation.
    pub fn interpolate(self, method: InterpolationMethod) -> Expr {
        self.apply_private(FunctionExpr::Interpolate(method))
    }

    #[cfg(feature = "rolling_window")]
    #[allow(clippy::type_complexity)]
    fn finish_rolling(
        self,
        options: RollingOptions,
        rolling_function: &dyn Fn(RollingOptions) -> RollingFunction,
        rolling_function_by: &dyn Fn(RollingOptions) -> RollingFunction,
    ) -> Expr {
        if let Some(ref by) = options.by {
            let name = by.clone();
            self.apply_many_private(
                FunctionExpr::RollingExpr(rolling_function_by(options)),
                &[col(&name)],
                true,
                false,
            )
        } else {
            if !options.window_size.parsed_int {
                panic!("if dynamic windows are used in a rolling aggregation, the 'by' argument must be set")
            }
            self.apply_private(FunctionExpr::RollingExpr(rolling_function(options)))
        }
    }

    /// Apply a rolling minimum.
    ///
    /// See: [`RollingAgg::rolling_min`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_min(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Min(options),
            &|options| RollingFunction::MinBy(options),
        )
    }

    /// Apply a rolling maximum.
    ///
    /// See: [`RollingAgg::rolling_max`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_max(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Max(options),
            &|options| RollingFunction::MaxBy(options),
        )
    }

    /// Apply a rolling mean.
    ///
    /// See: [`RollingAgg::rolling_mean`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_mean(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Mean(options),
            &|options| RollingFunction::MeanBy(options),
        )
    }

    /// Apply a rolling sum.
    ///
    /// See: [`RollingAgg::rolling_sum`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_sum(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Sum(options),
            &|options| RollingFunction::SumBy(options),
        )
    }

    /// Apply a rolling median.
    ///
    /// See: [`RollingAgg::rolling_median`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_median(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Median(options),
            &|options| RollingFunction::MedianBy(options),
        )
    }

    /// Apply a rolling quantile.
    ///
    /// See: [`RollingAgg::rolling_quantile`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_quantile(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Quantile(options),
            &|options| RollingFunction::QuantileBy(options),
        )
    }

    /// Apply a rolling variance.
    #[cfg(feature = "rolling_window")]
    pub fn rolling_var(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Var(options),
            &|options| RollingFunction::VarBy(options),
        )
    }

    /// Apply a rolling std-dev.
    #[cfg(feature = "rolling_window")]
    pub fn rolling_std(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            &|options| RollingFunction::Std(options),
            &|options| RollingFunction::StdBy(options),
        )
    }

    /// Apply a rolling skew.
    #[cfg(feature = "rolling_window")]
    #[cfg(feature = "moment")]
    pub fn rolling_skew(self, window_size: usize, bias: bool) -> Expr {
        self.apply_private(FunctionExpr::RollingExpr(RollingFunction::Skew(
            window_size,
            bias,
        )))
    }

    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    pub fn rolling_map(
        self,
        f: Arc<dyn Fn(&Series) -> Series + Send + Sync>,
        output_type: GetOutput,
        options: RollingOptionsFixedWindow,
    ) -> Expr {
        self.apply(
            move |s| s.rolling_map(f.as_ref(), options.clone()).map(Some),
            output_type,
        )
        .with_fmt("rolling_map")
    }

    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// Prefer this over rolling_apply in case of floating point numbers as this is faster.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    pub fn rolling_map_float<F>(self, window_size: usize, f: F) -> Expr
    where
        F: 'static + FnMut(&mut Float64Chunked) -> Option<f64> + Send + Sync + Copy,
    {
        self.apply(
            move |s| {
                let out = match s.dtype() {
                    DataType::Float64 => s
                        .f64()
                        .unwrap()
                        .rolling_map_float(window_size, f)
                        .map(|ca| ca.into_series()),
                    _ => s
                        .cast(&DataType::Float64)?
                        .f64()
                        .unwrap()
                        .rolling_map_float(window_size, f)
                        .map(|ca| ca.into_series()),
                }?;
                if let DataType::Float32 = s.dtype() {
                    out.cast(&DataType::Float32).map(Some)
                } else {
                    Ok(Some(out))
                }
            },
            GetOutput::map_field(|field| match field.data_type() {
                DataType::Float64 => field.clone(),
                DataType::Float32 => Field::new(field.name(), DataType::Float32),
                _ => Field::new(field.name(), DataType::Float64),
            }),
        )
        .with_fmt("rolling_map_float")
    }

    #[cfg(feature = "peaks")]
    pub fn peak_min(self) -> Expr {
        self.apply_private(FunctionExpr::PeakMin)
    }

    #[cfg(feature = "peaks")]
    pub fn peak_max(self) -> Expr {
        self.apply_private(FunctionExpr::PeakMax)
    }

    #[cfg(feature = "rank")]
    /// Assign ranks to data, dealing with ties appropriately.
    pub fn rank(self, options: RankOptions, seed: Option<u64>) -> Expr {
        self.apply_private(FunctionExpr::Rank { options, seed })
    }

    #[cfg(feature = "cutqcut")]
    /// Bin continuous values into discrete categories.
    pub fn cut(
        self,
        breaks: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        include_breaks: bool,
    ) -> Expr {
        self.apply_private(FunctionExpr::Cut {
            breaks,
            labels,
            left_closed,
            include_breaks,
        })
    }

    #[cfg(feature = "cutqcut")]
    /// Bin continuous values into discrete categories based on their quantiles.
    pub fn qcut(
        self,
        probs: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    ) -> Expr {
        self.apply_private(FunctionExpr::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        })
    }

    #[cfg(feature = "cutqcut")]
    /// Bin continuous values into discrete categories using uniform quantile probabilities.
    pub fn qcut_uniform(
        self,
        n_bins: usize,
        labels: Option<Vec<String>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    ) -> Expr {
        let probs = (1..n_bins).map(|b| b as f64 / n_bins as f64).collect();
        self.apply_private(FunctionExpr::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        })
    }

    #[cfg(feature = "rle")]
    /// Get the lengths of runs of identical values.
    pub fn rle(self) -> Expr {
        self.apply_private(FunctionExpr::RLE)
    }

    #[cfg(feature = "rle")]
    /// Similar to `rle`, but maps values to run IDs.
    pub fn rle_id(self) -> Expr {
        self.apply_private(FunctionExpr::RLEID)
    }

    #[cfg(feature = "diff")]
    /// Calculate the n-th discrete difference between values.
    pub fn diff(self, n: i64, null_behavior: NullBehavior) -> Expr {
        self.apply_private(FunctionExpr::Diff(n, null_behavior))
    }

    #[cfg(feature = "pct_change")]
    /// Computes percentage change between values.
    pub fn pct_change(self, n: Expr) -> Expr {
        self.apply_many_private(FunctionExpr::PctChange, &[n], false, false)
    }

    #[cfg(feature = "moment")]
    /// Compute the sample skewness of a data set.
    ///
    /// For normally distributed data, the skewness should be about zero. For
    /// uni-modal continuous distributions, a skewness value greater than zero means
    /// that there is more weight in the right tail of the distribution. The
    /// function `skewtest` can be used to determine if the skewness value
    /// is close enough to zero, statistically speaking.
    ///
    /// see: [scipy](https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1024)
    pub fn skew(self, bias: bool) -> Expr {
        self.apply_private(FunctionExpr::Skew(bias))
            .with_function_options(|mut options| {
                options.returns_scalar = true;
                options
            })
    }

    #[cfg(feature = "moment")]
    /// Compute the kurtosis (Fisher or Pearson).
    ///
    /// Kurtosis is the fourth central moment divided by the square of the
    /// variance. If Fisher's definition is used, then 3.0 is subtracted from
    /// the result to give 0.0 for a normal distribution.
    /// If bias is False then the kurtosis is calculated using k statistics to
    /// eliminate bias coming from biased moment estimators.
    pub fn kurtosis(self, fisher: bool, bias: bool) -> Expr {
        self.apply_private(FunctionExpr::Kurtosis(fisher, bias))
            .with_function_options(|mut options| {
                options.returns_scalar = true;
                options
            })
    }

    /// Get maximal value that could be hold by this dtype.
    pub fn upper_bound(self) -> Expr {
        self.map_private(FunctionExpr::UpperBound)
    }

    /// Get minimal value that could be hold by this dtype.
    pub fn lower_bound(self) -> Expr {
        self.map_private(FunctionExpr::LowerBound)
    }

    pub fn reshape(self, dims: &[i64]) -> Self {
        let dims = dims.to_vec();
        self.apply_private(FunctionExpr::Reshape(dims))
    }

    #[cfg(feature = "ewma")]
    /// Calculate the exponentially-weighted moving average.
    pub fn ewm_mean(self, options: EWMOptions) -> Self {
        self.apply_private(FunctionExpr::EwmMean { options })
    }

    #[cfg(feature = "ewma")]
    /// Calculate the exponentially-weighted moving standard deviation.
    pub fn ewm_std(self, options: EWMOptions) -> Self {
        self.apply_private(FunctionExpr::EwmStd { options })
    }

    #[cfg(feature = "ewma")]
    /// Calculate the exponentially-weighted moving variance.
    pub fn ewm_var(self, options: EWMOptions) -> Self {
        self.apply_private(FunctionExpr::EwmVar { options })
    }

    /// Returns whether any of the values in the column are `true`.
    ///
    /// If `ignore_nulls` is `False`, [Kleene logic] is used to deal with nulls:
    /// if the column contains any null values and no `true` values, the output
    /// is null.
    ///
    /// [Kleene logic]: https://en.wikipedia.org/wiki/Three-valued_logic
    pub fn any(self, ignore_nulls: bool) -> Self {
        self.apply_private(BooleanFunction::Any { ignore_nulls }.into())
            .with_function_options(|mut opt| {
                opt.returns_scalar = true;
                opt
            })
    }

    /// Returns whether all values in the column are `true`.
    ///
    /// If `ignore_nulls` is `False`, [Kleene logic] is used to deal with nulls:
    /// if the column contains any null values and no `true` values, the output
    /// is null.
    ///
    /// [Kleene logic]: https://en.wikipedia.org/wiki/Three-valued_logic
    pub fn all(self, ignore_nulls: bool) -> Self {
        self.apply_private(BooleanFunction::All { ignore_nulls }.into())
            .with_function_options(|mut opt| {
                opt.returns_scalar = true;
                opt
            })
    }

    /// Shrink numeric columns to the minimal required datatype
    /// needed to fit the extrema of this [`Series`].
    /// This can be used to reduce memory pressure.
    pub fn shrink_dtype(self) -> Self {
        self.map_private(FunctionExpr::ShrinkType)
    }

    #[cfg(feature = "dtype-struct")]
    /// Count all unique values and create a struct mapping value to count.
    /// (Note that it is better to turn parallel off in the aggregation context).
    pub fn value_counts(self, sort: bool, parallel: bool) -> Self {
        self.apply_private(FunctionExpr::ValueCounts { sort, parallel })
            .with_function_options(|mut opts| {
                opts.pass_name_to_apply = true;
                opts
            })
    }

    #[cfg(feature = "unique_counts")]
    /// Returns a count of the unique values in the order of appearance.
    /// This method differs from [`Expr::value_counts]` in that it does not return the
    /// values, only the counts and might be faster.
    pub fn unique_counts(self) -> Self {
        self.apply_private(FunctionExpr::UniqueCounts)
    }

    #[cfg(feature = "log")]
    /// Compute the logarithm to a given base.
    pub fn log(self, base: f64) -> Self {
        self.map_private(FunctionExpr::Log { base })
    }

    #[cfg(feature = "log")]
    /// Compute the natural logarithm of all elements plus one in the input array.
    pub fn log1p(self) -> Self {
        self.map_private(FunctionExpr::Log1p)
    }

    #[cfg(feature = "log")]
    /// Calculate the exponential of all elements in the input array.
    pub fn exp(self) -> Self {
        self.map_private(FunctionExpr::Exp)
    }

    #[cfg(feature = "log")]
    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    pub fn entropy(self, base: f64, normalize: bool) -> Self {
        self.apply_private(FunctionExpr::Entropy { base, normalize })
            .with_function_options(|mut options| {
                options.returns_scalar = true;
                options
            })
    }
    /// Get the null count of the column/group.
    pub fn null_count(self) -> Expr {
        self.apply_private(FunctionExpr::NullCount)
            .with_function_options(|mut options| {
                options.returns_scalar = true;
                options
            })
    }

    /// Set this `Series` as `sorted` so that downstream code can use
    /// fast paths for sorted arrays.
    /// # Warning
    /// This can lead to incorrect results if this `Series` is not sorted!!
    /// Use with care!
    pub fn set_sorted_flag(self, sorted: IsSorted) -> Expr {
        self.apply_private(FunctionExpr::SetSortedFlag(sorted))
    }

    #[cfg(feature = "row_hash")]
    /// Compute the hash of every element.
    pub fn hash(self, k0: u64, k1: u64, k2: u64, k3: u64) -> Expr {
        self.map_private(FunctionExpr::Hash(k0, k1, k2, k3))
    }

    pub fn to_physical(self) -> Expr {
        self.map_private(FunctionExpr::ToPhysical)
    }

    #[cfg(feature = "strings")]
    /// Get the [`string::StringNameSpace`]
    pub fn str(self) -> string::StringNameSpace {
        string::StringNameSpace(self)
    }

    /// Get the [`binary::BinaryNameSpace`]
    pub fn binary(self) -> binary::BinaryNameSpace {
        binary::BinaryNameSpace(self)
    }

    #[cfg(feature = "temporal")]
    /// Get the [`dt::DateLikeNameSpace`]
    pub fn dt(self) -> dt::DateLikeNameSpace {
        dt::DateLikeNameSpace(self)
    }

    /// Get the [`list::ListNameSpace`]
    pub fn list(self) -> list::ListNameSpace {
        list::ListNameSpace(self)
    }

    /// Get the [`name::ExprNameNameSpace`]
    pub fn name(self) -> name::ExprNameNameSpace {
        name::ExprNameNameSpace(self)
    }

    /// Get the [`array::ArrayNameSpace`].
    #[cfg(feature = "dtype-array")]
    pub fn arr(self) -> array::ArrayNameSpace {
        array::ArrayNameSpace(self)
    }

    /// Get the [`CategoricalNameSpace`].
    #[cfg(feature = "dtype-categorical")]
    pub fn cat(self) -> cat::CategoricalNameSpace {
        cat::CategoricalNameSpace(self)
    }

    /// Get the [`struct_::StructNameSpace`].
    #[cfg(feature = "dtype-struct")]
    pub fn struct_(self) -> struct_::StructNameSpace {
        struct_::StructNameSpace(self)
    }

    /// Get the [`meta::MetaNameSpace`]
    #[cfg(feature = "meta")]
    pub fn meta(self) -> meta::MetaNameSpace {
        meta::MetaNameSpace(self)
    }
}

/// Apply a function/closure over multiple columns once the logical plan get executed.
///
/// This function is very similar to `[apply_mul]`, but differs in how it handles aggregations.
///
///  * `map_mul` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
///  * `apply_mul` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
///
/// It is the responsibility of the caller that the schema is correct by giving
/// the correct output_type. If None given the output type of the input expr is used.
pub fn map_multiple<F, E>(function: F, expr: E, output_type: GetOutput) -> Expr
where
    F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: SpecialEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            fmt_str: "",
            ..Default::default()
        },
    }
}

/// Apply a function/closure over multiple columns once the logical plan get executed.
///
/// This function is very similar to `[apply_mul]`, but differs in how it handles aggregations.
///
///  * `map_mul` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
///  * `apply_mul` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
///  * `map_list_mul` should be used when the function expects a list aggregated series.
pub fn map_list_multiple<F, E>(function: F, expr: E, output_type: GetOutput) -> Expr
where
    F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: SpecialEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyList,
            returns_scalar: true,
            fmt_str: "",
            ..Default::default()
        },
    }
}

/// Apply a function/closure over the groups of multiple columns. This should only be used in a group_by aggregation.
///
/// It is the responsibility of the caller that the schema is correct by giving
/// the correct output_type. If None given the output type of the input expr is used.
///
/// This difference with `[map_mul]` is that `[apply_mul]` will create a separate `[Series]` per group.
///
/// * `[map_mul]` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
/// * `[apply_mul]` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
pub fn apply_multiple<F, E>(
    function: F,
    expr: E,
    output_type: GetOutput,
    returns_scalar: bool,
) -> Expr
where
    F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: SpecialEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            // don't set this to true
            // this is for the caller to decide
            returns_scalar,
            fmt_str: "",
            ..Default::default()
        },
    }
}

/// Count expression.
pub fn count() -> Expr {
    Expr::Count
}

/// First column in DataFrame.
pub fn first() -> Expr {
    Expr::Nth(0)
}

/// Last column in DataFrame.
pub fn last() -> Expr {
    Expr::Nth(-1)
}
