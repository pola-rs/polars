//! Domain specific language for the Lazy api.
#[cfg(feature = "dtype-categorical")]
pub mod cat;
#[cfg(feature = "dtype-categorical")]
pub use cat::*;
#[cfg(feature = "temporal")]
mod dt;
mod expr;
mod from;
pub(crate) mod function_expr;
#[cfg(feature = "compile")]
pub mod functions;
mod list;
#[cfg(feature = "meta")]
mod meta;
pub(crate) mod names;
mod options;
#[cfg(feature = "strings")]
pub mod string;
#[cfg(feature = "dtype-struct")]
mod struct_;

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::Arc;

pub use expr::*;
pub use function_expr::*;
pub use functions::*;
pub use list::*;
pub use options::*;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::prelude::*;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_core::series::IsSorted;
use polars_core::utils::{try_get_supertype, NoNull};
use polars_ops::prelude::SeriesOps;
#[cfg(feature = "rolling_window")]
use polars_time::series::SeriesOpsTime;

pub use crate::logical_plan::lit;
use crate::prelude::*;
use crate::utils::has_expr;
#[cfg(feature = "is_in")]
use crate::utils::has_root_literal_expr;

pub fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

/// Intermediate state of `when(..).then(..).otherwise(..)` expr.
#[derive(Clone)]
pub struct When {
    predicate: Expr,
}

/// Intermediate state of `when(..).then(..).otherwise(..)` expr.
#[derive(Clone)]
pub struct WhenThen {
    predicate: Expr,
    then: Expr,
}

/// Intermediate state of chain when then exprs.
///
/// ```text
/// when(..).then(..)
/// when(..).then(..)
/// when(..).then(..)
/// .otherwise(..)`
/// ```
#[derive(Clone)]
#[must_use]
pub struct WhenThenThen {
    predicates: Vec<Expr>,
    thens: Vec<Expr>,
}

impl When {
    pub fn then<E: Into<Expr>>(self, expr: E) -> WhenThen {
        WhenThen {
            predicate: self.predicate,
            then: expr.into(),
        }
    }
}

impl WhenThen {
    pub fn when<E: Into<Expr>>(self, predicate: E) -> WhenThenThen {
        WhenThenThen {
            predicates: vec![self.predicate, predicate.into()],
            thens: vec![self.then],
        }
    }

    pub fn otherwise<E: Into<Expr>>(self, expr: E) -> Expr {
        Expr::Ternary {
            predicate: Box::new(self.predicate),
            truthy: Box::new(self.then),
            falsy: Box::new(expr.into()),
        }
    }
}

impl WhenThenThen {
    pub fn then(mut self, expr: Expr) -> Self {
        self.thens.push(expr);
        self
    }

    pub fn when(mut self, predicate: Expr) -> Self {
        self.predicates.push(predicate);
        self
    }

    pub fn otherwise(self, expr: Expr) -> Expr {
        // we iterate the preds/ exprs last in first out
        // and nest them.
        //
        // // this expr:
        //   when((col('x') == 'a')).then(1)
        //         .when(col('x') == 'a').then(2)
        //         .when(col('x') == 'b').then(3)
        //         .otherwise(4)
        //
        // needs to become:
        //       when((col('x') == 'a')).then(1)                        -
        //         .otherwise(                                           |
        //             when(col('x') == 'a').then(2)            -        |
        //             .otherwise(                               |       |
        //                 pl.when(col('x') == 'b').then(3)      |       |
        //                 .otherwise(4)                         | inner | outer
        //             )                                         |       |
        //         )                                            _|      _|
        //
        // by iterating lifo we first create
        // `inner` and then assign that to `otherwise`,
        // which will be used in the next layer `outer`
        //

        let pred_iter = self.predicates.into_iter().rev();
        let mut then_iter = self.thens.into_iter().rev();

        let mut otherwise = expr;

        for e in pred_iter {
            otherwise = Expr::Ternary {
                predicate: Box::new(e),
                truthy: Box::new(
                    then_iter
                        .next()
                        .expect("expr expected, did you call when().then().otherwise?"),
                ),
                falsy: Box::new(otherwise),
            }
        }
        if then_iter.next().is_some() {
            panic!(
                "this expr is not properly constructed. \
            Every `when` should have an accompanied `then` call."
            )
        }
        otherwise
    }
}

/// Start a when-then-otherwise expression
pub fn when<E: Into<Expr>>(predicate: E) -> When {
    When {
        predicate: predicate.into(),
    }
}

pub fn ternary_expr(predicate: Expr, truthy: Expr, falsy: Expr) -> Expr {
    Expr::Ternary {
        predicate: Box::new(predicate),
        truthy: Box::new(truthy),
        falsy: Box::new(falsy),
    }
}

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
            }
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
            }
            _ => {
                panic!("implementation error")
            }
        }
    }

    /// Overwrite the function name used for formatting
    /// this is not intended to be used
    #[doc(hidden)]
    pub fn with_fmt(self, name: &'static str) -> Expr {
        self.with_function_options(|mut options| {
            options.fmt_str = name;
            options
        })
    }

    /// Compare `Expr` with other `Expr` on equality
    pub fn eq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::Eq, other.into())
    }

    /// Compare `Expr` with other `Expr` on non-equality
    pub fn neq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::NotEq, other.into())
    }

    /// Check if `Expr` < `Expr`
    pub fn lt<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::Lt, other.into())
    }

    /// Check if `Expr` > `Expr`
    pub fn gt<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::Gt, other.into())
    }

    /// Check if `Expr` >= `Expr`
    pub fn gt_eq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::GtEq, other.into())
    }

    /// Check if `Expr` <= `Expr`
    pub fn lt_eq<E: Into<Expr>>(self, other: E) -> Expr {
        binary_expr(self, Operator::LtEq, other.into())
    }

    /// Negate `Expr`
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        self.map_private(FunctionExpr::Not)
    }

    /// Rename Column.
    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), Arc::from(name))
    }

    /// Run is_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_null(self) -> Self {
        self.map_private(FunctionExpr::IsNull)
    }

    /// Run is_not_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_null(self) -> Self {
        self.map_private(FunctionExpr::IsNotNull)
    }

    /// Drop null values
    pub fn drop_nulls(self) -> Self {
        self.apply(|s| Ok(s.drop_nulls()), GetOutput::same_type())
    }

    /// Drop NaN values
    pub fn drop_nans(self) -> Self {
        self.apply_private(NanFunction::DropNans.into())
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

    /// Aggregate the group to a Series
    pub fn list(self) -> Self {
        AggExpr::List(Box::new(self)).into()
    }

    /// Compute the quantile per group.
    pub fn quantile(self, quantile: f64, interpol: QuantileInterpolOptions) -> Self {
        AggExpr::Quantile {
            expr: Box::new(self),
            quantile,
            interpol,
        }
        .into()
    }

    /// Get the group indexes of the group by operation.
    pub fn agg_groups(self) -> Self {
        AggExpr::AggGroups(Box::new(self)).into()
    }

    /// Alias for explode
    pub fn flatten(self) -> Self {
        self.explode()
    }

    /// Explode the utf8/ list column
    pub fn explode(self) -> Self {
        let has_filter = has_expr(&self, |e| matches!(e, Expr::Filter { .. }));

        // if we explode right after a window function we don't self join, but just flatten
        // the expression
        if let Expr::Window {
            function,
            partition_by,
            order_by,
            mut options,
        } = self
        {
            if has_filter {
                panic!("A Filter of a window function is not allowed in combination with explode/flatten.\
                The resulting column may not fit the DataFrame/ or the groups
                ")
            }

            options.explode = true;

            Expr::Explode(Box::new(Expr::Window {
                function,
                partition_by,
                order_by,
                options,
            }))
        } else {
            Expr::Explode(Box::new(self))
        }
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
                Ok(a)
            },
            output_type,
        )
    }

    /// Get the first `n` elements of the Expr result
    pub fn head(self, length: Option<usize>) -> Self {
        self.slice(lit(0), lit(length.unwrap_or(10) as u64))
    }

    /// Get the last `n` elements of the Expr result
    pub fn tail(self, length: Option<usize>) -> Self {
        let len = length.unwrap_or(10);
        self.slice(lit(-(len as i64)), lit(len as u64))
    }

    /// Get unique values of this expression.
    pub fn unique(self) -> Self {
        self.apply(|s: Series| s.unique(), GetOutput::same_type())
            .with_fmt("unique")
    }

    /// Get unique values of this expression, while maintaining order.
    /// This requires more work than [`Expr::unique`].
    pub fn unique_stable(self) -> Self {
        self.apply(|s: Series| s.unique_stable(), GetOutput::same_type())
            .with_fmt("unique_stable")
    }

    /// Get the first index of unique values of this expression.
    pub fn arg_unique(self) -> Self {
        self.apply(
            |s: Series| s.arg_unique().map(|ca| ca.into_series()),
            GetOutput::from_type(IDX_DTYPE),
        )
        .with_fmt("arg_unique")
    }

    /// Get the index value that has the minimum value
    pub fn arg_min(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            auto_explode: true,
            fmt_str: "arg_min",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| Ok(Series::new(s.name(), &[s.arg_min().map(|idx| idx as u32)])),
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index value that has the maximum value
    pub fn arg_max(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            auto_explode: true,
            fmt_str: "arg_max",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| Ok(Series::new(s.name(), &[s.arg_max().map(|idx| idx as u32)])),
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index values that would sort this expression.
    pub fn arg_sort(self, sort_options: SortOptions) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            fmt_str: "arg_sort",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| Ok(s.argsort(sort_options).into_series()),
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    #[cfg(feature = "search_sorted")]
    /// Find indices where elements should be inserted to maintain order.
    pub fn search_sorted<E: Into<Expr>>(self, element: E) -> Expr {
        let element = element.into();
        Expr::Function {
            input: vec![self, element],
            function: FunctionExpr::SearchSorted,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                auto_explode: true,
                fmt_str: "search_sorted",
                ..Default::default()
            },
        }
    }

    /// Cast expression to another data type.
    /// Throws an error if conversion had overflows
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
        }
    }

    /// Sort in increasing order. See [the eager implementation](Series::sort).
    pub fn sort(self, reverse: bool) -> Self {
        Expr::Sort {
            expr: Box::new(self),
            options: SortOptions {
                descending: reverse,
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
    pub fn top_k(self, k: usize, reverse: bool) -> Self {
        self.apply_private(FunctionExpr::TopK { k, reverse })
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
        F: Fn(Series) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: false,
                auto_explode: false,
                fmt_str: "map",
                cast_to_supertypes: false,
                allow_rename: false,
            },
        }
    }

    fn map_private(self, function_expr: FunctionExpr) -> Self {
        Expr::Function {
            input: vec![self],
            function: function_expr,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: false,
                auto_explode: false,
                cast_to_supertypes: false,
                allow_rename: false,
                ..Default::default()
            },
        }
    }

    /// Apply a function/closure once the logical plan get executed with many arguments
    ///
    /// See the [`Expr::map`] function for the differences between [`map`](Expr::map) and [`apply`](Expr::apply).
    pub fn map_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Series]) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::AnonymousFunction {
            input,
            function: SpecialEq::new(Arc::new(function)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
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
        F: Fn(Series) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyList,
                fmt_str: "map_list",
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
        F: Fn(Series) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options,
        }
    }

    /// Apply a function/closure over the groups. This should only be used in a groupby aggregation.
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
        F: Fn(Series) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: SpecialEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
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
                collect_groups: ApplyOptions::ApplyGroups,
                ..Default::default()
            },
        }
    }

    /// Apply a function/closure over the groups with many arguments. This should only be used in a groupby aggregation.
    ///
    /// See the [`Expr::apply`] function for the differences between [`map`](Expr::map) and [`apply`](Expr::apply).
    pub fn apply_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Series]) -> PolarsResult<Series> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::AnonymousFunction {
            input,
            function: SpecialEq::new(Arc::new(function)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                fmt_str: "",
                auto_explode: true,
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
                collect_groups: ApplyOptions::ApplyGroups,
                auto_explode,
                cast_to_supertypes,
                ..Default::default()
            },
        }
    }

    pub fn map_many_private(
        self,
        function_expr: FunctionExpr,
        arguments: &[Expr],
        cast_to_supertypes: bool,
    ) -> Self {
        let mut input = Vec::with_capacity(arguments.len() + 1);
        input.push(self);
        input.extend_from_slice(arguments);

        Expr::Function {
            input,
            function: function_expr,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                auto_explode: true,
                cast_to_supertypes,
                ..Default::default()
            },
        }
    }

    /// Get mask of finite values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_finite(self) -> Self {
        self.map(
            |s: Series| s.is_finite().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
        .with_fmt("is_finite")
    }

    /// Get mask of infinite values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_infinite(self) -> Self {
        self.map(
            |s: Series| s.is_infinite().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
        .with_fmt("is_infinite")
    }

    /// Get mask of NaN values if dtype is Float
    pub fn is_nan(self) -> Self {
        self.map_private(NanFunction::IsNan.into())
    }

    /// Get inverse mask of NaN values if dtype is Float
    pub fn is_not_nan(self) -> Self {
        self.map_private(NanFunction::IsNotNan.into())
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::shift).
    pub fn shift(self, periods: i64) -> Self {
        self.apply_private(FunctionExpr::Shift(periods))
    }

    /// Shift the values in the array by some period and fill the resulting empty values.
    pub fn shift_and_fill<E: Into<Expr>>(self, periods: i64, fill_value: E) -> Self {
        self.apply_many_private(
            FunctionExpr::ShiftAndFill { periods },
            &[fill_value.into()],
            false,
            true,
        )
    }

    /// Get an array with the cumulative sum computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cumsum(self, reverse: bool) -> Self {
        self.apply(
            move |s: Series| Ok(s.cumsum(reverse)),
            GetOutput::same_type(),
        )
        .with_fmt("cumsum")
    }

    /// Get an array with the cumulative product computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cumprod(self, reverse: bool) -> Self {
        self.apply(
            move |s: Series| Ok(s.cumprod(reverse)),
            GetOutput::map_dtype(|dt| {
                use DataType::*;
                match dt {
                    Float32 => Float32,
                    Float64 => Float64,
                    _ => Int64,
                }
            }),
        )
        .with_fmt("cumprod")
    }

    /// Get an array with the cumulative min computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cummin(self, reverse: bool) -> Self {
        self.apply(
            move |s: Series| Ok(s.cummin(reverse)),
            GetOutput::same_type(),
        )
        .with_fmt("cummin")
    }

    /// Get an array with the cumulative max computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cummax(self, reverse: bool) -> Self {
        self.apply(
            move |s: Series| Ok(s.cummax(reverse)),
            GetOutput::map_dtype(|dt| {
                use DataType::*;
                match dt {
                    Float32 => Float32,
                    Float64 => Float64,
                    _ => Int64,
                }
            }),
        )
        .with_fmt("cummax")
    }

    /// Get the product aggregation of an expression
    #[cfg_attr(docsrs, doc(cfg(feature = "product")))]
    pub fn product(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            auto_explode: true,
            fmt_str: "product",
            ..Default::default()
        };

        self.function_with_options(
            move |s: Series| Ok(s.product()),
            GetOutput::map_dtype(|dt| {
                use DataType::*;
                match dt {
                    Float32 => Float32,
                    Float64 => Float64,
                    _ => Int64,
                }
            }),
            options,
        )
    }

    /// Fill missing value with next non-null.
    pub fn backward_fill(self, limit: FillNullLimit) -> Self {
        self.apply(
            move |s: Series| s.fill_null(FillNullStrategy::Backward(limit)),
            GetOutput::same_type(),
        )
        .with_fmt("backward_fill")
    }

    /// Fill missing value with previous non-null.
    pub fn forward_fill(self, limit: FillNullLimit) -> Self {
        self.apply(
            move |s: Series| s.fill_null(FillNullStrategy::Forward(limit)),
            GetOutput::same_type(),
        )
        .with_fmt("forward_fill")
    }

    /// Round underlying floating point array to given decimal numbers.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn round(self, decimals: u32) -> Self {
        self.map(move |s: Series| s.round(decimals), GetOutput::same_type())
            .with_fmt("round")
    }

    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn floor(self) -> Self {
        self.map(move |s: Series| s.floor(), GetOutput::same_type())
            .with_fmt("floor")
    }

    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn ceil(self) -> Self {
        self.map(move |s: Series| s.ceil(), GetOutput::same_type())
            .with_fmt("ceil")
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn clip(self, min: AnyValue<'_>, max: AnyValue<'_>) -> Self {
        self.map_private(FunctionExpr::Clip {
            min: Some(min.into_static().unwrap()),
            max: Some(max.into_static().unwrap()),
        })
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn clip_max(self, max: AnyValue<'_>) -> Self {
        self.map_private(FunctionExpr::Clip {
            min: None,
            max: Some(max.into_static().unwrap()),
        })
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn clip_min(self, min: AnyValue<'_>) -> Self {
        self.map_private(FunctionExpr::Clip {
            min: Some(min.into_static().unwrap()),
            max: None,
        })
    }

    /// Convert all values to their absolute/positive value.
    #[cfg(feature = "abs")]
    #[cfg_attr(docsrs, doc(cfg(feature = "abs")))]
    pub fn abs(self) -> Self {
        self.map(move |s: Series| s.abs(), GetOutput::same_type())
            .with_fmt("abs")
    }

    /// Apply window function over a subgroup.
    /// This is similar to a groupby + aggregation + self join.
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
    ///     dbg!(&out);
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
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 1      ┆ 16     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 2      ┆ 13     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 2      ┆ 13     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ ...    ┆ ...    │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 1      ┆ 16     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 2      ┆ 13     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 3      ┆ 15     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 3      ┆ 15     │
    /// ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
    /// │ 1      ┆ 16     │
    /// ╰────────┴────────╯
    /// ```
    pub fn over<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(self, partition_by: E) -> Self {
        let partition_by = partition_by
            .as_ref()
            .iter()
            .map(|e| e.clone().into())
            .collect();
        Expr::Window {
            function: Box::new(self),
            partition_by,
            order_by: None,
            options: WindowOptions { explode: false },
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
                collect_groups: ApplyOptions::ApplyFlat,
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

    /// Standard deviation of the values of the Series
    pub fn std(self, ddof: u8) -> Self {
        AggExpr::Std(Box::new(self), ddof).into()
    }

    /// Variance of the values of the Series
    pub fn var(self, ddof: u8) -> Self {
        AggExpr::Var(Box::new(self), ddof).into()
    }

    /// Get a mask of duplicated values
    #[allow(clippy::wrong_self_convention)]
    pub fn is_duplicated(self) -> Self {
        self.apply_private(FunctionExpr::IsDuplicated)
    }

    /// Get a mask of unique values
    #[allow(clippy::wrong_self_convention)]
    pub fn is_unique(self) -> Self {
        self.apply_private(FunctionExpr::IsUnique)
    }

    /// and operation
    pub fn and<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::And, expr.into())
    }

    // xor operation
    pub fn xor<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::Xor, expr.into())
    }

    /// or operation
    pub fn or<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::Or, expr.into())
    }

    /// Raise expression to the power `exponent`
    pub fn pow<E: Into<Expr>>(self, exponent: E) -> Self {
        Expr::Function {
            input: vec![self, exponent.into()],
            function: FunctionExpr::Pow,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn sin(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Sin),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cos(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Cos),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn tan(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Tan),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arcsin(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcSin),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arccos(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcCos),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arctan(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcTan),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the hyperbolic sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn sinh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Sinh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the hyperbolic cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cosh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Cosh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the hyperbolic tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn tanh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Tanh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse hyperbolic sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arcsinh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcSinh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse hyperbolic cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arccosh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcCosh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse hyperbolic tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arctanh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcTanh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                fmt_str: "arctanh",
                ..Default::default()
            },
        }
    }

    /// Compute the sign of the given expression
    #[cfg(feature = "sign")]
    pub fn sign(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Sign,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Filter a single column
    /// Should be used in aggregation context. If you want to filter on a DataFrame level, use
    /// [LazyFrame::filter](LazyFrame::filter)
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
    #[cfg_attr(docsrs, doc(cfg(feature = "is_in")))]
    pub fn is_in<E: Into<Expr>>(self, other: E) -> Self {
        let other = other.into();
        let has_literal = has_root_literal_expr(&other);
        if has_literal
            && match &other {
                Expr::Literal(LiteralValue::Series(s)) if s.is_empty() => true,
                Expr::Literal(LiteralValue::Null) => true,
                _ => false,
            }
        {
            return Expr::Literal(LiteralValue::Boolean(false));
        }

        let arguments = &[other];
        // we don't have to apply on groups, so this is faster
        if has_literal {
            self.map_many_private(FunctionExpr::IsIn, arguments, true)
        } else {
            self.apply_many_private(FunctionExpr::IsIn, arguments, true, true)
        }
    }

    /// Sort this column by the ordering of another column.
    /// Can also be used in a groupby context to sort the groups.
    pub fn sort_by<E: AsRef<[IE]>, IE: Into<Expr> + Clone, R: AsRef<[bool]>>(
        self,
        by: E,
        reverse: R,
    ) -> Expr {
        let by = by.as_ref().iter().map(|e| e.clone().into()).collect();
        let reverse = reverse.as_ref().to_vec();
        Expr::SortBy {
            expr: Box::new(self),
            by,
            reverse,
        }
    }

    #[cfg(feature = "repeat_by")]
    fn repeat_by_impl(self, by: Expr) -> Expr {
        let function = |s: &mut [Series]| {
            let by = &s[1];
            let s = &s[0];
            let by = by.cast(&IDX_DTYPE)?;
            Ok(s.repeat_by(by.idx()?).into_series())
        };

        self.apply_many(
            function,
            &[by],
            GetOutput::map_dtype(|dt| DataType::List(dt.clone().into())),
        )
        .with_fmt("repeat_by")
    }

    #[cfg(feature = "repeat_by")]
    #[cfg_attr(docsrs, doc(cfg(feature = "repeat_by")))]
    /// Repeat the column `n` times, where `n` is determined by the values in `by`.
    /// This yields an `Expr` of dtype `List`
    pub fn repeat_by<E: Into<Expr>>(self, by: E) -> Expr {
        self.repeat_by_impl(by.into())
    }

    #[cfg(feature = "is_first")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_first")))]
    #[allow(clippy::wrong_self_convention)]
    /// Get a mask of the first unique value.
    pub fn is_first(self) -> Expr {
        self.apply(
            |s| s.is_first().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
        .with_fmt("is_first")
    }

    #[cfg(feature = "dot_product")]
    fn dot_impl(self, other: Expr) -> Expr {
        let function = |s: &mut [Series]| Ok((&s[0] * &s[1]).sum_as_series());

        self.apply_many(function, &[other], GetOutput::same_type())
            .with_fmt("dot")
    }

    #[cfg(feature = "dot_product")]
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_product")))]
    pub fn dot<E: Into<Expr>>(self, other: E) -> Expr {
        self.dot_impl(other.into())
    }

    #[cfg(feature = "mode")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mode")))]
    /// Compute the mode(s) of this column. This is the most occurring value.
    pub fn mode(self) -> Expr {
        self.apply(
            |s| s.mode().map(|ca| ca.into_series()),
            GetOutput::same_type(),
        )
        .with_fmt("mode")
    }

    /// Keep the original root name
    ///
    /// ```
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: LazyFrame) -> LazyFrame {
    ///     df.select([
    /// // even thought the alias yields a different column name,
    /// // `keep_name` will make sure that the original column name is used
    ///         col("*").alias("foo").keep_name()
    /// ])
    /// }
    /// ```
    pub fn keep_name(self) -> Expr {
        Expr::KeepName(Box::new(self))
    }

    /// Define an alias by mapping a function over the original root column name.
    pub fn map_alias<F>(self, function: F) -> Expr
    where
        F: Fn(&str) -> PolarsResult<String> + 'static + Send + Sync,
    {
        let function = SpecialEq::new(Arc::new(function) as Arc<dyn RenameAliasFn>);
        Expr::RenameAlias {
            expr: Box::new(self),
            function,
        }
    }

    /// Add a suffix to the root column name.
    pub fn suffix(self, suffix: &str) -> Expr {
        let suffix = suffix.to_string();
        self.map_alias(move |name| Ok(format!("{}{}", name, suffix)))
    }

    /// Add a prefix to the root column name.
    pub fn prefix(self, prefix: &str) -> Expr {
        let prefix = prefix.to_string();
        self.map_alias(move |name| Ok(format!("{}{}", prefix, name)))
    }

    /// Exclude a column from a wildcard/regex selection.
    ///
    /// You may also use regexes in the exclude as long as they start with `^` and end with `$`/
    ///
    /// # Example
    ///
    /// ```rust
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// // Select all columns except foo.
    /// fn example(df: DataFrame) -> LazyFrame {
    ///       df.lazy()
    ///         .select(&[
    ///                 col("*").exclude(&["foo"])
    ///                 ])
    /// }
    /// ```
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

    // Interpolate None values
    #[cfg(feature = "interpolate")]
    #[cfg_attr(docsrs, doc(cfg(feature = "interpolate")))]
    pub fn interpolate(self) -> Expr {
        self.apply(|s| Ok(s.interpolate()), GetOutput::same_type())
            .with_fmt("interpolate")
    }

    #[cfg(feature = "rolling_window")]
    #[allow(clippy::type_complexity)]
    fn finish_rolling(
        self,
        options: RollingOptions,
        expr_name: &'static str,
        expr_name_by: &'static str,
        rolling_fn: Arc<
            dyn (Fn(&Series, RollingOptionsImpl) -> PolarsResult<Series>) + Send + Sync,
        >,
    ) -> Expr {
        if let Some(ref by) = options.by {
            self.apply_many(
                move |s| {
                    let mut by = s[1].clone();
                    by = by.rechunk();
                    let s = &s[0];

                    if options.weights.is_some() {
                        return Err(PolarsError::ComputeError(
                            "weights not supported in 'rolling by' expression".into(),
                        ));
                    }

                    if matches!(by.dtype(), DataType::Datetime(_, _)) {
                        by = by.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
                    }
                    let by = by.datetime().unwrap();
                    let by_values = by.cont_slice().map_err(|_| {
                        PolarsError::ComputeError(
                            "'by' column should not have null values in 'rolling by'".into(),
                        )
                    })?;
                    let tu = by.time_unit();

                    let options = RollingOptionsImpl {
                        window_size: options.window_size,
                        min_periods: options.min_periods,
                        weights: None,
                        center: options.center,
                        by: Some(by_values),
                        tu: Some(tu),
                        closed_window: options.closed_window,
                    };

                    rolling_fn(s, options)
                },
                &[col(by)],
                GetOutput::same_type(),
            )
            .with_fmt(expr_name_by)
        } else {
            if !options.window_size.parsed_int {
                panic!("if dynamic windows are used in a rolling aggregation, the 'by' argument must be set")
            }

            self.apply(
                move |s| rolling_fn(&s, options.clone().into()),
                GetOutput::same_type(),
            )
            .with_fmt(expr_name)
        }
    }

    /// Apply a rolling min See:
    /// [ChunkedArray::rolling_min]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_min(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_min",
            "rolling_min_by",
            Arc::new(|s, options| s.rolling_min(options)),
        )
    }

    /// Apply a rolling max See:
    /// [ChunkedArray::rolling_max]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_max(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_max",
            "rolling_max_by",
            Arc::new(|s, options| s.rolling_max(options)),
        )
    }

    /// Apply a rolling mean See:
    /// [ChunkedArray::rolling_mean]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_mean(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_mean",
            "rolling_mean_by",
            Arc::new(|s, options| s.rolling_mean(options)),
        )
    }

    /// Apply a rolling sum See:
    /// [ChunkedArray::rolling_sum]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_sum(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_sum",
            "rolling_sum_by",
            Arc::new(|s, options| s.rolling_sum(options)),
        )
    }

    /// Apply a rolling median See:
    /// [`ChunkedArray::rolling_median`]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_median(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_median",
            "rolling_median_by",
            Arc::new(|s, options| s.rolling_median(options)),
        )
    }

    /// Apply a rolling quantile See:
    /// [`ChunkedArray::rolling_quantile`]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_quantile(
        self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptions,
    ) -> Expr {
        self.finish_rolling(
            options,
            "rolling_quantile",
            "rolling_quantile_by",
            Arc::new(move |s, options| s.rolling_quantile(quantile, interpolation, options)),
        )
    }

    /// Apply a rolling variance
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_var(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_var",
            "rolling_var_by",
            Arc::new(|s, options| s.rolling_var(options)),
        )
    }

    /// Apply a rolling std-dev
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_std(self, options: RollingOptions) -> Expr {
        self.finish_rolling(
            options,
            "rolling_std",
            "rolling_std_by",
            Arc::new(|s, options| s.rolling_std(options)),
        )
    }

    /// Apply a rolling skew
    #[cfg_attr(docsrs, doc(cfg(all(feature = "rolling_window", feature = "moment"))))]
    #[cfg(feature = "rolling_window")]
    #[cfg(feature = "moment")]
    pub fn rolling_skew(self, window_size: usize, bias: bool) -> Expr {
        self.apply_private(FunctionExpr::RollingSkew { window_size, bias })
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    pub fn rolling_apply(
        self,
        f: Arc<dyn Fn(&Series) -> Series + Send + Sync>,
        output_type: GetOutput,
        options: RollingOptionsFixedWindow,
    ) -> Expr {
        self.apply(
            move |s| s.rolling_apply(f.as_ref(), options.clone()),
            output_type,
        )
        .with_fmt("rolling_apply")
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// Prefer this over rolling_apply in case of floating point numbers as this is faster.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    pub fn rolling_apply_float<F>(self, window_size: usize, f: F) -> Expr
    where
        F: 'static + FnMut(&mut Float64Chunked) -> Option<f64> + Send + Sync + Copy,
    {
        self.apply(
            move |s| {
                let out = match s.dtype() {
                    DataType::Float64 => s
                        .f64()
                        .unwrap()
                        .rolling_apply_float(window_size, f)
                        .map(|ca| ca.into_series()),
                    _ => s
                        .cast(&DataType::Float64)?
                        .f64()
                        .unwrap()
                        .rolling_apply_float(window_size, f)
                        .map(|ca| ca.into_series()),
                }?;
                if let DataType::Float32 = s.dtype() {
                    out.cast(&DataType::Float32)
                } else {
                    Ok(out)
                }
            },
            GetOutput::map_field(|field| match field.data_type() {
                DataType::Float64 => field.clone(),
                DataType::Float32 => Field::new(field.name(), DataType::Float32),
                _ => Field::new(field.name(), DataType::Float64),
            }),
        )
        .with_fmt("rolling_apply_float")
    }

    #[cfg(feature = "rank")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rank")))]
    pub fn rank(self, options: RankOptions) -> Expr {
        self.apply(
            move |s| Ok(s.rank(options)),
            GetOutput::map_field(move |fld| match options.method {
                RankMethod::Average => Field::new(fld.name(), DataType::Float32),
                _ => Field::new(fld.name(), IDX_DTYPE),
            }),
        )
        .with_fmt("rank")
    }

    #[cfg(feature = "diff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    pub fn diff(self, n: usize, null_behavior: NullBehavior) -> Expr {
        self.apply_private(FunctionExpr::Diff(n, null_behavior))
    }

    #[cfg(feature = "pct_change")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pct_change")))]
    pub fn pct_change(self, n: usize) -> Expr {
        use DataType::*;
        self.apply(
            move |s| s.pct_change(n),
            GetOutput::map_dtype(|dt| match dt {
                Float64 | Float32 => dt.clone(),
                _ => Float64,
            }),
        )
        .with_fmt("pct_change")
    }

    #[cfg(feature = "moment")]
    #[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
    /// Compute the sample skewness of a data set.
    ///
    /// For normally distributed data, the skewness should be about zero. For
    /// uni-modal continuous distributions, a skewness value greater than zero means
    /// that there is more weight in the right tail of the distribution. The
    /// function `skewtest` can be used to determine if the skewness value
    /// is close enough to zero, statistically speaking.
    ///
    /// see: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L1024
    pub fn skew(self, bias: bool) -> Expr {
        self.apply(
            move |s| s.skew(bias).map(|opt_v| Series::new(s.name(), &[opt_v])),
            GetOutput::from_type(DataType::Float64),
        )
        .with_function_options(|mut options| {
            options.fmt_str = "skew";
            options.auto_explode = true;
            options
        })
    }

    #[cfg(feature = "moment")]
    #[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
    pub fn kurtosis(self, fisher: bool, bias: bool) -> Expr {
        self.apply(
            move |s| {
                s.kurtosis(fisher, bias)
                    .map(|opt_v| Series::new(s.name(), &[opt_v]))
            },
            GetOutput::from_type(DataType::Float64),
        )
        .with_function_options(|mut options| {
            options.fmt_str = "kurtosis";
            options.auto_explode = true;
            options
        })
    }

    /// Get maximal value that could be hold by this dtype.
    pub fn upper_bound(self) -> Expr {
        self.map(
            |s| {
                let name = s.name();
                use DataType::*;
                let s = match s.dtype().to_physical() {
                    #[cfg(feature = "dtype-i8")]
                    Int8 => Series::new(name, &[i8::MAX]),
                    #[cfg(feature = "dtype-i16")]
                    Int16 => Series::new(name, &[i16::MAX]),
                    Int32 => Series::new(name, &[i32::MAX]),
                    Int64 => Series::new(name, &[i64::MAX]),
                    #[cfg(feature = "dtype-u8")]
                    UInt8 => Series::new(name, &[u8::MAX]),
                    #[cfg(feature = "dtype-u16")]
                    UInt16 => Series::new(name, &[u16::MAX]),
                    UInt32 => Series::new(name, &[u32::MAX]),
                    UInt64 => Series::new(name, &[u64::MAX]),
                    Float32 => Series::new(name, &[f32::INFINITY]),
                    Float64 => Series::new(name, &[f64::INFINITY]),
                    dt => {
                        return Err(PolarsError::ComputeError(
                            format!("cannot determine upper bound of dtype {}", dt).into(),
                        ))
                    }
                };
                Ok(s)
            },
            GetOutput::same_type(),
        )
        .with_fmt("upper_bound")
    }

    /// Get minimal value that could be hold by this dtype.
    pub fn lower_bound(self) -> Expr {
        self.map(
            |s| {
                let name = s.name();
                use DataType::*;
                let s = match s.dtype().to_physical() {
                    #[cfg(feature = "dtype-i8")]
                    Int8 => Series::new(name, &[i8::MIN]),
                    #[cfg(feature = "dtype-i16")]
                    Int16 => Series::new(name, &[i16::MIN]),
                    Int32 => Series::new(name, &[i32::MIN]),
                    Int64 => Series::new(name, &[i64::MIN]),
                    #[cfg(feature = "dtype-u8")]
                    UInt8 => Series::new(name, &[u8::MIN]),
                    #[cfg(feature = "dtype-u16")]
                    UInt16 => Series::new(name, &[u16::MIN]),
                    UInt32 => Series::new(name, &[u32::MIN]),
                    UInt64 => Series::new(name, &[u64::MIN]),
                    Float32 => Series::new(name, &[f32::NEG_INFINITY]),
                    Float64 => Series::new(name, &[f64::NEG_INFINITY]),
                    dt => {
                        return Err(PolarsError::ComputeError(
                            format!("cannot determine lower bound of dtype {}", dt).into(),
                        ))
                    }
                };
                Ok(s)
            },
            GetOutput::same_type(),
        )
        .with_fmt("lower_bound")
    }

    pub fn reshape(self, dims: &[i64]) -> Self {
        let dims = dims.to_vec();
        let output_type = if dims.len() == 1 {
            GetOutput::map_field(|fld| {
                Field::new(
                    fld.name(),
                    fld.data_type()
                        .inner_dtype()
                        .unwrap_or_else(|| fld.data_type())
                        .clone(),
                )
            })
        } else {
            GetOutput::map_field(|fld| {
                let dtype = fld
                    .data_type()
                    .inner_dtype()
                    .unwrap_or_else(|| fld.data_type())
                    .clone();

                Field::new(fld.name(), DataType::List(Box::new(dtype)))
            })
        };
        self.apply(move |s| s.reshape(&dims), output_type)
            .with_fmt("reshape")
    }

    /// Cumulatively count values from 0 to len.
    pub fn cumcount(self, reverse: bool) -> Self {
        self.apply(
            move |s| {
                if reverse {
                    let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).rev().collect();
                    let mut ca = ca.into_inner();
                    ca.rename(s.name());
                    Ok(ca.into_series())
                } else {
                    let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).collect();
                    let mut ca = ca.into_inner();
                    ca.rename(s.name());
                    Ok(ca.into_series())
                }
            },
            GetOutput::from_type(IDX_DTYPE),
        )
        .with_fmt("cumcount")
    }

    #[cfg(feature = "random")]
    pub fn shuffle(self, seed: Option<u64>) -> Self {
        self.apply(move |s| Ok(s.shuffle(seed)), GetOutput::same_type())
            .with_fmt("shuffle")
    }

    #[cfg(feature = "random")]
    pub fn sample_n(
        self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.apply(
            move |s| s.sample_n(n, with_replacement, shuffle, seed),
            GetOutput::same_type(),
        )
        .with_fmt("sample_n")
    }

    #[cfg(feature = "random")]
    pub fn sample_frac(
        self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.apply(
            move |s| s.sample_frac(frac, with_replacement, shuffle, seed),
            GetOutput::same_type(),
        )
        .with_fmt("sample_frac")
    }

    #[cfg(feature = "ewma")]
    pub fn ewm_mean(self, options: EWMOptions) -> Self {
        use DataType::*;
        self.apply(
            move |s| s.ewm_mean(options),
            GetOutput::map_dtype(|dt| match dt {
                Float64 | Float32 => dt.clone(),
                _ => Float64,
            }),
        )
        .with_fmt("ewm_mean")
    }

    #[cfg(feature = "ewma")]
    pub fn ewm_std(self, options: EWMOptions) -> Self {
        use DataType::*;
        self.apply(
            move |s| s.ewm_std(options),
            GetOutput::map_dtype(|dt| match dt {
                Float64 | Float32 => dt.clone(),
                _ => Float64,
            }),
        )
        .with_fmt("ewm_std")
    }

    #[cfg(feature = "ewma")]
    pub fn ewm_var(self, options: EWMOptions) -> Self {
        use DataType::*;
        self.apply(
            move |s| s.ewm_var(options),
            GetOutput::map_dtype(|dt| match dt {
                Float64 | Float32 => dt.clone(),
                _ => Float64,
            }),
        )
        .with_fmt("ewm_var")
    }

    /// Check if any boolean value is `true`
    pub fn any(self) -> Self {
        self.apply(
            move |s| {
                let boolean = s.bool()?;
                if boolean.any() {
                    Ok(Series::new(s.name(), [true]))
                } else {
                    Ok(Series::new(s.name(), [false]))
                }
            },
            GetOutput::from_type(DataType::Boolean),
        )
        .with_function_options(|mut opt| {
            opt.fmt_str = "any";
            opt.auto_explode = true;
            opt
        })
    }

    /// Shrink numeric columns to the minimal required datatype
    /// needed to fit the extrema of this [`Series`].
    /// This can be used to reduce memory pressure.
    pub fn shrink_dtype(self) -> Self {
        self.map_private(FunctionExpr::ShrinkType)
    }

    /// Check if all boolean values are `true`
    pub fn all(self) -> Self {
        self.apply(
            move |s| {
                let boolean = s.bool()?;
                if boolean.all() {
                    Ok(Series::new(s.name(), [true]))
                } else {
                    Ok(Series::new(s.name(), [false]))
                }
            },
            GetOutput::from_type(DataType::Boolean),
        )
        .with_function_options(|mut opt| {
            opt.fmt_str = "all";
            opt.auto_explode = true;
            opt
        })
    }

    #[cfg(feature = "dtype-struct")]
    #[cfg_attr(docsrs, doc(cfg(feature = "dtype-struct")))]
    /// Count all unique values and create a struct mapping value to count
    /// Note that it is better to turn multithreaded off in the aggregation context
    pub fn value_counts(self, multithreaded: bool, sorted: bool) -> Self {
        self.apply(
            move |s| {
                s.value_counts(multithreaded, sorted)
                    .map(|df| df.into_struct(s.name()).into_series())
            },
            GetOutput::map_field(|fld| {
                Field::new(
                    fld.name(),
                    DataType::Struct(vec![fld.clone(), Field::new("counts", IDX_DTYPE)]),
                )
            }),
        )
        .with_fmt("value_counts")
    }

    #[cfg(feature = "unique_counts")]
    #[cfg_attr(docsrs, doc(cfg(feature = "unique_counts")))]
    /// Returns a count of the unique values in the order of appearance.
    /// This method differs from [`Expr::value_counts]` in that it does not return the
    /// values, only the counts and might be faster
    pub fn unique_counts(self) -> Self {
        self.apply(
            |s| Ok(s.unique_counts().into_series()),
            GetOutput::from_type(IDX_DTYPE),
        )
        .with_fmt("unique_counts")
    }

    #[cfg(feature = "log")]
    #[cfg_attr(docsrs, doc(cfg(feature = "log")))]
    /// Compute the logarithm to a given base
    pub fn log(self, base: f64) -> Self {
        self.map(
            move |s| Ok(s.log(base)),
            GetOutput::map_dtype(|dt| {
                if matches!(dt, DataType::Float32) {
                    DataType::Float32
                } else {
                    DataType::Float64
                }
            }),
        )
        .with_fmt("log")
    }

    #[cfg(feature = "log")]
    #[cfg_attr(docsrs, doc(cfg(feature = "log")))]
    /// Calculate the exponential of all elements in the input array
    pub fn exp(self) -> Self {
        self.map(
            move |s| Ok(s.exp()),
            GetOutput::map_dtype(|dt| {
                if matches!(dt, DataType::Float32) {
                    DataType::Float32
                } else {
                    DataType::Float64
                }
            }),
        )
        .with_fmt("exp")
    }

    #[cfg(feature = "log")]
    #[cfg_attr(docsrs, doc(cfg(feature = "log")))]
    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    pub fn entropy(self, base: f64, normalize: bool) -> Self {
        self.apply(
            move |s| Ok(Series::new(s.name(), [s.entropy(base, normalize)])),
            GetOutput::map_dtype(|dt| {
                if matches!(dt, DataType::Float32) {
                    DataType::Float32
                } else {
                    DataType::Float64
                }
            }),
        )
        .with_function_options(|mut options| {
            options.fmt_str = "entropy";
            options.auto_explode = true;
            options
        })
    }
    /// Get the null count of the column/group
    pub fn null_count(self) -> Expr {
        self.apply_private(FunctionExpr::NullCount)
            .with_function_options(|mut options| {
                options.auto_explode = true;
                options
            })
    }

    /// Set this `Series` as `sorted` so that downstream code can use
    /// fast paths for sorted arrays.
    /// # Warning
    /// This can lead to incorrect results if this `Series` is not sorted!!
    /// Use with care!
    pub fn set_sorted(self, sorted: IsSorted) -> Expr {
        self.apply(
            move |mut s| {
                s.set_sorted(sorted);
                Ok(s)
            },
            GetOutput::same_type(),
        )
    }

    #[cfg(feature = "row_hash")]
    /// Compute the hash of every element
    pub fn hash(self, k0: u64, k1: u64, k2: u64, k3: u64) -> Expr {
        self.map_private(FunctionExpr::Hash(k0, k1, k2, k3))
    }

    #[cfg(feature = "strings")]
    pub fn str(self) -> string::StringNameSpace {
        string::StringNameSpace(self)
    }

    #[cfg(feature = "temporal")]
    pub fn dt(self) -> dt::DateLikeNameSpace {
        dt::DateLikeNameSpace(self)
    }
    pub fn arr(self) -> list::ListNameSpace {
        list::ListNameSpace(self)
    }
    #[cfg(feature = "dtype-categorical")]
    pub fn cat(self) -> cat::CategoricalNameSpace {
        cat::CategoricalNameSpace(self)
    }
    #[cfg(feature = "dtype-struct")]
    pub fn struct_(self) -> struct_::StructNameSpace {
        struct_::StructNameSpace(self)
    }
    #[cfg(feature = "meta")]
    pub fn meta(self) -> meta::MetaNameSpace {
        meta::MetaNameSpace(self)
    }
}

// Arithmetic ops
impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Plus, rhs)
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Minus, rhs)
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Divide, rhs)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Multiply, rhs)
    }
}

impl Rem for Expr {
    type Output = Expr;

    fn rem(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Modulus, rhs)
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
    F: Fn(&mut [Series]) -> PolarsResult<Series> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: SpecialEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
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
    F: Fn(&mut [Series]) -> PolarsResult<Series> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: SpecialEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyList,
            auto_explode: true,
            fmt_str: "",
            ..Default::default()
        },
    }
}

/// Apply a function/closure over the groups of multiple columns. This should only be used in a groupby aggregation.
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
    F: Fn(&mut [Series]) -> PolarsResult<Series> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: SpecialEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            // don't set this to true
            // this is for the caller to decide
            auto_explode: returns_scalar,
            fmt_str: "",
            ..Default::default()
        },
    }
}

/// Count expression
pub fn count() -> Expr {
    Expr::Count
}

/// First column in DataFrame
pub fn first() -> Expr {
    Expr::Nth(0)
}

/// Last column in DataFrame
pub fn last() -> Expr {
    Expr::Nth(-1)
}
