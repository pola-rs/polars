#![allow(ambiguous_glob_reexports)]
//! Domain specific language for the Lazy API.
#[cfg(feature = "dtype-categorical")]
pub mod cat;

#[cfg(feature = "dtype-categorical")]
pub use cat::*;
#[cfg(feature = "rolling_window_by")]
pub(crate) use polars_time::prelude::*;

mod arithmetic;
mod arity;
#[cfg(feature = "dtype-array")]
mod array;
pub mod binary;
#[cfg(feature = "bitwise")]
mod bitwise;
mod builder_dsl;
pub use builder_dsl::*;
#[cfg(feature = "temporal")]
pub mod dt;
mod expr;
mod format;
mod from;
pub mod function_expr;
pub mod functions;
mod list;
mod match_to_schema;
#[cfg(feature = "meta")]
mod meta;
mod name;
mod options;
#[cfg(feature = "python")]
pub mod python_dsl;
#[cfg(feature = "random")]
mod random;
mod scan_sources;
mod selector;
mod statistics;
#[cfg(feature = "strings")]
pub mod string;
#[cfg(feature = "dtype-struct")]
mod struct_;
pub mod udf;

use std::fmt::Debug;
use std::sync::Arc;

mod plan;
pub use arity::*;
#[cfg(feature = "dtype-array")]
pub use array::*;
pub use expr::*;
pub use function_expr::schema::FieldsMapper;
pub use function_expr::*;
pub use functions::*;
pub use list::*;
pub use match_to_schema::*;
#[cfg(feature = "meta")]
pub use meta::*;
pub use name::*;
pub use options::*;
pub use plan::*;
use polars_compute::rolling::QuantileMethod;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
#[cfg(any(
    feature = "search_sorted",
    feature = "is_between",
    feature = "list_sets"
))]
use polars_core::utils::SuperTypeFlags;
use polars_core::utils::{SuperTypeOptions, try_get_supertype};
pub use selector::Selector;
#[cfg(feature = "dtype-struct")]
pub use struct_::*;
pub use udf::UserDefinedFunction;
mod file_scan;
pub use file_scan::*;
pub use scan_sources::{ScanSource, ScanSourceIter, ScanSourceRef, ScanSources};

pub use crate::plans::lit;
use crate::prelude::*;

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
        self.map_unary(BooleanFunction::Not)
    }

    /// Rename Column.
    pub fn alias<S>(self, name: S) -> Expr
    where
        S: Into<PlSmallStr>,
    {
        Expr::Alias(Arc::new(self), name.into())
    }

    /// Run is_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_null(self) -> Self {
        self.map_unary(BooleanFunction::IsNull)
    }

    /// Run is_not_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_null(self) -> Self {
        self.map_unary(BooleanFunction::IsNotNull)
    }

    /// Drop null values.
    pub fn drop_nulls(self) -> Self {
        self.map_unary(FunctionExpr::DropNulls)
    }

    /// Drop NaN values.
    pub fn drop_nans(self) -> Self {
        self.map_unary(FunctionExpr::DropNans)
    }

    /// Get the number of unique values in the groups.
    pub fn n_unique(self) -> Self {
        AggExpr::NUnique(Arc::new(self)).into()
    }

    /// Get the first value in the group.
    pub fn first(self) -> Self {
        AggExpr::First(Arc::new(self)).into()
    }

    /// Get the last value in the group.
    pub fn last(self) -> Self {
        AggExpr::Last(Arc::new(self)).into()
    }

    /// GroupBy the group to a Series.
    pub fn implode(self) -> Self {
        AggExpr::Implode(Arc::new(self)).into()
    }

    /// Compute the quantile per group.
    pub fn quantile(self, quantile: Expr, method: QuantileMethod) -> Self {
        AggExpr::Quantile {
            expr: Arc::new(self),
            quantile: Arc::new(quantile),
            method,
        }
        .into()
    }

    /// Get the group indexes of the group by operation.
    pub fn agg_groups(self) -> Self {
        AggExpr::AggGroups(Arc::new(self)).into()
    }

    /// Alias for `explode`.
    pub fn flatten(self) -> Self {
        self.explode()
    }

    /// Explode the String/List column.
    pub fn explode(self) -> Self {
        Expr::Explode {
            input: Arc::new(self),
            skip_empty: false,
        }
    }

    /// Slice the Series.
    /// `offset` may be negative.
    pub fn slice<E: Into<Expr>, F: Into<Expr>>(self, offset: E, length: F) -> Self {
        Expr::Slice {
            input: Arc::new(self),
            offset: Arc::new(offset.into()),
            length: Arc::new(length.into()),
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
        self.map_unary(FunctionExpr::Unique(false))
    }

    /// Get unique values of this expression, while maintaining order.
    /// This requires more work than [`Expr::unique`].
    pub fn unique_stable(self) -> Self {
        self.map_unary(FunctionExpr::Unique(true))
    }

    /// Get the first index of unique values of this expression.
    pub fn arg_unique(self) -> Self {
        self.map_unary(FunctionExpr::ArgUnique)
    }

    /// Get the index value that has the minimum value.
    pub fn arg_min(self) -> Self {
        let options = FunctionOptions::aggregation().with_fmt_str("arg_min");
        self.function_with_options(
            move |c: Column| {
                Ok(Some(Column::new(
                    c.name().clone(),
                    &[c.as_materialized_series().arg_min().map(|idx| idx as u32)],
                )))
            },
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index value that has the maximum value.
    pub fn arg_max(self) -> Self {
        let options = FunctionOptions::aggregation().with_fmt_str("arg_max");
        self.function_with_options(
            move |c: Column| {
                Ok(Some(Column::new(
                    c.name().clone(),
                    &[c.as_materialized_series()
                        .arg_max()
                        .map(|idx| idx as IdxSize)],
                )))
            },
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index values that would sort this expression.
    pub fn arg_sort(self, sort_options: SortOptions) -> Self {
        let options = FunctionOptions::groupwise().with_fmt_str("arg_sort");
        self.function_with_options(
            move |c: Column| {
                Ok(Some(
                    c.as_materialized_series()
                        .arg_sort(sort_options)
                        .into_column(),
                ))
            },
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    #[cfg(feature = "index_of")]
    /// Find the index of a value.
    pub fn index_of<E: Into<Expr>>(self, element: E) -> Expr {
        self.map_binary(FunctionExpr::IndexOf, element.into())
    }

    #[cfg(feature = "search_sorted")]
    /// Find indices where elements should be inserted to maintain order.
    pub fn search_sorted<E: Into<Expr>>(self, element: E, side: SearchSortedSide) -> Expr {
        self.map_binary(FunctionExpr::SearchSorted(side), element.into())
    }

    /// Cast expression to another data type.
    /// Throws an error if conversion had overflows.
    /// Returns an Error if cast is invalid on rows after predicates are pushed down.
    pub fn strict_cast(self, dtype: DataType) -> Self {
        Expr::Cast {
            expr: Arc::new(self),
            dtype,
            options: CastOptions::Strict,
        }
    }

    /// Cast expression to another data type.
    pub fn cast(self, dtype: DataType) -> Self {
        Expr::Cast {
            expr: Arc::new(self),
            dtype,
            options: CastOptions::NonStrict,
        }
    }

    /// Cast expression to another data type.
    pub fn cast_with_options(self, dtype: DataType, cast_options: CastOptions) -> Self {
        Expr::Cast {
            expr: Arc::new(self),
            dtype,
            options: cast_options,
        }
    }

    /// Take the values by idx.
    pub fn gather<E: Into<Expr>>(self, idx: E) -> Self {
        Expr::Gather {
            expr: Arc::new(self),
            idx: Arc::new(idx.into()),
            returns_scalar: false,
        }
    }

    /// Take the values by a single index.
    pub fn get<E: Into<Expr>>(self, idx: E) -> Self {
        Expr::Gather {
            expr: Arc::new(self),
            idx: Arc::new(idx.into()),
            returns_scalar: true,
        }
    }

    /// Sort with given options.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// # fn main() -> PolarsResult<()> {
    /// let lf = df! {
    ///    "a" => [Some(5), Some(4), Some(3), Some(2), None]
    /// }?
    /// .lazy();
    ///
    /// let sorted = lf
    ///     .select(
    ///         vec![col("a").sort(SortOptions::default())],
    ///     )
    ///     .collect()?;
    ///
    /// assert_eq!(
    ///     sorted,
    ///     df! {
    ///         "a" => [None, Some(2), Some(3), Some(4), Some(5)]
    ///     }?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    /// See [`SortOptions`] for more options.
    pub fn sort(self, options: SortOptions) -> Self {
        Expr::Sort {
            expr: Arc::new(self),
            options,
        }
    }

    /// Returns the `k` largest elements.
    ///
    /// This has time complexity `O(n + k log(n))`.
    #[cfg(feature = "top_k")]
    pub fn top_k(self, k: Expr) -> Self {
        self.map_binary(FunctionExpr::TopK { descending: false }, k)
    }

    /// Returns the `k` largest rows by given column.
    ///
    /// For single column, use [`Expr::top_k`].
    #[cfg(feature = "top_k")]
    pub fn top_k_by<K: Into<Expr>, E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
        self,
        k: K,
        by: E,
        descending: Vec<bool>,
    ) -> Self {
        self.map_n_ary(
            FunctionExpr::TopKBy { descending },
            [k.into()]
                .into_iter()
                .chain(by.as_ref().iter().map(|e| -> Expr { e.clone().into() })),
        )
    }

    /// Returns the `k` smallest elements.
    ///
    /// This has time complexity `O(n + k log(n))`.
    #[cfg(feature = "top_k")]
    pub fn bottom_k(self, k: Expr) -> Self {
        self.map_binary(FunctionExpr::TopK { descending: true }, k)
    }

    /// Returns the `k` smallest rows by given column.
    ///
    /// For single column, use [`Expr::bottom_k`].
    // #[cfg(feature = "top_k")]
    #[cfg(feature = "top_k")]
    pub fn bottom_k_by<K: Into<Expr>, E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
        self,
        k: K,
        by: E,
        descending: Vec<bool>,
    ) -> Self {
        let descending = descending.into_iter().map(|x| !x).collect();
        self.map_n_ary(
            FunctionExpr::TopKBy { descending },
            [k.into()]
                .into_iter()
                .chain(by.as_ref().iter().map(|e| -> Expr { e.clone().into() })),
        )
    }

    /// Reverse column
    pub fn reverse(self) -> Self {
        self.map_unary(FunctionExpr::Reverse)
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
        F: Fn(Column) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    {
        let f = move |c: &mut [Column]| function(std::mem::take(&mut c[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: new_column_udf(f),
            output_type,
            options: FunctionOptions::elementwise()
                .with_fmt_str("map")
                .with_flags(|f| f | FunctionFlags::OPTIONAL_RE_ENTRANT),
        }
    }

    /// Apply a function/closure once the logical plan get executed with many arguments.
    ///
    /// See the [`Expr::map`] function for the differences between [`map`](Expr::map) and [`apply`](Expr::apply).
    pub fn map_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Column]) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::AnonymousFunction {
            input,
            function: new_column_udf(function),
            output_type,
            options: FunctionOptions::elementwise().with_fmt_str(""),
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
        F: Fn(Column) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    {
        let f = move |c: &mut [Column]| function(std::mem::take(&mut c[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: new_column_udf(f),
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
        F: Fn(Column) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    {
        let f = move |c: &mut [Column]| function(std::mem::take(&mut c[0]));

        Expr::AnonymousFunction {
            input: vec![self],
            function: new_column_udf(f),
            output_type,
            options: FunctionOptions::groupwise().with_fmt_str(""),
        }
    }

    /// Apply a function/closure over the groups with many arguments. This should only be used in a group_by aggregation.
    ///
    /// See the [`Expr::apply`] function for the differences between [`map`](Expr::map) and [`apply`](Expr::apply).
    pub fn apply_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Column]) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::AnonymousFunction {
            input,
            function: new_column_udf(function),
            output_type,
            options: FunctionOptions::groupwise().with_fmt_str(""),
        }
    }

    /// Get mask of finite values if dtype is Float.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_finite(self) -> Self {
        self.map_unary(BooleanFunction::IsFinite)
    }

    /// Get mask of infinite values if dtype is Float.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_infinite(self) -> Self {
        self.map_unary(BooleanFunction::IsInfinite)
    }

    /// Get mask of NaN values if dtype is Float.
    pub fn is_nan(self) -> Self {
        self.map_unary(BooleanFunction::IsNan)
    }

    /// Get inverse mask of NaN values if dtype is Float.
    pub fn is_not_nan(self) -> Self {
        self.map_unary(BooleanFunction::IsNotNan)
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::shift).
    pub fn shift(self, n: Expr) -> Self {
        self.map_binary(FunctionExpr::Shift, n)
    }

    /// Shift the values in the array by some period and fill the resulting empty values.
    pub fn shift_and_fill<E: Into<Expr>, IE: Into<Expr>>(self, n: E, fill_value: IE) -> Self {
        self.map_ternary(FunctionExpr::ShiftAndFill, n.into(), fill_value.into())
    }

    /// Cumulatively count values from 0 to len.
    #[cfg(feature = "cum_agg")]
    pub fn cum_count(self, reverse: bool) -> Self {
        self.map_unary(FunctionExpr::CumCount { reverse })
    }

    /// Get an array with the cumulative sum computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cum_sum(self, reverse: bool) -> Self {
        self.map_unary(FunctionExpr::CumSum { reverse })
    }

    /// Get an array with the cumulative product computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cum_prod(self, reverse: bool) -> Self {
        self.map_unary(FunctionExpr::CumProd { reverse })
    }

    /// Get an array with the cumulative min computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cum_min(self, reverse: bool) -> Self {
        self.map_unary(FunctionExpr::CumMin { reverse })
    }

    /// Get an array with the cumulative max computed at every element.
    #[cfg(feature = "cum_agg")]
    pub fn cum_max(self, reverse: bool) -> Self {
        self.map_unary(FunctionExpr::CumMax { reverse })
    }

    /// Get the product aggregation of an expression.
    pub fn product(self) -> Self {
        let options = FunctionOptions::aggregation().with_fmt_str("product");
        self.function_with_options(
            move |c: Column| {
                Some(
                    c.product()
                        .map(|sc| sc.into_series(c.name().clone()).into_column()),
                )
                .transpose()
            },
            GetOutput::map_dtype(|dt| {
                use DataType as T;
                Ok(match dt {
                    T::Float32 => T::Float32,
                    T::Float64 => T::Float64,
                    T::UInt64 => T::UInt64,
                    #[cfg(feature = "dtype-i128")]
                    T::Int128 => T::Int128,
                    _ => T::Int64,
                })
            }),
            options,
        )
    }

    /// Round underlying floating point array to given decimal numbers.
    #[cfg(feature = "round_series")]
    pub fn round(self, decimals: u32, mode: RoundMode) -> Self {
        self.map_unary(FunctionExpr::Round { decimals, mode })
    }

    /// Round to a number of significant figures.
    #[cfg(feature = "round_series")]
    pub fn round_sig_figs(self, digits: i32) -> Self {
        self.map_unary(FunctionExpr::RoundSF { digits })
    }

    /// Floor underlying floating point array to the lowest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    pub fn floor(self) -> Self {
        self.map_unary(FunctionExpr::Floor)
    }

    /// Constant Pi
    #[cfg(feature = "round_series")]
    pub fn pi() -> Self {
        lit(std::f64::consts::PI)
    }

    /// Ceil underlying floating point array to the highest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    pub fn ceil(self) -> Self {
        self.map_unary(FunctionExpr::Ceil)
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    pub fn clip(self, min: Expr, max: Expr) -> Self {
        self.map_ternary(
            FunctionExpr::Clip {
                has_min: true,
                has_max: true,
            },
            min,
            max,
        )
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    pub fn clip_max(self, max: Expr) -> Self {
        self.map_binary(
            FunctionExpr::Clip {
                has_min: false,
                has_max: true,
            },
            max,
        )
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    pub fn clip_min(self, min: Expr) -> Self {
        self.map_binary(
            FunctionExpr::Clip {
                has_min: true,
                has_max: false,
            },
            min,
        )
    }

    /// Convert all values to their absolute/positive value.
    #[cfg(feature = "abs")]
    pub fn abs(self) -> Self {
        self.map_unary(FunctionExpr::Abs)
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
        self.over_with_options(Some(partition_by), None, Default::default())
            .expect("We explicitly passed `partition_by`")
    }

    pub fn over_with_options<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
        self,
        partition_by: Option<E>,
        order_by: Option<(E, SortOptions)>,
        options: WindowMapping,
    ) -> PolarsResult<Self> {
        polars_ensure!(partition_by.is_some() || order_by.is_some(), InvalidOperation: "At least one of `partition_by` and `order_by` must be specified in `over`");
        let partition_by = if let Some(partition_by) = partition_by {
            partition_by
                .as_ref()
                .iter()
                .map(|e| e.clone().into())
                .collect()
        } else {
            vec![lit(1)]
        };

        let order_by = order_by.map(|(e, options)| {
            let e = e.as_ref();
            let e = if e.len() == 1 {
                Arc::new(e[0].clone().into())
            } else {
                feature_gated!["dtype-struct", {
                    let e = e.iter().map(|e| e.clone().into()).collect::<Vec<_>>();
                    Arc::new(as_struct(e))
                }]
            };
            (e, options)
        });

        Ok(Expr::Window {
            function: Arc::new(self),
            partition_by,
            order_by,
            options: options.into(),
        })
    }

    #[cfg(feature = "dynamic_group_by")]
    pub fn rolling(self, options: RollingGroupOptions) -> Self {
        // We add the index column as `partition expr` so that the optimizer will
        // not ignore it.
        let index_col = col(options.index_column.clone());
        Expr::Window {
            function: Arc::new(self),
            partition_by: vec![index_col],
            order_by: None,
            options: WindowType::Rolling(options),
        }
    }

    fn fill_null_impl(self, fill_value: Expr) -> Self {
        self.map_binary(FunctionExpr::FillNull, fill_value)
    }

    /// Replace the null values by a value.
    pub fn fill_null<E: Into<Expr>>(self, fill_value: E) -> Self {
        self.fill_null_impl(fill_value.into())
    }

    pub fn fill_null_with_strategy(self, strategy: FillNullStrategy) -> Self {
        self.map_unary(FunctionExpr::FillNullWithStrategy(strategy))
    }

    /// Replace the floating point `NaN` values by a value.
    pub fn fill_nan<E: Into<Expr>>(self, fill_value: E) -> Self {
        // we take the not branch so that self is truthy value of `when -> then -> otherwise`
        // and that ensure we keep the name of `self`

        when(self.clone().is_not_nan().or(self.clone().is_null()))
            .then(self)
            .otherwise(fill_value.into())
    }
    /// Count the values of the Series
    /// or
    /// Get counts of the group by operation.
    pub fn count(self) -> Self {
        AggExpr::Count(Arc::new(self), false).into()
    }

    pub fn len(self) -> Self {
        AggExpr::Count(Arc::new(self), true).into()
    }

    /// Get a mask of duplicated values.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_unique")]
    pub fn is_duplicated(self) -> Self {
        self.map_unary(BooleanFunction::IsDuplicated)
    }

    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_between")]
    pub fn is_between<E: Into<Expr>>(self, lower: E, upper: E, closed: ClosedInterval) -> Self {
        self.map_ternary(
            BooleanFunction::IsBetween { closed },
            lower.into(),
            upper.into(),
        )
    }

    /// Get a mask of unique values.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_unique")]
    pub fn is_unique(self) -> Self {
        self.map_unary(BooleanFunction::IsUnique)
    }

    /// Get the approximate count of unique values.
    #[cfg(feature = "approx_unique")]
    pub fn approx_n_unique(self) -> Self {
        self.map_unary(FunctionExpr::ApproxNUnique)
    }

    /// Bitwise "and" operation.
    pub fn and<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::And, expr.into())
    }

    /// Bitwise "xor" operation.
    pub fn xor<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::Xor, expr.into())
    }

    /// Bitwise "or" operation.
    pub fn or<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::Or, expr.into())
    }

    /// Logical "or" operation.
    pub fn logical_or<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::LogicalOr, expr.into())
    }

    /// Logical "and" operation.
    pub fn logical_and<E: Into<Expr>>(self, expr: E) -> Self {
        binary_expr(self, Operator::LogicalAnd, expr.into())
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
            input: Arc::new(self),
            by: Arc::new(predicate.into()),
        }
    }

    /// Check if the values of the left expression are in the lists of the right expr.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_in")]
    pub fn is_in<E: Into<Expr>>(self, other: E, nulls_equal: bool) -> Self {
        let other = other.into();
        let function = BooleanFunction::IsIn { nulls_equal };
        let options = function.function_options();
        let function = function.into();
        Expr::Function {
            input: vec![self, other],
            function,
            options,
        }
    }

    /// Sort this column by the ordering of another column evaluated from given expr.
    /// Can also be used in a group_by context to sort the groups.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// # fn main() -> PolarsResult<()> {
    /// let lf = df! {
    ///     "a" => [1, 2, 3, 4, 5],
    ///     "b" => [5, 4, 3, 2, 1]
    /// }?.lazy();
    ///
    /// let sorted = lf
    ///     .select(
    ///         vec![col("a").sort_by(col("b"), SortOptions::default())],
    ///     )
    ///     .collect()?;
    ///
    /// assert_eq!(
    ///     sorted,
    ///     df! { "a" => [5, 4, 3, 2, 1] }?
    /// );
    /// # Ok(())
    /// # }
    pub fn sort_by<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
        self,
        by: E,
        sort_options: SortMultipleOptions,
    ) -> Expr {
        let by = by.as_ref().iter().map(|e| e.clone().into()).collect();
        Expr::SortBy {
            expr: Arc::new(self),
            by,
            sort_options,
        }
    }

    #[cfg(feature = "repeat_by")]
    /// Repeat the column `n` times, where `n` is determined by the values in `by`.
    /// This yields an `Expr` of dtype `List`.
    pub fn repeat_by<E: Into<Expr>>(self, by: E) -> Expr {
        self.map_binary(FunctionExpr::RepeatBy, by.into())
    }

    #[cfg(feature = "is_first_distinct")]
    #[allow(clippy::wrong_self_convention)]
    /// Get a mask of the first unique value.
    pub fn is_first_distinct(self) -> Expr {
        self.map_unary(BooleanFunction::IsFirstDistinct)
    }

    #[cfg(feature = "is_last_distinct")]
    #[allow(clippy::wrong_self_convention)]
    /// Get a mask of the last unique value.
    pub fn is_last_distinct(self) -> Expr {
        self.map_unary(BooleanFunction::IsLastDistinct)
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
        self.map_unary(FunctionExpr::Mode)
    }

    /// Exclude a column from a wildcard/regex selection.
    ///
    /// You may also use regexes in the exclude as long as they start with `^` and end with `$`.
    pub fn exclude(self, columns: impl IntoVec<PlSmallStr>) -> Expr {
        let v = columns.into_vec().into_iter().map(Excluded::Name).collect();
        Expr::Exclude(Arc::new(self), v)
    }

    pub fn exclude_dtype<D: AsRef<[DataType]>>(self, dtypes: D) -> Expr {
        let v = dtypes
            .as_ref()
            .iter()
            .map(|dt| Excluded::Dtype(dt.clone()))
            .collect();
        Expr::Exclude(Arc::new(self), v)
    }

    #[cfg(feature = "interpolate")]
    /// Interpolate intermediate values.
    /// Nulls at the beginning and end of the series remain null.
    pub fn interpolate(self, method: InterpolationMethod) -> Expr {
        self.map_unary(FunctionExpr::Interpolate(method))
    }

    #[cfg(feature = "rolling_window_by")]
    #[allow(clippy::type_complexity)]
    fn finish_rolling_by(
        self,
        by: Expr,
        options: RollingOptionsDynamicWindow,
        rolling_function_by: fn(RollingOptionsDynamicWindow) -> RollingFunctionBy,
    ) -> Expr {
        self.map_binary(
            FunctionExpr::RollingExprBy(rolling_function_by(options)),
            by,
        )
    }

    #[cfg(feature = "interpolate_by")]
    /// Interpolate intermediate values.
    /// Nulls at the beginning and end of the series remain null.
    /// The `by` column provides the x-coordinates for interpolation and must not contain nulls.
    pub fn interpolate_by(self, by: Expr) -> Expr {
        self.map_binary(FunctionExpr::InterpolateBy, by)
    }

    #[cfg(feature = "rolling_window")]
    #[allow(clippy::type_complexity)]
    fn finish_rolling(
        self,
        options: RollingOptionsFixedWindow,
        rolling_function: fn(RollingOptionsFixedWindow) -> RollingFunction,
    ) -> Expr {
        self.map_unary(FunctionExpr::RollingExpr(rolling_function(options)))
    }

    /// Apply a rolling minimum based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_min_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.finish_rolling_by(by, options, RollingFunctionBy::MinBy)
    }

    /// Apply a rolling maximum based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_max_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.finish_rolling_by(by, options, RollingFunctionBy::MaxBy)
    }

    /// Apply a rolling mean based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_mean_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.finish_rolling_by(by, options, RollingFunctionBy::MeanBy)
    }

    /// Apply a rolling sum based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_sum_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.finish_rolling_by(by, options, RollingFunctionBy::SumBy)
    }

    /// Apply a rolling quantile based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_quantile_by(
        self,
        by: Expr,
        method: QuantileMethod,
        quantile: f64,
        mut options: RollingOptionsDynamicWindow,
    ) -> Expr {
        use polars_compute::rolling::{RollingFnParams, RollingQuantileParams};
        options.fn_params = Some(RollingFnParams::Quantile(RollingQuantileParams {
            prob: quantile,
            method,
        }));

        self.finish_rolling_by(by, options, RollingFunctionBy::QuantileBy)
    }

    /// Apply a rolling variance based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_var_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.finish_rolling_by(by, options, RollingFunctionBy::VarBy)
    }

    /// Apply a rolling std-dev based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_std_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.finish_rolling_by(by, options, RollingFunctionBy::StdBy)
    }

    /// Apply a rolling median based on another column.
    #[cfg(feature = "rolling_window_by")]
    pub fn rolling_median_by(self, by: Expr, options: RollingOptionsDynamicWindow) -> Expr {
        self.rolling_quantile_by(by, QuantileMethod::Linear, 0.5, options)
    }

    /// Apply a rolling minimum.
    ///
    /// See: [`RollingAgg::rolling_min`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_min(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Min)
    }

    /// Apply a rolling maximum.
    ///
    /// See: [`RollingAgg::rolling_max`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_max(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Max)
    }

    /// Apply a rolling mean.
    ///
    /// See: [`RollingAgg::rolling_mean`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_mean(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Mean)
    }

    /// Apply a rolling sum.
    ///
    /// See: [`RollingAgg::rolling_sum`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_sum(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Sum)
    }

    /// Apply a rolling median.
    ///
    /// See: [`RollingAgg::rolling_median`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_median(self, options: RollingOptionsFixedWindow) -> Expr {
        self.rolling_quantile(QuantileMethod::Linear, 0.5, options)
    }

    /// Apply a rolling quantile.
    ///
    /// See: [`RollingAgg::rolling_quantile`]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_quantile(
        self,
        method: QuantileMethod,
        quantile: f64,
        mut options: RollingOptionsFixedWindow,
    ) -> Expr {
        use polars_compute::rolling::{RollingFnParams, RollingQuantileParams};

        options.fn_params = Some(RollingFnParams::Quantile(RollingQuantileParams {
            prob: quantile,
            method,
        }));

        self.finish_rolling(options, RollingFunction::Quantile)
    }

    /// Apply a rolling variance.
    #[cfg(feature = "rolling_window")]
    pub fn rolling_var(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Var)
    }

    /// Apply a rolling std-dev.
    #[cfg(feature = "rolling_window")]
    pub fn rolling_std(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Std)
    }

    /// Apply a rolling skew.
    #[cfg(feature = "rolling_window")]
    #[cfg(feature = "moment")]
    pub fn rolling_skew(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Skew)
    }

    /// Apply a rolling skew.
    #[cfg(feature = "rolling_window")]
    #[cfg(feature = "moment")]
    pub fn rolling_kurtosis(self, options: RollingOptionsFixedWindow) -> Expr {
        self.finish_rolling(options, RollingFunction::Kurtosis)
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
            move |c: Column| {
                c.as_materialized_series()
                    .rolling_map(f.as_ref(), options.clone())
                    .map(Column::from)
                    .map(Some)
            },
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
            move |c: Column| {
                let out = match c.dtype() {
                    DataType::Float64 => c
                        .f64()
                        .unwrap()
                        .rolling_map_float(window_size, f)
                        .map(|ca| ca.into_column()),
                    _ => c
                        .cast(&DataType::Float64)?
                        .f64()
                        .unwrap()
                        .rolling_map_float(window_size, f)
                        .map(|ca| ca.into_column()),
                }?;
                if let DataType::Float32 = c.dtype() {
                    out.cast(&DataType::Float32).map(Some)
                } else {
                    Ok(Some(out))
                }
            },
            GetOutput::map_field(|field| {
                Ok(match field.dtype() {
                    DataType::Float64 => field.clone(),
                    DataType::Float32 => Field::new(field.name().clone(), DataType::Float32),
                    _ => Field::new(field.name().clone(), DataType::Float64),
                })
            }),
        )
        .with_fmt("rolling_map_float")
    }

    #[cfg(feature = "peaks")]
    pub fn peak_min(self) -> Expr {
        self.map_unary(FunctionExpr::PeakMin)
    }

    #[cfg(feature = "peaks")]
    pub fn peak_max(self) -> Expr {
        self.map_unary(FunctionExpr::PeakMax)
    }

    #[cfg(feature = "rank")]
    /// Assign ranks to data, dealing with ties appropriately.
    pub fn rank(self, options: RankOptions, seed: Option<u64>) -> Expr {
        self.map_unary(FunctionExpr::Rank { options, seed })
    }

    #[cfg(feature = "replace")]
    /// Replace the given values with other values.
    pub fn replace<E: Into<Expr>>(self, old: E, new: E) -> Expr {
        let old = old.into();
        let new = new.into();
        self.map_n_ary(FunctionExpr::Replace, [old, new])
    }

    #[cfg(feature = "replace")]
    /// Replace the given values with other values.
    pub fn replace_strict<E: Into<Expr>>(
        self,
        old: E,
        new: E,
        default: Option<E>,
        return_dtype: Option<DataType>,
    ) -> Expr {
        let old = old.into();
        let new = new.into();
        let mut args = vec![old, new];
        args.extend(default.map(Into::into));
        self.map_n_ary(FunctionExpr::ReplaceStrict { return_dtype }, args)
    }

    #[cfg(feature = "cutqcut")]
    /// Bin continuous values into discrete categories.
    pub fn cut(
        self,
        breaks: Vec<f64>,
        labels: Option<impl IntoVec<PlSmallStr>>,
        left_closed: bool,
        include_breaks: bool,
    ) -> Expr {
        self.map_unary(FunctionExpr::Cut {
            breaks,
            labels: labels.map(|x| x.into_vec()),
            left_closed,
            include_breaks,
        })
    }

    #[cfg(feature = "cutqcut")]
    /// Bin continuous values into discrete categories based on their quantiles.
    pub fn qcut(
        self,
        probs: Vec<f64>,
        labels: Option<impl IntoVec<PlSmallStr>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    ) -> Expr {
        self.map_unary(FunctionExpr::QCut {
            probs,
            labels: labels.map(|x| x.into_vec()),
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
        labels: Option<impl IntoVec<PlSmallStr>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    ) -> Expr {
        let probs = (1..n_bins).map(|b| b as f64 / n_bins as f64).collect();
        self.map_unary(FunctionExpr::QCut {
            probs,
            labels: labels.map(|x| x.into_vec()),
            left_closed,
            allow_duplicates,
            include_breaks,
        })
    }

    #[cfg(feature = "rle")]
    /// Get the lengths of runs of identical values.
    pub fn rle(self) -> Expr {
        self.map_unary(FunctionExpr::RLE)
    }

    #[cfg(feature = "rle")]
    /// Similar to `rle`, but maps values to run IDs.
    pub fn rle_id(self) -> Expr {
        self.map_unary(FunctionExpr::RLEID)
    }

    #[cfg(feature = "diff")]
    /// Calculate the n-th discrete difference between values.
    pub fn diff(self, n: Expr, null_behavior: NullBehavior) -> Expr {
        self.map_binary(FunctionExpr::Diff(null_behavior), n)
    }

    #[cfg(feature = "pct_change")]
    /// Computes percentage change between values.
    pub fn pct_change(self, n: Expr) -> Expr {
        self.map_binary(FunctionExpr::PctChange, n)
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
        self.map_unary(FunctionExpr::Skew(bias))
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
        self.map_unary(FunctionExpr::Kurtosis(fisher, bias))
    }

    /// Get maximal value that could be hold by this dtype.
    pub fn upper_bound(self) -> Expr {
        self.map_unary(FunctionExpr::UpperBound)
    }

    /// Get minimal value that could be hold by this dtype.
    pub fn lower_bound(self) -> Expr {
        self.map_unary(FunctionExpr::LowerBound)
    }

    #[cfg(feature = "dtype-array")]
    pub fn reshape(self, dimensions: &[i64]) -> Self {
        let dimensions = dimensions
            .iter()
            .map(|&v| ReshapeDimension::new(v))
            .collect();
        self.map_unary(FunctionExpr::Reshape(dimensions))
    }

    #[cfg(feature = "ewma")]
    /// Calculate the exponentially-weighted moving average.
    pub fn ewm_mean(self, options: EWMOptions) -> Self {
        self.map_unary(FunctionExpr::EwmMean { options })
    }

    #[cfg(feature = "ewma_by")]
    /// Calculate the exponentially-weighted moving average by a time column.
    pub fn ewm_mean_by(self, times: Expr, half_life: Duration) -> Self {
        self.map_binary(FunctionExpr::EwmMeanBy { half_life }, times)
    }

    #[cfg(feature = "ewma")]
    /// Calculate the exponentially-weighted moving standard deviation.
    pub fn ewm_std(self, options: EWMOptions) -> Self {
        self.map_unary(FunctionExpr::EwmStd { options })
    }

    #[cfg(feature = "ewma")]
    /// Calculate the exponentially-weighted moving variance.
    pub fn ewm_var(self, options: EWMOptions) -> Self {
        self.map_unary(FunctionExpr::EwmVar { options })
    }

    /// Returns whether any of the values in the column are `true`.
    ///
    /// If `ignore_nulls` is `False`, [Kleene logic] is used to deal with nulls:
    /// if the column contains any null values and no `true` values, the output
    /// is null.
    ///
    /// [Kleene logic]: https://en.wikipedia.org/wiki/Three-valued_logic
    pub fn any(self, ignore_nulls: bool) -> Self {
        self.map_unary(BooleanFunction::Any { ignore_nulls })
    }

    /// Returns whether all values in the column are `true`.
    ///
    /// If `ignore_nulls` is `False`, [Kleene logic] is used to deal with nulls:
    /// if the column contains any null values and no `false` values, the output
    /// is null.
    ///
    /// [Kleene logic]: https://en.wikipedia.org/wiki/Three-valued_logic
    pub fn all(self, ignore_nulls: bool) -> Self {
        self.map_unary(BooleanFunction::All { ignore_nulls })
    }

    /// Shrink numeric columns to the minimal required datatype
    /// needed to fit the extrema of this [`Series`].
    /// This can be used to reduce memory pressure.
    pub fn shrink_dtype(self) -> Self {
        self.map_unary(FunctionExpr::ShrinkType)
    }

    #[cfg(feature = "dtype-struct")]
    /// Count all unique values and create a struct mapping value to count.
    /// (Note that it is better to turn parallel off in the aggregation context).
    /// The name of the struct field with the counts is given by the parameter `name`.
    pub fn value_counts(self, sort: bool, parallel: bool, name: &str, normalize: bool) -> Self {
        self.map_unary(FunctionExpr::ValueCounts {
            sort,
            parallel,
            name: name.into(),
            normalize,
        })
    }

    #[cfg(feature = "unique_counts")]
    /// Returns a count of the unique values in the order of appearance.
    /// This method differs from [`Expr::value_counts`] in that it does not return the
    /// values, only the counts and might be faster.
    pub fn unique_counts(self) -> Self {
        self.map_unary(FunctionExpr::UniqueCounts)
    }

    #[cfg(feature = "log")]
    /// Compute the logarithm to a given base.
    pub fn log(self, base: f64) -> Self {
        self.map_unary(FunctionExpr::Log { base })
    }

    #[cfg(feature = "log")]
    /// Compute the natural logarithm of all elements plus one in the input array.
    pub fn log1p(self) -> Self {
        self.map_unary(FunctionExpr::Log1p)
    }

    #[cfg(feature = "log")]
    /// Calculate the exponential of all elements in the input array.
    pub fn exp(self) -> Self {
        self.map_unary(FunctionExpr::Exp)
    }

    #[cfg(feature = "log")]
    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    pub fn entropy(self, base: f64, normalize: bool) -> Self {
        self.map_unary(FunctionExpr::Entropy { base, normalize })
    }
    /// Get the null count of the column/group.
    pub fn null_count(self) -> Expr {
        self.map_unary(FunctionExpr::NullCount)
    }

    /// Set this `Series` as `sorted` so that downstream code can use
    /// fast paths for sorted arrays.
    /// # Warning
    /// This can lead to incorrect results if this `Series` is not sorted!!
    /// Use with care!
    pub fn set_sorted_flag(self, sorted: IsSorted) -> Expr {
        // This is `map`. If a column is sorted. Chunks of that column are also sorted.
        self.map_unary(FunctionExpr::SetSortedFlag(sorted))
    }

    #[cfg(feature = "row_hash")]
    /// Compute the hash of every element.
    pub fn hash(self, k0: u64, k1: u64, k2: u64, k3: u64) -> Expr {
        self.map_unary(FunctionExpr::Hash(k0, k1, k2, k3))
    }

    pub fn to_physical(self) -> Expr {
        self.map_unary(FunctionExpr::ToPhysical)
    }

    pub fn gather_every(self, n: usize, offset: usize) -> Expr {
        self.map_unary(FunctionExpr::GatherEvery { n, offset })
    }

    #[cfg(feature = "reinterpret")]
    pub fn reinterpret(self, signed: bool) -> Expr {
        self.map_unary(FunctionExpr::Reinterpret(signed))
    }

    pub fn extend_constant(self, value: Expr, n: Expr) -> Expr {
        self.map_ternary(FunctionExpr::ExtendConstant, value, n)
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
/// This function is very similar to [`apply_multiple`], but differs in how it handles aggregations.
///
///  * [`map_multiple`] should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
///  * [`apply_multiple`] should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
///
/// It is the responsibility of the caller that the schema is correct by giving
/// the correct output_type. If None given the output type of the input expr is used.
pub fn map_multiple<F, E>(function: F, expr: E, output_type: GetOutput) -> Expr
where
    F: Fn(&mut [Column]) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::AnonymousFunction {
        input,
        function: new_column_udf(function),
        output_type,
        options: FunctionOptions::elementwise().with_fmt_str(""),
    }
}

/// Apply a function/closure over the groups of multiple columns. This should only be used in a group_by aggregation.
///
/// It is the responsibility of the caller that the schema is correct by giving
/// the correct output_type. If None given the output type of the input expr is used.
///
/// This difference with [`map_multiple`] is that [`apply_multiple`] will create a separate [`Series`] per group.
///
/// * [`map_multiple`] should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
/// * [`apply_multiple`] should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
pub fn apply_multiple<F, E>(
    function: F,
    expr: E,
    output_type: GetOutput,
    returns_scalar: bool,
) -> Expr
where
    F: Fn(&mut [Column]) -> PolarsResult<Option<Column>> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();
    Expr::AnonymousFunction {
        input,
        function: new_column_udf(function),
        output_type,
        options: FunctionOptions::groupwise()
            .with_fmt_str("")
            .with_flags(|mut f| {
                f.set(FunctionFlags::RETURNS_SCALAR, returns_scalar);
                f
            }),
    }
}

/// Return the number of rows in the context.
pub fn len() -> Expr {
    Expr::Len
}

/// First column in a DataFrame.
pub fn first() -> Expr {
    Expr::Nth(0)
}

/// Last column in a DataFrame.
pub fn last() -> Expr {
    Expr::Nth(-1)
}

/// Nth column in a DataFrame.
pub fn nth(n: i64) -> Expr {
    Expr::Nth(n)
}
