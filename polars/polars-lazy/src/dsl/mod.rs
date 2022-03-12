//! Domain specific language for the Lazy api.
#[cfg(feature = "dtype-categorical")]
pub mod cat;
#[cfg(feature = "dtype-categorical")]
pub use cat::*;
#[cfg(feature = "temporal")]
mod dt;
#[cfg(feature = "compile")]
mod functions;
#[cfg(feature = "list")]
mod list;
mod options;
#[cfg(feature = "strings")]
pub mod string;
#[cfg(feature = "dtype-struct")]
mod struct_;

use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::has_expr;

#[cfg(feature = "is_in")]
use crate::utils::has_root_literal_expr;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_core::export::arrow::{array::BooleanArray, bitmap::MutableBitmap};
use polars_core::prelude::*;

use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::{
    fmt,
    ops::{Add, Div, Mul, Rem, Sub},
    sync::Arc,
};
// reexport the lazy method
pub use crate::frame::IntoLazy;
pub use crate::logical_plan::lit;
pub use functions::*;
pub use options::*;

use polars_arrow::array::default_arrays::FromData;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_core::utils::{get_supertype, NoNull};

/// A wrapper trait for any closure `Fn(Vec<Series>) -> Result<Series>`
pub trait SeriesUdf: Send + Sync {
    fn call_udf(&self, s: &mut [Series]) -> Result<Series>;
}

impl<F> SeriesUdf for F
where
    F: Fn(&mut [Series]) -> Result<Series> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Series]) -> Result<Series> {
        self(s)
    }
}

impl Debug for dyn SeriesUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesUdf")
    }
}

/// A wrapper trait for any binary closure `Fn(Series, Series) -> Result<Series>`
pub trait SeriesBinaryUdf: Send + Sync {
    fn call_udf(&self, a: Series, b: Series) -> Result<Series>;
}

impl<F> SeriesBinaryUdf for F
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync,
{
    fn call_udf(&self, a: Series, b: Series) -> Result<Series> {
        self(a, b)
    }
}

impl Debug for dyn SeriesBinaryUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesBinaryUdf")
    }
}

pub trait RenameAliasFn: Send + Sync {
    fn call(&self, name: &str) -> String;
}

impl<F: Fn(&str) -> String + Send + Sync> RenameAliasFn for F {
    fn call(&self, name: &str) -> String {
        self(name)
    }
}

impl Debug for dyn RenameAliasFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenameAliasFn")
    }
}

#[derive(Clone)]
/// Wrapper type that indicates that the inner type is not equal to anything
pub struct NoEq<T>(T);

impl<T> NoEq<T> {
    pub fn new(val: T) -> Self {
        NoEq(val)
    }
}

impl<T> PartialEq for NoEq<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T> Debug for NoEq<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "no_eq")
    }
}

impl<T> Deref for NoEq<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait BinaryUdfOutputField: Send + Sync {
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field>;
}

impl<F> BinaryUdfOutputField for F
where
    F: Fn(&Schema, Context, &Field, &Field) -> Option<Field> + Send + Sync,
{
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field> {
        self(input_schema, cntxt, field_a, field_b)
    }
}

pub trait FunctionOutputField: Send + Sync {
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field;
}

pub type GetOutput = NoEq<Arc<dyn FunctionOutputField>>;

impl Default for GetOutput {
    fn default() -> Self {
        NoEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| fields[0].clone(),
        ))
    }
}

impl GetOutput {
    pub fn same_type() -> Self {
        Default::default()
    }

    pub fn from_type(dt: DataType) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            Field::new(flds[0].name(), dt.clone())
        }))
    }

    pub fn map_field<F: 'static + Fn(&Field) -> Field + Send + Sync>(f: F) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(&flds[0])
        }))
    }

    pub fn map_fields<F: 'static + Fn(&[Field]) -> Field + Send + Sync>(f: F) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(flds)
        }))
    }

    pub fn map_dtype<F: 'static + Fn(&DataType) -> DataType + Send + Sync>(f: F) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let new_type = f(fld.data_type());
            fld.coerce(new_type);
            fld
        }))
    }

    pub fn super_type() -> Self {
        Self::map_dtypes(|dtypes| {
            let mut st = dtypes[0].clone();
            for dt in &dtypes[1..] {
                st = get_supertype(&st, dt).unwrap()
            }
            st
        })
    }

    pub fn map_dtypes<F>(f: F) -> Self
    where
        F: 'static + Fn(&[&DataType]) -> DataType + Send + Sync,
    {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let dtypes = flds.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
            let new_type = f(&dtypes);
            fld.coerce(new_type);
            fld
        }))
    }
}

impl<F> FunctionOutputField for F
where
    F: Fn(&Schema, Context, &[Field]) -> Field + Send + Sync,
{
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field {
        self(input_schema, cntxt, fields)
    }
}

#[derive(PartialEq, Clone)]
pub enum AggExpr {
    Min(Box<Expr>),
    Max(Box<Expr>),
    Median(Box<Expr>),
    NUnique(Box<Expr>),
    First(Box<Expr>),
    Last(Box<Expr>),
    Mean(Box<Expr>),
    List(Box<Expr>),
    Count(Box<Expr>),
    Quantile {
        expr: Box<Expr>,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    },
    Sum(Box<Expr>),
    AggGroups(Box<Expr>),
    Std(Box<Expr>),
    Var(Box<Expr>),
}

impl AsRef<Expr> for AggExpr {
    fn as_ref(&self) -> &Expr {
        use AggExpr::*;
        match self {
            Min(e) => e,
            Max(e) => e,
            Median(e) => e,
            NUnique(e) => e,
            First(e) => e,
            Last(e) => e,
            Mean(e) => e,
            List(e) => e,
            Count(e) => e,
            Quantile { expr, .. } => expr,
            Sum(e) => e,
            AggGroups(e) => e,
            Std(e) => e,
            Var(e) => e,
        }
    }
}

/// Queries consists of multiple expressions.
#[derive(Clone, PartialEq)]
#[must_use]
pub enum Expr {
    Alias(Box<Expr>, Arc<str>),
    Column(Arc<str>),
    Columns(Vec<String>),
    DtypeColumn(Vec<DataType>),
    Literal(LiteralValue),
    BinaryExpr {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    Not(Box<Expr>),
    IsNotNull(Box<Expr>),
    IsNull(Box<Expr>),
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
        strict: bool,
    },
    Sort {
        expr: Box<Expr>,
        options: SortOptions,
    },
    Take {
        expr: Box<Expr>,
        idx: Box<Expr>,
    },
    SortBy {
        expr: Box<Expr>,
        by: Vec<Expr>,
        reverse: Vec<bool>,
    },
    Agg(AggExpr),
    /// A ternary operation
    /// if true then "foo" else "bar"
    Ternary {
        predicate: Box<Expr>,
        truthy: Box<Expr>,
        falsy: Box<Expr>,
    },
    Function {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: NoEq<Arc<dyn SeriesUdf>>,
        /// output dtype of the function
        output_type: GetOutput,
        options: FunctionOptions,
    },
    Shift {
        input: Box<Expr>,
        periods: i64,
    },
    Reverse(Box<Expr>),
    Duplicated(Box<Expr>),
    IsUnique(Box<Expr>),
    Explode(Box<Expr>),
    Filter {
        input: Box<Expr>,
        by: Box<Expr>,
    },
    /// See postgres window functions
    Window {
        /// Also has the input. i.e. avg("foo")
        function: Box<Expr>,
        partition_by: Vec<Expr>,
        order_by: Option<Box<Expr>>,
        options: WindowOptions,
    },
    Wildcard,
    Slice {
        input: Box<Expr>,
        /// length is not yet known so we accept negative offsets
        offset: Box<Expr>,
        length: Box<Expr>,
    },
    /// Can be used in a select statement to exclude a column from selection
    Exclude(Box<Expr>, Vec<Excluded>),
    /// Set root name as Alias
    KeepName(Box<Expr>),
    RenameAlias {
        function: NoEq<Arc<dyn RenameAliasFn>>,
        expr: Box<Expr>,
    },
    /// Special case that does not need columns
    Count,
    /// Take the nth column in the `DataFrame`
    Nth(i64),
}

impl Default for Expr {
    fn default() -> Self {
        Expr::Literal(LiteralValue::Null)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Excluded {
    Name(Arc<str>),
    Dtype(DataType),
}

impl Expr {
    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(&self, schema: &Schema, ctxt: Context) -> Result<Field> {
        // this is not called much and th expression depth is typically shallow
        let mut arena = Arena::with_capacity(5);
        let root = to_aexpr(self.clone(), &mut arena);
        arena.get(root).to_field(schema, ctxt, &arena)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Operator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    TrueDivide,
    Modulus,
    And,
    Or,
    Xor,
}

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
    pub fn then(self, expr: Expr) -> WhenThen {
        WhenThen {
            predicate: self.predicate,
            then: expr,
        }
    }
}

impl WhenThen {
    pub fn when(self, predicate: Expr) -> WhenThenThen {
        WhenThenThen {
            predicates: vec![self.predicate, predicate],
            thens: vec![self.then],
        }
    }

    pub fn otherwise(self, expr: Expr) -> Expr {
        Expr::Ternary {
            predicate: Box::new(self.predicate),
            truthy: Box::new(self.then),
            falsy: Box::new(expr),
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
        // `inner` and then assighn that to `otherwise`,
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
pub fn when(predicate: Expr) -> When {
    When { predicate }
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
        if let Self::Function {
            input,
            function,
            output_type,
            mut options,
        } = self
        {
            options = func(options);
            Self::Function {
                input,
                function,
                output_type,
                options,
            }
        } else {
            panic!("implementation error")
        }
    }

    /// Overwrite the function name used for formatting
    /// this is not intended to be used
    #[cfg(feature = "private")]
    #[doc(hidden)]
    pub fn with_fmt(self, name: &'static str) -> Expr {
        self.with_function_options(|mut options| {
            options.fmt_str = name;
            options
        })
    }

    /// Compare `Expr` with other `Expr` on equality
    pub fn eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Eq, other)
    }

    /// Compare `Expr` with other `Expr` on non-equality
    pub fn neq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::NotEq, other)
    }

    /// Check if `Expr` < `Expr`
    pub fn lt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Lt, other)
    }

    /// Check if `Expr` > `Expr`
    pub fn gt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Gt, other)
    }

    /// Check if `Expr` >= `Expr`
    pub fn gt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::GtEq, other)
    }

    /// Check if `Expr` <= `Expr`
    pub fn lt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::LtEq, other)
    }

    /// Negate `Expr`
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }

    /// Rename Column.
    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), Arc::from(name))
    }

    /// Run is_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_null(self) -> Self {
        Expr::IsNull(Box::new(self))
    }

    /// Run is_not_null operation on `Expr`.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_null(self) -> Self {
        Expr::IsNotNull(Box::new(self))
    }

    /// Reduce groups to minimal value.
    pub fn min(self) -> Self {
        AggExpr::Min(Box::new(self)).into()
    }

    /// Reduce groups to maximum value.
    pub fn max(self) -> Self {
        AggExpr::Max(Box::new(self)).into()
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
    pub fn slice(self, offset: Expr, length: Expr) -> Self {
        Expr::Slice {
            input: Box::new(self),
            offset: Box::new(offset),
            length: Box::new(length),
        }
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

    /// Get the index value that has the minumum value
    pub fn arg_min(self) -> Self {
        let options = FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: false,
            auto_explode: true,
            fmt_str: "arg_min",
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
            input_wildcard_expansion: false,
            auto_explode: true,
            fmt_str: "arg_max",
        };

        self.function_with_options(
            move |s: Series| Ok(Series::new(s.name(), &[s.arg_max().map(|idx| idx as u32)])),
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
    }

    /// Get the index values that would sort this expression.
    pub fn arg_sort(self, reverse: bool) -> Self {
        assert!(
            !has_expr(&self, |e| matches!(e, Expr::Wildcard)),
            "wildcard not supported in argsort expr"
        );
        let options = FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: false,
            auto_explode: false,
            fmt_str: "arg_sort",
        };

        self.function_with_options(
            move |s: Series| {
                Ok(s.argsort(SortOptions {
                    descending: reverse,
                    ..Default::default()
                })
                .into_series())
            },
            GetOutput::from_type(IDX_DTYPE),
            options,
        )
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
    pub fn take(self, idx: Expr) -> Self {
        Expr::Take {
            expr: Box::new(self),
            idx: Box::new(idx),
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

    /// Reverse column
    pub fn reverse(self) -> Self {
        Expr::Reverse(Box::new(self))
    }

    /// Apply a function/closure once the logical plan get executed.
    ///
    /// This function is very similar to [`apply`], but differs in how it handles aggregations.
    ///
    ///  * `map` should be used for operations that are independent of groups, e.g. `multiply * 2`, or `raise to the power`
    ///  * `apply` should be used for operations that work on a group of data. e.g. `sum`, `count`, etc.
    ///
    /// It is the responsibility of the caller that the schema is correct by giving
    /// the correct output_type. If None given the output type of the input expr is used.
    pub fn map<F>(self, function: F, output_type: GetOutput) -> Self
    where
        F: Fn(Series) -> Result<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::Function {
            input: vec![self],
            function: NoEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: false,
                auto_explode: false,
                fmt_str: "map",
            },
        }
    }

    /// Apply a function/closure once the logical plan get executed with many arguments
    ///
    /// See the [`map`] function for the differences between [`map`] and [`apply`].
    pub fn map_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Series]) -> Result<Series> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::Function {
            input,
            function: NoEq::new(Arc::new(function)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: false,
                auto_explode: false,
                fmt_str: "",
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
        F: Fn(Series) -> Result<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::Function {
            input: vec![self],
            function: NoEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyList,
                input_wildcard_expansion: false,
                auto_explode: false,
                fmt_str: "",
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
        F: Fn(Series) -> Result<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::Function {
            input: vec![self],
            function: NoEq::new(Arc::new(f)),
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
        F: Fn(Series) -> Result<Series> + 'static + Send + Sync,
    {
        let f = move |s: &mut [Series]| function(std::mem::take(&mut s[0]));

        Expr::Function {
            input: vec![self],
            function: NoEq::new(Arc::new(f)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: false,
                auto_explode: false,
                fmt_str: "",
            },
        }
    }

    /// Apply a function/closure over the groups with many arguments. This should only be used in a groupby aggregation.
    ///
    /// See the [`apply`] function for the differences between [`map`] and [`apply`].
    pub fn apply_many<F>(self, function: F, arguments: &[Expr], output_type: GetOutput) -> Self
    where
        F: Fn(&mut [Series]) -> Result<Series> + 'static + Send + Sync,
    {
        let mut input = vec![self];
        input.extend_from_slice(arguments);

        Expr::Function {
            input,
            function: NoEq::new(Arc::new(function)),
            output_type,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: false,
                auto_explode: true,
                fmt_str: "",
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
    #[allow(clippy::wrong_self_convention)]
    pub fn is_nan(self) -> Self {
        self.map(
            |s: Series| s.is_nan().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
        .with_fmt("is_nan")
    }

    /// Get inverse mask of NaN values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_nan(self) -> Self {
        self.map(
            |s: Series| s.is_not_nan().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
        .with_fmt("is_not_nan")
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::shift).
    pub fn shift(self, periods: i64) -> Self {
        Expr::Shift {
            input: Box::new(self),
            periods,
        }
    }

    /// Shift the values in the array by some period and fill the resulting empty values.
    pub fn shift_and_fill(self, periods: i64, fill_value: Expr) -> Self {
        // Note:
        // The order of the then | otherwise is important
        if periods > 0 {
            when(self.clone().apply(
                move |s: Series| {
                    let len = s.len();
                    let mut bits = MutableBitmap::with_capacity(s.len());
                    bits.extend_constant(periods as usize, false);
                    bits.extend_constant(len.saturating_sub(periods as usize), true);
                    let mask = BooleanArray::from_data_default(bits.into(), None);
                    let ca: BooleanChunked = mask.into();
                    Ok(ca.into_series())
                },
                GetOutput::from_type(DataType::Boolean),
            ))
            .then(self.shift(periods))
            .otherwise(fill_value)
        } else {
            when(self.clone().apply(
                move |s: Series| {
                    let length = s.len() as i64;
                    // periods is negative, so subtraction.
                    let tipping_point = std::cmp::max(length + periods, 0);
                    let mut bits = MutableBitmap::with_capacity(s.len());
                    bits.extend_constant(tipping_point as usize, true);
                    bits.extend_constant(-periods as usize, false);
                    let mask = BooleanArray::from_data_default(bits.into(), None);
                    let ca: BooleanChunked = mask.into();
                    Ok(ca.into_series())
                },
                GetOutput::from_type(DataType::Boolean),
            ))
            .then(self.shift(periods))
            .otherwise(fill_value)
        }
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

    /// Get the product aggreagtion of an expresion
    #[cfg_attr(docsrs, doc(cfg(feature = "product")))]
    pub fn product(self) -> Self {
        self.apply(
            move |s: Series| Ok(s.product()),
            GetOutput::map_dtype(|dt| {
                use DataType::*;
                match dt {
                    Float32 => Float32,
                    Float64 => Float64,
                    _ => Int64,
                }
            }),
        )
        .with_fmt("product")
    }

    /// Fill missing value with next non-null.
    pub fn backward_fill(self) -> Self {
        self.apply(
            move |s: Series| s.fill_null(FillNullStrategy::Backward),
            GetOutput::same_type(),
        )
        .with_fmt("backward_fill")
    }

    /// Fill missing value with previous non-null.
    pub fn forward_fill(self) -> Self {
        self.apply(
            move |s: Series| s.fill_null(FillNullStrategy::Forward),
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

    /// Ceil underlying floating point array to the heighest integers smaller or equal to the float value.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn ceil(self) -> Self {
        self.map(move |s: Series| s.ceil(), GetOutput::same_type())
            .with_fmt("ceil")
    }

    /// Clip underlying values to a set boundary.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn clip(self, min: f64, max: f64) -> Self {
        self.map(move |s: Series| s.clip(min, max), GetOutput::same_type())
            .with_fmt("clip")
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
    /// fn example() -> Result<()> {
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
    pub fn over<E: AsRef<[Expr]>>(self, partition_by: E) -> Self {
        Expr::Window {
            function: Box::new(self),
            partition_by: partition_by.as_ref().to_vec(),
            order_by: None,
            options: WindowOptions { explode: false },
        }
    }

    /// Replace the null values by a value.
    pub fn fill_null(self, fill_value: Expr) -> Self {
        self.map_many(
            |s| {
                let a = &s[0];
                let b = &s[1];

                if !a.null_count() == 0 {
                    Ok(a.clone())
                } else {
                    let st = get_supertype(a.dtype(), b.dtype())?;
                    let a = a.cast(&st)?;
                    let b = b.cast(&st)?;
                    let mask = a.is_not_null();
                    a.zip_with_same_type(&mask, &b)
                }
            },
            &[fill_value],
            GetOutput::super_type(),
        )
        .with_fmt("fill_null")
    }

    /// Replace the floating point `NaN` values by a value.
    pub fn fill_nan(self, fill_value: Expr) -> Self {
        when(self.clone().is_nan()).then(fill_value).otherwise(self)
    }
    /// Count the values of the Series
    /// or
    /// Get counts of the group by operation.
    pub fn count(self) -> Self {
        AggExpr::Count(Box::new(self)).into()
    }

    /// Standard deviation of the values of the Series
    pub fn std(self) -> Self {
        AggExpr::Std(Box::new(self)).into()
    }

    /// Variance of the values of the Series
    pub fn var(self) -> Self {
        AggExpr::Var(Box::new(self)).into()
    }

    /// Get a mask of duplicated values
    #[allow(clippy::wrong_self_convention)]
    pub fn is_duplicated(self) -> Self {
        Expr::Duplicated(Box::new(self))
    }

    /// Get a mask of unique values
    #[allow(clippy::wrong_self_convention)]
    pub fn is_unique(self) -> Self {
        Expr::IsUnique(Box::new(self))
    }

    /// and operation
    pub fn and(self, expr: Expr) -> Self {
        binary_expr(self, Operator::And, expr)
    }

    // xor operation
    pub fn xor(self, expr: Expr) -> Self {
        binary_expr(self, Operator::Xor, expr)
    }

    /// or operation
    pub fn or(self, expr: Expr) -> Self {
        binary_expr(self, Operator::Or, expr)
    }

    /// Raise expression to the power `exponent`
    pub fn pow(self, exponent: f64) -> Self {
        self.map(
            move |s: Series| s.pow(exponent),
            GetOutput::from_type(DataType::Float64),
        )
        .with_fmt("pow")
    }

    /// Filter a single column
    /// Should be used in aggregation context. If you want to filter on a DataFrame level, use
    /// [LazyFrame::filter](LazyFrame::filter)
    pub fn filter(self, predicate: Expr) -> Self {
        if has_expr(&self, |e| matches!(e, Expr::Wildcard)) {
            panic!("filter '*' not allowed, use LazyFrame::filter")
        };
        Expr::Filter {
            input: Box::new(self),
            by: Box::new(predicate),
        }
    }

    /// Check if the values of the left expression are in the lists of the right expr.
    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "is_in")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_in")))]
    pub fn is_in(self, other: Expr) -> Self {
        let has_literal = has_root_literal_expr(&other);
        if has_literal {
            if let Expr::Literal(LiteralValue::Series(s)) = &other {
                // nothing is in an empty list return all False
                if s.is_empty() {
                    return Expr::Literal(LiteralValue::Boolean(false));
                }
            }
        }

        let f = |s: &mut [Series]| {
            let left = &s[0];
            let other = &s[1];

            left.is_in(other).map(|ca| ca.into_series())
        };
        let arguments = &[other];
        let output_type = GetOutput::from_type(DataType::Boolean);

        // we don't have to apply on groups, so this is faster
        if has_literal {
            self.map_many(f, arguments, output_type)
        } else {
            self.apply_many(f, arguments, output_type)
        }
        .with_fmt("is_in")
    }

    /// Sort this column by the ordering of another column.
    /// Can also be used in a groupby context to sort the groups.
    pub fn sort_by<E: AsRef<[Expr]>, R: AsRef<[bool]>>(self, by: E, reverse: R) -> Expr {
        let by = by.as_ref().to_vec();
        let reverse = reverse.as_ref().to_vec();
        Expr::SortBy {
            expr: Box::new(self),
            by,
            reverse,
        }
    }

    #[cfg(feature = "repeat_by")]
    #[cfg_attr(docsrs, doc(cfg(feature = "repeat_by")))]
    pub fn repeat_by(self, by: Expr) -> Expr {
        let function = |s: &mut [Series]| {
            let by = &s[1];
            let s = &s[0];
            let by = by.cast(&IDX_DTYPE)?;
            Ok(s.repeat_by(by.idx()?).into_series())
        };

        self.map_many(
            function,
            &[by],
            GetOutput::map_dtype(|dt| DataType::List(dt.clone().into())),
        )
        .with_fmt("repeat_by")
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
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_product")))]
    pub fn dot(self, other: Expr) -> Expr {
        let function = |s: &mut [Series]| Ok((&s[0] * &s[1]).sum_as_series());

        self.map_many(function, &[other], GetOutput::same_type())
            .with_fmt("dot")
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
        F: Fn(&str) -> String + 'static + Send + Sync,
    {
        let function = NoEq::new(Arc::new(function) as Arc<dyn RenameAliasFn>);
        Expr::RenameAlias {
            expr: Box::new(self),
            function,
        }
    }

    /// Add a suffix to the root column name.
    pub fn suffix(self, suffix: &str) -> Expr {
        let suffix = suffix.to_string();
        self.map_alias(move |name| format!("{}{}", name, suffix))
    }

    /// Add a prefix to the root column name.
    pub fn prefix(self, prefix: &str) -> Expr {
        let prefix = prefix.to_string();
        self.map_alias(move |name| format!("{}{}", prefix, name))
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

    /// Apply a rolling min See:
    /// [ChunkedArray::rolling_min]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_min(self, options: RollingOptions) -> Expr {
        self.apply(
            move |s| s.rolling_min(options.clone()),
            GetOutput::same_type(),
        )
        .with_fmt("rolling_min")
    }

    /// Apply a rolling max See:
    /// [ChunkedArray::rolling_max]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_max(self, options: RollingOptions) -> Expr {
        self.apply(
            move |s| s.rolling_max(options.clone()),
            GetOutput::same_type(),
        )
        .with_fmt("rolling_max")
    }

    /// Apply a rolling mean See:
    /// [ChunkedArray::rolling_mean]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_mean(self, options: RollingOptions) -> Expr {
        self.apply(
            move |s| s.rolling_mean(options.clone()),
            GetOutput::same_type(),
        )
        .with_fmt("rolling_mean")
    }

    /// Apply a rolling sum See:
    /// [ChunkedArray::rolling_sum]
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_sum(self, options: RollingOptions) -> Expr {
        self.apply(
            move |s| s.rolling_sum(options.clone()),
            GetOutput::same_type(),
        )
        .with_fmt("rolling_sum")
    }

    /// Apply a rolling median See:
    /// [ChunkedArray::rolling_median](polars::prelude::ChunkWindow::rolling_median).
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_median(self, options: RollingOptions) -> Expr {
        self.apply(
            move |s| s.rolling_median(options.clone()),
            GetOutput::same_type(),
        )
        .with_fmt("rolling_median")
    }

    /// Apply a rolling quantile See:
    /// [ChunkedArray::rolling_quantile](polars::prelude::ChunkWindow::rolling_quantile).
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_quantile(
        self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptions,
    ) -> Expr {
        self.apply(
            move |s| s.rolling_quantile(quantile, interpolation, options.clone()),
            GetOutput::same_type(),
        )
        .with_fmt("rolling_quantile")
    }

    /// Apply a rolling variance
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_var(self, options: RollingOptions) -> Expr {
        self.to_float()
            .apply(
                move |s| match s.dtype() {
                    DataType::Float32 => s.f32().unwrap().rolling_var(options.clone()),
                    DataType::Float64 => s.f64().unwrap().rolling_var(options.clone()),
                    _ => unreachable!(),
                },
                GetOutput::same_type(),
            )
            .with_fmt("rolling_var")
    }

    /// Apply a rolling std-dev
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_std(self, options: RollingOptions) -> Expr {
        self.to_float()
            .apply(
                move |s| match s.dtype() {
                    DataType::Float32 => s.f32().unwrap().rolling_std(options.clone()),
                    DataType::Float64 => s.f64().unwrap().rolling_std(options.clone()),
                    _ => unreachable!(),
                },
                GetOutput::same_type(),
            )
            .with_fmt("rolling_std")
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    pub fn rolling_apply(
        self,
        f: Arc<dyn Fn(&Series) -> Series + Send + Sync>,
        output_type: GetOutput,
        options: RollingOptions,
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
        F: 'static + Fn(&Float64Chunked) -> Option<f64> + Send + Sync + Copy,
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
        self.apply(
            move |s| Ok(s.diff(n, null_behavior)),
            GetOutput::same_type(),
        )
        .with_fmt("diff")
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
    pub fn skew(self, bias: bool) -> Expr {
        self.apply(
            move |s| s.skew(bias).map(|opt_v| Series::new(s.name(), &[opt_v])),
            GetOutput::from_type(DataType::Float64),
        )
        .with_fmt("skew")
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
        .with_fmt("kurtosis")
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
    pub fn shuffle(self, seed: u64) -> Self {
        self.apply(move |s| Ok(s.shuffle(seed)), GetOutput::same_type())
            .with_fmt("shuffle")
    }

    #[cfg(feature = "random")]
    pub fn sample_frac(self, frac: f64, with_replacement: bool, seed: u64) -> Self {
        self.apply(
            move |s| s.sample_frac(frac, with_replacement, seed),
            GetOutput::same_type(),
        )
        .with_fmt("shuffle")
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
        .with_fmt("emw_mean")
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
        .with_fmt("emw_std")
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
        .with_fmt("emw_var")
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

    /// This is useful if an `apply` function needs a floating point type.
    /// Because this cast is done on a `map` level, it will be faster.
    pub fn to_float(self) -> Self {
        self.map(
            |s| match s.dtype() {
                DataType::Float32 | DataType::Float64 => Ok(s),
                _ => s.cast(&DataType::Float64),
            },
            GetOutput::map_dtype(|dt| {
                if matches!(dt, DataType::Float32) {
                    DataType::Float32
                } else {
                    DataType::Float64
                }
            }),
        )
    }

    #[cfg(feature = "strings")]
    pub fn str(self) -> string::StringNameSpace {
        string::StringNameSpace(self)
    }

    #[cfg(feature = "temporal")]
    pub fn dt(self) -> dt::DateLikeNameSpace {
        dt::DateLikeNameSpace(self)
    }
    #[cfg(feature = "list")]
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
    F: Fn(&mut [Series]) -> Result<Series> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::Function {
        input,
        function: NoEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: false,
            auto_explode: false,
            fmt_str: "",
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
    F: Fn(&mut [Series]) -> Result<Series> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::Function {
        input,
        function: NoEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyList,
            input_wildcard_expansion: false,
            auto_explode: true,
            fmt_str: "",
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
pub fn apply_multiple<F, E>(function: F, expr: E, output_type: GetOutput) -> Expr
where
    F: Fn(&mut [Series]) -> Result<Series> + 'static + Send + Sync,
    E: AsRef<[Expr]>,
{
    let input = expr.as_ref().to_vec();

    Expr::Function {
        input,
        function: NoEq::new(Arc::new(function)),
        output_type,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            input_wildcard_expansion: false,
            auto_explode: true,
            fmt_str: "",
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
