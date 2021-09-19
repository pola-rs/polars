//! Domain specific language for the Lazy api.
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{expr_to_root_column_name, has_expr, has_wildcard};
use polars_core::prelude::*;

#[cfg(feature = "temporal")]
use polars_core::utils::chrono::{NaiveDate, NaiveDateTime};
use std::fmt::{Debug, Formatter};
use std::ops::{BitAnd, BitOr, Deref};
use std::{
    fmt,
    ops::{Add, Div, Mul, Rem, Sub},
    sync::Arc,
};
// reexport the lazy method
pub use crate::frame::IntoLazy;
use polars_core::frame::select::Selection;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_core::utils::get_supertype;

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
}

impl<F> FunctionOutputField for F
where
    F: Fn(&Schema, Context, &[Field]) -> Field + Send + Sync,
{
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field {
        self(input_schema, cntxt, fields)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ApplyOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    ApplyGroups,
    // collect groups to a list and then apply
    ApplyList,
    // do not collect before apply
    ApplyFlat,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WindowOptions {
    /// Explode the aggregated list and just do a hstack instead of a join
    /// this requires the groups to be sorted to make any sense
    pub(crate) explode: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FunctionOptions {
    /// Collect groups to a list and apply the function over the groups.
    /// This can be important in aggregation context.
    pub(crate) collect_groups: ApplyOptions,
    /// There can be two ways of expanding wildcards:
    ///
    /// Say the schema is 'a', 'b' and there is a function f
    /// f('*')
    /// can expand to:
    /// 1.
    ///     f('a', 'b')
    /// or
    /// 2.
    ///     f('a'), f('b')
    ///
    /// setting this to true, will lead to behavior 1.
    ///
    /// this also accounts for regex expansion
    pub(crate) input_wildcard_expansion: bool,
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
    Quantile { expr: Box<Expr>, quantile: f64 },
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

impl From<AggExpr> for Expr {
    fn from(agg: AggExpr) -> Self {
        Expr::Agg(agg)
    }
}

/// Queries consists of multiple expressions.
#[derive(Clone, PartialEq)]
pub enum Expr {
    Alias(Box<Expr>, Arc<String>),
    Column(Arc<String>),
    Columns(Vec<String>),
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
        reverse: bool,
    },
    Take {
        expr: Box<Expr>,
        idx: Box<Expr>,
    },
    SortBy {
        expr: Box<Expr>,
        by: Box<Expr>,
        reverse: bool,
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
        offset: i64,
        length: usize,
    },
    BinaryFunction {
        input_a: Box<Expr>,
        input_b: Box<Expr>,
        function: NoEq<Arc<dyn SeriesBinaryUdf>>,
        /// Delays output type evaluation until input schema is known.
        output_field: NoEq<Arc<dyn BinaryUdfOutputField>>,
    },
    /// Can be used in a select statement to exclude a column from selection
    Exclude(Box<Expr>, Vec<Arc<String>>),
    /// Set root name as Alias
    KeepName(Box<Expr>),
    SufPreFix {
        is_suffix: bool,
        value: String,
        expr: Box<Expr>,
    },
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

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;
        match self {
            Window {
                function,
                partition_by,
                order_by,
                ..
            } => write!(
                f,
                "{:?} OVER (PARTITION BY {:?} ORDER BY {:?}",
                function, partition_by, order_by
            ),
            IsUnique(expr) => write!(f, "UNIQUE {:?}", expr),
            Explode(expr) => write!(f, "EXPLODE {:?}", expr),
            Duplicated(expr) => write!(f, "DUPLICATED {:?}", expr),
            Reverse(expr) => write!(f, "REVERSE {:?}", expr),
            Alias(expr, name) => write!(f, "{:?} AS {}", expr, name),
            Column(name) => write!(f, "{}", name),
            Literal(v) => {
                match v {
                    LiteralValue::Utf8(v) => {
                        // dot breaks with debug fmt due to \"
                        write!(f, "Utf8({})", v)
                    }
                    _ => {
                        write!(f, "{:?}", v)
                    }
                }
            }
            BinaryExpr { left, op, right } => write!(f, "[({:?}) {:?} ({:?})]", left, op, right),
            Not(expr) => write!(f, "NOT {:?}", expr),
            IsNull(expr) => write!(f, "{:?} IS NULL", expr),
            IsNotNull(expr) => write!(f, "{:?} IS NOT NULL", expr),
            Sort { expr, reverse } => match reverse {
                true => write!(f, "{:?} DESC", expr),
                false => write!(f, "{:?} ASC", expr),
            },
            SortBy { expr, by, reverse } => match reverse {
                true => write!(f, "{:?} DESC BY {:?}", expr, by),
                false => write!(f, "{:?} ASC BY {:?}", expr, by),
            },
            Filter { input, by } => {
                write!(f, "FILTER {:?} BY {:?}", input, by)
            }
            Take { expr, idx } => {
                write!(f, "TAKE {:?} AT {:?}", expr, idx)
            }
            Agg(agg) => {
                use AggExpr::*;
                match agg {
                    Min(expr) => write!(f, "AGG MIN {:?}", expr),
                    Max(expr) => write!(f, "AGG MAX {:?}", expr),
                    Median(expr) => write!(f, "AGG MEDIAN {:?}", expr),
                    Mean(expr) => write!(f, "AGG MEAN {:?}", expr),
                    First(expr) => write!(f, "AGG FIRST {:?}", expr),
                    Last(expr) => write!(f, "AGG LAST {:?}", expr),
                    List(expr) => write!(f, "AGG LIST {:?}", expr),
                    NUnique(expr) => write!(f, "AGG N UNIQUE {:?}", expr),
                    Sum(expr) => write!(f, "AGG SUM {:?}", expr),
                    AggGroups(expr) => write!(f, "AGG GROUPS {:?}", expr),
                    Count(expr) => write!(f, "AGG COUNT {:?}", expr),
                    Var(expr) => write!(f, "AGG VAR {:?}", expr),
                    Std(expr) => write!(f, "AGG STD {:?}", expr),
                    Quantile { expr, .. } => write!(f, "AGG QUANTILE {:?}", expr),
                }
            }
            Cast {
                expr, data_type, ..
            } => write!(f, "CAST {:?} TO {:?}", expr, data_type),
            Ternary {
                predicate,
                truthy,
                falsy,
            } => write!(
                f,
                "\nWHEN {:?}\n\t{:?}\nOTHERWISE\n\t{:?}",
                predicate, truthy, falsy
            ),
            Function { input, .. } => write!(f, "APPLY({:?})", input),
            BinaryFunction {
                input_a, input_b, ..
            } => write!(f, "BinaryFunction({:?}, {:?})", input_a, input_b),
            Shift { input, periods, .. } => write!(f, "SHIFT {:?} by {}", input, periods),
            Slice {
                input,
                offset,
                length,
            } => write!(f, "SLICE {:?} offset: {} len: {}", input, offset, length),
            Wildcard => write!(f, "*"),
            Exclude(column, names) => write!(f, "{:?}, EXCEPT {:?}", column, names),
            KeepName(e) => write!(f, "KEEP NAME {:?}", e),
            SufPreFix { expr, .. } => write!(f, "SUF-PREFIX {:?}", expr),
            Columns(names) => write!(f, "COLUMNS({:?})", names),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
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
    #[cfg(feature = "true_div")]
    TrueDivide,
    Modulus,
    And,
    Or,
}

pub fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

/// Intermediate state of `when(..).then(..).otherwise(..)` expr.
pub struct When {
    predicate: Expr,
}

/// Intermediate state of `when(..).then(..).otherwise(..)` expr.
pub struct WhenThen {
    predicate: Expr,
    then: Expr,
}

/// Intermediate state of chain when then exprs.
///
/// ```ignore
/// when(..).then(..)
/// when(..).then(..)
/// when(..).then(..)
/// .otherwise(..)`
/// ```
#[derive(Clone)]
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
        Expr::Alias(Box::new(self), Arc::new(name.into()))
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
    pub fn quantile(self, quantile: f64) -> Self {
        AggExpr::Quantile {
            expr: Box::new(self),
            quantile,
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
    pub fn slice(self, offset: i64, length: usize) -> Self {
        Expr::Slice {
            input: Box::new(self),
            offset,
            length,
        }
    }

    /// Get the first `n` elements of the Expr result
    pub fn head(self, length: Option<usize>) -> Self {
        self.slice(0, length.unwrap_or(10))
    }

    /// Get the last `n` elements of the Expr result
    pub fn tail(self, length: Option<usize>) -> Self {
        let len = length.unwrap_or(10);
        self.slice(-(len as i64), len)
    }

    /// Get unique values of this expression.
    pub fn unique(self) -> Self {
        if has_expr(&self, |e| matches!(e, Expr::Wildcard)) {
            panic!("wildcard not supperted in unique expr");
        }
        self.apply(|s: Series| s.unique(), GetOutput::same_type())
    }

    /// Get the first index of unique values of this expression.
    pub fn arg_unique(self) -> Self {
        if has_expr(&self, |e| matches!(e, Expr::Wildcard)) {
            panic!("wildcard not supported in unique expr");
        }
        self.apply(
            |s: Series| s.arg_unique().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::UInt32),
        )
    }

    /// Get the index values that would sort this expression.
    pub fn arg_sort(self, reverse: bool) -> Self {
        if has_expr(&self, |e| matches!(e, Expr::Wildcard)) {
            panic!("wildcard not supported in unique expr");
        }
        self.apply(
            move |s: Series| Ok(s.argsort(reverse).into_series()),
            GetOutput::from_type(DataType::UInt32),
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

    /// Sort in increasing order. See [the eager implementation](polars_core::series::SeriesTrait::sort).
    ///
    /// Can be used in `default` and `aggregation` context.
    pub fn sort(self, reverse: bool) -> Self {
        Expr::Sort {
            expr: Box::new(self),
            reverse,
        }
    }

    /// Reverse column
    pub fn reverse(self) -> Self {
        Expr::Reverse(Box::new(self))
    }

    /// Apply a function/closure once the logical plan get executed.
    ///
    /// This function is very similar to [apply](Expr::apply), but differs in how it handles aggregations.
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
            },
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
    }

    /// Get mask of infinite values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_infinite(self) -> Self {
        self.map(
            |s: Series| s.is_infinite().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
    }

    /// Get mask of NaN values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_nan(self) -> Self {
        self.map(
            |s: Series| s.is_nan().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
    }

    /// Get inverse mask of NaN values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_nan(self) -> Self {
        self.map(
            |s: Series| s.is_not_nan().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Boolean),
        )
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
                    let ca: BooleanChunked = (0..s.len() as i64).map(|i| i >= periods).collect();
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
                    let tipping_point = length + periods;
                    let ca: BooleanChunked = (0..length).map(|i| i < tipping_point).collect();
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
    }

    /// Get an array with the cumulative min computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cummin(self, reverse: bool) -> Self {
        self.apply(
            move |s: Series| Ok(s.cummin(reverse)),
            GetOutput::same_type(),
        )
    }

    /// Get an array with the cumulative max computed at every element
    #[cfg_attr(docsrs, doc(cfg(feature = "cum_agg")))]
    pub fn cummax(self, reverse: bool) -> Self {
        self.apply(
            move |s: Series| Ok(s.cummax(reverse)),
            GetOutput::same_type(),
        )
    }

    /// Fill missing value with next non-null.
    pub fn backward_fill(self) -> Self {
        self.apply(
            move |s: Series| s.fill_null(FillNullStrategy::Backward),
            GetOutput::same_type(),
        )
    }

    /// Fill missing value with previous non-null.
    pub fn forward_fill(self) -> Self {
        self.apply(
            move |s: Series| s.fill_null(FillNullStrategy::Forward),
            GetOutput::same_type(),
        )
    }

    /// Round underlying floating point array to given decimal numbers.
    #[cfg(feature = "round_series")]
    #[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
    pub fn round(self, decimals: u32) -> Self {
        self.map(move |s: Series| s.round(decimals), GetOutput::same_type())
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
    ///          sum("values").over(vec![col("groups")]),
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
    pub fn over(self, partition_by: Vec<Expr>) -> Self {
        Expr::Window {
            function: Box::new(self),
            partition_by,
            order_by: None,
            options: WindowOptions { explode: false },
        }
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::fill_null).
    pub fn fill_null(self, fill_value: Expr) -> Self {
        map_binary_lazy_field(
            self,
            fill_value,
            |a, b| {
                if a.null_count() == 0 {
                    Ok(a)
                } else {
                    let st = get_supertype(a.dtype(), b.dtype())?;
                    let a = a.cast_with_dtype(&st)?;
                    let b = b.cast_with_dtype(&st)?;
                    let mask = a.is_not_null();
                    a.zip_with_same_type(&mask, &b)
                }
            },
            |_schema, _ctx, a, b| {
                let st = get_supertype(a.data_type(), b.data_type()).unwrap();
                Some(Field::new(a.name(), st))
            },
        )
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
        let name = expr_to_root_column_name(&self).unwrap();
        let output_field = Some(Field::new(&name, DataType::Boolean));
        map_binary(
            self,
            other,
            move |left, other| {
                left.is_in(&other).map(|ca| {
                    let mut s = ca.into_series();
                    s.rename(&name);
                    s
                })
            },
            output_field,
        )
    }

    /// Get the year of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn year(self) -> Expr {
        let function = move |s: Series| s.year().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }

    /// Get the month of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn month(self) -> Expr {
        let function = move |s: Series| s.month().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }
    /// Extract the week from the underlying Date representation.
    /// Can be performed on Date32 and Date64

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    #[cfg(feature = "temporal")]
    pub fn week(self) -> Expr {
        let function = move |s: Series| s.week().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }

    /// Extract the week day from the underlying Date representation.
    /// Can be performed on Date32 and Date64.

    /// Returns the weekday number where monday = 0 and sunday = 6
    #[cfg(feature = "temporal")]
    pub fn weekday(self) -> Expr {
        let function = move |s: Series| s.weekday().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }

    /// Get the month of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn day(self) -> Expr {
        let function = move |s: Series| s.day().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }
    /// Get the ordinal_day of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn ordinal_day(self) -> Expr {
        let function = move |s: Series| s.ordinal_day().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }
    /// Get the hour of a Date64/Time64
    #[cfg(feature = "temporal")]
    pub fn hour(self) -> Expr {
        let function = move |s: Series| s.hour().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }
    /// Get the minute of a Date64/Time64
    #[cfg(feature = "temporal")]
    pub fn minute(self) -> Expr {
        let function = move |s: Series| s.minute().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }

    /// Get the second of a Date64/Time64
    #[cfg(feature = "temporal")]
    pub fn second(self) -> Expr {
        let function = move |s: Series| s.second().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }
    /// Get the nanosecond of a Time64
    #[cfg(feature = "temporal")]
    pub fn nanosecond(self) -> Expr {
        let function = move |s: Series| s.nanosecond().map(|ca| ca.into_series());
        self.map(function, GetOutput::from_type(DataType::UInt32))
    }

    /// Sort this column by the ordering of another column.
    /// Can also be used in a groupby context to sort the groups.
    pub fn sort_by(self, by: Expr, reverse: bool) -> Expr {
        Expr::SortBy {
            expr: Box::new(self),
            by: Box::new(by),
            reverse,
        }
    }

    #[cfg(feature = "repeat_by")]
    #[cfg_attr(docsrs, doc(cfg(feature = "repeat_by")))]
    pub fn repeat_by(self, by: Expr) -> Expr {
        let function = |s: Series, by: Series| {
            let by = by.cast_with_dtype(&DataType::UInt32)?;
            Ok(s.repeat_by(by.u32()?).into_series())
        };
        map_binary_lazy_field(self, by, function, |_schema, _ctxt, l, _r| {
            Some(Field::new(
                l.name(),
                DataType::List(l.data_type().to_arrow()),
            ))
        })
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
    }

    #[cfg(feature = "dot_product")]
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_product")))]
    pub fn dot(self, other: Expr) -> Expr {
        let function = |s: Series, other: Series| Ok((&s * &other).sum_as_series());

        map_binary_lazy_field(self, other, function, |_schema, _ctxt, l, _r| {
            Some(Field::new(l.name(), l.data_type().clone()))
        })
    }

    #[cfg(feature = "mode")]
    #[cfg_attr(docsrs, doc(cfg(feature = "mode")))]
    /// Compute the mode(s) of this column. This is the most occurring value.
    pub fn mode(self) -> Expr {
        self.apply(
            |s| s.mode().map(|ca| ca.into_series()),
            GetOutput::same_type(),
        )
    }

    /// Keep the original root name
    ///
    /// ```
    /// use polars_core::prelude::*;
    /// use polars_lazy::prelude::*;
    ///
    /// fn example(df: LazyFrame) -> LazyFrame {
    ///     df.select(vec![
    /// // even thought the alias yields a different column name,
    /// // `keep_name` will make sure that the original column name is used
    ///         col("*").alias("foo").keep_name()
    /// ])
    /// }
    /// ```
    pub fn keep_name(self) -> Expr {
        Expr::KeepName(Box::new(self))
    }

    /// Add a suffix to the root column name.
    pub fn suffix(self, suffix: &str) -> Expr {
        Expr::SufPreFix {
            is_suffix: true,
            value: suffix.to_string(),
            expr: Box::new(self),
        }
    }

    /// Add a prefix to the root column name.
    pub fn prefix(self, prefix: &str) -> Expr {
        Expr::SufPreFix {
            is_suffix: false,
            value: prefix.to_string(),
            expr: Box::new(self),
        }
    }

    /// Exclude a column from a wildcard selection
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
    pub fn exclude<'a, S, J>(self, columns: S) -> Expr
    where
        S: Selection<'a, J>,
    {
        let v = columns
            .to_selection_vec()
            .iter()
            .map(|s| Arc::new(s.to_string()))
            .collect();
        Expr::Exclude(Box::new(self), v)
    }

    // Interpolate None values
    #[cfg(feature = "interpolate")]
    #[cfg_attr(docsrs, doc(cfg(feature = "interpolate")))]
    pub fn interpolate(self) -> Expr {
        self.apply(|s| Ok(s.interpolate()), GetOutput::same_type())
    }

    /// Apply a rolling min See:
    /// [ChunkedArray::rolling_min](polars::prelude::ChunkWindow::rolling_min).
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_min(
        self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Expr {
        let weight = weight.map(|v| v.to_vec());
        self.apply(
            move |s| s.rolling_min(window_size, weight.as_deref(), ignore_null, min_periods),
            GetOutput::same_type(),
        )
    }

    /// Apply a rolling max See:
    /// [ChunkedArray::rolling_max](polars::prelude::ChunkWindow::rolling_max).
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_max(
        self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Expr {
        let weight = weight.map(|v| v.to_vec());
        self.apply(
            move |s| s.rolling_max(window_size, weight.as_deref(), ignore_null, min_periods),
            GetOutput::same_type(),
        )
    }

    /// Apply a rolling mean See:
    /// [ChunkedArray::rolling_mean](polars::prelude::ChunkWindow::rolling_mean).
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_mean(
        self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Expr {
        let weight = weight.map(|v| v.to_vec());
        self.apply(
            move |s| s.rolling_mean(window_size, weight.as_deref(), ignore_null, min_periods),
            GetOutput::same_type(),
        )
    }

    /// Apply a rolling sum See:
    /// [ChunkedArray::rolling_sum](polars::prelude::ChunkWindow::rolling_sum).
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_sum(
        self,
        window_size: u32,
        weight: Option<&[f64]>,
        ignore_null: bool,
        min_periods: u32,
    ) -> Expr {
        let weight = weight.map(|v| v.to_vec());
        self.apply(
            move |s| s.rolling_sum(window_size, weight.as_deref(), ignore_null, min_periods),
            GetOutput::same_type(),
        )
    }

    /// Apply a rolling variance
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_var(self, window_size: usize) -> Expr {
        self.apply(
            move |s| match s.dtype() {
                DataType::Float32 => Ok(s.f32().unwrap().rolling_var(window_size).into_series()),
                DataType::Float64 => Ok(s.f64().unwrap().rolling_var(window_size).into_series()),
                _ => Ok(s
                    .cast_with_dtype(&DataType::Float64)?
                    .f64()
                    .unwrap()
                    .rolling_var(window_size)
                    .into_series()),
            },
            GetOutput::map_field(|field| match field.data_type() {
                DataType::Float64 => field.clone(),
                DataType::Float32 => Field::new(field.name(), DataType::Float32),
                _ => Field::new(field.name(), DataType::Float64),
            }),
        )
    }

    /// Apply a rolling std-dev
    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    pub fn rolling_std(self, window_size: usize) -> Expr {
        self.apply(
            move |s| match s.dtype() {
                DataType::Float32 => Ok(s.f32().unwrap().rolling_std(window_size).into_series()),
                DataType::Float64 => Ok(s.f64().unwrap().rolling_std(window_size).into_series()),
                _ => Ok(s
                    .cast_with_dtype(&DataType::Float64)?
                    .f64()
                    .unwrap()
                    .rolling_std(window_size)
                    .into_series()),
            },
            GetOutput::map_field(|field| match field.data_type() {
                DataType::Float64 => field.clone(),
                DataType::Float32 => Field::new(field.name(), DataType::Float32),
                _ => Field::new(field.name(), DataType::Float64),
            }),
        )
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "rolling_window")))]
    #[cfg(feature = "rolling_window")]
    /// Apply a custom function over a rolling/ moving window of the array.
    /// This has quite some dynamic dispatch, so prefer rolling_min, max, mean, sum over this.
    pub fn rolling_apply(
        self,
        window_size: usize,
        f: Arc<dyn Fn(&Series) -> Series + Send + Sync>,
        output_type: GetOutput,
    ) -> Expr {
        self.apply(
            move |s| s.rolling_apply(window_size, f.as_ref()),
            output_type,
        )
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
                        .cast::<Float64Type>()?
                        .f64()
                        .unwrap()
                        .rolling_apply_float(window_size, f)
                        .map(|ca| ca.into_series()),
                }?;
                if let DataType::Float32 = s.dtype() {
                    out.cast_with_dtype(&DataType::Float32)
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
    }

    #[cfg(feature = "rank")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rank")))]
    pub fn rank(self, method: RankMethod) -> Expr {
        self.apply(
            move |s| Ok(s.rank(method)),
            GetOutput::map_field(move |fld| match method {
                RankMethod::Average => Field::new(fld.name(), DataType::Float32),
                _ => Field::new(fld.name(), DataType::UInt32),
            }),
        )
    }

    #[cfg(feature = "diff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    pub fn diff(self, n: usize, null_behavior: NullBehavior) -> Expr {
        self.apply(
            move |s| Ok(s.diff(n, null_behavior)),
            GetOutput::same_type(),
        )
    }

    #[cfg(feature = "moment")]
    #[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
    pub fn skew(self, bias: bool) -> Expr {
        self.apply(
            move |s| s.skew(bias).map(|opt_v| Series::new(s.name(), &[opt_v])),
            GetOutput::from_type(DataType::Float64),
        )
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
    }
}

/// Create a Column Expression based on a column name.
///
/// # Arguments
///
/// * `name` - A string slice that holds the name of the column
///
/// # Examples
///
/// ```ignore
/// // select a column name
/// col("foo")
/// ```
///
/// ```ignore
/// // select all columns by using a wildcard
/// col("*")
/// ```
///
/// ```ignore
/// // select specific column by writing a regular expression that starts with `^` and ends with `$`
/// // only if regex features is activated
/// col("^foo.*$")
/// ```
pub fn col(name: &str) -> Expr {
    match name {
        "*" => Expr::Wildcard,
        _ => Expr::Column(Arc::new(name.to_owned())),
    }
}

/// Select multiple columns by name
pub fn cols(names: Vec<String>) -> Expr {
    Expr::Columns(names)
}

/// Count the number of values in this Expression.
pub fn count(name: &str) -> Expr {
    match name {
        "" => col(name).count().alias("count"),
        _ => col(name).count(),
    }
}

/// Sum all the values in this Expression.
pub fn sum(name: &str) -> Expr {
    col(name).sum()
}

/// Find the minimum of all the values in this Expression.
pub fn min(name: &str) -> Expr {
    col(name).min()
}

/// Find the maximum of all the values in this Expression.
pub fn max(name: &str) -> Expr {
    col(name).max()
}

/// Find the mean of all the values in this Expression.
pub fn mean(name: &str) -> Expr {
    col(name).mean()
}

/// Find the mean of all the values in this Expression.
pub fn avg(name: &str) -> Expr {
    col(name).mean()
}

/// Find the median of all the values in this Expression.
pub fn median(name: &str) -> Expr {
    col(name).median()
}

/// Find a specific quantile of all the values in this Expression.
pub fn quantile(name: &str, quantile: f64) -> Expr {
    col(name).quantile(quantile)
}

/// Apply a closure on the two columns that are evaluated from `Expr` a and `Expr` b.
pub fn map_binary<F: 'static>(a: Expr, b: Expr, f: F, output_field: Option<Field>) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync,
{
    let output_field = move |_: &Schema, _: Context, _: &Field, _: &Field| output_field.clone();

    Expr::BinaryFunction {
        input_a: Box::new(a),
        input_b: Box::new(b),
        function: NoEq::new(Arc::new(f)),
        output_field: NoEq::new(Arc::new(output_field)),
    }
}

/// Binary function where the output type is determined at runtime when the schema is known.
pub fn map_binary_lazy_field<F: 'static, Fld: 'static>(
    a: Expr,
    b: Expr,
    f: F,
    output_field: Fld,
) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync,
    Fld: Fn(&Schema, Context, &Field, &Field) -> Option<Field> + Send + Sync,
{
    Expr::BinaryFunction {
        input_a: Box::new(a),
        input_b: Box::new(b),
        function: NoEq::new(Arc::new(f)),
        output_field: NoEq::new(Arc::new(output_field)),
    }
}

/// Accumulate over multiple columns horizontally / row wise.
pub fn fold_exprs<F: 'static>(mut acc: Expr, f: F, mut exprs: Vec<Expr>) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync + Clone,
{
    if exprs.iter().any(has_wildcard) {
        exprs.push(acc);

        let function = NoEq::new(Arc::new(move |series: &mut [Series]| {
            let mut series = series.to_vec();
            let mut acc = series.pop().unwrap();

            for s in series {
                acc = f(acc, s)?
            }
            Ok(acc)
        }) as Arc<dyn SeriesUdf>);

        // Todo! make sure that output type is correct
        Expr::Function {
            input: exprs,
            function,
            output_type: GetOutput::same_type(),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: true,
            },
        }
    } else {
        for e in exprs {
            acc = map_binary_lazy_field(
                acc,
                e,
                f.clone(),
                // written inline due to lifetime inference issues.
                |_schema, _ctxt, f_l: &Field, f_r: &Field| {
                    get_supertype(f_l.data_type(), f_r.data_type())
                        .ok()
                        .map(|dt| Field::new(f_l.name(), dt))
                },
            );
        }
        acc
    }
}

/// Get the the sum of the values per row
pub fn sum_exprs(exprs: Vec<Expr>) -> Expr {
    let func = |s1, s2| Ok(&s1 + &s2);
    fold_exprs(lit(0), func, exprs)
}

/// Get the the minimum value per row
pub fn max_exprs(exprs: Vec<Expr>) -> Expr {
    let func = |s1: Series, s2: Series| {
        let mask = s1.gt(&s2);
        s1.zip_with(&mask, &s2)
    };
    fold_exprs(lit(0), func, exprs)
}

/// Get the the minimum value per row
pub fn min_exprs(exprs: Vec<Expr>) -> Expr {
    let func = |s1: Series, s2: Series| {
        let mask = s1.lt(&s2);
        s1.zip_with(&mask, &s2)
    };
    fold_exprs(lit(0), func, exprs)
}

/// Evaluate all the expressions with a bitwise or
pub fn any_exprs(exprs: Vec<Expr>) -> Expr {
    let func = |s1: Series, s2: Series| Ok(s1.bool()?.bitor(s2.bool()?).into_series());
    fold_exprs(lit(false), func, exprs)
}

/// Evaluate all the expressions with a bitwise and
pub fn all_exprs(exprs: Vec<Expr>) -> Expr {
    let func = |s1: Series, s2: Series| Ok(s1.bool()?.bitand(s2.bool()?).into_series());
    fold_exprs(lit(true), func, exprs)
}

pub trait Literal {
    /// [Literal](Expr::Literal) expression.
    fn lit(self) -> Expr;
}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Utf8(self))
    }
}

impl<'a> Literal for &'a str {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Utf8(self.to_owned()))
    }
}

macro_rules! make_literal {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(LiteralValue::$SCALAR(self))
            }
        }
    };
}

make_literal!(bool, Boolean);
make_literal!(f32, Float32);
make_literal!(f64, Float64);
#[cfg(feature = "dtype-i8")]
make_literal!(i8, Int8);
#[cfg(feature = "dtype-i16")]
make_literal!(i16, Int16);
make_literal!(i32, Int32);
make_literal!(i64, Int64);
#[cfg(feature = "dtype-u8")]
make_literal!(u8, UInt8);
#[cfg(feature = "dtype-u16")]
make_literal!(u16, UInt16);
make_literal!(u32, UInt32);
#[cfg(feature = "dtype-u64")]
make_literal!(u64, UInt64);

/// The literal Null
pub struct Null {}

impl Literal for Null {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Null)
    }
}

#[cfg(all(feature = "temporal", feature = "dtype-date64"))]
impl Literal for NaiveDateTime {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::DateTime(self))
    }
}

#[cfg(all(feature = "temporal", feature = "dtype-date64"))]
impl Literal for NaiveDate {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::DateTime(self.and_hms(0, 0, 0)))
    }
}

impl Literal for Series {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Series(NoEq::new(self)))
    }
}

/// Create a Literal Expression from `L`
pub fn lit<L: Literal>(t: L) -> Expr {
    t.lit()
}

/// [Not](Expr::Not) expression.
pub fn not(expr: Expr) -> Expr {
    Expr::Not(Box::new(expr))
}

/// [IsNull](Expr::IsNotNull) expression
pub fn is_null(expr: Expr) -> Expr {
    Expr::IsNull(Box::new(expr))
}

/// [IsNotNull](Expr::IsNotNull) expression.
pub fn is_not_null(expr: Expr) -> Expr {
    Expr::IsNotNull(Box::new(expr))
}

/// [Cast](Expr::Cast) expression.
pub fn cast(expr: Expr, data_type: DataType) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        data_type,
        strict: false,
    }
}

pub trait Range<T> {
    fn into_range(self, high: T) -> Expr;
}

macro_rules! impl_into_range {
    ($dt: ty) => {
        impl Range<$dt> for $dt {
            fn into_range(self, high: $dt) -> Expr {
                Expr::Literal(LiteralValue::Range {
                    low: self as i64,
                    high: high as i64,
                    data_type: DataType::Int32,
                })
            }
        }
    };
}

impl_into_range!(i32);
impl_into_range!(i64);
impl_into_range!(u32);

/// Create a range literal.
pub fn range<T: Range<T>>(low: T, high: T) -> Expr {
    low.into_range(high)
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

#[cfg(test)]
mod test {
    use super::*;
    use polars_core::df;

    #[test]
    #[cfg(feature = "is_in")]
    fn test_is_in() -> Result<()> {
        let df = df![
            "x" => [1, 2, 3],
            "y" => ["a", "b", "c"]
        ]?;
        let s = Series::new("a", ["a", "b"]);

        let out = df
            .lazy()
            .select([col("y").is_in(lit(s)).alias("isin")])
            .collect()?;
        assert_eq!(
            Vec::from(out.column("isin")?.bool()?),
            &[Some(true), Some(true), Some(false)]
        );
        Ok(())
    }

    #[test]
    fn test_when_then_when_then() {
        let e = when(col("a"))
            .then(col("b"))
            .when(col("c"))
            .then(col("d"))
            .otherwise(col("f"));

        dbg!(e);
    }
}
