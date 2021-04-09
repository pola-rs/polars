//! Domain specific language for the Lazy api.
use crate::logical_plan::Context;
use crate::prelude::*;
use crate::utils::{output_name, rename_field};
use polars_core::{
    frame::groupby::{fmt_groupby_column, GroupByMethod},
    prelude::*,
    utils::get_supertype,
};

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

pub trait SeriesUdf: Send + Sync {
    fn call_udf(&self, s: Series) -> Result<Series>;
}

impl<F> SeriesUdf for F
where
    F: Fn(Series) -> Result<Series> + Send + Sync,
{
    fn call_udf(&self, s: Series) -> Result<Series> {
        self(s)
    }
}

impl Debug for dyn SeriesUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesUdf")
    }
}

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
        _input_schema: &Schema,
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
    Literal(LiteralValue),
    BinaryExpr {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    // Nested(Box<Expr>),
    Not(Box<Expr>),
    IsNotNull(Box<Expr>),
    IsNull(Box<Expr>),
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
    },
    Sort {
        expr: Box<Expr>,
        reverse: bool,
    },
    SortBy {
        expr: Box<Expr>,
        by: Box<Expr>,
        reverse: bool,
    },
    Agg(AggExpr),
    Ternary {
        predicate: Box<Expr>,
        truthy: Box<Expr>,
        falsy: Box<Expr>,
    },
    Udf {
        input: Box<Expr>,
        function: NoEq<Arc<dyn SeriesUdf>>,
        output_type: Option<DataType>,
    },
    Shift {
        input: Box<Expr>,
        periods: i64,
    },
    Reverse(Box<Expr>),
    Duplicated(Box<Expr>),
    IsUnique(Box<Expr>),
    Explode(Box<Expr>),
    /// See postgres window functions
    Window {
        /// Also has the input. i.e. avg("foo")
        function: Box<Expr>,
        partition_by: Box<Expr>,
        order_by: Option<Box<Expr>>,
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
    Except(Box<Expr>),
}

impl Expr {
    /// Get DataType result of the expression. The schema is the input data.
    pub fn get_type(&self, schema: &Schema, context: Context) -> Result<DataType> {
        self.to_field(schema, context)
            .map(|f| f.data_type().clone())
    }

    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(&self, schema: &Schema, ctxt: Context) -> Result<Field> {
        use Expr::*;
        match self {
            Window { function, .. } => function.to_field(schema, ctxt),
            IsUnique(expr) => {
                let field = expr.to_field(&schema, ctxt)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Duplicated(expr) => {
                let field = expr.to_field(&schema, ctxt)?;
                Ok(Field::new(field.name(), DataType::Boolean))
            }
            Reverse(expr) => expr.to_field(&schema, ctxt),
            Explode(expr) => expr.to_field(&schema, ctxt),
            Alias(expr, name) => Ok(Field::new(name, expr.get_type(schema, ctxt)?)),
            Column(name) => {
                let field = schema.field_with_name(name).map(|f| f.clone())?;
                Ok(field)
            }
            Literal(sv) => Ok(Field::new("lit", sv.get_datatype())),
            BinaryExpr { left, right, op } => {
                let left_type = left.get_type(schema, ctxt)?;
                let right_type = right.get_type(schema, ctxt)?;

                let expr_type = match op {
                    Operator::Not
                    | Operator::Lt
                    | Operator::Gt
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::And
                    | Operator::LtEq
                    | Operator::GtEq
                    | Operator::Or
                    | Operator::NotLike
                    | Operator::Like => DataType::Boolean,
                    _ => get_supertype(&left_type, &right_type)?,
                };

                use Operator::*;
                let out_field;
                let out_name = match op {
                    Plus | Minus | Multiply | Divide | Modulus => {
                        out_field = left.to_field(schema, ctxt)?;
                        out_field.name().as_str()
                    }
                    Eq | Lt | GtEq | LtEq => "",
                    _ => "binary_expr",
                };

                Ok(Field::new(out_name, expr_type))
            }
            Not(_) => Ok(Field::new("not", DataType::Boolean)),
            IsNull(_) => Ok(Field::new("is_null", DataType::Boolean)),
            IsNotNull(_) => Ok(Field::new("is_not_null", DataType::Boolean)),
            Sort { expr, .. } => expr.to_field(schema, ctxt),
            SortBy { expr, .. } => expr.to_field(schema, ctxt),
            Agg(agg) => {
                use AggExpr::*;
                let field = match agg {
                    Min(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::Min)
                    }
                    Max(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::Max)
                    }
                    Median(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::Median)
                    }
                    Mean(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::Mean)
                    }
                    First(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::First)
                    }
                    Last(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::Last)
                    }
                    List(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::List)
                    }
                    NUnique(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let field = Field::new(field.name(), DataType::UInt32);
                        match ctxt {
                            Context::Default => field,
                            Context::Aggregation => {
                                let new_name =
                                    fmt_groupby_column(field.name(), GroupByMethod::NUnique);
                                rename_field(&field, &new_name)
                            }
                        }
                    }
                    Sum(expr) => {
                        field_by_context(expr.to_field(schema, ctxt)?, ctxt, GroupByMethod::Sum)
                    }
                    Std(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let field = Field::new(field.name(), DataType::Float64);
                        field_by_context(field, ctxt, GroupByMethod::Std)
                    }
                    Var(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let field = Field::new(field.name(), DataType::Float64);
                        field_by_context(field, ctxt, GroupByMethod::Var)
                    }
                    Count(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let field = Field::new(field.name(), DataType::UInt32);
                        match ctxt {
                            Context::Default => field,
                            Context::Aggregation => {
                                let new_name =
                                    fmt_groupby_column(field.name(), GroupByMethod::Count);
                                rename_field(&field, &new_name)
                            }
                        }
                    }
                    AggGroups(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Groups);
                        Field::new(&new_name, DataType::List(ArrowDataType::UInt32))
                    }
                    Quantile { expr, quantile } => field_by_context(
                        expr.to_field(schema, ctxt)?,
                        ctxt,
                        GroupByMethod::Quantile(*quantile),
                    ),
                };
                Ok(field)
            }
            Cast { expr, data_type } => {
                let field = expr.to_field(schema, ctxt)?;
                Ok(Field::new(field.name(), data_type.clone()))
            }
            Ternary { truthy, .. } => truthy.to_field(schema, ctxt),
            Udf {
                output_type, input, ..
            } => match output_type {
                None => input.to_field(schema, ctxt),
                Some(output_type) => {
                    let input_field = input.to_field(schema, ctxt)?;
                    Ok(Field::new(input_field.name(), output_type.clone()))
                }
            },
            BinaryFunction {
                input_a,
                input_b,
                output_field,
                ..
            } => {
                let field_a = input_a.to_field(schema, ctxt)?;
                let field_b = input_b.to_field(schema, ctxt)?;
                // if field is unknown we try to guess a return type. May fail.
                Ok(output_field
                    .get_field(schema, ctxt, &field_a, &field_b)
                    .unwrap_or_else(|| {
                        Field::new(
                            "binary_expr",
                            get_supertype(field_a.data_type(), field_b.data_type())
                                .unwrap_or(DataType::Null),
                        )
                    }))
            }
            Shift { input, .. } => input.to_field(schema, ctxt),
            Slice { input, .. } => input.to_field(schema, ctxt),
            Wildcard => panic!("should be no wildcard at this point"),
            Except(_) => panic!("should be no except at this point"),
        }
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
            Literal(v) => write!(f, "{:?}", v),
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
            Cast { expr, data_type } => write!(f, "CAST {:?} TO {:?}", expr, data_type),
            Ternary {
                predicate,
                truthy,
                falsy,
            } => write!(
                f,
                "\nWHEN {:?}\n\t{:?}\nOTHERWISE\n\t{:?}",
                predicate, truthy, falsy
            ),
            Udf { input, .. } => write!(f, "APPLY({:?})", input),
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
            Except(column) => write!(f, "EXCEPT {:?}", column),
        }
    }
}

/// Exclude a column from selection.
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
///                 col("*"), except("foo")
///                 ])
/// }
/// ```
pub fn except(name: &str) -> Expr {
    match name {
        "*" => panic!("cannot use a wildcard as a column exception"),
        _ => Expr::Except(Box::new(col(name))),
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
    Modulus,
    And,
    Or,
    Not,
    Like,
    NotLike,
}

pub fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

pub struct When {
    predicate: Expr,
}

pub struct WhenThen {
    predicate: Expr,
    then: Expr,
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
    pub fn otherwise(self, expr: Expr) -> Expr {
        Expr::Ternary {
            predicate: Box::new(self.predicate),
            truthy: Box::new(self.then),
            falsy: Box::new(expr),
        }
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

    /// Explode the utf8/ list column
    pub fn explode(self) -> Self {
        Expr::Explode(Box::new(self))
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

    /// Cast expression to another data type.
    pub fn cast(self, data_type: DataType) -> Self {
        Expr::Cast {
            expr: Box::new(self),
            data_type,
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
    /// It is the responsibility of the caller that the schema is correct by giving
    /// the correct output_type. If None given the output type of the input expr is used.
    pub fn map<F>(self, function: F, output_type: Option<DataType>) -> Self
    where
        F: SeriesUdf + 'static,
    {
        Expr::Udf {
            input: Box::new(self),
            function: NoEq::new(Arc::new(function)),
            output_type,
        }
    }

    /// Get mask of finite values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_finite(self) -> Self {
        self.map(
            |s: Series| s.is_finite().map(|ca| ca.into_series()),
            Some(DataType::Boolean),
        )
    }

    /// Get mask of infinite values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_infinite(self) -> Self {
        self.map(
            |s: Series| s.is_infinite().map(|ca| ca.into_series()),
            Some(DataType::Boolean),
        )
    }

    /// Get mask of NaN values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_nan(self) -> Self {
        self.map(
            |s: Series| s.is_nan().map(|ca| ca.into_series()),
            Some(DataType::Boolean),
        )
    }

    /// Get inverse mask of NaN values if dtype is Float
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_nan(self) -> Self {
        self.map(
            |s: Series| s.is_not_nan().map(|ca| ca.into_series()),
            Some(DataType::Boolean),
        )
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::shift).
    pub fn shift(self, periods: i64) -> Self {
        Expr::Shift {
            input: Box::new(self),
            periods,
        }
    }

    /// Shift the valus in the array by some period and fill the resulting empty values.
    pub fn shift_and_fill(self, periods: i64, fill_value: Expr) -> Self {
        // let name = output_name(&self).unwrap();
        // Note:
        // The order of the then | otherwise is im
        if periods > 0 {
            when(self.clone().map(
                move |s: Series| {
                    let ca: BooleanChunked = (0..s.len() as i64).map(|i| i >= periods).collect();
                    Ok(ca.into_series())
                },
                Some(DataType::Boolean),
            ))
            .then(self.shift(periods))
            .otherwise(fill_value)
            // .alias(&name)
        } else {
            when(self.clone().map(
                move |s: Series| {
                    let length = s.len() as i64;
                    // periods is negative, so subtraction.
                    let tipping_point = length + periods;
                    let ca: BooleanChunked = (0..length).map(|i| i < tipping_point).collect();
                    Ok(ca.into_series())
                },
                Some(DataType::Boolean),
            ))
            .then(self.shift(periods))
            .otherwise(fill_value)
            // .alias(&name)
        }
    }

    /// Get an array with the cumulative sum computed at every element
    pub fn cum_sum(self, reverse: bool) -> Self {
        self.map(move |s: Series| Ok(s.cum_sum(reverse)), None)
    }

    /// Get an array with the cumulative min computed at every element
    pub fn cum_min(self, reverse: bool) -> Self {
        self.map(move |s: Series| Ok(s.cum_min(reverse)), None)
    }

    /// Get an array with the cumulative max computed at every element
    pub fn cum_max(self, reverse: bool) -> Self {
        self.map(move |s: Series| Ok(s.cum_max(reverse)), None)
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
    ///          sum("values").over(col("groups")),
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
    pub fn over(self, partition_by: Expr) -> Self {
        Expr::Window {
            function: Box::new(self),
            partition_by: Box::new(partition_by),
            order_by: None,
        }
    }

    /// Shift the values in the array by some period. See [the eager implementation](polars_core::series::SeriesTrait::fill_none).
    pub fn fill_none(self, fill_value: Expr) -> Self {
        let name = output_name(&self).unwrap();
        when(self.is_null())
            .then(fill_value)
            .otherwise(col(&*name))
            .alias(&*name)
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
        self.map(move |s: Series| s.pow(exponent), Some(DataType::Float64))
    }

    /// Check if the values of the left expression are in the lists of the right expr.
    #[allow(clippy::wrong_self_convention)]
    pub fn is_in(self, list_expr: Expr) -> Self {
        map_binary(
            self,
            list_expr,
            |left, list_ex| {
                let list_array = list_ex.list()?;
                left.is_in(list_array).map(|ca| ca.into_series())
            },
            Some(Field::new("", DataType::Boolean)),
        )
    }

    /// Get the year of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn year(self) -> Expr {
        let function = move |s: Series| s.year().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }

    /// Get the month of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn month(self) -> Expr {
        let function = move |s: Series| s.month().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }
    /// Get the month of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn day(self) -> Expr {
        let function = move |s: Series| s.day().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }
    /// Get the ordinal_day of a Date32/Date64
    #[cfg(feature = "temporal")]
    pub fn ordinal_day(self) -> Expr {
        let function = move |s: Series| s.ordinal_day().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }
    /// Get the hour of a Date64/Time64
    #[cfg(feature = "temporal")]
    pub fn hour(self) -> Expr {
        let function = move |s: Series| s.hour().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }
    /// Get the minute of a Date64/Time64
    #[cfg(feature = "temporal")]
    pub fn minute(self) -> Expr {
        let function = move |s: Series| s.minute().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }

    /// Get the second of a Date64/Time64
    #[cfg(feature = "temporal")]
    pub fn second(self) -> Expr {
        let function = move |s: Series| s.second().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
    }
    /// Get the nanosecond of a Time64
    #[cfg(feature = "temporal")]
    pub fn nanosecond(self) -> Expr {
        let function = move |s: Series| s.nanosecond().map(|ca| ca.into_series());
        self.map(function, Some(DataType::UInt32))
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
}

/// Create a Column Expression based on a column name.
pub fn col(name: &str) -> Expr {
    match name {
        "*" => Expr::Wildcard,
        _ => Expr::Column(Arc::new(name.to_owned())),
    }
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
pub fn fold_exprs<F: 'static>(mut acc: Expr, f: F, exprs: Vec<Expr>) -> Expr
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync + Copy,
{
    for e in exprs {
        acc = map_binary(acc, e, f, None);
    }
    acc
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
