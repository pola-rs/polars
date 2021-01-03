//! Domain specific language for the Lazy api.
use crate::frame::group_by::{fmt_groupby_column, GroupByMethod};
use crate::lazy::logical_plan::Context;
use crate::lazy::utils::{output_name, rename_field};
use crate::{lazy::prelude::*, prelude::*, utils::get_supertype};
use arrow::datatypes::{Field, Schema};
use std::fmt::{Debug, Formatter};
use std::{
    fmt,
    ops::{Add, Div, Mul, Rem, Sub},
    sync::Arc,
};

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
#[derive(Clone)]
pub enum Expr {
    Alias(Box<Expr>, Arc<String>),
    Column(Arc<String>),
    Literal(ScalarValue),
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
        data_type: ArrowDataType,
    },
    Sort {
        expr: Box<Expr>,
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
        function: Arc<dyn SeriesUdf>,
        output_type: Option<ArrowDataType>,
    },
    Shift {
        input: Box<Expr>,
        periods: i32,
    },
    Reverse(Box<Expr>),
    Duplicated(Box<Expr>),
    Unique(Box<Expr>),
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
        offset: isize,
        length: usize,
    },
    BinaryFunction {
        input_a: Box<Expr>,
        input_b: Box<Expr>,
        function: Arc<dyn SeriesBinaryUdf>,
        /// Delays output type evaluation until input schema is known.
        output_field: Arc<dyn BinaryUdfOutputField>,
    },
}

macro_rules! impl_partial_eq {
    ($variant:ident, $left:expr, $other:expr) => {{
        if let Expr::$variant(right) = $other {
            $left.eq(right)
        } else {
            false
        }
    }};
}
macro_rules! impl_partial_eq_agg {
    ($variant:ident, $left:expr, $other:expr) => {{
        if let AggExpr::$variant(right) = $other {
            $left.eq(right)
        } else {
            false
        }
    }};
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Expr::Duplicated(left) => impl_partial_eq!(Duplicated, left, other),
            Expr::Unique(left) => impl_partial_eq!(Unique, left, other),
            Expr::Reverse(left) => impl_partial_eq!(Reverse, left, other),
            Expr::Explode(left) => impl_partial_eq!(Explode, left, other),
            Expr::Agg(agg) => {
                if let Expr::Agg(other) = other {
                    match agg {
                        AggExpr::Mean(left) => impl_partial_eq_agg!(Mean, left, other),
                        AggExpr::Median(left) => impl_partial_eq_agg!(Median, left, other),
                        AggExpr::First(left) => impl_partial_eq_agg!(First, left, other),
                        AggExpr::Last(left) => impl_partial_eq_agg!(Last, left, other),
                        AggExpr::AggGroups(left) => impl_partial_eq_agg!(AggGroups, left, other),
                        AggExpr::NUnique(left) => impl_partial_eq_agg!(NUnique, left, other),
                        AggExpr::Min(left) => impl_partial_eq_agg!(Min, left, other),
                        AggExpr::Max(left) => impl_partial_eq_agg!(Max, left, other),
                        AggExpr::Sum(left) => impl_partial_eq_agg!(Sum, left, other),
                        AggExpr::List(left) => impl_partial_eq_agg!(List, left, other),
                        AggExpr::Count(left) => impl_partial_eq_agg!(Count, left, other),
                        AggExpr::Std(left) => impl_partial_eq_agg!(Std, left, other),
                        AggExpr::Var(left) => impl_partial_eq_agg!(Var, left, other),
                        AggExpr::Quantile { expr, quantile } => {
                            let left = expr;
                            let left_q = quantile;
                            if let AggExpr::Quantile { expr, quantile } = other {
                                if left_q == quantile {
                                    left.eq(expr)
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                    }
                } else {
                    false
                }
            }
            Expr::Column(left) => impl_partial_eq!(Column, left, other),
            Expr::Literal(left) => impl_partial_eq!(Literal, left, other),
            Expr::Wildcard => matches!(other, Expr::Wildcard),
            Expr::Udf { .. } => false,
            Expr::BinaryFunction { .. } => false,
            Expr::BinaryExpr { .. } => false, // todo: should it?
            Expr::Ternary { .. } => false,
            Expr::IsNull(left) => impl_partial_eq!(IsNull, left, other),
            Expr::IsNotNull(left) => impl_partial_eq!(IsNotNull, left, other),
            Expr::Not(left) => impl_partial_eq!(Not, left, other),
            Expr::Alias(left, name) => {
                if let Expr::Alias(right, r_name) = other {
                    if name == r_name {
                        left.eq(right)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Expr::Cast { expr, data_type } => {
                let left = expr;
                let ldtype = data_type;
                if let Expr::Cast { expr, data_type } = other {
                    let right = expr;
                    let rdtype = data_type;
                    if ldtype == rdtype {
                        left.eq(right)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Expr::Sort { expr, reverse } => {
                let left = expr;
                let lreverse = reverse;
                if let Expr::Sort { expr, reverse } = other {
                    let right = expr;
                    let rreverse = reverse;
                    if lreverse == rreverse {
                        left.eq(right)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Expr::Window {
                function,
                partition_by,
                order_by,
            } => {
                let function_left = function;
                let partition_by_left = partition_by;
                let order_by_left = order_by;
                if let Expr::Window {
                    function,
                    partition_by,
                    order_by,
                } = other
                {
                    function_left.eq(function)
                        && partition_by_left.eq(partition_by)
                        && order_by_left.eq(order_by)
                } else {
                    false
                }
            }
            Expr::Shift { input, periods } => {
                let left = input;
                let lperiods = periods;
                if let Expr::Shift { input, periods } = other {
                    let right = input;
                    let rperiods = periods;
                    if lperiods == rperiods {
                        left.eq(right)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Expr::Slice {
                input: left,
                offset: offset_left,
                length: length_left,
            } => {
                if let Expr::Slice {
                    input,
                    offset,
                    length,
                } = other
                {
                    offset == offset_left && length == length_left && left.eq(input)
                } else {
                    false
                }
            }
        }
    }
}

impl Expr {
    /// Get DataType result of the expression. The schema is the input data.
    pub fn get_type(&self, schema: &Schema, context: Context) -> Result<ArrowDataType> {
        self.to_field(schema, context)
            .map(|f| f.data_type().clone())
    }

    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(&self, schema: &Schema, ctxt: Context) -> Result<Field> {
        use Expr::*;
        match self {
            Window { function, .. } => function.to_field(schema, ctxt),
            Unique(expr) => {
                let field = expr.to_field(&schema, ctxt)?;
                Ok(Field::new(field.name(), ArrowDataType::Boolean, true))
            }
            Duplicated(expr) => {
                let field = expr.to_field(&schema, ctxt)?;
                Ok(Field::new(field.name(), ArrowDataType::Boolean, true))
            }
            Reverse(expr) => expr.to_field(&schema, ctxt),
            Explode(expr) => expr.to_field(&schema, ctxt),
            Alias(expr, name) => Ok(Field::new(name, expr.get_type(schema, ctxt)?, true)),
            Column(name) => {
                let field = schema.field_with_name(name).map(|f| f.clone())?;
                Ok(field)
            }
            Literal(sv) => Ok(Field::new("lit", sv.get_datatype(), true)),
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
                    | Operator::Like => ArrowDataType::Boolean,
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

                Ok(Field::new(out_name, expr_type, true))
            }
            Not(_) => Ok(Field::new("not", ArrowDataType::Boolean, true)),
            IsNull(_) => Ok(Field::new("is_null", ArrowDataType::Boolean, true)),
            IsNotNull(_) => Ok(Field::new("is_not_null", ArrowDataType::Boolean, true)),
            Sort { expr, .. } => expr.to_field(schema, ctxt),
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
                        let field =
                            Field::new(field.name(), ArrowDataType::UInt32, field.is_nullable());
                        match ctxt {
                            Context::Other => field,
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
                        let field =
                            Field::new(field.name(), ArrowDataType::Float64, field.is_nullable());
                        field_by_context(field, ctxt, GroupByMethod::Std)
                    }
                    Var(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let field =
                            Field::new(field.name(), ArrowDataType::Float64, field.is_nullable());
                        field_by_context(field, ctxt, GroupByMethod::Var)
                    }
                    Count(expr) => {
                        let field = expr.to_field(schema, ctxt)?;
                        let field =
                            Field::new(field.name(), ArrowDataType::UInt32, field.is_nullable());
                        match ctxt {
                            Context::Other => field,
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
                        Field::new(
                            &new_name,
                            ArrowDataType::List(Box::new(ArrowDataType::UInt32)),
                            field.is_nullable(),
                        )
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
                Ok(Field::new(
                    field.name(),
                    data_type.clone(),
                    field.is_nullable(),
                ))
            }
            Ternary { truthy, .. } => truthy.to_field(schema, ctxt),
            Udf {
                output_type, input, ..
            } => match output_type {
                None => input.to_field(schema, ctxt),
                Some(output_type) => {
                    let input_field = input.to_field(schema, ctxt)?;
                    Ok(Field::new(
                        input_field.name(),
                        output_type.clone(),
                        input_field.is_nullable(),
                    ))
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
                output_field
                    .get_field(schema, ctxt, &field_a, &field_b)
                    .ok_or_else(|| panic!("field expected"))
            }
            Shift { input, .. } => input.to_field(schema, ctxt),
            Slice { input, .. } => input.to_field(schema, ctxt),
            Wildcard => panic!("should be no wildcard at this point"),
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
                "{:?} OVER (PARTION BY {:?} ORDER BY {:?}",
                function, partition_by, order_by
            ),
            Unique(expr) => write!(f, "UNIQUE {:?}", expr),
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
        }
    }
}

#[derive(Debug, Copy, Clone)]
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
    pub fn slice(self, offset: isize, length: usize) -> Self {
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
        self.slice(-(len as isize), len)
    }

    /// Cast expression to another data type.
    pub fn cast(self, data_type: ArrowDataType) -> Self {
        Expr::Cast {
            expr: Box::new(self),
            data_type,
        }
    }

    /// Sort expression. See [the eager implementation](crate::series::SeriesTrait::sort).
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
    pub fn map<F>(self, function: F, output_type: Option<ArrowDataType>) -> Self
    where
        F: SeriesUdf + 'static,
    {
        Expr::Udf {
            input: Box::new(self),
            function: Arc::new(function),
            output_type,
        }
    }

    /// Shift the values in the array by some period. See [the eager implementation](crate::series::SeriesTrait::shift).
    pub fn shift(self, periods: i32) -> Self {
        Expr::Shift {
            input: Box::new(self),
            periods,
        }
    }

    /// Apply window function over a subgroup.
    /// This is similar to a groupby + aggregation + self join.
    /// Or similar to [window functions in Postgres](https://www.postgresql.org/docs/9.1/tutorial-window.html).
    ///
    /// # Example
    ///
    /// ``` rust
    /// #[macro_use] extern crate polars;
    /// use polars::prelude::*;
    /// use polars::lazy::dsl::*;
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

    /// Shift the values in the array by some period. See [the eager implementation](crate::series::SeriesTrait::fill_none).
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
        Expr::Unique(Box::new(self))
    }

    /// and operation
    pub fn and(self, expr: Expr) -> Self {
        binary_expr(self, Operator::And, expr)
    }

    /// Raise expression to the power `exponent`
    pub fn pow(self, exponent: f64) -> Self {
        self.map(
            move |s: Series| s.pow(exponent),
            Some(ArrowDataType::Float64),
        )
    }
}

/// Create a Column Expression based on a column name.
pub fn col(name: &str) -> Expr {
    match name {
        "*" => Expr::Wildcard,
        _ => Expr::Column(Arc::new(name.to_owned())),
    }
}

pub fn count(name: &str) -> Expr {
    col(name).count()
}
pub fn sum(name: &str) -> Expr {
    col(name).sum()
}

pub fn min(name: &str) -> Expr {
    col(name).min()
}

pub fn max(name: &str) -> Expr {
    col(name).max()
}

pub fn mean(name: &str) -> Expr {
    col(name).mean()
}

pub fn avg(name: &str) -> Expr {
    col(name).mean()
}

pub fn median(name: &str) -> Expr {
    col(name).median()
}

pub fn quantile(name: &str, quantile: f64) -> Expr {
    col(name).quantile(quantile)
}

pub fn pow(expr: Expr, exponent: f64) -> Expr {
    expr.pow(exponent)
}

pub trait Literal {
    /// [Literal](Expr::Literal) expression.
    fn lit(self) -> Expr;
}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(ScalarValue::Utf8(self))
    }
}

impl<'a> Literal for &'a str {
    fn lit(self) -> Expr {
        Expr::Literal(ScalarValue::Utf8(self.to_owned()))
    }
}

macro_rules! make_literal {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(ScalarValue::$SCALAR(self))
            }
        }
    };
}

make_literal!(bool, Boolean);
make_literal!(f32, Float32);
make_literal!(f64, Float64);
make_literal!(i8, Int8);
make_literal!(i16, Int16);
make_literal!(i32, Int32);
make_literal!(i64, Int64);
make_literal!(u8, UInt8);
make_literal!(u16, UInt16);
make_literal!(u32, UInt32);
make_literal!(u64, UInt64);

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
pub fn cast(expr: Expr, data_type: ArrowDataType) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        data_type,
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
