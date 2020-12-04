//! Domain specific language for the Lazy api.
use crate::frame::group_by::{fmt_groupby_column, GroupByMethod};
use crate::lazy::logical_plan::Context;
use crate::lazy::utils::{output_name, rename_field};
use crate::{lazy::prelude::*, prelude::*, utils::get_supertype};
use arrow::datatypes::{Field, Schema};
use std::{
    fmt,
    ops::{Add, Div, Mul, Rem, Sub},
    sync::Arc,
};

pub trait Udf: Send + Sync {
    fn call_udf(&self, s: Series) -> Result<Series>;
}

impl<F> Udf for F
where
    F: Fn(Series) -> Result<Series> + Send + Sync,
{
    fn call_udf(&self, s: Series) -> Result<Series> {
        self(s)
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
    },
    Sum(Box<Expr>),
    AggGroups(Box<Expr>),
    Ternary {
        predicate: Box<Expr>,
        truthy: Box<Expr>,
        falsy: Box<Expr>,
    },
    Apply {
        input: Box<Expr>,
        function: Arc<dyn Udf>,
        output_type: Option<ArrowDataType>,
    },
    Shift {
        input: Box<Expr>,
        periods: i32,
    },
    Reverse(Box<Expr>),
    Duplicated(Box<Expr>),
    Unique(Box<Expr>),
    /// See postgres window functions
    Window {
        /// Also has the input. i.e. avg("foo")
        function: Box<Expr>,
        partition_by: Box<Expr>,
        order_by: Option<Box<Expr>>,
    },
    Wildcard,
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

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Expr::Duplicated(left) => impl_partial_eq!(Duplicated, left, other),
            Expr::Unique(left) => impl_partial_eq!(Unique, left, other),
            Expr::Reverse(left) => impl_partial_eq!(Reverse, left, other),
            Expr::Mean(left) => impl_partial_eq!(Mean, left, other),
            Expr::Median(left) => impl_partial_eq!(Median, left, other),
            Expr::First(left) => impl_partial_eq!(First, left, other),
            Expr::Last(left) => impl_partial_eq!(Last, left, other),
            Expr::AggGroups(left) => impl_partial_eq!(AggGroups, left, other),
            Expr::NUnique(left) => impl_partial_eq!(NUnique, left, other),
            Expr::Min(left) => impl_partial_eq!(Min, left, other),
            Expr::Max(left) => impl_partial_eq!(Max, left, other),
            Expr::Sum(left) => impl_partial_eq!(Sum, left, other),
            Expr::List(left) => impl_partial_eq!(List, left, other),
            Expr::Count(left) => impl_partial_eq!(Count, left, other),
            Expr::Column(left) => impl_partial_eq!(Column, left, other),
            Expr::Literal(left) => impl_partial_eq!(Literal, left, other),
            Expr::Wildcard => matches!(other, Expr::Wildcard),
            Expr::Quantile { expr, quantile } => {
                let left = expr;
                let left_q = quantile;
                if let Expr::Quantile { expr, quantile } = other {
                    if left_q == quantile {
                        left.eq(expr)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Expr::Apply { .. } => false,
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
        }
    }
}

fn get_field_by_context(
    expr: &Expr,
    schema: &Schema,
    ctxt: Context,
    groupby_method: GroupByMethod,
) -> Result<Field> {
    let mut field = expr.to_field(schema, ctxt)?;
    if &ArrowDataType::Boolean == field.data_type() {
        field = Field::new(field.name(), ArrowDataType::UInt32, field.is_nullable())
    }

    match ctxt {
        Context::Other => Ok(field),
        Context::Aggregation => {
            let new_name = fmt_groupby_column(field.name(), groupby_method);
            Ok(rename_field(&field, &new_name))
        }
    }
}

impl Expr {
    /// Get DataType result of the expression. The schema is the input data.
    pub fn get_type(&self, schema: &Schema) -> Result<ArrowDataType> {
        use Expr::*;
        match self {
            Window { function, .. } => function.get_type(schema),
            Unique(_) => Ok(ArrowDataType::Boolean),
            Duplicated(_) => Ok(ArrowDataType::Boolean),
            Reverse(expr) => expr.get_type(schema),
            Alias(expr, ..) => expr.get_type(schema),
            Column(name) => Ok(schema.field_with_name(name)?.data_type().clone()),
            Literal(sv) => Ok(sv.get_datatype()),
            BinaryExpr { left, op, right } => match op {
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
                | Operator::Like => Ok(ArrowDataType::Boolean),
                _ => {
                    let left_type = left.get_type(schema)?;
                    let right_type = right.get_type(schema)?;
                    get_supertype(&left_type, &right_type)
                }
            },
            Not(_) => Ok(ArrowDataType::Boolean),
            IsNull(_) => Ok(ArrowDataType::Boolean),
            IsNotNull(_) => Ok(ArrowDataType::Boolean),
            Sort { expr, .. } => expr.get_type(schema),
            Min(expr) => expr.get_type(schema),
            Max(expr) => expr.get_type(schema),
            Sum(expr) => expr.get_type(schema),
            First(expr) => expr.get_type(schema),
            Last(expr) => expr.get_type(schema),
            Count(expr) => expr.get_type(schema),
            List(expr) => Ok(ArrowDataType::List(Box::new(expr.get_type(schema)?))),
            Mean(expr) => expr.get_type(schema),
            Median(expr) => expr.get_type(schema),
            AggGroups(_) => Ok(ArrowDataType::List(Box::new(ArrowDataType::UInt32))),
            NUnique(_) => Ok(ArrowDataType::UInt32),
            Quantile { expr, .. } => expr.get_type(schema),
            Cast { data_type, .. } => Ok(data_type.clone()),
            Ternary { truthy, .. } => truthy.get_type(schema),
            Apply {
                input, output_type, ..
            } => match output_type {
                Some(output_type) => Ok(output_type.clone()),
                None => input.get_type(schema),
            },
            Shift { input, .. } => input.get_type(schema),
            Wildcard => panic!("should be no wildcard at this point"),
        }
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
            Alias(expr, name) => Ok(Field::new(name, expr.get_type(schema)?, true)),
            Column(name) => {
                let field = schema.field_with_name(name).map(|f| f.clone())?;
                Ok(field)
            }
            Literal(sv) => Ok(Field::new("lit", sv.get_datatype(), true)),
            BinaryExpr { left, right, op } => {
                let left_type = left.get_type(schema)?;
                let right_type = right.get_type(schema)?;
                let expr_type = get_supertype(&left_type, &right_type)?;

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
            Min(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::Min),
            Max(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::Max),
            Median(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::Median),
            Mean(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::Mean),
            First(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::First),
            Last(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::Last),
            List(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::List),
            NUnique(expr) => {
                let field = expr.to_field(schema, ctxt)?;
                let field = Field::new(field.name(), ArrowDataType::UInt32, field.is_nullable());
                match ctxt {
                    Context::Other => Ok(field),
                    Context::Aggregation => {
                        let new_name = fmt_groupby_column(field.name(), GroupByMethod::NUnique);
                        Ok(rename_field(&field, &new_name))
                    }
                }
            }
            Sum(expr) => get_field_by_context(expr, schema, ctxt, GroupByMethod::Sum),
            Count(expr) => {
                let field = expr.to_field(schema, ctxt)?;
                let field = Field::new(field.name(), ArrowDataType::UInt32, field.is_nullable());
                match ctxt {
                    Context::Other => Ok(field),
                    Context::Aggregation => {
                        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Count);
                        Ok(rename_field(&field, &new_name))
                    }
                }
            }
            AggGroups(expr) => {
                let field = expr.to_field(schema, ctxt)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Groups);
                let new_field = Field::new(
                    &new_name,
                    ArrowDataType::List(Box::new(ArrowDataType::UInt32)),
                    field.is_nullable(),
                );
                Ok(new_field)
            }
            Quantile { expr, quantile } => {
                get_field_by_context(expr, schema, ctxt, GroupByMethod::Quantile(*quantile))
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
            Apply {
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
            Shift { input, .. } => input.to_field(schema, ctxt),
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
            Duplicated(expr) => write!(f, "DUPLICATED {:?}", expr),
            Reverse(expr) => write!(f, "REVERSE {:?}", expr),
            Alias(expr, name) => write!(f, "{:?} AS {}", expr, name),
            Column(name) => write!(f, "COLUMN {}", name),
            Literal(v) => write!(f, "LIT {:?}", v),
            BinaryExpr { left, op, right } => write!(f, "[({:?}) {:?} ({:?})]", left, op, right),
            Not(expr) => write!(f, "NOT {:?}", expr),
            IsNull(expr) => write!(f, "{:?} IS NULL", expr),
            IsNotNull(expr) => write!(f, "{:?} IS NOT NULL", expr),
            Sort { expr, reverse } => match reverse {
                true => write!(f, "{:?} DESC", expr),
                false => write!(f, "{:?} ASC", expr),
            },
            Min(expr) => write!(f, "AGGREGATE MIN {:?}", expr),
            Max(expr) => write!(f, "AGGREGATE MAX {:?}", expr),
            Median(expr) => write!(f, "AGGREGATE MEDIAN {:?}", expr),
            Mean(expr) => write!(f, "AGGREGATE MEAN {:?}", expr),
            First(expr) => write!(f, "AGGREGATE FIRST {:?}", expr),
            Last(expr) => write!(f, "AGGREGATE LAST {:?}", expr),
            List(expr) => write!(f, "AGGREGATE LIST {:?}", expr),
            NUnique(expr) => write!(f, "AGGREGATE N UNIQUE {:?}", expr),
            Sum(expr) => write!(f, "AGGREGATE SUM {:?}", expr),
            AggGroups(expr) => write!(f, "AGGREGATE GROUPS {:?}", expr),
            Count(expr) => write!(f, "AGGREGATE COUNT {:?}", expr),
            Quantile { expr, .. } => write!(f, "AGGREGATE QUANTILE {:?}", expr),
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
            Apply { input, .. } => write!(f, "APPLY({:?})", input),
            Shift { input, periods, .. } => write!(f, "SHIFT {:?} by {}", input, periods),
            Wildcard => write!(f, "*"),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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

impl From<u8> for Operator {
    fn from(op: u8) -> Self {
        match op {
            0 => Operator::Eq,
            1 => Operator::NotEq,
            2 => Operator::Lt,
            3 => Operator::LtEq,
            4 => Operator::Gt,
            5 => Operator::GtEq,
            6 => Operator::Plus,
            7 => Operator::Minus,
            8 => Operator::Multiply,
            9 => Operator::Divide,
            10 => Operator::Modulus,
            11 => Operator::And,
            12 => Operator::Or,
            13 => Operator::Not,
            14 => Operator::Like,
            15 => Operator::NotLike,
            _ => panic!("not an operator"),
        }
    }
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
        Expr::Min(Box::new(self))
    }

    /// Reduce groups to maximum value.
    pub fn max(self) -> Self {
        Expr::Max(Box::new(self))
    }

    /// Reduce groups to the mean value.
    pub fn mean(self) -> Self {
        Expr::Mean(Box::new(self))
    }

    /// Reduce groups to the median value.
    pub fn median(self) -> Self {
        Expr::Median(Box::new(self))
    }

    /// Reduce groups to the sum of all the values.
    pub fn sum(self) -> Self {
        Expr::Sum(Box::new(self))
    }

    /// Get the number of unique values in the groups.
    pub fn n_unique(self) -> Self {
        Expr::NUnique(Box::new(self))
    }

    /// Get the first value in the group.
    pub fn first(self) -> Self {
        Expr::First(Box::new(self))
    }

    /// Get the last value in the group.
    pub fn last(self) -> Self {
        Expr::Last(Box::new(self))
    }

    /// Aggregate the group to a Series
    pub fn list(self) -> Self {
        Expr::List(Box::new(self))
    }

    /// Compute the quantile per group.
    pub fn quantile(self, quantile: f64) -> Self {
        Expr::Quantile {
            expr: Box::new(self),
            quantile,
        }
    }

    /// Get the group indexes of the group by operation.
    pub fn agg_groups(self) -> Self {
        Expr::AggGroups(Box::new(self))
    }

    /// Cast expression to another data type.
    pub fn cast(self, data_type: ArrowDataType) -> Self {
        Expr::Cast {
            expr: Box::new(self),
            data_type,
        }
    }

    /// Sort expression. See [the eager implementation](Series::sort).
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
    pub fn apply<F>(self, function: F, output_type: Option<ArrowDataType>) -> Self
    where
        F: Udf + 'static,
    {
        Expr::Apply {
            input: Box::new(self),
            function: Arc::new(function),
            output_type,
        }
    }

    /// Shift the values in the array by some period. See [the eager implementation](Series::shift).
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

    /// Shift the values in the array by some period. See [the eager implementation](Series::fill_none).
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
        Expr::Count(Box::new(self))
    }

    /// Standard deviation of the values of the Series
    pub fn std(self) -> Self {
        let function = move |s: Series| {
            s.std_as_series()
                .cast_with_arrow_datatype(&ArrowDataType::Float64)
        };
        self.apply(function, Some(ArrowDataType::Float64))
    }

    /// Variance of the values of the Series
    pub fn var(self) -> Self {
        let function = move |s: Series| {
            s.var_as_series()
                .cast_with_arrow_datatype(&ArrowDataType::Float64)
        };
        self.apply(function, Some(ArrowDataType::Float64))
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
