//! Domain specific language for the Lazy api.
use crate::frame::group_by::{fmt_groupby_column, GroupByMethod};
use crate::lazy::utils::{get_supertype, rename_field};
use crate::{lazy::prelude::*, prelude::*};
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
    AggMin(Box<Expr>),
    AggMax(Box<Expr>),
    AggMedian(Box<Expr>),
    AggNUnique(Box<Expr>),
    AggFirst(Box<Expr>),
    AggLast(Box<Expr>),
    AggMean(Box<Expr>),
    AggQuantile {
        expr: Box<Expr>,
        quantile: f64,
    },
    AggSum(Box<Expr>),
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
    }, // ScalarFunction {
       //     name: String,
       //     args: Vec<Expr>,
       //     return_type: ArrowDataType,
       // },
       // Wildcard
}

impl Expr {
    /// Get DataType result of the expression. The schema is the input data.
    pub fn get_type(&self, schema: &Schema) -> Result<ArrowDataType> {
        use Expr::*;
        match self {
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
            AggMin(expr) => expr.get_type(schema),
            AggMax(expr) => expr.get_type(schema),
            AggSum(expr) => expr.get_type(schema),
            AggFirst(expr) => expr.get_type(schema),
            AggLast(expr) => expr.get_type(schema),
            AggMean(expr) => expr.get_type(schema),
            AggMedian(expr) => expr.get_type(schema),
            AggGroups(_) => Ok(ArrowDataType::UInt32),
            AggNUnique(_) => Ok(ArrowDataType::UInt32),
            AggQuantile { expr, .. } => expr.get_type(schema),
            Cast { data_type, .. } => Ok(data_type.clone()),
            Ternary { truthy, .. } => truthy.get_type(schema),
            Apply {
                input, output_type, ..
            } => match output_type {
                Some(output_type) => Ok(output_type.clone()),
                None => input.get_type(schema),
            },
            Shift { input, .. } => input.get_type(schema),
        }
    }

    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(&self, schema: &Schema) -> Result<Field> {
        use Expr::*;
        match self {
            Alias(expr, name) => Ok(Field::new(name, expr.get_type(schema)?, true)),
            Column(name) => {
                let field = schema.field_with_name(name).map(|f| f.clone())?;
                Ok(field)
            }
            Literal(sv) => Ok(Field::new("lit", sv.get_datatype(), true)),
            BinaryExpr { left, right, .. } => {
                let left_type = left.get_type(schema)?;
                let right_type = right.get_type(schema)?;
                let expr_type = get_supertype(&left_type, &right_type)?;
                Ok(Field::new("binary_expr", expr_type, true))
            }
            Not(_) => Ok(Field::new("not", ArrowDataType::Boolean, true)),
            IsNull(_) => Ok(Field::new("is_null", ArrowDataType::Boolean, true)),
            IsNotNull(_) => Ok(Field::new("is_not_null", ArrowDataType::Boolean, true)),
            Sort { expr, .. } => expr.to_field(schema),
            AggMin(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Min);
                Ok(rename_field(&field, &new_name))
            }
            AggMax(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Max);
                Ok(rename_field(&field, &new_name))
            }
            AggMedian(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Median);
                Ok(rename_field(&field, &new_name))
            }
            AggMean(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Mean);
                Ok(rename_field(&field, &new_name))
            }
            AggFirst(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::First);
                Ok(rename_field(&field, &new_name))
            }
            AggLast(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Last);
                Ok(rename_field(&field, &new_name))
            }
            AggNUnique(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::NUnique);
                let new_field = Field::new(&new_name, ArrowDataType::UInt32, field.is_nullable());
                Ok(new_field)
            }
            AggSum(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Sum);
                Ok(rename_field(&field, &new_name))
            }
            AggGroups(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Groups);
                let new_field = Field::new(&new_name, ArrowDataType::UInt32, field.is_nullable());
                Ok(new_field)
            }
            AggQuantile { expr, quantile } => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Quantile(*quantile));
                Ok(rename_field(&field, &new_name))
            }
            Cast { expr, data_type } => {
                let field = expr.to_field(schema)?;
                Ok(Field::new(
                    field.name(),
                    data_type.clone(),
                    field.is_nullable(),
                ))
            }
            Ternary { truthy, .. } => truthy.to_field(schema),
            Apply {
                output_type, input, ..
            } => match output_type {
                None => input.to_field(schema),
                Some(output_type) => {
                    let input_field = input.to_field(schema)?;
                    Ok(Field::new(
                        input_field.name(),
                        output_type.clone(),
                        input_field.is_nullable(),
                    ))
                }
            },
            Shift { input, .. } => input.to_field(schema),
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;
        match self {
            Alias(expr, name) => write!(f, "{:?} AS {}", expr, name),
            Column(name) => write!(f, "COLUMN {}", name),
            Literal(v) => write!(f, "LIT {:?}", v),
            BinaryExpr { left, op, right } => write!(f, "{:?} {:?} {:?}", left, op, right),
            Not(expr) => write!(f, "NOT {:?}", expr),
            IsNull(expr) => write!(f, "{:?} IS NULL", expr),
            IsNotNull(expr) => write!(f, "{:?} IS NOT NULL", expr),
            Sort { expr, reverse } => match reverse {
                true => write!(f, "{:?} DESC", expr),
                false => write!(f, "{:?} ASC", expr),
            },
            AggMin(expr) => write!(f, "AGGREGATE MIN {:?}", expr),
            AggMax(expr) => write!(f, "AGGREGATE MAX {:?}", expr),
            AggMedian(expr) => write!(f, "AGGREGATE MEDIAN {:?}", expr),
            AggMean(expr) => write!(f, "AGGREGATE MEAN {:?}", expr),
            AggFirst(expr) => write!(f, "AGGREGATE FIRST {:?}", expr),
            AggLast(expr) => write!(f, "AGGREGATE LAST {:?}", expr),
            AggNUnique(expr) => write!(f, "AGGREGATE N UNIQUE {:?}", expr),
            AggSum(expr) => write!(f, "AGGREGATE SUM {:?}", expr),
            AggGroups(expr) => write!(f, "AGGREGATE GROUPS {:?}", expr),
            AggQuantile { expr, .. } => write!(f, "AGGREGATE QUANTILE {:?}", expr),
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
    pub fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }

    /// Rename Column.
    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), Arc::new(name.into()))
    }

    /// Run is_null operation on `Expr`.
    pub fn is_null(self) -> Self {
        Expr::IsNull(Box::new(self))
    }

    /// Run is_not_null operation on `Expr`.
    pub fn is_not_null(self) -> Self {
        Expr::IsNotNull(Box::new(self))
    }

    /// Reduce groups to minimal value.
    pub fn agg_min(self) -> Self {
        Expr::AggMin(Box::new(self))
    }

    /// Reduce groups to maximum value.
    pub fn agg_max(self) -> Self {
        Expr::AggMax(Box::new(self))
    }

    /// Reduce groups to the mean value.
    pub fn agg_mean(self) -> Self {
        Expr::AggMean(Box::new(self))
    }

    /// Reduce groups to the median value.
    pub fn agg_median(self) -> Self {
        Expr::AggMedian(Box::new(self))
    }

    /// Reduce groups to the sum of all the values.
    pub fn agg_sum(self) -> Self {
        Expr::AggSum(Box::new(self))
    }

    /// Get the number of unique values in the groups.
    pub fn agg_n_unique(self) -> Self {
        Expr::AggNUnique(Box::new(self))
    }

    /// Get the first value in the group.
    pub fn agg_first(self) -> Self {
        Expr::AggFirst(Box::new(self))
    }

    /// Get the last value in the group.
    pub fn agg_last(self) -> Self {
        Expr::AggLast(Box::new(self))
    }

    /// Compute the quantile per group.
    pub fn agg_quantile(self, quantile: f64) -> Self {
        Expr::AggQuantile {
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

    /// Shift the values in the array by some period. See [the eager implementation](Series::fill_none).
    pub fn fill_none(self, strategy: FillNoneStrategy) -> Self {
        let function = move |s: Series| s.fill_none(strategy);
        self.apply(function, None)
    }

    /// Get the maximum value of the Series.
    pub fn max(self) -> Self {
        let function = move |s: Series| Ok(s.max_as_series());
        self.apply(function, None)
    }

    /// Get the minimum value of the Series.
    pub fn min(self) -> Self {
        let function = move |s: Series| Ok(s.min_as_series());
        self.apply(function, None)
    }

    /// Get the sum value of the Series.
    pub fn sum(self) -> Self {
        let function = move |s: Series| Ok(s.sum_as_series());
        self.apply(function, None)
    }

    /// Get the mean value of the Series.
    pub fn mean(self) -> Self {
        let function = move |s: Series| Ok(s.mean_as_series());
        self.apply(function, None)
    }

    /// Get the median value of the Series.
    pub fn median(self) -> Self {
        let function = move |s: Series| Ok(s.median_as_series());
        self.apply(function, None)
    }

    /// Get the quantile value of the Series.
    pub fn quantile(self, quantile: f64) -> Self {
        let function = move |s: Series| s.quantile_as_series(quantile);
        self.apply(function, None)
    }
}

/// Create a Column Expression based on a column name.
pub fn col(name: &str) -> Expr {
    Expr::Column(Arc::new(name.to_owned()))
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
