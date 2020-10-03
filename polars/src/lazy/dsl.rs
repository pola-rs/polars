use crate::frame::group_by::{fmt_groupby_column, GroupByMethod};
use crate::lazy::utils::{get_supertype, rename_field};
use crate::{lazy::prelude::*, prelude::*};
use arrow::datatypes::{Field, Schema};
use std::fmt;
use std::rc::Rc;

#[derive(Clone)]
pub enum Expr {
    Alias(Box<Expr>, Rc<String>),
    Column(Rc<String>),
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
    // Cast {
    //     expr: Box<Expr>,
    //     data_type: ArrowDataType,
    // },
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
    AggGroups(Box<Expr>), // ScalarFunction {
                          //     name: String,
                          //     args: Vec<Expr>,
                          //     return_type: ArrowDataType,
                          // },
                          // Wildcard
}

impl Expr {
    /// Get DataType result of the expression. The schema is the input data.
    fn get_type(&self, schema: &Schema) -> Result<ArrowDataType> {
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
                Ok(rename_field(&field, &new_name))
            }
            AggSum(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Sum);
                Ok(rename_field(&field, &new_name))
            }
            AggGroups(expr) => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Groups);
                Ok(rename_field(&field, &new_name))
            }
            AggQuantile { expr, quantile } => {
                let field = expr.to_field(schema)?;
                let new_name = fmt_groupby_column(field.name(), GroupByMethod::Quantile(*quantile));
                Ok(rename_field(&field, &new_name))
            }
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;
        match self {
            Alias(expr, name) => write!(f, "{:?} AS {}", expr, name),
            Column(name) => write!(f, "COLUMN {}", name),
            Literal(v) => write!(f, "{:?}", v),
            BinaryExpr { left, op, right } => write!(f, "{:?} {:?} {:?}", left, op, right),
            Not(expr) => write!(f, "NOT {:?}", expr),
            IsNull(expr) => write!(f, "{:?} IS NULL", expr),
            IsNotNull(expr) => write!(f, "{:?} IS NOT NULL", expr),
            Sort { expr, reverse } => match reverse {
                true => write!(f, "{:?} DESC", expr),
                false => write!(f, "{:?} ASC", expr),
            },
            AggMin(expr) => write!(f, "AGGREGATE MIN {:?}", expr),
        }
    }
}

fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
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
        Expr::Alias(Box::new(self), Rc::new(name.into()))
    }

    /// Run is_null operation on `Expr`.
    pub fn is_null(self) -> Self {
        Expr::IsNull(Box::new(self))
    }

    /// Run is_not_null operation on `Expr`.
    pub fn is_not_null(self) -> Self {
        Expr::IsNotNull(Box::new(self))
    }

    /// Reduce column to minimal value.
    pub fn agg_min(self) -> Self {
        Expr::AggMin(Box::new(self))
    }
}

/// Create a Colum Expression based on a column name.
pub fn col(name: &str) -> Expr {
    Expr::Column(Rc::new(name.to_owned()))
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

/// [Not](Expr::Not) expression
pub fn not(expr: Expr) -> Expr {
    Expr::Not(Box::new(expr))
}

/// [IsNull](Expr::IsNotNull) expression
pub fn is_null(expr: Expr) -> Expr {
    Expr::IsNull(Box::new(expr))
}

/// [IsNotNull](Expr::IsNotNull) expression
pub fn is_not_null(expr: Expr) -> Expr {
    Expr::IsNotNull(Box::new(expr))
}
