use crate::lazy::prelude::*;
use std::rc::Rc;

#[derive(Clone, Debug)]
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
    // IsNotNull(Box<Expr>),
    // IsNull(Box<Expr>),
    // Cast {
    //     expr: Box<Expr>,
    //     data_type: ArrowDataType,
    // },
    Sort {
        expr: Box<Expr>,
        reverse: bool,
    },
    // ScalarFunction {
    //     name: String,
    //     args: Vec<Expr>,
    //     return_type: ArrowDataType,
    // },
    // AggregateFunction {
    //     name: String,
    //     args: Vec<Expr>,
    // },
    // Wildcard,
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

    pub fn alias(self, name: &str) -> Expr {
        Expr::Alias(Box::new(self), Rc::new(name.into()))
    }
}

/// Create a Colum Expression based on a column name.
pub fn col(name: &str) -> Expr {
    Expr::Column(Rc::new(name.to_owned()))
}

pub trait Literal {
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

pub fn not(expr: Expr) -> Expr {
    Expr::Not(Box::new(expr))
}
