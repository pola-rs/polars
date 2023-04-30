mod binary;
mod categorical;
mod datetime;
mod general;
mod list;
#[cfg(feature = "meta")]
mod meta;
mod string;
mod r#struct;

use polars::lazy::dsl;
use polars::lazy::dsl::Operator;
use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

impl From<dsl::Expr> for PyExpr {
    fn from(expr: dsl::Expr) -> Self {
        PyExpr { inner: expr }
    }
}

pub fn binary_expr(l: PyExpr, op: u8, r: PyExpr) -> PyExpr {
    let left = l.inner;
    let right = r.inner;

    let op = match op {
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
        _ => panic!("not an operator"),
    };

    dsl::binary_expr(left, op, right).into()
}

pub fn range(low: i64, high: i64, dtype: Wrap<DataType>) -> PyExpr {
    match dtype.0 {
        DataType::Int32 => dsl::range(low as i32, high as i32).into(),
        DataType::UInt32 => dsl::range(low as u32, high as u32).into(),
        _ => dsl::range(low, high).into(),
    }
}
