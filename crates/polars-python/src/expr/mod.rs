#[cfg(feature = "pymethods")]
mod array;
#[cfg(feature = "pymethods")]
mod binary;
#[cfg(feature = "pymethods")]
mod categorical;
#[cfg(feature = "pymethods")]
mod datetime;
#[cfg(feature = "pymethods")]
mod general;
#[cfg(feature = "pymethods")]
mod list;
#[cfg(all(feature = "meta", feature = "pymethods"))]
mod meta;
#[cfg(feature = "pymethods")]
mod name;
#[cfg(feature = "pymethods")]
mod rolling;
#[cfg(feature = "pymethods")]
mod serde;
#[cfg(feature = "pymethods")]
mod string;
#[cfg(feature = "pymethods")]
mod r#struct;

use std::mem::ManuallyDrop;

use polars::lazy::dsl::Expr;
use pyo3::pyclass;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: Expr,
}

impl From<Expr> for PyExpr {
    fn from(expr: Expr) -> Self {
        PyExpr { inner: expr }
    }
}

pub(crate) trait ToExprs {
    fn to_exprs(self) -> Vec<Expr>;
}

impl ToExprs for Vec<PyExpr> {
    fn to_exprs(self) -> Vec<Expr> {
        // SAFETY: repr is transparent.
        unsafe {
            let length = self.len();
            let capacity = self.capacity();
            let mut manual_drop_vec = ManuallyDrop::new(self);
            let vec_ptr: *mut PyExpr = manual_drop_vec.as_mut_ptr();
            let ptr: *mut Expr = vec_ptr as *mut Expr;
            Vec::from_raw_parts(ptr, length, capacity)
        }
    }
}

pub(crate) trait ToPyExprs {
    fn to_pyexprs(self) -> Vec<PyExpr>;
}

impl ToPyExprs for Vec<Expr> {
    fn to_pyexprs(self) -> Vec<PyExpr> {
        // SAFETY: repr is transparent.
        unsafe {
            let length = self.len();
            let capacity = self.capacity();
            let mut manual_drop_vec = ManuallyDrop::new(self);
            let vec_ptr: *mut Expr = manual_drop_vec.as_mut_ptr();
            let ptr: *mut PyExpr = vec_ptr as *mut PyExpr;
            Vec::from_raw_parts(ptr, length, capacity)
        }
    }
}
