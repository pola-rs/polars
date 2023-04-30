mod apply;
pub mod dataframe;
pub mod dsl;
#[cfg(feature = "meta")]
mod meta;
pub mod utils;

pub use apply::*;
use dsl::*;
use polars_lazy::prelude::*;

pub(crate) trait ToExprs {
    fn to_exprs(self) -> Vec<Expr>;
}

impl ToExprs for Vec<PyExpr> {
    fn to_exprs(self) -> Vec<Expr> {
        // Safety
        // repr is transparent
        // and has only got one inner field`
        unsafe { std::mem::transmute(self) }
    }
}

pub(crate) trait ToPyExprs {
    fn to_pyexprs(self) -> Vec<PyExpr>;
}

impl ToPyExprs for Vec<Expr> {
    fn to_pyexprs(self) -> Vec<PyExpr> {
        // Safety
        // repr is transparent
        // and has only got one inner field`
        unsafe { std::mem::transmute(self) }
    }
}
