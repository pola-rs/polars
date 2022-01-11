use crate::prelude::*;

impl From<AggExpr> for Expr {
    fn from(agg: AggExpr) -> Self {
        Expr::Agg(agg)
    }
}

impl<S: AsRef<str>> From<S> for Expr {
    fn from(s: S) -> Self {
        col(s.as_ref())
    }
}
