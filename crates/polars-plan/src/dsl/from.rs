use super::*;

impl From<AggExpr> for Expr {
    fn from(agg: AggExpr) -> Self {
        Expr::Agg(agg)
    }
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        col(s)
    }
}

macro_rules! from_literals {
    ($type:ty) => {
        impl From<$type> for Expr {
            fn from(val: $type) -> Self {
                lit(val)
            }
        }
    };
}

from_literals!(f32);
from_literals!(f64);
#[cfg(feature = "dtype-i8")]
from_literals!(i8);
#[cfg(feature = "dtype-i16")]
from_literals!(i16);
from_literals!(i32);
from_literals!(i64);
#[cfg(feature = "dtype-u8")]
from_literals!(u8);
#[cfg(feature = "dtype-u16")]
from_literals!(u16);
from_literals!(u32);
from_literals!(u64);
from_literals!(bool);
