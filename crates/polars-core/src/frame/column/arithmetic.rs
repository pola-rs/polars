use num_traits::{Num, NumCast};
use polars_error::PolarsResult;

use super::{Column, ScalarColumn, Series};

fn num_op_with_broadcast<T: Num + NumCast, F: Fn(&Series, T) -> Series>(
    c: &'_ Column,
    n: T,
    op: F,
) -> Column {
    match c {
        Column::Series(s) => op(s, n).into(),
        // @partition-opt
        Column::Partitioned(s) => op(s.as_materialized_series(), n).into(),
        Column::Scalar(s) => {
            ScalarColumn::from_single_value_series(op(&s.as_single_value_series(), n), s.len())
                .into()
        },
    }
}

macro_rules! broadcastable_ops {
    ($(($trait:ident, $op:ident))+) => {
        $(
        impl std::ops::$trait for Column {
            type Output = PolarsResult<Column>;

            #[inline]
            fn $op(self, rhs: Self) -> Self::Output {
                self.try_apply_broadcasting_binary_elementwise(&rhs, |l, r| l.$op(r))
            }
        }

        impl std::ops::$trait for &Column {
            type Output = PolarsResult<Column>;

            #[inline]
            fn $op(self, rhs: Self) -> Self::Output {
                self.try_apply_broadcasting_binary_elementwise(rhs, |l, r| l.$op(r))
            }
        }
        )+
    }
}

macro_rules! broadcastable_num_ops {
    ($(($trait:ident, $op:ident))+) => {
        $(
        impl<T> std::ops::$trait::<T> for Column
        where
            T: Num + NumCast,
        {
            type Output = Self;

            #[inline]
            fn $op(self, rhs: T) -> Self::Output {
                num_op_with_broadcast(&self, rhs, |l, r| l.$op(r))
            }
        }

        impl<T> std::ops::$trait::<T> for &Column
        where
            T: Num + NumCast,
        {
            type Output = Column;

            #[inline]
            fn $op(self, rhs: T) -> Self::Output {
                num_op_with_broadcast(self, rhs, |l, r| l.$op(r))
            }
        }
        )+
    };
}

broadcastable_ops! {
    (Add, add)
    (Sub, sub)
    (Mul, mul)
    (Div, div)
    (Rem, rem)
    (BitAnd, bitand)
    (BitOr, bitor)
    (BitXor, bitxor)
}

broadcastable_num_ops! {
    (Add, add)
    (Sub, sub)
    (Mul, mul)
    (Div, div)
    (Rem, rem)
}
