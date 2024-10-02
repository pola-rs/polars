use num_traits::{Num, NumCast};
use polars_error::{polars_bail, PolarsResult};

use super::{Column, ScalarColumn, Series};
use crate::utils::Container;

fn output_length(a: &Column, b: &Column) -> PolarsResult<usize> {
    match (a.len(), b.len()) {
        // broadcasting
        (1, o) | (o, 1) => Ok(o),
        // equal
        (a, b) if a == b => Ok(a),
        // unequal
        (a, b) => {
            polars_bail!(InvalidOperation: "cannot do arithmetic operation on series of different lengths: got {} and {}", a, b)
        },
    }
}

fn unit_series_op<F: Fn(&Series, &Series) -> PolarsResult<Series>>(
    l: &Series,
    r: &Series,
    op: F,
    length: usize,
) -> PolarsResult<Column> {
    debug_assert!(l.len() <= 1);
    debug_assert!(r.len() <= 1);

    op(l, r)
        .map(|s| ScalarColumn::from_single_value_series(s, length))
        .map(Column::from)
}

fn op_with_broadcast<F: Fn(&Series, &Series) -> PolarsResult<Series>>(
    l: &Column,
    r: &Column,
    op: F,
) -> PolarsResult<Column> {
    // Here we rely on the underlying broadcast operations.

    let length = output_length(l, r)?;
    match (l, r) {
        (Column::Series(l), Column::Series(r)) => op(l, r).map(Column::from),
        (Column::Series(l), Column::Scalar(r)) => {
            let r = r.as_single_value_series();
            if l.len() == 1 {
                unit_series_op(l, &r, op, length)
            } else {
                op(l, &r).map(Column::from)
            }
        },
        (Column::Scalar(l), Column::Series(r)) => {
            let l = l.as_single_value_series();
            if r.len() == 1 {
                unit_series_op(&l, r, op, length)
            } else {
                op(&l, r).map(Column::from)
            }
        },
        (Column::Scalar(l), Column::Scalar(r)) => unit_series_op(
            &l.as_single_value_series(),
            &r.as_single_value_series(),
            op,
            length,
        ),
    }
}

fn num_op_with_broadcast<T: Num + NumCast, F: Fn(&Series, T) -> Series>(
    c: &'_ Column,
    n: T,
    op: F,
) -> Column {
    match c {
        Column::Series(s) => op(s, n).into(),
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
                op_with_broadcast(&self, &rhs, |l, r| l.$op(r))
            }
        }

        impl std::ops::$trait for &Column {
            type Output = PolarsResult<Column>;

            #[inline]
            fn $op(self, rhs: Self) -> Self::Output {
                op_with_broadcast(self, rhs, |l, r| l.$op(r))
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
