use std::ops::{Add, Div, Mul, Rem, Sub};

use num_traits::{Num, NumCast};
use polars_error::PolarsResult;

use super::Column;

impl Add for Column {
    type Output = PolarsResult<Column>;

    fn add(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .add(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Add for &Column {
    type Output = PolarsResult<Column>;

    fn add(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .add(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Sub for Column {
    type Output = PolarsResult<Column>;

    fn sub(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .sub(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Sub for &Column {
    type Output = PolarsResult<Column>;

    fn sub(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .sub(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Mul for Column {
    type Output = PolarsResult<Column>;

    fn mul(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .mul(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Mul for &Column {
    type Output = PolarsResult<Column>;

    fn mul(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .mul(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl<T> Sub<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn sub(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().sub(rhs).into()
    }
}

impl<T> Sub<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().sub(rhs).into()
    }
}

impl<T> Add<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn add(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().add(rhs).into()
    }
}

impl<T> Add<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().add(rhs).into()
    }
}

impl<T> Div<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn div(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().div(rhs).into()
    }
}

impl<T> Div<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().div(rhs).into()
    }
}

impl<T> Mul<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn mul(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().mul(rhs).into()
    }
}

impl<T> Mul<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().mul(rhs).into()
    }
}

impl<T> Rem<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn rem(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().rem(rhs).into()
    }
}

impl<T> Rem<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().rem(rhs).into()
    }
}
