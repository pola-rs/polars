use crate::prelude::*;
use enum_dispatch::enum_dispatch;
use num::{Num, NumCast, ToPrimitive};
use std::ops;

#[enum_dispatch(Series)]
pub(super) trait NumOpsDispatch {
    fn subtract(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation)
    }
    fn add_to(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation)
    }
    fn multiply(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation)
    }
    fn divide(&self, _rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation)
    }
}

impl<T> NumOpsDispatch for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: ops::Add<Output = T::Native>
        + ops::Sub<Output = T::Native>
        + ops::Mul<Output = T::Native>
        + ops::Div<Output = T::Native>
        + num::Zero
        + num::One,
{
    fn subtract(&self, rhs: &Series) -> Result<Series> {
        let rhs = self.unpack_series_matching_type(rhs)?;
        let out = self - rhs;
        Ok(out.into_series())
    }
    fn add_to(&self, rhs: &Series) -> Result<Series> {
        let rhs = self.unpack_series_matching_type(rhs)?;
        let out = self + rhs;
        Ok(out.into_series())
    }
    fn multiply(&self, rhs: &Series) -> Result<Series> {
        let rhs = self.unpack_series_matching_type(rhs)?;
        let out = self * rhs;
        Ok(out.into_series())
    }
    fn divide(&self, rhs: &Series) -> Result<Series> {
        let rhs = self.unpack_series_matching_type(rhs)?;
        let out = self / rhs;
        Ok(out.into_series())
    }
}

impl NumOpsDispatch for Utf8Chunked {}
impl NumOpsDispatch for BooleanChunked {}

impl ops::Sub for Series {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).subtract(&rhs).expect("data types don't match")
    }
}

impl ops::Add for Series {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add_to(&rhs).expect("data types don't match")
    }
}

impl std::ops::Mul for Series {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).multiply(&rhs).expect("data types don't match")
    }
}

impl std::ops::Div for Series {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).divide(&rhs).expect("data types don't match")
    }
}

// Same only now for referenced data types

impl ops::Sub for &Series {
    type Output = Series;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).subtract(rhs).expect("data types don't match")
    }
}

impl ops::Add for &Series {
    type Output = Series;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add_to(rhs).expect("data types don't match")
    }
}

impl std::ops::Mul for &Series {
    type Output = Series;

    /// ```
    /// # use polars::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let out = &s * &s;
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        (&self).multiply(rhs).expect("data types don't match")
    }
}

impl std::ops::Div for &Series {
    type Output = Series;

    /// ```
    /// # use polars::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let out = &s / &s;
    /// ```
    fn div(self, rhs: Self) -> Self::Output {
        (&self).divide(rhs).expect("data types don't match")
    }
}

// Series +-/* numbers instead of Series

#[enum_dispatch(Series)]
pub(super) trait NumOpsDispatchSeriesSingleNumber {
    fn subtract_number<N: Num + NumCast>(&self, _rhs: N) -> Series {
        unimplemented!()
    }
    fn add_number<N: Num + NumCast>(&self, _rhs: N) -> Series {
        unimplemented!()
    }
    fn multiply_number<N: Num + NumCast>(&self, _rhs: N) -> Series {
        unimplemented!()
    }
    fn divide_number<N: Num + NumCast>(&self, _rhs: N) -> Series {
        unimplemented!()
    }
}

impl NumOpsDispatchSeriesSingleNumber for BooleanChunked {}
impl NumOpsDispatchSeriesSingleNumber for Utf8Chunked {}

impl<T> NumOpsDispatchSeriesSingleNumber for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Num
        + NumCast
        + ops::Add<Output = T::Native>
        + ops::Sub<Output = T::Native>
        + ops::Mul<Output = T::Native>
        + ops::Div<Output = T::Native>,
{
    fn subtract_number<N: Num + NumCast>(&self, rhs: N) -> Series {
        let rhs: T::Native = NumCast::from(rhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| v - rhs))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }

    fn add_number<N: Num + NumCast>(&self, rhs: N) -> Series {
        let rhs: T::Native = NumCast::from(rhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| v + rhs))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn multiply_number<N: Num + NumCast>(&self, rhs: N) -> Series {
        let rhs: T::Native = NumCast::from(rhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| v * rhs))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn divide_number<N: Num + NumCast>(&self, rhs: N) -> Series {
        let rhs: T::Native = NumCast::from(rhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| v / rhs))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }
}

impl<T> ops::Sub<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn sub(self, rhs: T) -> Self::Output {
        self.subtract_number(rhs)
    }
}

impl<T> ops::Sub<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl<T> ops::Add<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn add(self, rhs: T) -> Self::Output {
        self.add_number(rhs)
    }
}

impl<T> ops::Add<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<T> ops::Div<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn div(self, rhs: T) -> Self::Output {
        self.divide_number(rhs)
    }
}

impl<T> ops::Div<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        (&self).div(rhs)
    }
}

impl<T> ops::Mul<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn mul(self, rhs: T) -> Self::Output {
        self.multiply_number(rhs)
    }
}

impl<T> ops::Mul<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        (&self).mul(rhs)
    }
}

/// We cannot override the left hand side behaviour. So we create a trait Lhs num ops.
/// This allows for 1.add(&Series)

#[enum_dispatch(Series)]
pub(super) trait LhsNumOpsDispatch {
    fn lhs_subtract_number<N: Num + NumCast>(&self, _lhs: N) -> Series {
        unimplemented!()
    }
    fn lhs_add_number<N: Num + NumCast>(&self, _lhs: N) -> Series {
        unimplemented!()
    }
    fn lhs_multiply_number<N: Num + NumCast>(&self, _lhs: N) -> Series {
        unimplemented!()
    }
    fn lhs_divide_number<N: Num + NumCast>(&self, _lhs: N) -> Series {
        unimplemented!()
    }
}

impl LhsNumOpsDispatch for BooleanChunked {}
impl LhsNumOpsDispatch for Utf8Chunked {}

impl<T> LhsNumOpsDispatch for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Num
        + NumCast
        + ops::Add<Output = T::Native>
        + ops::Sub<Output = T::Native>
        + ops::Mul<Output = T::Native>
        + ops::Div<Output = T::Native>,
{
    fn lhs_subtract_number<N: Num + NumCast>(&self, lhs: N) -> Series {
        let lhs: T::Native = NumCast::from(lhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| lhs - v))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }

    fn lhs_add_number<N: Num + NumCast>(&self, lhs: N) -> Series {
        let lhs: T::Native = NumCast::from(lhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| lhs + v))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn lhs_multiply_number<N: Num + NumCast>(&self, lhs: N) -> Series {
        let lhs: T::Native = NumCast::from(lhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| lhs * v))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn lhs_divide_number<N: Num + NumCast>(&self, lhs: N) -> Series {
        let lhs: T::Native = NumCast::from(lhs).expect(&format!("could not cast"));
        let mut ca: ChunkedArray<T> = self
            .into_iter()
            .map(|opt_v| opt_v.map(|v| lhs / v))
            .collect();
        ca.rename(self.name());
        ca.into_series()
    }
}

pub trait LhsNumOps {
    type Output;

    fn add(self, rhs: &Series) -> Self::Output;
    fn sub(self, rhs: &Series) -> Self::Output;
    fn div(self, rhs: &Series) -> Self::Output;
    fn mul(self, rhs: &Series) -> Self::Output;
}

impl<T> LhsNumOps for T
where
    T: Num + NumCast,
{
    type Output = Series;

    fn add(self, rhs: &Series) -> Self::Output {
        rhs.lhs_add_number(self)
    }
    fn sub(self, rhs: &Series) -> Self::Output {
        rhs.lhs_subtract_number(self)
    }
    fn div(self, rhs: &Series) -> Self::Output {
        rhs.lhs_divide_number(self)
    }
    fn mul(self, rhs: &Series) -> Self::Output {
        rhs.lhs_multiply_number(self)
    }
}

// TODO: use enum dispatch
impl Series {
    fn pow<E: Num>(&self, exp: E) -> Series
    where
        E: ToPrimitive,
    {
        match self {
            Series::UInt8(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::UInt16(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::UInt32(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::UInt64(ca) => Series::Float64(ca.pow_f64(exp.to_f64().unwrap())),
            Series::Int8(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::Int16(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::Int32(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::Int64(ca) => Series::Float64(ca.pow_f64(exp.to_f64().unwrap())),
            Series::Float32(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::Float64(ca) => Series::Float64(ca.pow_f64(exp.to_f64().unwrap())),
            Series::Date32(ca) => Series::Float32(ca.pow_f32(exp.to_f32().unwrap())),
            Series::Date64(ca) => Series::Float64(ca.pow_f64(exp.to_f64().unwrap())),
            Series::Time64Nanosecond(ca) => Series::Float64(ca.pow_f64(exp.to_f64().unwrap())),
            Series::DurationNanosecond(ca) => Series::Float64(ca.pow_f64(exp.to_f64().unwrap())),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_arithmetic_series() {
        // Series +-/* Series
        let s: Series = [1, 2, 3].iter().collect();
        assert_eq!(
            Vec::from((&s * &s).i32().unwrap()),
            [Some(1), Some(4), Some(9)]
        );
        assert_eq!(
            Vec::from((&s / &s).i32().unwrap()),
            [Some(1), Some(1), Some(1)]
        );
        assert_eq!(
            Vec::from((&s - &s).i32().unwrap()),
            [Some(0), Some(0), Some(0)]
        );
        assert_eq!(
            Vec::from((&s + &s).i32().unwrap()),
            [Some(2), Some(4), Some(6)]
        );
        // Series +-/* Number
        assert_eq!(
            Vec::from((&s + 1).i32().unwrap()),
            [Some(2), Some(3), Some(4)]
        );
        assert_eq!(
            Vec::from((&s - 1).i32().unwrap()),
            [Some(0), Some(1), Some(2)]
        );
        assert_eq!(
            Vec::from((&s * 2).i32().unwrap()),
            [Some(2), Some(4), Some(6)]
        );
        assert_eq!(
            Vec::from((&s / 2).i32().unwrap()),
            [Some(0), Some(1), Some(1)]
        );

        // Lhs operations
        assert_eq!(
            Vec::from((1.add(&s)).i32().unwrap()),
            [Some(2), Some(3), Some(4)]
        );
        assert_eq!(
            Vec::from((1.sub(&s)).i32().unwrap()),
            [Some(0), Some(-1), Some(-2)]
        );
        assert_eq!(
            Vec::from((1.div(&s)).i32().unwrap()),
            [Some(1), Some(0), Some(0)]
        );
        assert_eq!(
            Vec::from((1.mul(&s)).i32().unwrap()),
            [Some(1), Some(2), Some(3)]
        );
    }
}
