use crate::prelude::*;
use num::{Num, NumCast, ToPrimitive};
use std::ops;

// TODO: implement type

impl Series {
    fn subtract(&self, rhs: &Series) -> Result<Self> {
        macro_rules! subtract {
            ($variant:path, $lhs:ident) => {{
                if let $variant(rhs_) = rhs {
                    Ok($variant($lhs - rhs_))
                } else {
                    Err(PolarsError::DataTypeMisMatch)
                }
            }};
        }
        match self {
            Series::UInt8(lhs) => subtract!(Series::UInt8, lhs),
            Series::UInt16(lhs) => subtract!(Series::UInt16, lhs),
            Series::UInt32(lhs) => subtract!(Series::UInt32, lhs),
            Series::UInt64(lhs) => subtract!(Series::UInt64, lhs),
            Series::Int8(lhs) => subtract!(Series::Int8, lhs),
            Series::Int16(lhs) => subtract!(Series::Int16, lhs),
            Series::Int32(lhs) => subtract!(Series::Int32, lhs),
            Series::Int64(lhs) => subtract!(Series::Int64, lhs),
            Series::Float32(lhs) => subtract!(Series::Float32, lhs),
            Series::Float64(lhs) => subtract!(Series::Float64, lhs),
            Series::Date32(lhs) => subtract!(Series::Date32, lhs),
            Series::Date64(lhs) => subtract!(Series::Date64, lhs),
            Series::Time64Nanosecond(lhs) => subtract!(Series::Time64Nanosecond, lhs),
            Series::DurationNanosecond(lhs) => subtract!(Series::DurationNanosecond, lhs),
            _ => Err(PolarsError::InvalidOperation),
        }
    }

    fn add_to(&self, rhs: &Series) -> Result<Self> {
        macro_rules! add {
            ($variant:path, $lhs:ident) => {{
                if let $variant(rhs_) = rhs {
                    Ok($variant($lhs + rhs_))
                } else {
                    Err(PolarsError::DataTypeMisMatch)
                }
            }};
        }
        match self {
            Series::UInt8(lhs) => add!(Series::UInt8, lhs),
            Series::UInt16(lhs) => add!(Series::UInt16, lhs),
            Series::UInt32(lhs) => add!(Series::UInt32, lhs),
            Series::UInt64(lhs) => add!(Series::UInt64, lhs),
            Series::Int8(lhs) => add!(Series::Int8, lhs),
            Series::Int16(lhs) => add!(Series::Int16, lhs),
            Series::Int32(lhs) => add!(Series::Int32, lhs),
            Series::Int64(lhs) => add!(Series::Int64, lhs),
            Series::Float32(lhs) => add!(Series::Float32, lhs),
            Series::Float64(lhs) => add!(Series::Float64, lhs),
            Series::Date32(lhs) => add!(Series::Date32, lhs),
            Series::Date64(lhs) => add!(Series::Date64, lhs),
            Series::Time64Nanosecond(lhs) => add!(Series::Time64Nanosecond, lhs),
            Series::DurationNanosecond(lhs) => add!(Series::DurationNanosecond, lhs),
            _ => Err(PolarsError::InvalidOperation),
        }
    }

    fn multiply(&self, rhs: &Series) -> Result<Self> {
        macro_rules! multiply {
            ($variant:path, $lhs:ident) => {{
                if let $variant(rhs_) = rhs {
                    Ok($variant($lhs * rhs_))
                } else {
                    Err(PolarsError::DataTypeMisMatch)
                }
            }};
        }
        match self {
            Series::UInt8(lhs) => multiply!(Series::UInt8, lhs),
            Series::UInt16(lhs) => multiply!(Series::UInt16, lhs),
            Series::UInt32(lhs) => multiply!(Series::UInt32, lhs),
            Series::UInt64(lhs) => multiply!(Series::UInt64, lhs),
            Series::Int8(lhs) => multiply!(Series::Int8, lhs),
            Series::Int16(lhs) => multiply!(Series::Int16, lhs),
            Series::Int32(lhs) => multiply!(Series::Int32, lhs),
            Series::Int64(lhs) => multiply!(Series::Int64, lhs),
            Series::Float32(lhs) => multiply!(Series::Float32, lhs),
            Series::Float64(lhs) => multiply!(Series::Float64, lhs),
            Series::Date32(lhs) => multiply!(Series::Date32, lhs),
            Series::Date64(lhs) => multiply!(Series::Date64, lhs),
            Series::Time64Nanosecond(lhs) => multiply!(Series::Time64Nanosecond, lhs),
            Series::DurationNanosecond(lhs) => multiply!(Series::DurationNanosecond, lhs),
            _ => Err(PolarsError::InvalidOperation),
        }
    }

    fn divide(&self, rhs: &Series) -> Result<Self> {
        macro_rules! divide {
            ($variant:path, $lhs:ident) => {{
                if let $variant(rhs_) = rhs {
                    Ok($variant($lhs / rhs_))
                } else {
                    Err(PolarsError::DataTypeMisMatch)
                }
            }};
        }
        match self {
            Series::UInt8(lhs) => divide!(Series::UInt8, lhs),
            Series::UInt16(lhs) => divide!(Series::UInt16, lhs),
            Series::UInt32(lhs) => divide!(Series::UInt32, lhs),
            Series::UInt64(lhs) => divide!(Series::UInt64, lhs),
            Series::Int8(lhs) => divide!(Series::Int8, lhs),
            Series::Int16(lhs) => divide!(Series::Int16, lhs),
            Series::Int32(lhs) => divide!(Series::Int32, lhs),
            Series::Int64(lhs) => divide!(Series::Int64, lhs),
            Series::Float32(lhs) => divide!(Series::Float32, lhs),
            Series::Float64(lhs) => divide!(Series::Float64, lhs),
            Series::Date32(lhs) => divide!(Series::Date32, lhs),
            Series::Date64(lhs) => divide!(Series::Date64, lhs),
            Series::Time64Nanosecond(lhs) => divide!(Series::Time64Nanosecond, lhs),
            Series::DurationNanosecond(lhs) => divide!(Series::DurationNanosecond, lhs),
            _ => Err(PolarsError::InvalidOperation),
        }
    }
}

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

// Series +-/* number

macro_rules! op_num_rhs {
    ($typ:ty, $ca:ident, $rhs:ident, $operand:tt) => {
    {
            let rhs: $typ = NumCast::from($rhs).expect(&format!("could not cast"));
            $ca.into_iter().map(|opt_v| opt_v.map(|v| v $operand rhs)).collect()
            }
    }
}

impl<T> ops::Sub<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn sub(self, rhs: T) -> Self::Output {
        match self {
            Series::UInt8(ca) => op_num_rhs!(u8, ca, rhs, -),
            Series::UInt16(ca) => op_num_rhs!(u16, ca, rhs, -),
            Series::UInt32(ca) => op_num_rhs!(u32, ca, rhs, -),
            Series::UInt64(ca) => op_num_rhs!(u64, ca, rhs, -),
            Series::Int8(ca) => op_num_rhs!(i8, ca, rhs, -),
            Series::Int16(ca) => op_num_rhs!(i16, ca, rhs, -),
            Series::Int32(ca) => op_num_rhs!(i32, ca, rhs, -),
            Series::Int64(ca) => op_num_rhs!(i64, ca, rhs, -),
            Series::Float32(ca) => op_num_rhs!(f32, ca, rhs, -),
            Series::Float64(ca) => op_num_rhs!(f64, ca, rhs, -),
            Series::Date32(ca) => op_num_rhs!(i32, ca, rhs, -),
            Series::Date64(ca) => op_num_rhs!(i64, ca, rhs, -),
            Series::Time64Nanosecond(ca) => op_num_rhs!(i64, ca, rhs, -),
            Series::DurationNanosecond(ca) => op_num_rhs!(i64, ca, rhs, -),
            _ => unimplemented!(),
        }
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
        match self {
            Series::UInt8(ca) => op_num_rhs!(u8, ca, rhs, +),
            Series::UInt16(ca) => op_num_rhs!(u16, ca, rhs, +),
            Series::UInt32(ca) => op_num_rhs!(u32, ca, rhs, +),
            Series::UInt64(ca) => op_num_rhs!(u64, ca, rhs, +),
            Series::Int8(ca) => op_num_rhs!(i8, ca, rhs, +),
            Series::Int16(ca) => op_num_rhs!(i16, ca, rhs, +),
            Series::Int32(ca) => op_num_rhs!(i32, ca, rhs, +),
            Series::Int64(ca) => op_num_rhs!(i64, ca, rhs, +),
            Series::Float32(ca) => op_num_rhs!(f32, ca, rhs, +),
            Series::Float64(ca) => op_num_rhs!(f64, ca, rhs, +),
            Series::Date32(ca) => op_num_rhs!(i32, ca, rhs, +),
            Series::Date64(ca) => op_num_rhs!(i64, ca, rhs, +),
            Series::Time64Nanosecond(ca) => op_num_rhs!(i64, ca, rhs, +),
            Series::DurationNanosecond(ca) => op_num_rhs!(i64, ca, rhs, +),
            _ => unimplemented!(),
        }
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
        match self {
            Series::UInt8(ca) => op_num_rhs!(u8, ca, rhs, /),
            Series::UInt16(ca) => op_num_rhs!(u16, ca, rhs, /),
            Series::UInt32(ca) => op_num_rhs!(u32, ca, rhs, /),
            Series::UInt64(ca) => op_num_rhs!(u64, ca, rhs, /),
            Series::Int8(ca) => op_num_rhs!(i8, ca, rhs, /),
            Series::Int16(ca) => op_num_rhs!(i16, ca, rhs, /),
            Series::Int32(ca) => op_num_rhs!(i32, ca, rhs, /),
            Series::Int64(ca) => op_num_rhs!(i64, ca, rhs, /),
            Series::Float32(ca) => op_num_rhs!(f32, ca, rhs, /),
            Series::Float64(ca) => op_num_rhs!(f64, ca, rhs, /),
            Series::Date32(ca) => op_num_rhs!(i32, ca, rhs, /),
            Series::Date64(ca) => op_num_rhs!(i64, ca, rhs, /),
            Series::Time64Nanosecond(ca) => op_num_rhs!(i64, ca, rhs, /),
            Series::DurationNanosecond(ca) => op_num_rhs!(i64, ca, rhs, /),
            _ => unimplemented!(),
        }
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
        match self {
            Series::UInt8(ca) => op_num_rhs!(u8, ca, rhs, *),
            Series::UInt16(ca) => op_num_rhs!(u16, ca, rhs, *),
            Series::UInt32(ca) => op_num_rhs!(u32, ca, rhs, *),
            Series::UInt64(ca) => op_num_rhs!(u64, ca, rhs, *),
            Series::Int8(ca) => op_num_rhs!(i8, ca, rhs, *),
            Series::Int16(ca) => op_num_rhs!(i16, ca, rhs, *),
            Series::Int32(ca) => op_num_rhs!(i32, ca, rhs, *),
            Series::Int64(ca) => op_num_rhs!(i64, ca, rhs, *),
            Series::Float32(ca) => op_num_rhs!(f32, ca, rhs, *),
            Series::Float64(ca) => op_num_rhs!(f64, ca, rhs, *),
            Series::Date32(ca) => op_num_rhs!(i32, ca, rhs, *),
            Series::Date64(ca) => op_num_rhs!(i64, ca, rhs, *),
            Series::Time64Nanosecond(ca) => op_num_rhs!(i64, ca, rhs, *),
            Series::DurationNanosecond(ca) => op_num_rhs!(i64, ca, rhs, *),
            _ => unimplemented!(),
        }
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

pub trait LhsNumOps<Rhs> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
    fn div(self, rhs: Rhs) -> Self::Output;
    fn mul(self, rhs: Rhs) -> Self::Output;
}

macro_rules! op_num_lhs {
    ($typ:ty, $ca:ident, $lhs:ident, $operand:tt) => {
    {
            let lhs: $typ = NumCast::from($lhs).expect(&format!("could not cast"));
            $ca.into_iter().map(|opt_v| opt_v.map(|v| lhs $operand v)).collect()
            }
    }
}

impl<T> LhsNumOps<&Series> for T
where
    T: Num + NumCast,
{
    type Output = Series;

    fn add(self, rhs: &Series) -> Self::Output {
        match rhs {
            Series::UInt8(ca) => op_num_lhs!(u8, ca, self, +),
            Series::UInt16(ca) => op_num_lhs!(u16, ca, self, +),
            Series::UInt32(ca) => op_num_lhs!(u32, ca, self, +),
            Series::UInt64(ca) => op_num_lhs!(u64, ca, self, +),
            Series::Int8(ca) => op_num_lhs!(i8, ca, self, +),
            Series::Int16(ca) => op_num_lhs!(i16, ca, self, +),
            Series::Int32(ca) => op_num_lhs!(i32, ca, self, +),
            Series::Int64(ca) => op_num_lhs!(i64, ca, self, +),
            Series::Float32(ca) => op_num_lhs!(f32, ca, self, +),
            Series::Float64(ca) => op_num_lhs!(f64, ca, self, +),
            Series::Date32(ca) => op_num_lhs!(i32, ca, self, +),
            Series::Date64(ca) => op_num_lhs!(i64, ca, self, +),
            Series::Time64Nanosecond(ca) => op_num_lhs!(i64, ca, self, +),
            Series::DurationNanosecond(ca) => op_num_lhs!(i64, ca, self, +),
            _ => unimplemented!(),
        }
    }
    fn sub(self, rhs: &Series) -> Self::Output {
        match rhs {
            Series::UInt8(ca) => op_num_lhs!(u8, ca, self, -),
            Series::UInt16(ca) => op_num_lhs!(u16, ca, self, -),
            Series::UInt32(ca) => op_num_lhs!(u32, ca, self, -),
            Series::UInt64(ca) => op_num_lhs!(u64, ca, self, -),
            Series::Int8(ca) => op_num_lhs!(i8, ca, self, -),
            Series::Int16(ca) => op_num_lhs!(i16, ca, self, -),
            Series::Int32(ca) => op_num_lhs!(i32, ca, self, -),
            Series::Int64(ca) => op_num_lhs!(i64, ca, self, -),
            Series::Float32(ca) => op_num_lhs!(f32, ca, self, -),
            Series::Float64(ca) => op_num_lhs!(f64, ca, self, -),
            Series::Date32(ca) => op_num_lhs!(i32, ca, self, -),
            Series::Date64(ca) => op_num_lhs!(i64, ca, self, -),
            Series::Time64Nanosecond(ca) => op_num_lhs!(i64, ca, self, -),
            Series::DurationNanosecond(ca) => op_num_lhs!(i64, ca, self, -),
            _ => unimplemented!(),
        }
    }
    fn div(self, rhs: &Series) -> Self::Output {
        match rhs {
            Series::UInt8(ca) => op_num_lhs!(u8, ca, self, /),
            Series::UInt16(ca) => op_num_lhs!(u16, ca, self, /),
            Series::UInt32(ca) => op_num_lhs!(u32, ca, self, /),
            Series::UInt64(ca) => op_num_lhs!(u64, ca, self, /),
            Series::Int8(ca) => op_num_lhs!(i8, ca, self, /),
            Series::Int16(ca) => op_num_lhs!(i16, ca, self, /),
            Series::Int32(ca) => op_num_lhs!(i32, ca, self, /),
            Series::Int64(ca) => op_num_lhs!(i64, ca, self, /),
            Series::Float32(ca) => op_num_lhs!(f32, ca, self, /),
            Series::Float64(ca) => op_num_lhs!(f64, ca, self, /),
            Series::Date32(ca) => op_num_lhs!(i32, ca, self, /),
            Series::Date64(ca) => op_num_lhs!(i64, ca, self, /),
            Series::Time64Nanosecond(ca) => op_num_lhs!(i64, ca, self, /),
            Series::DurationNanosecond(ca) => op_num_lhs!(i64, ca, self, /),
            _ => unimplemented!(),
        }
    }
    fn mul(self, rhs: &Series) -> Self::Output {
        match rhs {
            Series::UInt8(ca) => op_num_lhs!(u8, ca, self, *),
            Series::UInt16(ca) => op_num_lhs!(u16, ca, self, *),
            Series::UInt32(ca) => op_num_lhs!(u32, ca, self, *),
            Series::UInt64(ca) => op_num_lhs!(u64, ca, self, *),
            Series::Int8(ca) => op_num_lhs!(i8, ca, self, *),
            Series::Int16(ca) => op_num_lhs!(i16, ca, self, *),
            Series::Int32(ca) => op_num_lhs!(i32, ca, self, *),
            Series::Int64(ca) => op_num_lhs!(i64, ca, self, *),
            Series::Float32(ca) => op_num_lhs!(f32, ca, self, *),
            Series::Float64(ca) => op_num_lhs!(f64, ca, self, *),
            Series::Date32(ca) => op_num_lhs!(i32, ca, self, *),
            Series::Date64(ca) => op_num_lhs!(i64, ca, self, *),
            Series::Time64Nanosecond(ca) => op_num_lhs!(i64, ca, self, *),
            Series::DurationNanosecond(ca) => op_num_lhs!(i64, ca, self, *),
            _ => unimplemented!(),
        }
    }
}

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
