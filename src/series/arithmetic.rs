use crate::{
    error::{PolarsError, Result},
    series::series::Series,
};
use std::ops;

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
            Series::Int32(lhs) => subtract!(Series::Int32, lhs),
            Series::Int64(lhs) => subtract!(Series::Int64, lhs),
            Series::Float32(lhs) => subtract!(Series::Float32, lhs),
            Series::Float64(lhs) => subtract!(Series::Float64, lhs),
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
            Series::Int32(lhs) => add!(Series::Int32, lhs),
            Series::Int64(lhs) => add!(Series::Int64, lhs),
            Series::Float32(lhs) => add!(Series::Float32, lhs),
            Series::Float64(lhs) => add!(Series::Float64, lhs),
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
            Series::Int32(lhs) => multiply!(Series::Int32, lhs),
            Series::Int64(lhs) => multiply!(Series::Int64, lhs),
            Series::Float32(lhs) => multiply!(Series::Float32, lhs),
            Series::Float64(lhs) => multiply!(Series::Float64, lhs),
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
            Series::Int32(lhs) => divide!(Series::Int32, lhs),
            Series::Int64(lhs) => divide!(Series::Int64, lhs),
            Series::Float32(lhs) => divide!(Series::Float32, lhs),
            Series::Float64(lhs) => divide!(Series::Float64, lhs),
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

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).multiply(rhs).expect("data types don't match")
    }
}

impl std::ops::Div for &Series {
    type Output = Series;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).divide(rhs).expect("data types don't match")
    }
}
