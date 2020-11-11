use crate::prelude::*;
use rayon::prelude::*;

use crate::lazy::utils::get_supertype;
use std::ops::{Add, Div, Mul, Rem, Sub};

macro_rules! impl_arithmetic {
    ($self:expr, $rhs:expr, $operand: tt) => {{
        let cols = $self.columns.par_iter().map(|s| {
            let st = get_supertype(s.dtype(), $rhs.dtype())?;
            Ok(s.cast_with_arrow_datatype(&st)? $operand $rhs.cast_with_arrow_datatype(&st)?)
        }).collect::<Result<_>>()?;
        Ok(DataFrame::new_no_checks(cols))
    }}
}

impl Add<&Series> for &DataFrame {
    type Output = Result<DataFrame>;

    fn add(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, +)
    }
}

impl Add<&Series> for DataFrame {
    type Output = Result<DataFrame>;

    fn add(self, rhs: &Series) -> Self::Output {
        (&self).add(rhs)
    }
}

impl Sub<&Series> for &DataFrame {
    type Output = Result<DataFrame>;

    fn sub(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, -)
    }
}

impl Sub<&Series> for DataFrame {
    type Output = Result<DataFrame>;

    fn sub(self, rhs: &Series) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl Mul<&Series> for &DataFrame {
    type Output = Result<DataFrame>;

    fn mul(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, *)
    }
}

impl Mul<&Series> for DataFrame {
    type Output = Result<DataFrame>;

    fn mul(self, rhs: &Series) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl Div<&Series> for &DataFrame {
    type Output = Result<DataFrame>;

    fn div(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, /)
    }
}

impl Div<&Series> for DataFrame {
    type Output = Result<DataFrame>;

    fn div(self, rhs: &Series) -> Self::Output {
        (&self).div(rhs)
    }
}

impl Rem<&Series> for &DataFrame {
    type Output = Result<DataFrame>;

    fn rem(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, %)
    }
}

impl Rem<&Series> for DataFrame {
    type Output = Result<DataFrame>;

    fn rem(self, rhs: &Series) -> Self::Output {
        (&self).rem(rhs)
    }
}
