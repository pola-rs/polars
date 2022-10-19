use std::ops::{Add, Div, Mul, Rem, Sub};

use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::try_get_supertype;

/// Get the supertype that is valid for all columns in the DataFrame.
/// This reduces casting of the rhs in arithmetic.
fn get_supertype_all(df: &DataFrame, rhs: &Series) -> PolarsResult<DataType> {
    df.columns
        .iter()
        .fold(Ok(rhs.dtype().clone()), |dt, s| match dt {
            Ok(dt) => try_get_supertype(s.dtype(), &dt),
            e => e,
        })
}

macro_rules! impl_arithmetic {
    ($self:expr, $rhs:expr, $operand: tt) => {{
        let st = get_supertype_all($self, $rhs)?;
        let rhs = $rhs.cast(&st)?;
        let cols = $self.columns.par_iter().map(|s| {
            Ok(&s.cast(&st)? $operand &rhs)
        }).collect::<PolarsResult<_>>()?;
        Ok(DataFrame::new_no_checks(cols))
    }}
}

impl Add<&Series> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn add(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, +)
    }
}

impl Add<&Series> for DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn add(self, rhs: &Series) -> Self::Output {
        (&self).add(rhs)
    }
}

impl Sub<&Series> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn sub(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, -)
    }
}

impl Sub<&Series> for DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn sub(self, rhs: &Series) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl Mul<&Series> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn mul(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, *)
    }
}

impl Mul<&Series> for DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn mul(self, rhs: &Series) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl Div<&Series> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn div(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, /)
    }
}

impl Div<&Series> for DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn div(self, rhs: &Series) -> Self::Output {
        (&self).div(rhs)
    }
}

impl Rem<&Series> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn rem(self, rhs: &Series) -> Self::Output {
        impl_arithmetic!(self, rhs, %)
    }
}

impl Rem<&Series> for DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn rem(self, rhs: &Series) -> Self::Output {
        (&self).rem(rhs)
    }
}

impl DataFrame {
    fn binary_aligned(
        &self,
        other: &DataFrame,
        f: &(dyn Fn(&Series, &Series) -> PolarsResult<Series> + Sync + Send),
    ) -> PolarsResult<DataFrame> {
        let max_len = std::cmp::max(self.height(), other.height());
        let max_width = std::cmp::max(self.width(), other.width());
        let mut cols = self
            .get_columns()
            .par_iter()
            .zip(other.get_columns().par_iter())
            .map(|(l, r)| {
                let diff_l = max_len - l.len();
                let diff_r = max_len - r.len();

                let st = try_get_supertype(l.dtype(), r.dtype())?;
                let mut l = l.cast(&st)?;
                let mut r = r.cast(&st)?;

                if diff_l > 0 {
                    l = l.extend_constant(AnyValue::Null, diff_l)?;
                };
                if diff_r > 0 {
                    r = r.extend_constant(AnyValue::Null, diff_r)?;
                };

                f(&l, &r)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        let col_len = cols.len();
        if col_len < max_width {
            let df = if col_len < self.width() { self } else { other };

            for i in col_len..max_len {
                let s = &df.get_columns()[i];
                let name = s.name();
                let dtype = s.dtype();

                // trick to fill a series with nulls
                let vals: &[Option<i32>] = &[None];
                let s = Series::new(name, vals).cast(dtype)?;
                cols.push(s.new_from_index(0, max_len))
            }
        }
        DataFrame::new(cols)
    }
}

impl Add<&DataFrame> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn add(self, rhs: &DataFrame) -> Self::Output {
        self.binary_aligned(rhs, &|a, b| Ok(a + b))
    }
}

impl Sub<&DataFrame> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn sub(self, rhs: &DataFrame) -> Self::Output {
        self.binary_aligned(rhs, &|a, b| Ok(a - b))
    }
}

impl Div<&DataFrame> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn div(self, rhs: &DataFrame) -> Self::Output {
        self.binary_aligned(rhs, &|a, b| Ok(a / b))
    }
}

impl Mul<&DataFrame> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn mul(self, rhs: &DataFrame) -> Self::Output {
        self.binary_aligned(rhs, &|a, b| Ok(a * b))
    }
}

impl Rem<&DataFrame> for &DataFrame {
    type Output = PolarsResult<DataFrame>;

    fn rem(self, rhs: &DataFrame) -> Self::Output {
        self.binary_aligned(rhs, &|a, b| Ok(a % b))
    }
}
