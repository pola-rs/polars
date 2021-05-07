use crate::prelude::*;
use crate::utils::get_supertype;
use num::{Num, NumCast};
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops;

pub trait NumOpsDispatch: Debug {
    fn subtract(&self, rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!(
                "subtraction operation not supported for {:?} and {:?}",
                self, rhs
            )
            .into(),
        ))
    }
    fn add_to(&self, rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!(
                "addition operation not supported for {:?} and {:?}",
                self, rhs
            )
            .into(),
        ))
    }
    fn multiply(&self, rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!(
                "multiplication operation not supported for {:?} and {:?}",
                self, rhs
            )
            .into(),
        ))
    }
    fn divide(&self, rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!(
                "division operation not supported for {:?} and {:?}",
                self, rhs
            )
            .into(),
        ))
    }
    fn remainder(&self, rhs: &Series) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            format!(
                "remainder operation not supported for {:?} and {:?}",
                self, rhs
            )
            .into(),
        ))
    }
}

impl<T> NumOpsDispatch for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: ops::Add<Output = T::Native>
        + ops::Sub<Output = T::Native>
        + ops::Mul<Output = T::Native>
        + ops::Div<Output = T::Native>
        + ops::Rem<Output = T::Native>
        + num::Zero
        + num::One,
    ChunkedArray<T>: IntoSeries,
{
    fn subtract(&self, rhs: &Series) -> Result<Series> {
        // Safety:
        // The ChunkedArray with the wrong dtype is dropped after this operation
        let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
        let out = self - rhs;
        Ok(out.into_series())
    }
    fn add_to(&self, rhs: &Series) -> Result<Series> {
        let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
        let out = self + rhs;
        Ok(out.into_series())
    }
    fn multiply(&self, rhs: &Series) -> Result<Series> {
        let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
        let out = self * rhs;
        Ok(out.into_series())
    }
    fn divide(&self, rhs: &Series) -> Result<Series> {
        let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
        let out = self / rhs;
        Ok(out.into_series())
    }
    fn remainder(&self, rhs: &Series) -> Result<Series> {
        let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
        let out = self % rhs;
        Ok(out.into_series())
    }
}

impl NumOpsDispatch for Utf8Chunked {
    fn add_to(&self, rhs: &Series) -> Result<Series> {
        let rhs = self.unpack_series_matching_type(rhs)?;
        let out = self + rhs;
        Ok(out.into_series())
    }
}
impl NumOpsDispatch for BooleanChunked {}
impl NumOpsDispatch for ListChunked {}
impl NumOpsDispatch for CategoricalChunked {}

pub(crate) fn coerce_lhs_rhs<'a>(
    lhs: &'a Series,
    rhs: &'a Series,
) -> Result<(Cow<'a, Series>, Cow<'a, Series>)> {
    let dtype = get_supertype(lhs.dtype(), rhs.dtype())?;
    let left = if lhs.dtype() == &dtype {
        Cow::Borrowed(lhs)
    } else {
        Cow::Owned(lhs.cast_with_dtype(&dtype)?)
    };
    let right = if rhs.dtype() == &dtype {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.cast_with_dtype(&dtype)?)
    };
    Ok((left, right))
}

impl ops::Sub for &Series {
    type Output = Series;

    fn sub(self, rhs: Self) -> Self::Output {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        lhs.subtract(rhs.as_ref()).expect("data types don't match")
    }
}

impl ops::Add for &Series {
    type Output = Series;

    fn add(self, rhs: Self) -> Self::Output {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        lhs.add_to(rhs.as_ref()).expect("data types don't match")
    }
}

impl ops::Mul for &Series {
    type Output = Series;

    /// ```
    /// # use polars_core::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let out = &s * &s;
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        lhs.multiply(rhs.as_ref()).expect("data types don't match")
    }
}

impl ops::Div for &Series {
    type Output = Series;

    /// ```
    /// # use polars_core::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let out = &s / &s;
    /// ```
    fn div(self, rhs: Self) -> Self::Output {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        lhs.divide(rhs.as_ref()).expect("data types don't match")
    }
}

impl ops::Rem for &Series {
    type Output = Series;

    /// ```
    /// # use polars_core::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let out = &s / &s;
    /// ```
    fn rem(self, rhs: Self) -> Self::Output {
        let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
        lhs.remainder(rhs.as_ref()).expect("data types don't match")
    }
}

// Series +-/* numbers instead of Series

impl<T> ops::Sub<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn sub(self, rhs: T) -> Self::Output {
        macro_rules! numeric {
            ($ca:expr) => {{
                $ca.sub(rhs).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(self, numeric, noop, noop)
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
        macro_rules! numeric {
            ($ca:expr) => {{
                $ca.add(rhs).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(self, numeric, noop, noop)
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
        macro_rules! numeric {
            ($ca:expr) => {{
                $ca.div(rhs).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(self, numeric, noop, noop)
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
        macro_rules! numeric {
            ($ca:expr) => {{
                $ca.mul(rhs).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(self, numeric, noop, noop)
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

impl<T> ops::Rem<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn rem(self, rhs: T) -> Self::Output {
        macro_rules! numeric {
            ($ca:expr) => {{
                $ca.rem(rhs).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(self, numeric, noop, noop)
    }
}

impl<T> ops::Rem<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        (&self).rem(rhs)
    }
}

/// We cannot override the left hand side behaviour. So we create a trait LhsNumOps.
/// This allows for 1.add(&Series)
///
impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Num + NumCast + ops::Sub<Output = T::Native> + ops::Div<Output = T::Native>,
    ChunkedArray<T>: IntoSeries,
{
    /// Apply lhs - self
    pub fn lhs_sub<N: Num + NumCast>(&self, lhs: N) -> Self {
        let lhs: T::Native = NumCast::from(lhs).expect("could not cast");
        self.apply(|v| lhs - v)
    }

    /// Apply lhs / self
    pub fn lhs_div<N: Num + NumCast>(&self, lhs: N) -> Self {
        let lhs: T::Native = NumCast::from(lhs).expect("could not cast");
        self.apply(|v| lhs / v)
    }

    /// Apply lhs % self
    pub fn lhs_rem<N: Num + NumCast>(&self, lhs: N) -> Self {
        let lhs: T::Native = NumCast::from(lhs).expect("could not cast");
        self.apply(|v| lhs % v)
    }
}

pub trait LhsNumOps {
    type Output;

    fn add(self, rhs: &Series) -> Self::Output;
    fn sub(self, rhs: &Series) -> Self::Output;
    fn div(self, rhs: &Series) -> Self::Output;
    fn mul(self, rhs: &Series) -> Self::Output;
    fn rem(self, rem: &Series) -> Self::Output;
}

impl<T> LhsNumOps for T
where
    T: Num + NumCast,
{
    type Output = Series;

    fn add(self, rhs: &Series) -> Self::Output {
        // order doesn't matter, dispatch to rhs + lhs
        rhs + self
    }
    fn sub(self, rhs: &Series) -> Self::Output {
        macro_rules! numeric {
            ($rhs:expr) => {{
                $rhs.lhs_sub(self).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(rhs, numeric, noop, noop)
    }
    fn div(self, rhs: &Series) -> Self::Output {
        macro_rules! numeric {
            ($rhs:expr) => {{
                $rhs.lhs_div(self).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(rhs, numeric, noop, noop)
    }
    fn mul(self, rhs: &Series) -> Self::Output {
        // order doesn't matter, dispatch to rhs * lhs
        rhs * self
    }
    fn rem(self, rhs: &Series) -> Self::Output {
        macro_rules! numeric {
            ($rhs:expr) => {{
                $rhs.lhs_rem(self).into_series()
            }};
        }

        macro_rules! noop {
            ($ca:expr) => {{
                unimplemented!()
            }};
        }
        match_arrow_data_type_apply_macro_ca!(rhs, numeric, noop, noop)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    #[allow(clippy::eq_op)]
    fn test_arithmetic_series() {
        // Series +-/* Series
        let s = Series::new("foo", [1, 2, 3]);
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
        assert_eq!(
            Vec::from((1.rem(&s)).i32().unwrap()),
            [Some(0), Some(1), Some(1)]
        );

        assert_eq!((&s * &s).name(), "foo");
        assert_eq!((&s * 1).name(), "foo");
        assert_eq!((1.div(&s)).name(), "foo");
    }

    #[test]
    #[cfg(feature = "dtype-date64")]
    fn test_arithmetic_series_date() {
        // Date Series have a different dispatch, so let's test that.
        let s = Date64Chunked::new_from_slice("foo", &[0, 1, 2, 3]).into_series();

        // test if it runs.
        let _ = &s * &s;

        let _ = s.minute().map(|m| &m / 5);
        let _ = s.minute().map(|m| m / 5);
        let _ = s.minute().map(|m| m.into_series() / 5);
    }
}
