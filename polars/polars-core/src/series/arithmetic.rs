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
        // There will be UB if a ChunkedArray is alive with the wrong datatype.
        // we now only create the potentially wrong dtype for a short time.
        // Note that the physical type correctness is checked!
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

#[cfg(feature = "checked_arithmetic")]
pub mod checked {
    use super::*;
    use crate::utils::align_chunks_binary;
    use num::{CheckedDiv, ToPrimitive, Zero};

    pub trait NumOpsDispatchChecked: Debug {
        /// Checked integer division. Computes self / rhs, returning None if rhs == 0 or the division results in overflow.
        fn checked_div(&self, rhs: &Series) -> Result<Series> {
            Err(PolarsError::InvalidOperation(
                format!(
                    "checked division operation not supported for {:?} and {:?}",
                    self, rhs
                )
                .into(),
            ))
        }
        fn checked_div_num<T: ToPrimitive>(&self, _rhs: T) -> Result<Series> {
            Err(PolarsError::InvalidOperation(
                format!(
                    "checked division by number operation not supported for {:?}",
                    self
                )
                .into(),
            ))
        }
    }

    impl<T> NumOpsDispatchChecked for ChunkedArray<T>
    where
        T: PolarsIntegerType,
        T::Native:
            CheckedDiv<Output = T::Native> + CheckedDiv<Output = T::Native> + num::Zero + num::One,
        ChunkedArray<T>: IntoSeries,
    {
        fn checked_div(&self, rhs: &Series) -> Result<Series> {
            let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
            let (l, r) = align_chunks_binary(self, rhs);

            Ok((l)
                .downcast_iter()
                .zip(r.downcast_iter())
                .map(|(l_arr, r_arr)| {
                    l_arr
                        .into_iter()
                        .zip(r_arr)
                        // we don't use a kernel, because the checked div also supplies nulls.
                        // so the usual bit combining is not enough.
                        .map(|(opt_l, opt_r)| match (opt_l, opt_r) {
                            (Some(l), Some(r)) => l.checked_div(r),
                            _ => None,
                        })
                })
                .flatten()
                .collect::<ChunkedArray<T>>()
                .into_series())
        }
    }

    impl NumOpsDispatchChecked for Float32Chunked {
        fn checked_div(&self, rhs: &Series) -> Result<Series> {
            let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
            let (l, r) = align_chunks_binary(self, rhs);

            Ok((l)
                .downcast_iter()
                .zip(r.downcast_iter())
                .map(|(l_arr, r_arr)| {
                    l_arr
                        .into_iter()
                        .zip(r_arr)
                        // we don't use a kernel, because the checked div also supplies nulls.
                        // so the usual bit combining is not enough.
                        .map(|(opt_l, opt_r)| match (opt_l, opt_r) {
                            (Some(l), Some(r)) => {
                                if r.is_zero() {
                                    None
                                } else {
                                    Some(l / r)
                                }
                            }
                            _ => None,
                        })
                })
                .flatten()
                .collect::<Float32Chunked>()
                .into_series())
        }
    }

    impl NumOpsDispatchChecked for Float64Chunked {
        fn checked_div(&self, rhs: &Series) -> Result<Series> {
            let rhs = unsafe { self.unpack_series_matching_physical_type(rhs)? };
            let (l, r) = align_chunks_binary(self, rhs);

            Ok((l)
                .downcast_iter()
                .zip(r.downcast_iter())
                .map(|(l_arr, r_arr)| {
                    l_arr
                        .into_iter()
                        .zip(r_arr)
                        // we don't use a kernel, because the checked div also supplies nulls.
                        // so the usual bit combining is not enough.
                        .map(|(opt_l, opt_r)| match (opt_l, opt_r) {
                            (Some(l), Some(r)) => {
                                if r.is_zero() {
                                    None
                                } else {
                                    Some(l / r)
                                }
                            }
                            _ => None,
                        })
                })
                .flatten()
                .collect::<Float64Chunked>()
                .into_series())
        }
    }

    impl NumOpsDispatchChecked for BooleanChunked {}
    impl NumOpsDispatchChecked for ListChunked {}
    impl NumOpsDispatchChecked for CategoricalChunked {}
    impl NumOpsDispatchChecked for Utf8Chunked {}

    impl NumOpsDispatchChecked for Series {
        fn checked_div(&self, rhs: &Series) -> Result<Series> {
            let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
            lhs.as_ref().as_ref().checked_div(rhs.as_ref())
        }

        fn checked_div_num<T: ToPrimitive>(&self, rhs: T) -> Result<Series> {
            use DataType::*;
            let s = self.to_physical_repr();

            let out = match s.dtype() {
                #[cfg(feature = "dtype-u8")]
                UInt8 => s
                    .u8()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_u8().unwrap())))
                    .into_series(),
                #[cfg(feature = "dtype-i8")]
                Int8 => s
                    .i8()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_i8().unwrap())))
                    .into_series(),
                #[cfg(feature = "dtype-i16")]
                Int16 => s
                    .i16()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_i16().unwrap())))
                    .into_series(),
                #[cfg(feature = "dtype-u16")]
                UInt16 => s
                    .u16()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_u16().unwrap())))
                    .into_series(),
                UInt32 => s
                    .u32()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_u32().unwrap())))
                    .into_series(),
                Int32 => s
                    .i32()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_i32().unwrap())))
                    .into_series(),
                #[cfg(feature = "dtype-u64")]
                UInt64 => s
                    .u64()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_u64().unwrap())))
                    .into_series(),
                Int64 => s
                    .i64()
                    .unwrap()
                    .apply_on_opt(|opt_v| opt_v.and_then(|v| v.checked_div(rhs.to_i64().unwrap())))
                    .into_series(),
                Float32 => s
                    .f32()
                    .unwrap()
                    .apply_on_opt(|opt_v| {
                        opt_v.and_then(|v| {
                            let res = rhs.to_f32().unwrap();
                            if res.is_zero() {
                                None
                            } else {
                                Some(v / res)
                            }
                        })
                    })
                    .into_series(),
                Float64 => s
                    .f64()
                    .unwrap()
                    .apply_on_opt(|opt_v| {
                        opt_v.and_then(|v| {
                            let res = rhs.to_f64().unwrap();
                            if res.is_zero() {
                                None
                            } else {
                                Some(v / res)
                            }
                        })
                    })
                    .into_series(),
                _ => panic!("dtype not yet supported in checked div"),
            };
            out.cast_with_dtype(self.dtype())
        }
    }
}

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
        macro_rules! sub {
            ($ca:expr) => {{
                $ca.sub(rhs).into_series()
            }};
        }

        match_arrow_data_type_apply_macro_ca_logical_num!(self, sub)
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
        macro_rules! add {
            ($ca:expr) => {{
                $ca.add(rhs).into_series()
            }};
        }
        match_arrow_data_type_apply_macro_ca_logical_num!(self, add)
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
        macro_rules! div {
            ($ca:expr) => {{
                $ca.div(rhs).into_series()
            }};
        }

        match_arrow_data_type_apply_macro_ca_logical_num!(self, div)
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
        macro_rules! mul {
            ($ca:expr) => {{
                $ca.mul(rhs).into_series()
            }};
        }
        match_arrow_data_type_apply_macro_ca_logical_num!(self, mul)
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
        macro_rules! rem {
            ($ca:expr) => {{
                $ca.rem(rhs).into_series()
            }};
        }
        match_arrow_data_type_apply_macro_ca_logical_num!(self, rem)
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
        macro_rules! sub {
            ($rhs:expr) => {{
                $rhs.lhs_sub(self).into_series()
            }};
        }
        match_arrow_data_type_apply_macro_ca_logical_num!(rhs, sub)
    }
    fn div(self, rhs: &Series) -> Self::Output {
        macro_rules! div {
            ($rhs:expr) => {{
                $rhs.lhs_div(self).into_series()
            }};
        }
        match_arrow_data_type_apply_macro_ca_logical_num!(rhs, div)
    }
    fn mul(self, rhs: &Series) -> Self::Output {
        // order doesn't matter, dispatch to rhs * lhs
        rhs * self
    }
    fn rem(self, rhs: &Series) -> Self::Output {
        macro_rules! rem {
            ($rhs:expr) => {{
                $rhs.lhs_rem(self).into_series()
            }};
        }

        match_arrow_data_type_apply_macro_ca_logical_num!(rhs, rem)
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
    #[cfg(feature = "checked_arithmetic")]
    fn test_checked_div() {
        let s = Series::new("foo", [1i32, 0, 1]);
        let out = s.checked_div(&s).unwrap();
        assert_eq!(Vec::from(out.i32().unwrap()), &[Some(1), None, Some(1)]);
        let out = s.checked_div_num(0).unwrap();
        assert_eq!(Vec::from(out.i32().unwrap()), &[None, None, None]);

        let s_f32 = Series::new("float32", [1.0f32, 0.0, 1.0]);
        let out = s_f32.checked_div(&s_f32).unwrap();
        assert_eq!(
            Vec::from(out.f32().unwrap()),
            &[Some(1.0f32), None, Some(1.0f32)]
        );
        let out = s_f32.checked_div_num(0.0f32).unwrap();
        assert_eq!(Vec::from(out.f32().unwrap()), &[None, None, None]);

        let s_f64 = Series::new("float64", [1.0f64, 0.0, 1.0]);
        let out = s_f64.checked_div(&s_f64).unwrap();
        assert_eq!(
            Vec::from(out.f64().unwrap()),
            &[Some(1.0f64), None, Some(1.0f64)]
        );
        let out = s_f64.checked_div_num(0.0f64).unwrap();
        assert_eq!(Vec::from(out.f64().unwrap()), &[None, None, None]);
    }
}
