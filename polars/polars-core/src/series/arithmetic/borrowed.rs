use super::*;

macro_rules! impl_series_arithmetic_op {
    ($trait:ident, $op:ident, $try_trait:ident, $try_op:ident, $series_op:ident) => {
        impl<'lhs, 'rhs, T> $try_trait<&'rhs Series> for &'lhs ChunkedArray<T>
        where
            T: PolarsDataType + 'rhs,
            Self: $try_trait<&'rhs ChunkedArray<T>, Output = ChunkedArray<T>, Error = PolarsError>,
            ChunkedArray<T>: IntoSeries,
        {
            type Output = Series;
            type Error = PolarsError;

            fn $try_op(self, rhs: &'rhs Series) -> PolarsResult<Series> {
                Ok(self
                    .$try_op(self.unpack_series_matching_type(rhs)?)?
                    .into_series())
            }
        }

        impl<'lhs, 'rhs, T> $trait<&'rhs Series> for &'lhs ChunkedArray<T>
        where
            T: PolarsDataType,
            Self: $try_trait<&'rhs Series, Output = Series, Error = PolarsError>,
        {
            type Output = Series;

            fn $op(self, rhs: &'rhs Series) -> Series {
                self.$try_op(rhs).unwrap()
            }
        }

        impl<'lhs, 'rhs> $try_trait<&'rhs Series> for &'lhs Series {
            type Output = Series;
            type Error = PolarsError;

            fn $try_op(self, rhs: &'rhs Series) -> PolarsResult<Series> {
                let lhs = self;
                match (lhs.dtype(), rhs.dtype()) {
                    #[cfg(feature = "dtype-struct")]
                    (DataType::Struct(_), DataType::Struct(_)) => {
                        struct_arithmetic_try_op(lhs, rhs, |a, b| $try_trait::$try_op(a, b))
                    }
                    _ => {
                        let (lhs, rhs) = coerce_lhs_rhs(lhs, rhs)?;
                        lhs.$series_op(rhs.as_ref())
                    }
                }
            }
        }

        impl<'lhs, 'rhs> $trait<&'rhs Series> for &'lhs Series {
            type Output = Series;

            fn $op(self, rhs: &'rhs Series) -> Series {
                self.$try_op(rhs).unwrap()
            }
        }
    };
}

impl_series_arithmetic_op!(Add, add, TryAdd, try_add, series_add);
impl_series_arithmetic_op!(Sub, sub, TrySub, try_sub, series_sub);
impl_series_arithmetic_op!(Mul, mul, TryMul, try_mul, series_mul);
impl_series_arithmetic_op!(Div, div, TryDiv, try_div, series_div);
impl_series_arithmetic_op!(Rem, rem, TryRem, try_rem, series_rem);

#[cfg(feature = "checked_arithmetic")]
pub mod checked {
    use num_traits::{CheckedDiv, One, ToPrimitive, Zero};

    use super::*;
    use crate::utils::align_chunks_binary;

    pub trait NumOpsDispatchCheckedInner: PolarsDataType + Sized {
        /// Checked integer division. Computes self / rhs, returning None if rhs == 0 or the division results in overflow.
        fn checked_div(lhs: &ChunkedArray<Self>, rhs: &Series) -> PolarsResult<Series> {
            polars_bail!(opq = checked_div, lhs.dtype(), rhs.dtype());
        }
        fn checked_div_num<T: ToPrimitive>(
            lhs: &ChunkedArray<Self>,
            _rhs: T,
        ) -> PolarsResult<Series> {
            polars_bail!(opq = checked_div_num, lhs.dtype(), Self::get_dtype());
        }
    }

    pub trait NumOpsDispatchChecked {
        /// Checked integer division. Computes self / rhs, returning None if rhs == 0 or the division results in overflow.
        fn checked_div(&self, rhs: &Series) -> PolarsResult<Series>;
        fn checked_div_num<T: ToPrimitive>(&self, _rhs: T) -> PolarsResult<Series>;
    }

    impl<S: NumOpsDispatchCheckedInner> NumOpsDispatchChecked for ChunkedArray<S> {
        fn checked_div(&self, rhs: &Series) -> PolarsResult<Series> {
            S::checked_div(self, rhs)
        }
        fn checked_div_num<T: ToPrimitive>(&self, rhs: T) -> PolarsResult<Series> {
            S::checked_div_num(self, rhs)
        }
    }

    impl<T> NumOpsDispatchCheckedInner for T
    where
        T: PolarsIntegerType,
        T::Native: CheckedDiv<Output = T::Native> + CheckedDiv<Output = T::Native> + Zero + One,
        ChunkedArray<T>: IntoSeries,
    {
        fn checked_div(lhs: &ChunkedArray<T>, rhs: &Series) -> PolarsResult<Series> {
            // Safety:
            // There will be UB if a ChunkedArray is alive with the wrong datatype.
            // we now only create the potentially wrong dtype for a short time.
            // Note that the physical type correctness is checked!
            // The ChunkedArray with the wrong dtype is dropped after this operation
            let rhs = unsafe { lhs.unpacked_series_matching_type_unchecked(rhs) };
            let (l, r) = align_chunks_binary(lhs, rhs);

            Ok((l)
                .downcast_iter()
                .zip(r.downcast_iter())
                .flat_map(|(l_arr, r_arr)| {
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
                .collect::<ChunkedArray<T>>()
                .into_series())
        }
    }

    impl NumOpsDispatchCheckedInner for Float32Type {
        fn checked_div(lhs: &Float32Chunked, rhs: &Series) -> PolarsResult<Series> {
            // Safety:
            // see check_div for chunkedarray<T>
            let rhs = unsafe { lhs.unpacked_series_matching_type_unchecked(rhs) };
            let (l, r) = align_chunks_binary(lhs, rhs);

            Ok((l)
                .downcast_iter()
                .zip(r.downcast_iter())
                .flat_map(|(l_arr, r_arr)| {
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
                .collect::<Float32Chunked>()
                .into_series())
        }
    }

    impl NumOpsDispatchCheckedInner for Float64Type {
        fn checked_div(lhs: &Float64Chunked, rhs: &Series) -> PolarsResult<Series> {
            // Safety:
            // see check_div
            let rhs = unsafe { lhs.unpacked_series_matching_type_unchecked(rhs) };
            let (l, r) = align_chunks_binary(lhs, rhs);

            Ok((l)
                .downcast_iter()
                .zip(r.downcast_iter())
                .flat_map(|(l_arr, r_arr)| {
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
                .collect::<Float64Chunked>()
                .into_series())
        }
    }

    impl NumOpsDispatchChecked for Series {
        fn checked_div(&self, rhs: &Series) -> PolarsResult<Series> {
            let (lhs, rhs) = coerce_lhs_rhs(self, rhs).expect("cannot coerce datatypes");
            lhs.as_ref().as_ref().checked_div(rhs.as_ref())
        }

        fn checked_div_num<T: ToPrimitive>(&self, rhs: T) -> PolarsResult<Series> {
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
            out.cast(self.dtype())
        }
    }
}

pub(crate) fn coerce_lhs_rhs<'a>(
    lhs: &'a Series,
    rhs: &'a Series,
) -> PolarsResult<(Cow<'a, Series>, Cow<'a, Series>)> {
    if let Some(result) = coerce_time_units(lhs, rhs) {
        return Ok(result);
    }
    let dtype = match (lhs.dtype(), rhs.dtype()) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(_)) => {
            return Ok((Cow::Borrowed(lhs), Cow::Borrowed(rhs)))
        }
        _ => try_get_supertype(lhs.dtype(), rhs.dtype())?,
    };

    let left = if lhs.dtype() == &dtype {
        Cow::Borrowed(lhs)
    } else {
        Cow::Owned(lhs.cast(&dtype)?)
    };
    let right = if rhs.dtype() == &dtype {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.cast(&dtype)?)
    };
    Ok((left, right))
}

// Handle (Date | Datetime) +/- (Duration) | (Duration) +/- (Date | Datetime) | (Duration) +-
// (Duration)
// Time arithmetic is only implemented on the date / datetime so ensure that's on left

fn coerce_time_units<'a>(
    lhs: &'a Series,
    rhs: &'a Series,
) -> Option<(Cow<'a, Series>, Cow<'a, Series>)> {
    match (lhs.dtype(), rhs.dtype()) {
        (DataType::Datetime(lu, t), DataType::Duration(ru)) => {
            let units = get_time_units(lu, ru);
            let left = if *lu == units {
                Cow::Borrowed(lhs)
            } else {
                Cow::Owned(lhs.cast(&DataType::Datetime(units, t.clone())).ok()?)
            };
            let right = if *ru == units {
                Cow::Borrowed(rhs)
            } else {
                Cow::Owned(rhs.cast(&DataType::Duration(units)).ok()?)
            };
            Some((left, right))
        }
        // make sure to return Some here, so we don't cast to supertype.
        (DataType::Date, DataType::Duration(_)) => Some((Cow::Borrowed(lhs), Cow::Borrowed(rhs))),
        (DataType::Duration(lu), DataType::Duration(ru)) => {
            let units = get_time_units(lu, ru);
            let left = if *lu == units {
                Cow::Borrowed(lhs)
            } else {
                Cow::Owned(lhs.cast(&DataType::Duration(units)).ok()?)
            };
            let right = if *ru == units {
                Cow::Borrowed(rhs)
            } else {
                Cow::Owned(rhs.cast(&DataType::Duration(units)).ok()?)
            };
            Some((left, right))
        }
        // swap the order
        (DataType::Duration(_), DataType::Datetime(_, _))
        | (DataType::Duration(_), DataType::Date) => {
            let (right, left) = coerce_time_units(rhs, lhs)?;
            Some((left, right))
        }
        _ => None,
    }
}

#[cfg(feature = "dtype-struct")]
pub fn struct_arithmetic_try_op<F: FnMut(&Series, &Series) -> PolarsResult<Series>>(
    lhs: &Series,
    rhs: &Series,
    mut func: F,
) -> PolarsResult<Series> {
    let lhs = lhs.struct_().unwrap_or_else(|_| unreachable!());
    let rhs = rhs.struct_().unwrap_or_else(|_| unreachable!());
    let lhs_fields = lhs.fields();
    let rhs_fields = rhs.fields();
    Ok(match (lhs_fields.len(), rhs_fields.len()) {
        (_, 1) => {
            let rhs = &rhs.fields()[0];
            lhs.try_apply_fields(|s| func(s, rhs))?.into_series()
        }
        (1, _) => {
            let lhs = &lhs.fields()[0];
            rhs.try_apply_fields(|rhs| func(lhs, rhs))?.into_series()
        }
        _ => {
            let mut rhs_iter = rhs.fields().iter();
            lhs.try_apply_fields(|s| match rhs_iter.next() {
                Some(rhs) => func(s, rhs),
                None => Ok(s.clone()),
            })?
            .into_series()
        }
    })
}

// Series +-/* numbers instead of Series

fn finish_cast(inp: &Series, out: Series) -> Series {
    match inp.dtype() {
        #[cfg(feature = "dtype-date")]
        DataType::Date => out.into_date(),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, tz) => out.into_datetime(*tu, tz.clone()),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => out.into_duration(*tu),
        #[cfg(feature = "dtype-time")]
        DataType::Time => out.into_time(),
        _ => out,
    }
}

impl<T> Sub<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn sub(self, rhs: T) -> Self::Output {
        let s = self.to_physical_repr();
        macro_rules! sub {
            ($ca:expr) => {{
                $ca.sub(rhs).into_series()
            }};
        }

        let out = downcast_as_macro_arg_physical!(s, sub);
        finish_cast(self, out)
    }
}

impl<T> Sub<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl<T> Add<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn add(self, rhs: T) -> Self::Output {
        let s = self.to_physical_repr();
        macro_rules! add {
            ($ca:expr) => {{
                $ca.add(rhs).into_series()
            }};
        }
        let out = downcast_as_macro_arg_physical!(s, add);
        finish_cast(self, out)
    }
}

impl<T> Add<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<T> Div<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn div(self, rhs: T) -> Self::Output {
        let s = self.to_physical_repr();
        macro_rules! div {
            ($ca:expr) => {{
                $ca.div(rhs).into_series()
            }};
        }

        let out = downcast_as_macro_arg_physical!(s, div);
        finish_cast(self, out)
    }
}

impl<T> Div<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        (&self).div(rhs)
    }
}

impl<T> Mul<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn mul(self, rhs: T) -> Self::Output {
        let s = self.to_physical_repr();
        macro_rules! mul {
            ($ca:expr) => {{
                $ca.mul(rhs).into_series()
            }};
        }
        let out = downcast_as_macro_arg_physical!(s, mul);
        finish_cast(self, out)
    }
}

impl<T> Mul<T> for Series
where
    T: Num + NumCast,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl<T> Rem<T> for &Series
where
    T: Num + NumCast,
{
    type Output = Series;

    fn rem(self, rhs: T) -> Self::Output {
        let s = self.to_physical_repr();
        macro_rules! rem {
            ($ca:expr) => {{
                $ca.rem(rhs).into_series()
            }};
        }
        let out = downcast_as_macro_arg_physical!(s, rem);
        finish_cast(self, out)
    }
}

impl<T> Rem<T> for Series
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
    ChunkedArray<T>: IntoSeries,
{
    /// Apply lhs - self
    #[must_use]
    pub fn lhs_sub<N: Num + NumCast>(&self, lhs: N) -> Self {
        let lhs: T::Native = NumCast::from(lhs).expect("could not cast");
        self.apply(|v| lhs - v)
    }

    /// Apply lhs / self
    #[must_use]
    pub fn lhs_div<N: Num + NumCast>(&self, lhs: N) -> Self {
        let lhs: T::Native = NumCast::from(lhs).expect("could not cast");
        self.apply(|v| lhs / v)
    }

    /// Apply lhs % self
    #[must_use]
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
        let s = rhs.to_physical_repr();
        macro_rules! sub {
            ($rhs:expr) => {{
                $rhs.lhs_sub(self).into_series()
            }};
        }
        let out = downcast_as_macro_arg_physical!(s, sub);

        finish_cast(rhs, out)
    }
    fn div(self, rhs: &Series) -> Self::Output {
        let s = rhs.to_physical_repr();
        macro_rules! div {
            ($rhs:expr) => {{
                $rhs.lhs_div(self).into_series()
            }};
        }
        let out = downcast_as_macro_arg_physical!(s, div);

        finish_cast(rhs, out)
    }
    fn mul(self, rhs: &Series) -> Self::Output {
        // order doesn't matter, dispatch to rhs * lhs
        rhs * self
    }
    fn rem(self, rhs: &Series) -> Self::Output {
        let s = rhs.to_physical_repr();
        macro_rules! rem {
            ($rhs:expr) => {{
                $rhs.lhs_rem(self).into_series()
            }};
        }

        let out = downcast_as_macro_arg_physical!(s, rem);

        finish_cast(rhs, out)
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
