#[cfg(feature = "timezones")]
use chrono_tz::Tz;
use polars_arrow::kernels::rolling::no_nulls::{self, RollingAggWindowNoNulls};

use super::*;

// Use an aggregation window that maintains the state
pub(crate) fn rolling_apply_agg_window<'a, Agg, T, O>(
    values: &'a [T],
    offsets: O,
    params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    // items (offset, len) -> so offsets are offset, offset + len
    Agg: RollingAggWindowNoNulls<'a, T>,
    O: Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen,
    T: Debug + IsFloat + NativeType,
{
    if values.is_empty() {
        let out: Vec<T> = vec![];
        return Ok(Box::new(PrimitiveArray::new(
            T::PRIMITIVE.into(),
            out.into(),
            None,
        )));
    }
    // start with a dummy index, will be overwritten on first iteration.
    let mut agg_window = Agg::new(values, 0, 0, params);

    let out = offsets
        .map(|result| {
            result.map(|(start, len)| {
                let end = start + len;

                if start == end {
                    None
                } else {
                    // safety:
                    // we are in bounds
                    Some(unsafe { agg_window.update(start as usize, end as usize) })
                }
            })
        })
        .collect::<PolarsResult<PrimitiveArray<T>>>()?;

    Ok(Box::new(out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_min<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => groupby_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => groupby_values_iter(period, time, closed_window, tu, None),
    };
    rolling_apply_agg_window::<no_nulls::MinWindow<_>, _, _>(values, offset_iter, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_max<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => groupby_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => groupby_values_iter(period, time, closed_window, tu, None),
    };
    rolling_apply_agg_window::<no_nulls::MaxWindow<_>, _, _>(values, offset_iter, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_sum<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + std::iter::Sum + NumCast + Mul<Output = T> + AddAssign + SubAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => groupby_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => groupby_values_iter(period, time, closed_window, tu, None),
    };
    rolling_apply_agg_window::<no_nulls::SumWindow<_>, _, _>(values, offset_iter, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_mean<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => groupby_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => groupby_values_iter(period, time, closed_window, tu, None),
    };
    rolling_apply_agg_window::<no_nulls::MeanWindow<_>, _, _>(values, offset_iter, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_var<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    params: DynArgs,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => groupby_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => groupby_values_iter(period, time, closed_window, tu, None),
    };
    rolling_apply_agg_window::<no_nulls::VarWindow<_>, _, _>(values, offset_iter, params)
}
