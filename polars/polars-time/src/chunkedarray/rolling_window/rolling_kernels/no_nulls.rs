use polars_arrow::kernels::rolling::no_nulls::{self, RollingAggWindowNoNulls};
use polars_core::export::num;

use super::*;

// Use an aggregation window that maintains the state
pub(crate) fn rolling_apply_agg_window<'a, Agg, T, O>(values: &'a [T], offsets: O) -> ArrayRef
where
    // items (offset, len) -> so offsets are offset, offset + len
    Agg: RollingAggWindowNoNulls<'a, T>,
    O: Iterator<Item = (IdxSize, IdxSize)> + TrustedLen,
    T: Debug + IsFloat + NativeType,
{
    if values.is_empty() {
        let out: Vec<T> = vec![];
        return Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), out.into(), None));
    }
    // start with a dummy index, will be overwritten on first iteration.
    let mut agg_window = Agg::new(values, 0, 0);

    let out = offsets
        .map(|(start, len)| {
            let end = start + len;

            if start == end {
                None
            } else {
                // safety:
                // we are in bounds
                Some(unsafe { agg_window.update(start as usize, end as usize) })
            }
        })
        .collect::<PrimitiveArray<T>>();

    Box::new(out)
}

pub(crate) fn rolling_min<T>(
    values: &[T],
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    let offset_iter = groupby_values_iter(period, offset, time, closed_window, tu);
    rolling_apply_agg_window::<no_nulls::MinWindow<_>, _, _>(values, offset_iter)
}

pub(crate) fn rolling_max<T>(
    values: &[T],
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    let offset_iter = groupby_values_iter(period, offset, time, closed_window, tu);
    rolling_apply_agg_window::<no_nulls::MaxWindow<_>, _, _>(values, offset_iter)
}

pub(crate) fn rolling_sum<T>(
    values: &[T],
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + NumCast + Mul<Output = T> + AddAssign + SubAssign + IsFloat,
{
    let offset_iter = groupby_values_iter(period, offset, time, closed_window, tu);
    rolling_apply_agg_window::<no_nulls::SumWindow<_>, _, _>(values, offset_iter)
}

pub(crate) fn rolling_mean<T>(
    values: &[T],
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> ArrayRef
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = groupby_values_iter(period, offset, time, closed_window, tu);
    rolling_apply_agg_window::<no_nulls::MeanWindow<_>, _, _>(values, offset_iter)
}

pub(crate) fn rolling_var<T>(
    values: &[T],
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> ArrayRef
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = groupby_values_iter(period, offset, time, closed_window, tu);
    rolling_apply_agg_window::<no_nulls::VarWindow<_>, _, _>(values, offset_iter)
}

pub(crate) fn rolling_std<T>(
    values: &[T],
    period: Duration,
    offset: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> ArrayRef
where
    T: NativeType
        + Float
        + IsFloat
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + One
        + Sub<Output = T>
        + num::pow::Pow<T, Output = T>,
{
    let offset_iter = groupby_values_iter(period, offset, time, closed_window, tu);
    rolling_apply_agg_window::<no_nulls::StdWindow<_>, _, _>(values, offset_iter)
}
