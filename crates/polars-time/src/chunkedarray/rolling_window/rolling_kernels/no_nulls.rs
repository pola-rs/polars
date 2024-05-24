use arrow::bitmap::MutableBitmap;
use arrow::legacy::kernels::rolling::no_nulls::{self, RollingAggWindowNoNulls};
use bytemuck::allocation::zeroed_vec;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;

use super::*;

// Use an aggregation window that maintains the state.
// Fastpath if values were known to already be sorted by time.
pub(crate) fn rolling_apply_agg_window_sorted<'a, Agg, T, O>(
    values: &'a [T],
    offsets: O,
    min_periods: usize,
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

                // On the Python side, if `min_periods` wasn't specified, it is set to
                // `1`. In that case, this condition is the same as checking
                // `if start == end`.
                if len < (min_periods as IdxSize) {
                    None
                } else {
                    // SAFETY:
                    // we are in bounds
                    unsafe { agg_window.update(start as usize, end as usize) }
                }
            })
        })
        .collect::<PolarsResult<PrimitiveArray<T>>>()?;

    Ok(Box::new(out))
}

// Instantiate a bitmap when the first null value is encountered.
// Set the validity at index `idx` to `false`.
fn instantiate_bitmap_if_null_and_set_false_at_idx(
    validity: &mut Option<MutableBitmap>,
    len: usize,
    idx: usize,
) {
    let bitmap = validity.get_or_insert_with(|| {
        let mut bitmap = MutableBitmap::with_capacity(len);
        bitmap.extend_constant(len, true);
        bitmap
    });
    bitmap.set(idx, false);
}

// Use an aggregation window that maintains the state
pub(crate) fn rolling_apply_agg_window<'a, Agg, T, O>(
    values: &'a [T],
    offsets: O,
    min_periods: usize,
    params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
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
    let sorting_indices = sorting_indices.expect("`sorting_indices` should have been set");
    // start with a dummy index, will be overwritten on first iteration.
    let mut agg_window = Agg::new(values, 0, 0, params);

    let mut out = zeroed_vec(values.len());
    let mut validity: Option<MutableBitmap> = None;
    offsets.enumerate().try_for_each(|(idx, result)| {
        let (start, len) = result?;
        let end = start + len;
        let out_idx = unsafe { sorting_indices.get_unchecked(idx) };

        // On the Python side, if `min_periods` wasn't specified, it is set to
        // `1`. In that case, this condition is the same as checking
        // `if start == end`.
        if len >= (min_periods as IdxSize) {
            // SAFETY:
            // we are in bound
            let res = unsafe { agg_window.update(start as usize, end as usize) };

            if let Some(res) = res {
                // SAFETY: `idx` is in bounds because `sorting_indices` was just taken from
                // `by`, which has already been checked to be the same length as the values.
                unsafe { *out.get_unchecked_mut(*out_idx as usize) = res };
            } else {
                instantiate_bitmap_if_null_and_set_false_at_idx(
                    &mut validity,
                    values.len(),
                    *out_idx as usize,
                )
            }
        } else {
            instantiate_bitmap_if_null_and_set_false_at_idx(
                &mut validity,
                values.len(),
                *out_idx as usize,
            )
        }
        Ok::<(), PolarsError>(())
    })?;

    let out = PrimitiveArray::<T>::from_vec(out).with_validity(validity.map(|x| x.into()));

    Ok(Box::new(out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_min<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;
    if sorting_indices.is_none() {
        rolling_apply_agg_window_sorted::<no_nulls::MinWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
        )
    } else {
        rolling_apply_agg_window::<no_nulls::MinWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
            sorting_indices,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_max<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + IsFloat + Bounded + NumCast + Mul<Output = T>,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;
    if sorting_indices.is_none() {
        rolling_apply_agg_window_sorted::<no_nulls::MaxWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
        )
    } else {
        rolling_apply_agg_window::<no_nulls::MaxWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
            sorting_indices,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_sum<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + std::iter::Sum + NumCast + Mul<Output = T> + AddAssign + SubAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;
    if sorting_indices.is_none() {
        rolling_apply_agg_window_sorted::<no_nulls::SumWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
        )
    } else {
        rolling_apply_agg_window::<no_nulls::SumWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
            sorting_indices,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_mean<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;
    if sorting_indices.is_none() {
        rolling_apply_agg_window_sorted::<no_nulls::MeanWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
        )
    } else {
        rolling_apply_agg_window::<no_nulls::MeanWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            None,
            sorting_indices,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_var<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;
    if sorting_indices.is_none() {
        rolling_apply_agg_window_sorted::<no_nulls::VarWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            params,
        )
    } else {
        rolling_apply_agg_window::<no_nulls::VarWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            params,
            sorting_indices,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_quantile<T>(
    values: &[T],
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    params: DynArgs,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + Float + std::iter::Sum<T> + SubAssign + AddAssign + IsFloat,
{
    let offset_iter = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => group_by_values_iter(period, time, closed_window, tu, tz.parse::<Tz>().ok()),
        _ => group_by_values_iter(period, time, closed_window, tu, None),
    }?;
    if sorting_indices.is_none() {
        rolling_apply_agg_window_sorted::<no_nulls::QuantileWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            params,
        )
    } else {
        rolling_apply_agg_window::<no_nulls::QuantileWindow<_>, _, _>(
            values,
            offset_iter,
            min_periods,
            params,
            sorting_indices,
        )
    }
}
