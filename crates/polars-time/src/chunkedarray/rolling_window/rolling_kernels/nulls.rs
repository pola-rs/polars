use arrow::bitmap::{Bitmap, MutableBitmap};
use bytemuck::allocation::zeroed_vec;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
use polars_compute::rolling::nulls::RollingAggWindowNulls;
use polars_compute::rolling::{MeanWindow, RollingFnParams};

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn rolling_mean<'a, T>(
    values: &'a [T],
    validity: &'a Bitmap,
    period: Duration,
    time: &[i64],
    closed_window: ClosedWindow,
    min_periods: usize,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
    _params: Option<RollingFnParams>,
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
        rolling_apply_agg_window_sorted::<MeanWindow<_>, _, _, _>(
            values,
            validity,
            offset_iter,
            min_periods,
            None,
        )
    } else {
        rolling_apply_agg_window::<MeanWindow<_>, _, _, _>(
            values,
            validity,
            offset_iter,
            min_periods,
            None,
            sorting_indices,
        )
    }
}

// Use an aggregation window that maintains the state.
// Fastpath if values were known to already be sorted by time.
pub(crate) fn rolling_apply_agg_window_sorted<'a, Agg, T, O, Out>(
    values: &'a [T],
    validity: &'a Bitmap,
    offsets: O,
    min_periods: usize,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    // items (offset, len) -> so offsets are offset, offset + len
    Agg: RollingAggWindowNulls<'a, T, Out>,
    O: Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen,
    T: Debug + NativeType,
    Out: Debug + NativeType,
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
    let mut agg_window = Agg::new(values, validity, 0, 0, params, None);

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
        .collect::<PolarsResult<PrimitiveArray<Out>>>()?;

    Ok(Box::new(out))
}

// Use an aggregation window that maintains the state
pub(crate) fn rolling_apply_agg_window<'a, Agg, T, O, Out>(
    values: &'a [T],
    validity: &'a Bitmap,
    offsets: O,
    min_periods: usize,
    params: Option<RollingFnParams>,
    sorting_indices: Option<&[IdxSize]>,
) -> PolarsResult<ArrayRef>
where
    // items (offset, len) -> so offsets are offset, offset + len
    Agg: RollingAggWindowNulls<'a, T, Out>,
    O: Iterator<Item = PolarsResult<(IdxSize, IdxSize)>> + TrustedLen,
    T: Debug + NativeType,
    Out: Debug + NativeType,
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
    let mut agg_window = Agg::new(values, validity, 0, 0, params, None);

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

    let out = PrimitiveArray::<Out>::from_vec(out).with_validity(validity.map(|x| x.into()));

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
