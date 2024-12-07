mod mean;
mod min_max;
mod quantile;
mod sum;
mod variance;

pub use mean::*;
pub use min_max::*;
pub use quantile::*;
pub use sum::*;
pub use variance::*;

use super::*;
use crate::bitmap::utils::count_zeros;
use crate::datatypes::ArrowDataType;

pub trait RollingAggWindowNulls<'a, T: NativeType> {
    /// # Safety
    /// `start` and `end` must be in bounds for `slice` and `validity`
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
    ) -> Self;

    /// # Safety
    /// `start` and `end` must be in bounds of `slice` and `bitmap`
    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T>;

    fn is_valid(&self, min_periods: usize) -> bool;
}

pub trait RollingAggWindowBoolNulls<Fo: Fn(Idx, WindowSize, Len) -> (Start, End)> {
    /// set none values to true/false and treat it as no null
    fn result_values(
        bitmap: &Bitmap,
        validity: &Bitmap,
        window_size: WindowSize,
        det_effects_fn: Fo,
    ) -> Bitmap;
}

// Use an aggregation window that maintains the state
pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    validity: &'a Bitmap,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    Agg: RollingAggWindowNulls<'a, T>,
    T: IsFloat + NativeType,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    // SAFETY; we are in bounds
    let mut agg_window = unsafe { Agg::new(values, validity, start, end, params) };

    let mut validity = create_validity(min_periods, len, window_size, det_offsets_fn)
        .unwrap_or_else(|| {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            validity
        });

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            // SAFETY:
            // we are in bounds
            let agg = unsafe { agg_window.update(start, end) };
            match agg {
                Some(val) => {
                    if agg_window.is_valid(min_periods) {
                        val
                    } else {
                        // SAFETY: we are in bounds
                        unsafe { validity.set_unchecked(idx, false) };
                        T::default()
                    }
                },
                None => {
                    // SAFETY: we are in bounds
                    unsafe { validity.set_unchecked(idx, false) };
                    T::default()
                },
            }
        })
        .collect_trusted::<Vec<_>>();

    Box::new(PrimitiveArray::new(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

pub(super) fn rolling_apply_agg_window_bool<'a, Agg, Fo>(
    values: &'a Bitmap,
    validity: &'a Bitmap,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    det_effects_fn: Fo,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    Agg: RollingAggWindowBoolNulls<Fo>,
{
    let len = values.len();
    let mut new_validity = create_validity(min_periods, len, window_size, det_offsets_fn)
        .unwrap_or_else(|| MutableBitmap::from_len_set(len));

    let mut null_cnt = 0;
    let mut last_start = 0;
    let mut last_end = 0;
    let slice = validity.as_slice().0;

    for idx in 0..len {
        let (start, end) = det_offsets_fn(idx, window_size, len);

        null_cnt -= count_zeros(slice, last_start, start - last_start);
        null_cnt += count_zeros(slice, last_end, end - last_end);

        if (end - start) - null_cnt < min_periods {
            new_validity.set(idx, false);
        }

        last_start = start;
        last_end = end;
    }

    let out = Agg::result_values(values, validity, window_size, det_effects_fn);

    BooleanArray::new(ArrowDataType::Boolean, out, Some(new_validity.into())).boxed()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::array::{Array, Int32Array};
    use crate::buffer::Buffer;

    fn get_null_arr() -> PrimitiveArray<f64> {
        // 1, None, -1, 4
        let buf = Buffer::from(vec![1.0, 0.0, -1.0, 4.0]);
        PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        )
    }

    #[test]
    fn test_rolling_sum_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );

        let out = rolling_sum(arr, 2, 2, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(7.0)]);

        let out = rolling_sum(arr, 2, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(7.0)]);

        let out = rolling_sum(arr, 4, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(4.0), Some(8.0)]);

        let out = rolling_sum(arr, 4, 1, true, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(4.0), Some(8.0), Some(7.0)]);

        let out = rolling_sum(arr, 4, 4, true, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }

    #[test]
    fn test_rolling_mean_nulls() {
        let arr = get_null_arr();
        let arr = &arr;

        let out = rolling_mean(arr, 2, 2, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(1.5)]);

        let out = rolling_mean(arr, 2, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(-1.0), Some(1.5)]);

        let out = rolling_mean(arr, 4, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(0.0), Some(4.0 / 3.0)]);
    }

    #[test]
    fn test_rolling_var_nulls() {
        let arr = get_null_arr();
        let arr = &arr;

        let out = rolling_var(arr, 3, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();

        assert_eq!(out, &[None, None, Some(2.0), Some(12.5)]);

        let testpars = Some(RollingFnParams::Var(RollingVarParams { ddof: 0 }));
        let out = rolling_var(arr, 3, 1, false, None, testpars.clone());
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();

        assert_eq!(out, &[Some(0.0), Some(0.0), Some(1.0), Some(6.25)]);

        let out = rolling_var(arr, 4, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, Some(2.0), Some(6.333333333333334)]);

        let out = rolling_var(arr, 4, 1, false, None, testpars.clone());
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(
            out,
            &[Some(0.), Some(0.0), Some(1.0), Some(4.222222222222222)]
        );
    }

    #[test]
    fn test_rolling_max_no_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, true, true, true])),
        );
        let out = rolling_max(arr, 4, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);

        let out = rolling_max(arr, 2, 2, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(2.0), Some(3.0), Some(4.0)]);

        let out = rolling_max(arr, 4, 4, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(4.0)]);

        let buf = Buffer::from(vec![4.0, 3.0, 2.0, 1.0]);
        let arr = &PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, true, true, true])),
        );
        let out = rolling_max(arr, 2, 1, false, None, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(4.0), Some(4.0), Some(3.0), Some(2.0)]);

        let out =
            super::no_nulls::rolling_max(arr.values().as_slice(), 2, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(4.0), Some(4.0), Some(3.0), Some(2.0)]);

        // bool
        let values = Bitmap::from(&[false, false, true, true, false]);
        let arr = &BooleanArray::new(
            ArrowDataType::Boolean,
            values,
            Some(Bitmap::from(&[true, true, true, true, true])),
        );
        let out = rolling_max_bool(arr, 4, 1, false);
        let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        let out = out.into_iter().collect::<Vec<_>>();
        assert_eq!(
            out,
            &[Some(false), Some(false), Some(true), Some(true), Some(true)]
        );

        let out = rolling_max_bool(arr, 2, 2, false);
        let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        let out = out.into_iter().collect::<Vec<_>>();
        assert_eq!(
            out,
            &[None, Some(false), Some(true), Some(true), Some(true)]
        );

        let out = rolling_max_bool(arr, 4, 4, false);
        let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        let out = out.into_iter().collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(true), Some(true)]);

        let values = Bitmap::from(&[true, true, false, false, true]);
        let arr = &BooleanArray::new(
            ArrowDataType::Boolean,
            values,
            Some(Bitmap::from(&[true, true, true, true, true])),
        );
        let out = rolling_max_bool(arr, 2, 1, false);
        let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        let out = out.into_iter().collect::<Vec<_>>();
        assert_eq!(
            out,
            &[Some(true), Some(true), Some(true), Some(false), Some(true)]
        );

        let out = super::no_nulls::rolling_max_bool(arr.values(), 2, 1, false);
        let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        let out = out.into_iter().collect::<Vec<_>>();
        assert_eq!(
            out,
            &[Some(true), Some(true), Some(true), Some(false), Some(true)]
        );
    }

    #[test]
    fn test_rolling_extrema_nulls() {
        let vals = vec![3, 3, 3, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let mut validity = MutableBitmap::new();
        validity.extend_constant(vals.len(), true);

        let window_size = 3;
        let min_periods = 3;

        let arr = Int32Array::new(ArrowDataType::Int32, vals.into(), Some(validity.into()));

        let out = rolling_apply_agg_window::<MaxWindow<_>, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        );
        let arr = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(arr.null_count(), 2);
        assert_eq!(
            &arr.values().as_slice()[2..],
            &[3, 10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3]
        );
    }
}
