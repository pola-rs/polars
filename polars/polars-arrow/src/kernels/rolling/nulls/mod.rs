mod mean;
mod min_max;
mod quantile;
mod sum;
mod variance;

use super::*;
pub use mean::rolling_mean;
pub use min_max::{rolling_max, rolling_min};
pub use quantile::{rolling_median, rolling_quantile};
pub use sum::rolling_sum;
pub use variance::rolling_var;

pub(crate) trait RollingAggWindow<'a, T: NativeType> {
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        min_periods: usize,
    ) -> Self;

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T>;
}

// Use an aggregation window that maintains the state
pub(super) fn rolling_apply_agg_window<'a, Agg, T, Fo>(
    values: &'a [T],
    validity: &'a Bitmap,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    Agg: RollingAggWindow<'a, T>,
    T: IsFloat + NativeType,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    // Safety; we are in bounds
    let mut agg_window = unsafe { Agg::new(values, validity, start, end, min_periods) };

    let mut validity = match create_validity(min_periods, len as usize, window_size, det_offsets_fn)
    {
        Some(v) => v,
        None => {
            let mut validity = MutableBitmap::with_capacity(len);
            validity.extend_constant(len, true);
            validity
        }
    };

    let out = (0..len)
        .map(|idx| {
            let (start, end) = det_offsets_fn(idx, window_size, len);
            // safety:
            // we are in bounds
            let agg = unsafe { agg_window.update(start, end) };
            match agg {
                Some(val) => val,
                None => {
                    // safety: we are in bounds
                    unsafe { validity.set_unchecked(idx, false) };
                    T::default()
                }
            }
        })
        .collect_trusted::<Vec<_>>();

    Arc::new(PrimitiveArray::from_data(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::kernels::rolling::nulls::mean::rolling_mean;
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;

    fn get_null_arr() -> PrimitiveArray<f64> {
        // 1, None, -1, 4
        let buf = Buffer::from(vec![1.0, 0.0, -1.0, 4.0]);
        PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        )
    }

    #[test]
    fn test_rolling_sum_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );

        let out = rolling_sum(arr, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(7.0)]);

        let out = rolling_sum(arr, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(7.0)]);

        let out = rolling_sum(arr, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(4.0), Some(8.0)]);

        let out = rolling_sum(arr, 4, 1, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(4.0), Some(8.0), Some(7.0)]);

        let out = rolling_sum(arr, 4, 4, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }

    #[test]
    fn test_rolling_mean_nulls() {
        let arr = get_null_arr();
        let arr = &arr;

        let out = rolling_mean(arr, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(1.5)]);

        let out = rolling_mean(arr, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(-1.0), Some(1.5)]);

        let out = rolling_mean(arr, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(0.0), Some(4.0 / 3.0)]);
    }

    #[test]
    fn test_rolling_var_nulls() {
        let arr = get_null_arr();
        let arr = &arr;

        let out = rolling_var(arr, 3, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out
            .into_iter()
            .map(|v| v.copied().unwrap())
            .collect::<Vec<_>>();

        // we cannot compare nans, so we compare the string values
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!("{:?}", &[f64::nan(), f64::nan(), 2.0, 12.5])
        );

        let out = rolling_var(arr, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out
            .into_iter()
            .map(|v| v.copied().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            format!("{:?}", out.as_slice()),
            format!("{:?}", &[f64::nan(), f64::nan(), 2.0, 6.333333333333334])
        );
    }

    #[test]
    fn test_rolling_max_no_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::from_data(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, true, true, true])),
        );
        let out = rolling_max(arr, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);

        let out = rolling_max(arr, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, Some(2.0), Some(3.0), Some(4.0)]);

        let out = rolling_max(arr, 4, 4, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(4.0)])
    }
}
