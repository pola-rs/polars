use super::*;
use crate::index::IdxSize;
use crate::trusted_len::TrustedLen;

// used by agg_quantile
#[allow(clippy::too_many_arguments)]
pub fn rolling_quantile_by_iter<T, O>(
    values: &[T],
    bitmap: &Bitmap,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    offsets: O,
) -> ArrayRef
where
    O: Iterator<Item = (IdxSize, IdxSize)> + TrustedLen,
    T: std::iter::Sum<T>
        + NativeType
        + Copy
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + IsFloat
        + AddAssign
        + Zero,
{
    if values.is_empty() {
        let out: Vec<T> = vec![];
        return Box::new(PrimitiveArray::new(T::PRIMITIVE.into(), out.into(), None));
    }

    let len = values.len();
    // Safety
    // we are in bounds
    let mut sorted_window = unsafe { SortedBufNulls::new(values, bitmap, 0, 1) };

    let mut validity = MutableBitmap::with_capacity(len);
    validity.extend_constant(len, true);

    let out = offsets
        .enumerate()
        .map(|(idx, (start, len))| {
            let end = start + len;

            if start == end {
                validity.set(idx, false);
                T::default()
            } else {
                // safety
                // we are in bounds
                unsafe { sorted_window.update(start as usize, end as usize) };
                let null_count = sorted_window.null_count;
                let window = sorted_window.window();

                match compute_quantile(window, null_count, quantile, interpolation, 1) {
                    Some(val) => val,
                    None => {
                        validity.set(idx, false);
                        T::default()
                    }
                }
            }
        })
        .collect_trusted::<Vec<T>>();

    Box::new(PrimitiveArray::new(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

#[allow(clippy::too_many_arguments)]
fn rolling_apply_quantile<T, Fo, Fa>(
    values: &[T],
    bitmap: &Bitmap,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    // &[Option<T>] -> window values
    // usize -> null_count
    // f764 ->  quantile
    // QuantileInterpolOptions -> Interpolation option
    // usize -> min_periods
    Fa: Fn(&[Option<T>], usize, f64, QuantileInterpolOptions, usize) -> Option<T>,
    T: Default + NativeType + IsFloat + PartialOrd,
{
    let len = values.len();
    let (start, end) = det_offsets_fn(0, window_size, len);
    // Safety
    // we are in bounds
    let mut sorted_window = unsafe { SortedBufNulls::new(values, bitmap, start, end) };

    let mut validity = match create_validity(min_periods, len, window_size, det_offsets_fn) {
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

            // safety
            // we are in bounds
            unsafe { sorted_window.update(start, end) };
            let null_count = sorted_window.null_count;
            let window = sorted_window.window();

            match aggregator(window, null_count, quantile, interpolation, min_periods) {
                Some(val) => val,
                None => {
                    validity.set(idx, false);
                    T::default()
                }
            }
        })
        .collect_trusted::<Vec<T>>();

    Box::new(PrimitiveArray::new(
        T::PRIMITIVE.into(),
        out.into(),
        Some(validity.into()),
    ))
}

fn compute_quantile<T>(
    values: &[Option<T>],
    null_count: usize,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType
        + std::iter::Sum<T>
        + Zero
        + AddAssign
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + IsFloat,
{
    if (values.len() - null_count) < min_periods {
        return None;
    }
    // slice off nulls
    let values = &values[null_count..];
    let length = values.len();

    let mut idx = match interpolation {
        QuantileInterpolOptions::Nearest => ((length as f64) * quantile) as usize,
        QuantileInterpolOptions::Lower
        | QuantileInterpolOptions::Midpoint
        | QuantileInterpolOptions::Linear => ((length as f64 - 1.0) * quantile).floor() as usize,
        QuantileInterpolOptions::Higher => ((length as f64 - 1.0) * quantile).ceil() as usize,
    };

    idx = std::cmp::min(idx, length - 1);

    // we can unwrap because we sliced of the nulls
    match interpolation {
        QuantileInterpolOptions::Midpoint => {
            let top_idx = ((length as f64 - 1.0) * quantile).ceil() as usize;
            Some(
                (values[idx].unwrap() + values[top_idx].unwrap()) / T::from::<f64>(2.0f64).unwrap(),
            )
        }
        QuantileInterpolOptions::Linear => {
            let float_idx = (length as f64 - 1.0) * quantile;
            let top_idx = f64::ceil(float_idx) as usize;

            if top_idx == idx {
                Some(values[idx].unwrap())
            } else {
                let proportion = T::from(float_idx - idx as f64).unwrap();
                Some(
                    proportion * (values[top_idx].unwrap() - values[idx].unwrap())
                        + values[idx].unwrap(),
                )
            }
        }
        _ => Some(values[idx].unwrap()),
    }
}
pub fn rolling_median<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType
        + std::iter::Sum
        + Zero
        + AddAssign
        + Copy
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + IsFloat,
{
    rolling_quantile(
        arr,
        0.5,
        QuantileInterpolOptions::Linear,
        window_size,
        min_periods,
        center,
        weights,
    )
}

pub fn rolling_quantile<T>(
    arr: &PrimitiveArray<T>,
    quantile: f64,
    interpolation: QuantileInterpolOptions,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType
        + std::iter::Sum
        + Zero
        + AddAssign
        + Copy
        + std::cmp::PartialOrd
        + num::ToPrimitive
        + NumCast
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + IsFloat,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply_quantile(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets_center,
            compute_quantile,
        )
    } else {
        rolling_apply_quantile(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            quantile,
            interpolation,
            window_size,
            min_periods,
            det_offsets,
            compute_quantile,
        )
    }
}

#[cfg(test)]
mod test {
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;

    use super::*;
    use crate::kernels::rolling::nulls::{rolling_max, rolling_min};

    #[test]
    fn test_rolling_median_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::new(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 2, 2, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(3.5)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 2, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 4, 1, false, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(2.0), Some(3.0)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 4, 1, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 0.5, QuantileInterpolOptions::Linear, 4, 4, true, None);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }

    #[test]
    fn test_rolling_quantile_nulls_limits() {
        // compare quantiles to corresponding min/max/median values
        let buf = Buffer::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let values = &PrimitiveArray::new(
            DataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, false, true, true])),
        );

        let interpol_options = vec![
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            let out1 = rolling_min(values, 2, 1, false, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 0.0, interpol, 2, 1, false, None);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let out1 = rolling_max(values, 2, 1, false, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 1.0, interpol, 2, 1, false, None);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);
        }
    }
}
