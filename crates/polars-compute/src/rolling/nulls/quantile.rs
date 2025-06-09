#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::MutablePrimitiveArray;

use super::*;
use crate::rolling::quantile_filter::SealedRolling;

pub struct QuantileWindow<'a, T: NativeType + IsFloat + PartialOrd> {
    sorted: SortedBufNulls<'a, T>,
    prob: f64,
    method: QuantileMethod,
}

impl<
    'a,
    T: NativeType
        + IsFloat
        + Float
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + One
        + Zero
        + SealedRolling
        + PartialOrd
        + Sub<Output = T>,
> RollingAggWindowNulls<'a, T> for QuantileWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self {
        let params = params.unwrap();
        let RollingFnParams::Quantile(params) = params else {
            unreachable!("expected Quantile params");
        };
        Self {
            sorted: SortedBufNulls::new(slice, validity, start, end, window_size),
            prob: params.prob,
            method: params.method,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let null_count = self.sorted.update(start, end);
        let mut length = self.sorted.len();
        // The min periods_issue will be taken care of when actually rolling
        if null_count == length {
            return None;
        }
        // Nulls are guaranteed to be at the front
        length -= null_count;
        let mut idx = match self.method {
            QuantileMethod::Nearest => ((length as f64) * self.prob) as usize,
            QuantileMethod::Lower | QuantileMethod::Midpoint | QuantileMethod::Linear => {
                ((length as f64 - 1.0) * self.prob).floor() as usize
            },
            QuantileMethod::Higher => ((length as f64 - 1.0) * self.prob).ceil() as usize,
            QuantileMethod::Equiprobable => {
                ((length as f64 * self.prob).ceil() - 1.0).max(0.0) as usize
            },
        };

        idx = std::cmp::min(idx, length - 1);

        // we can unwrap because we sliced of the nulls
        match self.method {
            QuantileMethod::Midpoint => {
                let top_idx = ((length as f64 - 1.0) * self.prob).ceil() as usize;
                Some(
                    (self.sorted.get(idx + null_count).unwrap()
                        + self.sorted.get(top_idx + null_count).unwrap())
                        / T::from::<f64>(2.0f64).unwrap(),
                )
            },
            QuantileMethod::Linear => {
                let float_idx = (length as f64 - 1.0) * self.prob;
                let top_idx = f64::ceil(float_idx) as usize;

                if top_idx == idx {
                    Some(self.sorted.get(idx + null_count).unwrap())
                } else {
                    let proportion = T::from(float_idx - idx as f64).unwrap();
                    Some(
                        proportion
                            * (self.sorted.get(top_idx + null_count).unwrap()
                                - self.sorted.get(idx + null_count).unwrap())
                            + self.sorted.get(idx + null_count).unwrap(),
                    )
                }
            },
            _ => Some(self.sorted.get(idx + null_count).unwrap()),
        }
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.sorted.is_valid(min_periods)
    }
}

pub fn rolling_quantile<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType
        + IsFloat
        + Float
        + std::iter::Sum
        + AddAssign
        + SubAssign
        + Div<Output = T>
        + NumCast
        + One
        + Zero
        + SealedRolling
        + PartialOrd
        + Sub<Output = T>,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };
    if !center {
        let params = params.as_ref().unwrap();
        let RollingFnParams::Quantile(params) = params else {
            unreachable!("expected Quantile params");
        };

        let out = super::quantile_filter::rolling_quantile::<_, MutablePrimitiveArray<_>>(
            params.method,
            min_periods,
            window_size,
            arr.clone(),
            params.prob,
        );
        let out: PrimitiveArray<T> = out.into();
        return Box::new(out);
    }
    rolling_apply_agg_window::<QuantileWindow<_>, _, _>(
        arr.values().as_slice(),
        arr.validity().as_ref().unwrap(),
        window_size,
        min_periods,
        offset_fn,
        params,
    )
}

#[cfg(test)]
mod test {
    use arrow::buffer::Buffer;
    use arrow::datatypes::ArrowDataType;

    use super::*;

    #[test]
    fn test_rolling_median_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );
        let med_pars = Some(RollingFnParams::Quantile(RollingQuantileParams {
            prob: 0.5,
            method: QuantileMethod::Linear,
        }));

        let out = rolling_quantile(arr, 2, 2, false, None, med_pars);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(3.5)]);

        let out = rolling_quantile(arr, 2, 1, false, None, med_pars);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 4, 1, false, None, med_pars);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(2.0), Some(3.0)]);

        let out = rolling_quantile(arr, 4, 1, true, None, med_pars);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 4, 4, true, None, med_pars);
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, None]);
    }

    #[test]
    fn test_rolling_quantile_nulls_limits() {
        // compare quantiles to corresponding min/max/median values
        let buf = Buffer::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let values = &PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, false, true, true])),
        );

        let methods = vec![
            QuantileMethod::Lower,
            QuantileMethod::Higher,
            QuantileMethod::Nearest,
            QuantileMethod::Midpoint,
            QuantileMethod::Linear,
            QuantileMethod::Equiprobable,
        ];

        for method in methods {
            let min_pars = Some(RollingFnParams::Quantile(RollingQuantileParams {
                prob: 0.0,
                method,
            }));
            let out1 = rolling_min(values, 2, 1, false, None, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 2, 1, false, None, min_pars);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let max_pars = Some(RollingFnParams::Quantile(RollingQuantileParams {
                prob: 1.0,
                method,
            }));
            let out1 = rolling_max(values, 2, 1, false, None, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 2, 1, false, None, max_pars);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);
        }
    }
}
