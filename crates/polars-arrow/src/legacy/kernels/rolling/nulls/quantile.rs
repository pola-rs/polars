use polars_utils::slice::GetSaferUnchecked;

use super::*;
use crate::array::MutablePrimitiveArray;

pub struct QuantileWindow<'a, T: NativeType + IsFloat + PartialOrd> {
    sorted: SortedBufNulls<'a, T>,
    prob: f64,
    interpol: QuantileInterpolOptions,
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
            + PartialOrd
            + Sub<Output = T>,
    > RollingAggWindowNulls<'a, T> for QuantileWindow<'a, T>
{
    unsafe fn new(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: DynArgs,
    ) -> Self {
        let params = params.unwrap();
        let params = params.downcast_ref::<RollingQuantileParams>().unwrap();
        Self {
            sorted: SortedBufNulls::new(slice, validity, start, end),
            prob: params.prob,
            interpol: params.interpol,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) -> Option<T> {
        let (values, null_count) = self.sorted.update(start, end);
        // The min periods_issue will be taken care of when actually rolling
        if null_count == values.len() {
            return None;
        }
        // Nulls are guaranteed to be at the front
        let values = &values[null_count..];
        let length = values.len();

        let mut idx = match self.interpol {
            QuantileInterpolOptions::Nearest => ((length as f64) * self.prob) as usize,
            QuantileInterpolOptions::Lower
            | QuantileInterpolOptions::Midpoint
            | QuantileInterpolOptions::Linear => {
                ((length as f64 - 1.0) * self.prob).floor() as usize
            },
            QuantileInterpolOptions::Higher => ((length as f64 - 1.0) * self.prob).ceil() as usize,
        };

        idx = std::cmp::min(idx, length - 1);

        // we can unwrap because we sliced of the nulls
        match self.interpol {
            QuantileInterpolOptions::Midpoint => {
                let top_idx = ((length as f64 - 1.0) * self.prob).ceil() as usize;
                Some(
                    (values.get_unchecked_release(idx).unwrap()
                        + values.get_unchecked_release(top_idx).unwrap())
                        / T::from::<f64>(2.0f64).unwrap(),
                )
            },
            QuantileInterpolOptions::Linear => {
                let float_idx = (length as f64 - 1.0) * self.prob;
                let top_idx = f64::ceil(float_idx) as usize;

                if top_idx == idx {
                    Some(values.get_unchecked_release(idx).unwrap())
                } else {
                    let proportion = T::from(float_idx - idx as f64).unwrap();
                    Some(
                        proportion
                            * (values.get_unchecked_release(top_idx).unwrap()
                                - values.get_unchecked_release(idx).unwrap())
                            + values.get_unchecked_release(idx).unwrap(),
                    )
                }
            },
            _ => Some(values.get_unchecked_release(idx).unwrap()),
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
    params: DynArgs,
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
        let params = params.downcast_ref::<RollingQuantileParams>().unwrap();
        let out = super::quantile_filter::rolling_quantile::<_, MutablePrimitiveArray<_>>(
            params.interpol,
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
    use super::*;
    use crate::buffer::Buffer;
    use crate::datatypes::ArrowDataType;

    #[test]
    fn test_rolling_median_nulls() {
        let buf = Buffer::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = &PrimitiveArray::new(
            ArrowDataType::Float64,
            buf,
            Some(Bitmap::from(&[true, false, true, true])),
        );
        let med_pars = Some(Arc::new(RollingQuantileParams {
            prob: 0.5,
            interpol: QuantileInterpolOptions::Linear,
        }) as Arc<dyn Any + Send + Sync>);

        let out = rolling_quantile(arr, 2, 2, false, None, med_pars.clone());
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[None, None, None, Some(3.5)]);

        let out = rolling_quantile(arr, 2, 1, false, None, med_pars.clone());
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 4, 1, false, None, med_pars.clone());
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(1.0), Some(2.0), Some(3.0)]);

        let out = rolling_quantile(arr, 4, 1, true, None, med_pars.clone());
        let out = out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let out = out.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
        assert_eq!(out, &[Some(1.0), Some(2.0), Some(3.0), Some(3.5)]);

        let out = rolling_quantile(arr, 4, 4, true, None, med_pars.clone());
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

        let interpol_options = vec![
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            let min_pars = Some(Arc::new(RollingQuantileParams {
                prob: 0.0,
                interpol,
            }) as Arc<dyn Any + Send + Sync>);
            let out1 = rolling_min(values, 2, 1, false, None, None);
            let out1 = out1.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out1 = out1.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            let out2 = rolling_quantile(values, 2, 1, false, None, min_pars);
            let out2 = out2.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
            let out2 = out2.into_iter().map(|v| v.copied()).collect::<Vec<_>>();
            assert_eq!(out1, out2);

            let max_pars = Some(Arc::new(RollingQuantileParams {
                prob: 1.0,
                interpol,
            }) as Arc<dyn Any + Send + Sync>);
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
