use super::*;
use crate::utils::CustomIterTools;
use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::bitmap::utils::{count_zeros, get_bit_unchecked};
use arrow::buffer::MutableBuffer;
use arrow::types::NativeType;
use num::{Float, One, Zero};
use std::ops::AddAssign;
use std::sync::Arc;

fn rolling_apply<T, K, Fo, Fa>(
    values: &[T],
    bitmap: &Bitmap,
    window_size: usize,
    min_periods: usize,
    det_offsets_fn: Fo,
    aggregator: Fa,
) -> ArrayRef
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End) + Copy,
    // &[T] -> values of array
    // &[u8] -> validity bytes
    // usize -> offset in validity bytes array
    // usize -> min_periods
    Fa: Fn(&[T], &[u8], usize, usize) -> Option<K>,
    K: NativeType + Default,
{
    let len = values.len();
    let (validity_bytes, offset, _) = bitmap.as_slice();

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
            let vals = unsafe { values.get_unchecked(start..end) };
            match aggregator(vals, validity_bytes, offset + start, min_periods) {
                Some(val) => val,
                None => {
                    validity.set(idx, false);
                    K::default()
                }
            }
        })
        .collect_trusted::<MutableBuffer<K>>();

    Arc::new(PrimitiveArray::from_data(
        K::DATA_TYPE,
        out.into(),
        Some(validity.into()),
    ))
}

fn compute_sum<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_sum(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = Zero::zero();
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                out += *val;
            }
        }
        Some(out)
    }
}

fn compute_mean<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + Float,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_mean(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = T::zero();
        let mut count = T::zero();
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                out += *val;
                count += One::one()
            }
        }
        Some(out / count)
    }
}

pub(crate) fn compute_var<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + Float,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_var(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        match compute_mean(values, validity_bytes, offset, min_periods) {
            None => None,
            Some(mean) => {
                let mut sum = T::zero();
                let mut count = T::zero();
                for (i, val) in values.iter().enumerate() {
                    // Safety:
                    // in bounds
                    if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                        let v = *val - mean;
                        sum += v * v;
                        count += One::one()
                    }
                }
                Some(sum / (count - T::one()))
            }
        }
    }
}

fn compute_min<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + PartialOrd,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_min(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = None;
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                match out {
                    None => {
                        out = Some(*val);
                    }
                    Some(a) => {
                        if *val < a {
                            out = Some(*val)
                        }
                    }
                }
            }
        }
        out
    }
}

fn compute_max<T>(
    values: &[T],
    validity_bytes: &[u8],
    offset: usize,
    min_periods: usize,
) -> Option<T>
where
    T: NativeType + PartialOrd,
{
    let null_count = count_zeros(validity_bytes, offset, values.len());
    if null_count == 0 {
        Some(no_nulls::compute_min(values))
    } else if (values.len() - null_count) < min_periods {
        None
    } else {
        let mut out = None;
        for (i, val) in values.iter().enumerate() {
            // Safety:
            // in bounds
            if unsafe { get_bit_unchecked(validity_bytes, offset + i) } {
                match out {
                    None => {
                        out = Some(*val);
                    }
                    Some(a) => {
                        if *val > a {
                            out = Some(*val)
                        }
                    }
                }
            }
        }
        out
    }
}
pub fn rolling_var<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum<T> + Zero + AddAssign + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_var,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_var,
        )
    }
}

pub fn rolling_sum<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_sum,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_sum,
        )
    }
}

pub fn rolling_mean<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + Float,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_mean,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_mean,
        )
    }
}

pub fn rolling_min<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_min,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_min,
        )
    }
}

pub fn rolling_max<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
) -> ArrayRef
where
    T: NativeType + std::iter::Sum + Zero + AddAssign + Copy + PartialOrd,
{
    if weights.is_some() {
        panic!("weights not yet supported on array with null values")
    }
    if center {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            compute_max,
        )
    } else {
        rolling_apply(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            compute_max,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;

    #[test]
    fn test_rolling_sum_nulls() {
        let buf = Buffer::from([1.0, 2.0, 3.0, 4.0]);
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
}
