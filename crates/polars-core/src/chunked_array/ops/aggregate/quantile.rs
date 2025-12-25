use num_traits::AsPrimitive;
use polars_compute::rolling::QuantileMethod;
#[cfg(feature = "dtype-f16")]
use polars_utils::float16::pf16;

use super::*;

pub trait QuantileAggSeries {
    /// Get the median of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn median_reduce(&self) -> Scalar;

    /// Get the quantile of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn quantile_reduce(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Scalar>;

    /// Get the quantiles of the [`ChunkedArray`] as a new [`Series`] of same length as quantiles
    fn quantiles_reduce(&self, _quantiles: &[f64], _method: QuantileMethod)
    -> PolarsResult<Scalar>;
}

/// helper
fn quantile_idx(
    quantile: f64,
    length: usize,
    null_count: usize,
    method: QuantileMethod,
) -> (usize, f64, usize) {
    let nonnull_count = (length - null_count) as f64;
    let float_idx = (nonnull_count - 1.0) * quantile + null_count as f64;
    let mut base_idx = match method {
        QuantileMethod::Nearest => {
            let idx = float_idx.round() as usize;
            return (idx, 0.0, idx);
        },
        QuantileMethod::Lower | QuantileMethod::Midpoint | QuantileMethod::Linear => {
            float_idx as usize
        },
        QuantileMethod::Higher => float_idx.ceil() as usize,
        QuantileMethod::Equiprobable => {
            let idx = ((nonnull_count * quantile).ceil() - 1.0).max(0.0) as usize + null_count;
            return (idx, 0.0, idx);
        },
    };

    base_idx = base_idx.clamp(0, length - 1);
    let top_idx = f64::ceil(float_idx) as usize;

    (base_idx, float_idx, top_idx)
}

/// helper
fn linear_interpol<T: Float>(lower: T, upper: T, idx: usize, float_idx: f64) -> T {
    if lower == upper {
        lower
    } else {
        let proportion: T = T::from(float_idx).unwrap() - T::from(idx).unwrap();
        proportion * (upper - lower) + lower
    }
}
fn midpoint_interpol<T: Float>(lower: T, upper: T) -> T {
    if lower == upper {
        lower
    } else {
        (lower + upper) / (T::one() + T::one())
    }
}

// Quickselect algorithm is used when
//    1. The data is not already sorted
//    2. We can make a contiguous slice of the data
//    3. We only need a single quantile
fn quantile_slice<T: ToPrimitive + TotalOrd + Copy>(
    vals: &mut [T],
    quantile: f64,
    method: QuantileMethod,
) -> PolarsResult<Option<f64>> {
    polars_ensure!((0.0..=1.0).contains(&quantile),
        ComputeError: "quantile should be between 0.0 and 1.0",
    );
    if vals.is_empty() {
        return Ok(None);
    }
    if vals.len() == 1 {
        return Ok(vals[0].to_f64());
    }
    let (idx, float_idx, top_idx) = quantile_idx(quantile, vals.len(), 0, method);

    let (_lhs, lower, rhs) = vals.select_nth_unstable_by(idx, TotalOrd::tot_cmp);
    if idx == top_idx {
        Ok(lower.to_f64())
    } else {
        match method {
            QuantileMethod::Midpoint => {
                let upper = rhs.iter().copied().min_by(TotalOrd::tot_cmp).unwrap();
                Ok(Some(midpoint_interpol(
                    lower.to_f64().unwrap(),
                    upper.to_f64().unwrap(),
                )))
            },
            QuantileMethod::Linear => {
                let upper = rhs.iter().copied().min_by(TotalOrd::tot_cmp).unwrap();
                Ok(linear_interpol(
                    lower.to_f64().unwrap(),
                    upper.to_f64().unwrap(),
                    idx,
                    float_idx,
                )
                .to_f64())
            },
            _ => Ok(lower.to_f64()),
        }
    }
}

// This function is called if multiple quantiles are requested
// but we are able to make a contiguous slice of the data.
// Right now, we do the same thing as the generic function: sort once and
// get all quantiles from the sorted data. But we could consider multi-quickselect
// algorithms in the future.
fn quantiles_slice<T: ToPrimitive + TotalOrd + Copy>(
    vals: &mut [T],
    quantiles: &[f64],
    method: QuantileMethod,
) -> PolarsResult<Vec<Option<f64>>> {
    // Validate all quantiles
    for &q in quantiles {
        polars_ensure!(
            (0.0..=1.0).contains(&q),
            ComputeError: "quantile should be between 0.0 and 1.0"
        );
    }

    if vals.is_empty() {
        return Ok(vec![None; quantiles.len()]);
    }
    if vals.len() == 1 {
        let v = vals[0].to_f64();
        return Ok(vec![v; quantiles.len()]);
    }

    // For multiple quantiles, just sort once
    vals.sort_by(TotalOrd::tot_cmp);
    let n = vals.len();

    let mut out = Vec::with_capacity(quantiles.len());

    for &q in quantiles {
        let (idx, float_idx, top_idx) = quantile_idx(q, n, 0, method);

        // No nulls here, so unwrap is safe
        let lower = vals[idx].to_f64().unwrap();

        let opt = match method {
            QuantileMethod::Midpoint => {
                if top_idx == idx {
                    Some(lower)
                } else {
                    let upper = vals[idx + 1].to_f64().unwrap();
                    midpoint_interpol(lower, upper).to_f64()
                }
            },
            QuantileMethod::Linear => {
                if top_idx == idx {
                    Some(lower)
                } else {
                    let upper = vals[idx + 1].to_f64().unwrap();
                    linear_interpol(lower, upper, idx, float_idx).to_f64()
                }
            },
            _ => Some(lower),
        };

        out.push(opt);
    }

    Ok(out)
}

// This function is called if data is already sorted or we cannot make a contiguous slice
fn generic_quantiles<T>(
    ca: ChunkedArray<T>,
    quantiles: &[f64],
    method: QuantileMethod,
) -> PolarsResult<Vec<Option<f64>>>
where
    T: PolarsNumericType,
{
    // Validate all quantiles
    for &q in quantiles {
        polars_ensure!(
            (0.0..=1.0).contains(&q),
            ComputeError: "`quantile` should be between 0.0 and 1.0",
        );
    }

    let null_count = ca.null_count();
    let length = ca.len();

    if null_count == length {
        return Ok(vec![None; quantiles.len()]);
    }

    let sorted = ca.sort(false);
    let mut out = Vec::with_capacity(quantiles.len());

    for &q in quantiles {
        let (idx, float_idx, top_idx) = quantile_idx(q, length, null_count, method);

        let lower = sorted.get(idx).map(|v| v.to_f64().unwrap());

        let opt = match method {
            QuantileMethod::Midpoint => {
                if top_idx == idx {
                    lower
                } else {
                    let upper = sorted.get(idx + 1).map(|v| v.to_f64().unwrap());
                    midpoint_interpol(lower.unwrap(), upper.unwrap()).to_f64()
                }
            },
            QuantileMethod::Linear => {
                if top_idx == idx {
                    lower
                } else {
                    let upper = sorted.get(idx + 1).map(|v| v.to_f64().unwrap());
                    linear_interpol(lower.unwrap(), upper.unwrap(), idx, float_idx).to_f64()
                }
            },
            _ => lower,
        };

        out.push(opt);
    }

    Ok(out)
}

impl<T> ChunkQuantile<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: TotalOrd,
{
    fn quantile(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Option<f64>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        if let (Ok(slice), false) = (self.cont_slice(), self.is_sorted_ascending_flag()) {
            let mut owned = slice.to_vec();
            quantile_slice(&mut owned, quantile, method)
        } else {
            let re_val = generic_quantiles(self.clone(), &[quantile], method)?;
            Ok(re_val.into_iter().next().unwrap())
        }
    }

    fn quantiles(
        &self,
        quantiles: &[f64],
        method: QuantileMethod,
    ) -> PolarsResult<Vec<Option<f64>>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        if let (Ok(slice), false) = (self.cont_slice(), self.is_sorted_ascending_flag()) {
            let mut owned = slice.to_vec();
            quantiles_slice(&mut owned, quantiles, method)
        } else {
            generic_quantiles(self.clone(), quantiles, method)
        }
    }

    fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileMethod::Linear).unwrap() // unwrap fine since quantile in range
    }
}

// Version of quantile/median that don't need a memcpy
impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: TotalOrd,
{
    pub(crate) fn quantile_faster(
        mut self,
        quantile: f64,
        method: QuantileMethod,
    ) -> PolarsResult<Option<f64>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        let is_sorted = self.is_sorted_ascending_flag();
        if let (Some(slice), false) = (self.cont_slice_mut(), is_sorted) {
            quantile_slice(slice, quantile, method)
        } else {
            self.quantile(quantile, method)
        }
    }

    pub(crate) fn median_faster(self) -> Option<f64> {
        self.quantile_faster(0.5, QuantileMethod::Linear).unwrap()
    }
}

macro_rules! impl_chunk_quantile_for_float_chunked {
    ($T:ty, $CA:ty) => {
        impl ChunkQuantile<$T> for $CA {
            fn quantile(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Option<$T>> {
                // in case of sorted data, the sort is free, so don't take quickselect route
                let out = if let (Ok(slice), false) =
                    (self.cont_slice(), self.is_sorted_ascending_flag())
                {
                    let mut owned = slice.to_vec();
                    quantile_slice(&mut owned, quantile, method)
                } else {
                    let re_val = generic_quantiles(self.clone(), &[quantile], method)?;
                    Ok(re_val.into_iter().next().unwrap())
                };
                out.map(|v| v.map(|v| v.as_()))
            }

            fn quantiles(
                &self,
                quantiles: &[f64],
                method: QuantileMethod,
            ) -> PolarsResult<Vec<Option<$T>>> {
                // in case of sorted data, the sort is free, so don't take quickselect route
                let out = if let (Ok(slice), false) =
                    (self.cont_slice(), self.is_sorted_ascending_flag())
                {
                    let mut owned = slice.to_vec();
                    quantiles_slice(&mut owned, quantiles, method)
                } else {
                    generic_quantiles(self.clone(), quantiles, method)
                };

                out.map(|vec_t| {
                    vec_t
                        .into_iter()
                        .map(|opt| opt.map(|v| AsPrimitive::<$T>::as_(v)))
                        .collect::<Vec<Option<$T>>>()
                })
            }

            fn median(&self) -> Option<$T> {
                self.quantile(0.5, QuantileMethod::Linear).unwrap() // unwrap fine since quantile in range
            }
        }
    };
}

#[cfg(feature = "dtype-f16")]
impl_chunk_quantile_for_float_chunked!(pf16, Float16Chunked);
impl_chunk_quantile_for_float_chunked!(f32, Float32Chunked);
impl_chunk_quantile_for_float_chunked!(f64, Float64Chunked);

macro_rules! impl_float_chunked {
    ($T:ty, $CA:ty) => {
        impl $CA {
            pub(crate) fn quantile_faster(
                mut self,
                quantile: f64,
                method: QuantileMethod,
            ) -> PolarsResult<Option<$T>> {
                // in case of sorted data, the sort is free, so don't take quickselect route
                let is_sorted = self.is_sorted_ascending_flag();
                if let (Some(slice), false) = (self.cont_slice_mut(), is_sorted) {
                    Ok(quantile_slice(slice, quantile, method)?.map(AsPrimitive::as_))
                } else {
                    Ok(self.quantile(quantile, method)?.map(AsPrimitive::as_))
                }
            }

            pub(crate) fn median_faster(self) -> Option<$T> {
                self.quantile_faster(0.5.into(), QuantileMethod::Linear)
                    .unwrap()
            }
        }
    };
}

#[cfg(feature = "dtype-f16")]
impl_float_chunked!(pf16, Float16Chunked);
impl_float_chunked!(f32, Float32Chunked);
impl_float_chunked!(f64, Float64Chunked);

impl ChunkQuantile<String> for StringChunked {}
impl ChunkQuantile<Series> for ListChunked {}
#[cfg(feature = "dtype-array")]
impl ChunkQuantile<Series> for ArrayChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkQuantile<Series> for ObjectChunked<T> {}
impl ChunkQuantile<bool> for BooleanChunked {}
