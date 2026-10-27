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

/// Returns the ranks among `n_nonnull` sorted non-null values that `quantile` needs, and the
/// interpolation fraction between them: `(lower_rank, upper_rank, fraction)`.
///
/// `upper_rank == lower_rank` unless `method` interpolates between two distinct ranks.
pub fn quantile_ranks(
    n_nonnull: usize,
    quantile: f64,
    method: QuantileMethod,
) -> (usize, usize, f64) {
    assert!(n_nonnull > 0);
    let float_idx = (n_nonnull as f64 - 1.0) * quantile;
    let floor_idx = float_idx as usize;
    let ceil_idx = (float_idx.ceil() as usize).min(n_nonnull - 1);
    match method {
        QuantileMethod::Nearest => {
            let idx = float_idx.round() as usize;
            (idx, idx, 0.0)
        },
        QuantileMethod::Lower => (floor_idx, floor_idx, 0.0),
        QuantileMethod::Higher => (ceil_idx, ceil_idx, 0.0),
        QuantileMethod::Midpoint | QuantileMethod::Linear => {
            (floor_idx, ceil_idx, float_idx - floor_idx as f64)
        },
        QuantileMethod::Equiprobable => {
            let idx = ((n_nonnull as f64 * quantile).ceil() - 1.0).max(0.0) as usize;
            (idx, idx, 0.0)
        },
    }
}

/// [`quantile_ranks`] as indices into `length` sorted values with the `null_count` nulls first.
fn quantile_idx(
    quantile: f64,
    length: usize,
    null_count: usize,
    method: QuantileMethod,
) -> (usize, f64, usize) {
    let (lower, upper, fraction) = quantile_ranks(length - null_count, quantile, method);
    (lower + null_count, fraction, upper + null_count)
}

/// helper
fn linear_interpol<T: Float>(lower: T, upper: T, fraction: f64) -> T {
    if lower == upper {
        lower
    } else {
        let proportion: T = T::from(fraction).unwrap();
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
        ComputeError: "quantile should be between 0.0 and 1.0, got {}", quantile,
    );
    if vals.is_empty() {
        return Ok(None);
    }
    if vals.len() == 1 {
        return Ok(vals[0].to_f64());
    }
    let (idx, fraction, top_idx) = quantile_idx(quantile, vals.len(), 0, method);

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
                Ok(
                    linear_interpol(lower.to_f64().unwrap(), upper.to_f64().unwrap(), fraction)
                        .to_f64(),
                )
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
            ComputeError: "quantile should be between 0.0 and 1.0, got {}", q
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
        let (idx, fraction, top_idx) = quantile_idx(q, n, 0, method);

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
                    linear_interpol(lower, upper, fraction).to_f64()
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
            ComputeError: "`quantile` should be between 0.0 and 1.0, got {}", q,
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
        let (idx, fraction, top_idx) = quantile_idx(q, length, null_count, method);

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
                    linear_interpol(lower.unwrap(), upper.unwrap(), fraction).to_f64()
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
