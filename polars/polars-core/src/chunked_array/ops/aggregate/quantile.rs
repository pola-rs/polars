use polars_arrow::floats::{f32_to_ordablef32, f64_to_ordablef64};
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_utils::slice::Extrema;

use super::*;

pub trait QuantileAggSeries {
    /// Get the median of the ChunkedArray as a new Series of length 1.
    fn median_as_series(&self) -> Series;
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series>;
}

/// helper
fn quantile_idx(
    quantile: f64,
    length: usize,
    null_count: usize,
    interpol: QuantileInterpolOptions,
) -> (usize, f64, usize) {
    let mut base_idx = match interpol {
        QuantileInterpolOptions::Nearest => {
            (((length - null_count) as f64) * quantile + null_count as f64) as usize
        }
        QuantileInterpolOptions::Lower
        | QuantileInterpolOptions::Midpoint
        | QuantileInterpolOptions::Linear => {
            (((length - null_count) as f64 - 1.0) * quantile + null_count as f64) as usize
        }
        QuantileInterpolOptions::Higher => {
            (((length - null_count) as f64 - 1.0) * quantile + null_count as f64).ceil() as usize
        }
    };

    base_idx = base_idx.clamp(0, length - 1);
    let float_idx = ((length - null_count) as f64 - 1.0) * quantile + null_count as f64;
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

trait Sortable {
    fn sort(&self) -> Self;
}

// Utility trait for `generic_quantile`
impl<T> Sortable for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
{
    fn sort(&self) -> Self {
        ChunkSort::sort(self, false)
    }
}
impl Sortable for Float32Chunked {
    fn sort(&self) -> Self {
        ChunkSort::sort(self, false)
    }
}

impl Sortable for Float64Chunked {
    fn sort(&self) -> Self {
        ChunkSort::sort(self, false)
    }
}

// Uses quickselect instead of sorting all data
fn quantile_slice<T: ToPrimitive + Ord>(
    vals: &mut [T],
    quantile: f64,
    interpol: QuantileInterpolOptions,
) -> PolarsResult<Option<f64>> {
    if !(0.0..=1.0).contains(&quantile) {
        return Err(PolarsError::ComputeError(
            "quantile should be between 0.0 and 1.0".into(),
        ));
    }
    if vals.is_empty() {
        return Ok(None);
    }
    if vals.len() == 1 {
        return Ok(vals[0].to_f64());
    }
    let (idx, float_idx, top_idx) = quantile_idx(quantile, vals.len(), 0, interpol);

    let (_lhs, lower, rhs) = vals.select_nth_unstable(idx);
    if idx == top_idx {
        Ok(lower.to_f64())
    } else {
        match interpol {
            QuantileInterpolOptions::Midpoint => {
                let upper = rhs.min_value().unwrap();
                Ok(Some(midpoint_interpol(
                    lower.to_f64().unwrap(),
                    upper.to_f64().unwrap(),
                )))
            }
            QuantileInterpolOptions::Linear => {
                let upper = rhs.min_value().unwrap();
                Ok(linear_interpol(
                    lower.to_f64().unwrap(),
                    upper.to_f64().unwrap(),
                    idx,
                    float_idx,
                )
                .to_f64())
            }
            _ => Ok(lower.to_f64()),
        }
    }
}

fn generic_quantile<T>(
    ca: ChunkedArray<T>,
    quantile: f64,
    interpol: QuantileInterpolOptions,
) -> PolarsResult<Option<f64>>
where
    T: PolarsNumericType,
    ChunkedArray<T>: Sortable,
{
    if !(0.0..=1.0).contains(&quantile) {
        return Err(PolarsError::ComputeError(
            "quantile should be between 0.0 and 1.0".into(),
        ));
    }

    let null_count = ca.null_count();
    let length = ca.len();

    if null_count == length {
        return Ok(None);
    }

    let (idx, float_idx, top_idx) = quantile_idx(quantile, length, null_count, interpol);
    let sorted = ca.sort();
    let lower = sorted.get(idx).map(|v| v.to_f64().unwrap());

    let opt = match interpol {
        QuantileInterpolOptions::Midpoint => {
            if top_idx == idx {
                lower
            } else {
                let upper = sorted.get(idx + 1).map(|v| v.to_f64().unwrap());
                midpoint_interpol(lower.unwrap(), upper.unwrap()).to_f64()
            }
        }
        QuantileInterpolOptions::Linear => {
            if top_idx == idx {
                lower
            } else {
                let upper = sorted.get(idx + 1).map(|v| v.to_f64().unwrap());

                linear_interpol(lower.unwrap(), upper.unwrap(), idx, float_idx).to_f64()
            }
        }
        _ => lower,
    };
    Ok(opt)
}

impl<T> ChunkQuantile<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
{
    fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        if let (Ok(slice), false) = (self.cont_slice(), self.is_sorted_flag()) {
            let mut owned = slice.to_vec();
            quantile_slice(&mut owned, quantile, interpol)
        } else {
            generic_quantile(self.clone(), quantile, interpol)
        }
    }

    fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileInterpolOptions::Linear).unwrap() // unwrap fine since quantile in range
    }
}

// Version of quantile/median that don't need a memcpy
impl<T> ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
{
    pub(crate) fn quantile_faster(
        mut self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        let is_sorted = self.is_sorted_flag();
        if let (Some(slice), false) = (self.cont_slice_mut(), is_sorted) {
            quantile_slice(slice, quantile, interpol)
        } else {
            self.quantile(quantile, interpol)
        }
    }

    pub(crate) fn median_faster(self) -> Option<f64> {
        self.quantile_faster(0.5, QuantileInterpolOptions::Linear)
            .unwrap()
    }
}

impl ChunkQuantile<f32> for Float32Chunked {
    fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f32>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        let out = if let (Ok(slice), false) = (self.cont_slice(), self.is_sorted_flag()) {
            let mut owned = slice.to_vec();
            let owned = f32_to_ordablef32(&mut owned);
            quantile_slice(owned, quantile, interpol)
        } else {
            generic_quantile(self.clone(), quantile, interpol)
        };
        out.map(|v| v.map(|v| v as f32))
    }

    fn median(&self) -> Option<f32> {
        self.quantile(0.5, QuantileInterpolOptions::Linear).unwrap() // unwrap fine since quantile in range
    }
}

impl ChunkQuantile<f64> for Float64Chunked {
    fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        if let (Ok(slice), false) = (self.cont_slice(), self.is_sorted_flag()) {
            let mut owned = slice.to_vec();
            let owned = f64_to_ordablef64(&mut owned);
            quantile_slice(owned, quantile, interpol)
        } else {
            generic_quantile(self.clone(), quantile, interpol)
        }
    }

    fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileInterpolOptions::Linear).unwrap() // unwrap fine since quantile in range
    }
}

impl Float64Chunked {
    pub(crate) fn quantile_faster(
        mut self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        let is_sorted = self.is_sorted_flag();
        if let (Some(slice), false) = (self.cont_slice_mut(), is_sorted) {
            let slice = f64_to_ordablef64(slice);
            quantile_slice(slice, quantile, interpol)
        } else {
            self.quantile(quantile, interpol)
        }
    }

    pub(crate) fn median_faster(self) -> Option<f64> {
        self.quantile_faster(0.5, QuantileInterpolOptions::Linear)
            .unwrap()
    }
}

impl Float32Chunked {
    pub(crate) fn quantile_faster(
        mut self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f32>> {
        // in case of sorted data, the sort is free, so don't take quickselect route
        let is_sorted = self.is_sorted_flag();
        if let (Some(slice), false) = (self.cont_slice_mut(), is_sorted) {
            let slice = f32_to_ordablef32(slice);
            quantile_slice(slice, quantile, interpol).map(|v| v.map(|v| v as f32))
        } else {
            self.quantile(quantile, interpol)
        }
    }

    pub(crate) fn median_faster(self) -> Option<f32> {
        self.quantile_faster(0.5, QuantileInterpolOptions::Linear)
            .unwrap()
    }
}

impl ChunkQuantile<String> for Utf8Chunked {}
impl ChunkQuantile<Series> for ListChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkQuantile<Series> for ObjectChunked<T> {}
impl ChunkQuantile<bool> for BooleanChunked {}
