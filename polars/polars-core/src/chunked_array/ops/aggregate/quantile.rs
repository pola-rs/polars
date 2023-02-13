use polars_arrow::prelude::QuantileInterpolOptions;

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
) -> (i64, f64, i64) {
    let mut base_idx = match interpol {
        QuantileInterpolOptions::Nearest => {
            (((length - null_count) as f64) * quantile + null_count as f64) as i64
        }
        QuantileInterpolOptions::Lower
        | QuantileInterpolOptions::Midpoint
        | QuantileInterpolOptions::Linear => {
            (((length - null_count) as f64 - 1.0) * quantile + null_count as f64) as i64
        }
        QuantileInterpolOptions::Higher => {
            (((length - null_count) as f64 - 1.0) * quantile + null_count as f64).ceil() as i64
        }
    };

    base_idx = base_idx.clamp(0, (length - 1) as i64);
    let float_idx = ((length - null_count) as f64 - 1.0) * quantile + null_count as f64;
    let top_idx = f64::ceil(float_idx) as i64;

    (base_idx, float_idx, top_idx)
}

/// helper
fn linear_interpol<T: Float>(bounds: &[Option<T>], idx: i64, float_idx: f64) -> Option<T> {
    if bounds[0] == bounds[1] {
        Some(bounds[0].unwrap())
    } else {
        let proportion: T = T::from(float_idx).unwrap() - T::from(idx).unwrap();
        Some(proportion * (bounds[1].unwrap() - bounds[0].unwrap()) + bounds[0].unwrap())
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

fn generic_quantile<T>(
    ca: &ChunkedArray<T>,
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

    let opt = match interpol {
        QuantileInterpolOptions::Midpoint => {
            if top_idx == idx {
                ca.sort()
                    .slice(idx, 1)
                    .apply_cast_numeric::<_, Float64Type>(|value| value.to_f64().unwrap())
                    .into_iter()
                    .next()
                    .flatten()
            } else {
                let bounds: Vec<Option<f64>> = ca
                    .sort()
                    .slice(idx, 2)
                    .apply_cast_numeric::<_, Float64Type>(|value| value.to_f64().unwrap())
                    .into_iter()
                    .collect();

                Some((bounds[0].unwrap() + bounds[1].unwrap()) / 2.0f64)
            }
        }
        QuantileInterpolOptions::Linear => {
            if top_idx == idx {
                ca.sort()
                    .slice(idx, 1)
                    .apply_cast_numeric::<_, Float64Type>(|value| value.to_f64().unwrap())
                    .into_iter()
                    .next()
                    .flatten()
            } else {
                let bounds: Vec<Option<f64>> = ca
                    .sort()
                    .slice(idx, 2)
                    .apply_cast_numeric::<_, Float64Type>(|value| value.to_f64().unwrap())
                    .into_iter()
                    .collect();

                linear_interpol(&bounds, idx, float_idx)
            }
        }
        _ => ca
            .sort()
            .slice(idx, 1)
            .apply_cast_numeric::<_, Float64Type>(|value| value.to_f64().unwrap())
            .into_iter()
            .next()
            .flatten(),
    };
    Ok(opt)
}

impl<T> ChunkQuantile<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f64>> {
        generic_quantile(self, quantile, interpol)
    }

    fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileInterpolOptions::Linear).unwrap() // unwrap fine since quantile in range
    }
}

impl ChunkQuantile<f32> for Float32Chunked {
    fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Option<f32>> {
        generic_quantile(self, quantile, interpol).map(|v| v.map(|v| v as f32))
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
        generic_quantile(self, quantile, interpol)
    }

    fn median(&self) -> Option<f64> {
        self.quantile(0.5, QuantileInterpolOptions::Linear).unwrap() // unwrap fine since quantile in range
    }
}

impl ChunkQuantile<String> for Utf8Chunked {}
impl ChunkQuantile<Series> for ListChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkQuantile<Series> for ObjectChunked<T> {}
impl ChunkQuantile<bool> for BooleanChunked {}
