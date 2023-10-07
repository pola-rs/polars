use super::*;

pub trait VarAggSeries {
    /// Get the variance of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn var_as_series(&self, ddof: u8) -> Series;
    /// Get the standard deviation of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn std_as_series(&self, ddof: u8) -> Series;
}

impl<T> ChunkVar for ChunkedArray<T>
where
    T: PolarsNumericType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn var(&self, ddof: u8) -> Option<f64> {
        let n_values = self.len() - self.null_count();
        if n_values <= ddof as usize {
            return None;
        }

        let mean = self.mean()?;
        let squared: Float64Chunked = ChunkedArray::apply_values_generic(self, |value| {
            let tmp = value.to_f64().unwrap() - mean;
            tmp * tmp
        });

        squared
            .sum()
            .map(|sum| sum / (n_values as f64 - ddof as f64))
    }

    fn std(&self, ddof: u8) -> Option<f64> {
        self.var(ddof).map(|var| var.sqrt())
    }
}

impl ChunkVar for Utf8Chunked {}
impl ChunkVar for ListChunked {}
#[cfg(feature = "dtype-array")]
impl ChunkVar for ArrayChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkVar for ObjectChunked<T> {}
impl ChunkVar for BooleanChunked {}
