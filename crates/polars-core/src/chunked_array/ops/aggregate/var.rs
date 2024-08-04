use arity::unary_elementwise_values;

use super::*;

pub trait VarAggSeries {
    /// Get the variance of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn var_reduce(&self, ddof: u8) -> Scalar;
    /// Get the standard deviation of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn std_reduce(&self, ddof: u8) -> Scalar;
}

impl<T> ChunkVar for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    fn var(&self, ddof: u8) -> Option<f64> {
        let n_values = self.len() - self.null_count();
        if n_values <= ddof as usize {
            return None;
        }

        let mean = self.mean()?;
        let squared: Float64Chunked = unary_elementwise_values(self, |value| {
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

impl ChunkVar for StringChunked {}
impl ChunkVar for ListChunked {}
#[cfg(feature = "dtype-array")]
impl ChunkVar for ArrayChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkVar for ObjectChunked<T> {}
impl ChunkVar for BooleanChunked {}
