use polars_array::arrow::bridge::chunk_to_arrow;
use polars_compute::moment::VarState;

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
        let mut out = VarState::default();
        // The kernel is the Arrow one, so each chunk crosses over — see `polars_array::arrow::bridge`.
        for arr in self.downcast_iter() {
            out.combine(&polars_compute::moment::var(&chunk_to_arrow(arr)))
        }
        out.finalize(ddof)
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
