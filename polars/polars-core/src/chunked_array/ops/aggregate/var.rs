use super::*;

pub trait VarAggSeries {
    /// Get the variance of the ChunkedArray as a new Series of length 1.
    fn var_as_series(&self, ddof: u8) -> Series;
    /// Get the standard deviation of the ChunkedArray as a new Series of length 1.
    fn std_as_series(&self, ddof: u8) -> Series;
}

impl<T> ChunkVar<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn var(&self, ddof: u8) -> Option<f64> {
        if self.len() == 1 {
            return Some(0.0);
        }
        let n_values = self.len() - self.null_count();

        if ddof as usize > n_values {
            return None;
        }
        let n_values = n_values as f64;

        let mean = self.mean()?;
        let squared = self.apply_cast_numeric::<_, Float64Type>(|value| {
            let tmp = value.to_f64().unwrap() - mean;
            tmp * tmp
        });
        // Note, this is similar behavior to numpy if DDOF=1.
        // in statistics DDOF often = 1.
        // this last step is similar to mean, only now instead of 1/n it is 1/(n-1)
        squared.sum().map(|sum| sum / (n_values - ddof as f64))
    }
    fn std(&self, ddof: u8) -> Option<f64> {
        self.var(ddof).map(|var| var.sqrt())
    }
}

impl ChunkVar<f32> for Float32Chunked {
    fn var(&self, ddof: u8) -> Option<f32> {
        if self.len() == 1 {
            return Some(0.0);
        }
        let n_values = self.len() - self.null_count();

        if ddof as usize > n_values {
            return None;
        }
        let n_values = n_values as f32;

        let mean = self.mean()? as f32;
        let squared = self.apply(|value| {
            let tmp = value - mean;
            tmp * tmp
        });
        squared.sum().map(|sum| sum / (n_values - ddof as f32))
    }
    fn std(&self, ddof: u8) -> Option<f32> {
        self.var(ddof).map(|var| var.sqrt())
    }
}

impl ChunkVar<f64> for Float64Chunked {
    fn var(&self, ddof: u8) -> Option<f64> {
        if self.len() == 1 {
            return Some(0.0);
        }
        let n_values = self.len() - self.null_count();

        if ddof as usize > n_values {
            return None;
        }
        let n_values = n_values as f64;

        let mean = self.mean()?;
        let squared = self.apply(|value| {
            let tmp = value - mean;
            tmp * tmp
        });
        squared.sum().map(|sum| sum / (n_values - ddof as f64))
    }
    fn std(&self, ddof: u8) -> Option<f64> {
        self.var(ddof).map(|var| var.sqrt())
    }
}

impl ChunkVar<String> for Utf8Chunked {}
impl ChunkVar<Series> for ListChunked {}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkVar<Series> for ObjectChunked<T> {}
impl ChunkVar<bool> for BooleanChunked {}
