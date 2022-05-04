use polars_core::prelude::*;

pub trait CutQCut<T> {
    fn qcut(&self, _bins: Vec<f64>) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            "qcut is not implemented for the dtype".into(),
        ))
    }
}

//fn qcut_helper<T>(val: T, bins: &[T]) -> u32
//where
//    T: PolarsNumericType + std::cmp::PartialOrd
//{
//    for (idx, bin) in bins.iter().enumerate() {
//        if val < *bin {
//            return idx as u32
//        }
//    }
//    return bins.len() as u32
//}

impl<T> CutQCut<T> for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: NumericNative + From<f64>,
    ChunkedArray<T>: ChunkQuantile<f64>,
{
    fn qcut(&self, bins: Vec<f64>) -> Result<Series> {
        dbg!("qcut placeholder");
        dbg!(&bins);
        let bin_vals: Vec<T::Native> = bins
            .iter()
            .map(|f| {
                self.quantile(*f, QuantileInterpolOptions::Nearest)
                    .unwrap()
                    .unwrap()
                    .into()
            })
            .collect();

        Ok(self.cast(&DataType::Float64)?.into_series())
    }
}
