use polars_core::prelude::*;

pub trait CutQCut<T> {
    fn qcut(&self, _bins: Vec<f64>) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            "qcut is not implemented for the dtype".into(),
        ))
    }
    fn cut(&self, _bins: Vec<f64>) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            "qcut is not implemented for the dtype".into(),
        ))
    }
}

fn bin_categories(bins: &[f64]) -> Vec<Option<String>> {
    
    let mut cats: Vec<Option<String>> = Vec::new();
    let mut lag: String = "".to_string();
    
    cats.push(None);
    for (idx, val) in bins.iter().enumerate() {
        if idx > 0 {
            cats.push(Some(format!("({} - {}]", lag, val)));
        }
        lag = val.to_string();
    }
    cats.push(None);

    cats
}

impl<T> CutQCut<T> for ChunkedArray<T>
where
    T: PolarsNumericType + Sync,
    T::Native: NumericNative,
    ChunkedArray<T>: ChunkQuantile<f64>,
{
    fn qcut(&self, bins: Vec<f64>) -> Result<Series> {
        dbg!("qcut placeholder");

        let bin_vals: Vec<f64> = bins
            .iter()
            .map(|f| {
                self.quantile(*f, QuantileInterpolOptions::Nearest)
                    .unwrap()
                    .unwrap()
            })
            .collect();

        self.cut(bin_vals)
    }

    fn cut(&self, bins: Vec<f64>) -> Result<Series> {
        dbg!("cut placeholder");

        let bins_f: ChunkedArray<Float64Type> = ChunkedArray::from_vec("x", bins.clone());
        let categories = bin_categories(&bins);

        let output: Utf8Chunked = self
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|f| {

                if let Some(val) = f {
                    for (idx, bin_val) in bins_f.into_iter().enumerate() {
                        if bin_val.unwrap() > val {
                            return categories[idx].as_ref();
                        }
                    }
                    return None
                } else {
                    return None
                }
            })
            .collect();

        Ok(output.into_series().cast(&DataType::Categorical(None))?)
    }
}
