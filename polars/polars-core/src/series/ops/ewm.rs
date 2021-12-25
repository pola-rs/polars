use crate::prelude::*;
pub use polars_arrow::kernels::ewm::EWMOptions;
use polars_arrow::kernels::ewm::{ewma_inf_hist_no_nulls, ewma_no_nulls};

impl Series {
    pub fn ewm_mean(&self, options: EWMOptions) -> Result<Self> {
        if self.null_count() > 0 {
            return self
                .fill_null(FillNullStrategy::Zero)
                .unwrap()
                .ewm_mean(options);
        }

        match self.dtype() {
            DataType::Float32 => {
                let ca = self.f32().unwrap();
                match self.n_chunks() {
                    1 => {
                        let vals = ca.downcast_iter().next().unwrap();
                        let vals = vals.values().as_slice();
                        let out = if options.adjust {
                            ewma_no_nulls(vals.iter().copied(), options.alpha as f32)
                        } else {
                            ewma_inf_hist_no_nulls(vals.iter().copied(), options.alpha as f32)
                        };
                        Ok(Float32Chunked::new_vec(self.name(), out).into_series())
                    }
                    _ => {
                        let iter = ca.into_no_null_iter();
                        let out = if options.adjust {
                            ewma_no_nulls(iter, options.alpha as f32)
                        } else {
                            ewma_inf_hist_no_nulls(iter, options.alpha as f32)
                        };
                        Ok(Float32Chunked::new_vec(self.name(), out).into_series())
                    }
                }
            }
            DataType::Float64 => {
                let ca = self.f64().unwrap();
                match self.n_chunks() {
                    1 => {
                        let vals = ca.downcast_iter().next().unwrap();
                        let vals = vals.values().as_slice();
                        let out = if options.adjust {
                            ewma_no_nulls(vals.iter().copied(), options.alpha)
                        } else {
                            ewma_inf_hist_no_nulls(vals.iter().copied(), options.alpha)
                        };
                        Ok(Float64Chunked::new_vec(self.name(), out).into_series())
                    }
                    _ => {
                        let iter = ca.into_no_null_iter();
                        let out = if options.adjust {
                            ewma_no_nulls(iter, options.alpha)
                        } else {
                            ewma_inf_hist_no_nulls(iter, options.alpha)
                        };
                        Ok(Float64Chunked::new_vec(self.name(), out).into_series())
                    }
                }
            }
            _ => self.cast(&DataType::Float64)?.ewm_mean(options),
        }
    }
}
