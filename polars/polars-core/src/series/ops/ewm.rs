use crate::prelude::*;
use arrow::bitmap::MutableBitmap;
use arrow::types::NativeType;
pub use polars_arrow::kernels::ewm::EWMOptions;
use polars_arrow::kernels::ewm::{ewma_inf_hist_no_nulls, ewma_no_nulls};
use polars_arrow::prelude::FromData;
use std::convert::TryFrom;

fn prepare_primitive_array<T: NativeType>(vals: Vec<T>, min_periods: usize) -> PrimitiveArray<T> {
    if min_periods > 1 {
        let mut validity = MutableBitmap::with_capacity(vals.len());
        validity.extend_constant(min_periods, false);
        validity.extend_constant(vals.len() - min_periods, true);

        PrimitiveArray::from_data_default(vals.into(), Some(validity.into()))
    } else {
        PrimitiveArray::from_data_default(vals.into(), None)
    }
}

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
                        let arr = prepare_primitive_array(out, options.min_periods);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                    _ => {
                        let iter = ca.into_no_null_iter();
                        let out = if options.adjust {
                            ewma_no_nulls(iter, options.alpha as f32)
                        } else {
                            ewma_inf_hist_no_nulls(iter, options.alpha as f32)
                        };
                        let arr = prepare_primitive_array(out, options.min_periods);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
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
                        let arr = prepare_primitive_array(out, options.min_periods);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                    _ => {
                        let iter = ca.into_no_null_iter();
                        let out = if options.adjust {
                            ewma_no_nulls(iter, options.alpha)
                        } else {
                            ewma_inf_hist_no_nulls(iter, options.alpha)
                        };
                        let arr = prepare_primitive_array(out, options.min_periods);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                }
            }
            _ => self.cast(&DataType::Float64)?.ewm_mean(options),
        }
    }
}
