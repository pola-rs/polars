use crate::prelude::*;
use arrow::bitmap::MutableBitmap;
use arrow::types::NativeType;
pub use polars_arrow::kernels::ewm::EWMOptions;
use polars_arrow::kernels::ewm::{
    ewm_std, ewm_var, ewma, ewma_inf_hist_no_nulls, ewma_inf_hists, ewma_no_nulls,
};
use polars_arrow::prelude::FromData;
use polars_utils::mem::to_mutable_slice;
use std::convert::TryFrom;

fn prepare_primitive_array<T: NativeType>(
    vals: Vec<T>,
    min_periods: usize,
    leading_nulls: usize,
) -> PrimitiveArray<T> {
    let leading = std::cmp::max(min_periods, leading_nulls);
    if leading > 1 {
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
        match (self.dtype(), self.null_count()) {
            (DataType::Float32, 0) => {
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
                        let arr = prepare_primitive_array(out, options.min_periods, 0);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                    _ => {
                        let iter = ca.into_no_null_iter();
                        let out = if options.adjust {
                            ewma_no_nulls(iter, options.alpha as f32)
                        } else {
                            ewma_inf_hist_no_nulls(iter, options.alpha as f32)
                        };
                        let arr = prepare_primitive_array(out, options.min_periods, 0);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                }
            }
            (DataType::Float64, 0) => {
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
                        let arr = prepare_primitive_array(out, options.min_periods, 0);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                    _ => {
                        let iter = ca.into_no_null_iter();
                        let out = if options.adjust {
                            ewma_no_nulls(iter, options.alpha)
                        } else {
                            ewma_inf_hist_no_nulls(iter, options.alpha)
                        };
                        let arr = prepare_primitive_array(out, options.min_periods, 0);
                        Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
                    }
                }
            }
            (DataType::Float32, _) => {
                let ca = self.f32().unwrap();
                let iter = ca.into_iter();
                let (leading_nulls, out) = if options.adjust {
                    ewma(iter, options.alpha as f32)
                } else {
                    ewma_inf_hists(iter, options.alpha as f32)
                };
                let arr = prepare_primitive_array(out, options.min_periods, leading_nulls);
                Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
            }
            (DataType::Float64, _) => {
                let ca = self.f64().unwrap();
                let iter = ca.into_iter();
                let (leading_nulls, out) = if options.adjust {
                    ewma(iter, options.alpha as f64)
                } else {
                    ewma_inf_hists(iter, options.alpha)
                };
                let arr = prepare_primitive_array(out, options.min_periods, leading_nulls);
                Series::try_from((self.name(), Arc::new(arr) as ArrayRef))
            }
            _ => self.cast(&DataType::Float64)?.ewm_mean(options),
        }
    }

    pub fn ewm_std(&self, options: EWMOptions) -> Result<Self> {
        let ca = self.rechunk();
        match ca.dtype() {
            DataType::Float32 | DataType::Float64 => {}
            _ => return self.cast(&DataType::Float64)?.ewm_std(options),
        }
        let ewma = ca.ewm_mean(options)?;

        match ewma.dtype() {
            DataType::Float64 => {
                let ewma_arr = ewma.f64().unwrap().downcast_iter().next().unwrap();
                // Safety:
                // we are the only owners for arr;
                let ewma_slice = unsafe { to_mutable_slice(ewma_arr.values().as_slice()) };
                let arr = ca.f64().unwrap().downcast_iter().next().unwrap();
                let x_slice = arr.values().as_slice();

                ewm_std(x_slice, ewma_slice, options.alpha);
                // we mask the original null values until we know better how to deal with them.
                let out = Arc::new(ewma_arr.with_validity(arr.validity().cloned())) as ArrayRef;
                Series::try_from((self.name(), out))
            }
            DataType::Float32 => {
                let ewma_arr = ewma.f32().unwrap().downcast_iter().next().unwrap();
                // Safety:
                // we are the only owners for arr;
                let ewma_slice = unsafe { to_mutable_slice(ewma_arr.values().as_slice()) };
                let arr = ca.f32().unwrap().downcast_iter().next().unwrap();
                let x_slice = arr.values().as_slice();

                ewm_std(x_slice, ewma_slice, options.alpha as f32);
                // we mask the original null values until we know better how to deal with them.
                let out = Arc::new(ewma_arr.with_validity(arr.validity().cloned())) as ArrayRef;
                Series::try_from((self.name(), out))
            }
            _ => unimplemented!(),
        }
    }

    pub fn ewm_var(&self, options: EWMOptions) -> Result<Self> {
        let ca = self.rechunk();
        match ca.dtype() {
            DataType::Float32 | DataType::Float64 => {}
            _ => return self.cast(&DataType::Float64)?.ewm_var(options),
        }
        let ewma = ca.ewm_mean(options)?;

        match ewma.dtype() {
            DataType::Float64 => {
                let ewma_arr = ewma.f64().unwrap().downcast_iter().next().unwrap();
                // Safety:
                // we are the only owners for arr;
                let ewma_slice = unsafe { to_mutable_slice(ewma_arr.values().as_slice()) };
                let arr = ca.f64().unwrap().downcast_iter().next().unwrap();
                let x_slice = arr.values().as_slice();

                ewm_var(x_slice, ewma_slice, options.alpha);
                // we mask the original null values until we know better how to deal with them.
                let out = Arc::new(ewma_arr.with_validity(arr.validity().cloned())) as ArrayRef;
                Series::try_from((self.name(), out))
            }
            DataType::Float32 => {
                let ewma_arr = ewma.f32().unwrap().downcast_iter().next().unwrap();
                // Safety:
                // we are the only owners for arr;
                let ewma_slice = unsafe { to_mutable_slice(ewma_arr.values().as_slice()) };
                let arr = ca.f32().unwrap().downcast_iter().next().unwrap();
                let x_slice = arr.values().as_slice();

                ewm_var(x_slice, ewma_slice, options.alpha as f32);
                // we mask the original null values until we know better how to deal with them.
                let out = Arc::new(ewma_arr.with_validity(arr.validity().cloned())) as ArrayRef;
                Series::try_from((self.name(), out))
            }
            _ => unimplemented!(),
        }
    }
}
