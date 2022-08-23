use std::convert::TryFrom;

pub use polars_arrow::kernels::ewm::EWMOptions;
use polars_arrow::kernels::ewm::{ewm_mean, ewm_std, ewm_var};
use polars_utils::mem::to_mutable_slice;

use crate::prelude::*;

impl Series {
    pub fn ewm_mean(&self, options: EWMOptions) -> Result<Self> {
        match self.dtype() {
            DataType::Float32 => {
                let xs = self.f32().unwrap().downcast_iter().next().unwrap();
                let result = ewm_mean(
                    xs,
                    options.alpha as f32,
                    options.adjust,
                    options.min_periods,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            DataType::Float64 => {
                let xs = self.f64().unwrap().downcast_iter().next().unwrap();
                let result = ewm_mean(
                    xs,
                    options.alpha as f64,
                    options.adjust,
                    options.min_periods,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
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
                let out =
                    Box::new(ewma_arr.clone().with_validity(arr.validity().cloned())) as ArrayRef;
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
                let out =
                    Box::new(ewma_arr.clone().with_validity(arr.validity().cloned())) as ArrayRef;
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
                let out =
                    Box::new(ewma_arr.clone().with_validity(arr.validity().cloned())) as ArrayRef;
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
                let out =
                    Box::new(ewma_arr.clone().with_validity(arr.validity().cloned())) as ArrayRef;
                Series::try_from((self.name(), out))
            }
            _ => unimplemented!(),
        }
    }
}
