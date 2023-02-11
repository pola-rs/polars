use std::convert::TryFrom;

pub use polars_arrow::kernels::ewm::EWMOptions;
use polars_arrow::kernels::ewm::{ewm_mean, ewm_std, ewm_var};

use crate::prelude::*;

impl Series {
    pub fn ewm_mean(&self, options: EWMOptions) -> PolarsResult<Self> {
        if options.alpha <= 0. || options.alpha > 1. {
            return Err(PolarsError::ComputeError(
                "alpha must satisfy: 0 < alpha <= 1".into(),
            ));
        };
        match self.dtype() {
            DataType::Float32 => {
                let xs = self.f32().unwrap();
                let result = ewm_mean(
                    xs,
                    options.alpha as f32,
                    options.adjust,
                    options.min_periods,
                    options.ignore_nulls,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            DataType::Float64 => {
                let xs = self.f64().unwrap();
                let result = ewm_mean(
                    xs,
                    options.alpha,
                    options.adjust,
                    options.min_periods,
                    options.ignore_nulls,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            _ => self.cast(&DataType::Float64)?.ewm_mean(options),
        }
    }

    pub fn ewm_std(&self, options: EWMOptions) -> PolarsResult<Self> {
        if options.alpha <= 0. || options.alpha > 1. {
            return Err(PolarsError::ComputeError(
                "alpha must satisfy: 0 < alpha <= 1".into(),
            ));
        };
        match self.dtype() {
            DataType::Float32 => {
                let xs = self.f32().unwrap();
                let result = ewm_std(
                    xs,
                    options.alpha as f32,
                    options.adjust,
                    options.bias,
                    options.min_periods,
                    options.ignore_nulls,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            DataType::Float64 => {
                let xs = self.f64().unwrap();
                let result = ewm_std(
                    xs,
                    options.alpha,
                    options.adjust,
                    options.bias,
                    options.min_periods,
                    options.ignore_nulls,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            _ => self.cast(&DataType::Float64)?.ewm_std(options),
        }
    }

    pub fn ewm_var(&self, options: EWMOptions) -> PolarsResult<Self> {
        if options.alpha <= 0. || options.alpha > 1. {
            return Err(PolarsError::ComputeError(
                "alpha must satisfy: 0 < alpha <= 1".into(),
            ));
        };
        match self.dtype() {
            DataType::Float32 => {
                let xs = self.f32().unwrap();
                let result = ewm_var(
                    xs,
                    options.alpha as f32,
                    options.adjust,
                    options.bias,
                    options.min_periods,
                    options.ignore_nulls,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            DataType::Float64 => {
                let xs = self.f64().unwrap();
                let result = ewm_var(
                    xs,
                    options.alpha,
                    options.adjust,
                    options.bias,
                    options.min_periods,
                    options.ignore_nulls,
                );
                Series::try_from((self.name(), Box::new(result) as ArrayRef))
            }
            _ => self.cast(&DataType::Float64)?.ewm_var(options),
        }
    }
}
