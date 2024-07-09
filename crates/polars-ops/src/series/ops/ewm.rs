pub use arrow::legacy::kernels::ewm::EWMOptions;
use arrow::legacy::kernels::ewm::{
    ewm_mean as kernel_ewm_mean, ewm_std as kernel_ewm_std, ewm_var as kernel_ewm_var,
};
use polars_core::prelude::*;

fn check_alpha(alpha: f64) -> PolarsResult<()> {
    polars_ensure!((0.0..=1.0).contains(&alpha), ComputeError: "alpha must be in [0; 1]");
    Ok(())
}

pub fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;
    match s.dtype() {
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_mean(
                xs,
                options.alpha as f32,
                options.adjust,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name(), Box::new(result) as ArrayRef))
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_mean(
                xs,
                options.alpha,
                options.adjust,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name(), Box::new(result) as ArrayRef))
        },
        _ => ewm_mean(&s.cast(&DataType::Float64)?, options),
    }
}

pub fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;
    match s.dtype() {
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_std(
                xs,
                options.alpha as f32,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name(), Box::new(result) as ArrayRef))
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_std(
                xs,
                options.alpha,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name(), Box::new(result) as ArrayRef))
        },
        _ => ewm_std(&s.cast(&DataType::Float64)?, options),
    }
}

pub fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;
    match s.dtype() {
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_var(
                xs,
                options.alpha as f32,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name(), Box::new(result) as ArrayRef))
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_var(
                xs,
                options.alpha,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name(), Box::new(result) as ArrayRef))
        },
        _ => ewm_var(&s.cast(&DataType::Float64)?, options),
    }
}
