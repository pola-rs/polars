pub use polars_compute::ewm::EWMOptions;
use polars_compute::ewm::mean::ewm_mean as kernel_ewm_mean;
use polars_compute::ewm::sum::ewm_sum as kernel_ewm_sum;
use polars_compute::ewm::{ewm_std as kernel_ewm_std, ewm_var as kernel_ewm_var};
use polars_core::prelude::*;

fn check_alpha(alpha: f64) -> PolarsResult<()> {
    polars_ensure!((0.0..=1.0).contains(&alpha), ComputeError: "alpha must be in [0; 1]");
    Ok(())
}

macro_rules! dispatch_ewm_kernel {
    ($s:expr, $options:expr, $fallback:ident, |$xs:ident, $alpha:ident| $kernel:expr) => {{
        check_alpha($options.alpha).inspect_err(|_| {
            if cfg!(debug_assertions) {
                panic!()
            }
        })?;
        match $s.dtype() {
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => {
                use num_traits::AsPrimitive;

                let $xs = $s.f16().unwrap();
                let $alpha = $options.alpha.as_();
                let result = $kernel;
                Series::try_from(($s.name().clone(), Box::new(result) as ArrayRef))
            },
            DataType::Float32 => {
                let $xs = $s.f32().unwrap();
                let $alpha = $options.alpha as f32;
                let result = $kernel;
                Series::try_from(($s.name().clone(), Box::new(result) as ArrayRef))
            },
            DataType::Float64 => {
                let $xs = $s.f64().unwrap();
                let $alpha = $options.alpha;
                let result = $kernel;
                Series::try_from(($s.name().clone(), Box::new(result) as ArrayRef))
            },
            dt if cfg!(debug_assertions) => panic!("{:?}", dt),
            _ => $fallback($s, $options),
        }
    }};
}

pub fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    dispatch_ewm_kernel!(s, options, ewm_mean, |xs, alpha| kernel_ewm_mean(
        xs.iter(),
        alpha,
        options.adjust,
        options.min_periods,
        options.ignore_nulls,
    ))
}

pub fn ewm_sum(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    dispatch_ewm_kernel!(s, options, ewm_sum, |xs, alpha| kernel_ewm_sum(
        xs.iter(),
        alpha,
        options.min_periods,
        options.ignore_nulls,
    ))
}

pub fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            use num_traits::AsPrimitive;

            let xs = s.f16().unwrap();
            let result = kernel_ewm_std(
                xs.iter(),
                options.alpha.as_(),
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_std(
                xs.iter(),
                options.alpha as f32,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_std(
                xs.iter(),
                options.alpha,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        _ => ewm_std(&s.cast(&DataType::Float64)?, options),
    }
}

pub fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;
    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            use num_traits::AsPrimitive;

            let xs = s.f16().unwrap();
            let result = kernel_ewm_var(
                xs.iter(),
                options.alpha.as_(),
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_var(
                xs.iter(),
                options.alpha as f32,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_var(
                xs.iter(),
                options.alpha,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Series::try_from((s.name().clone(), Box::new(result) as ArrayRef))
        },
        _ => ewm_var(&s.cast(&DataType::Float64)?, options),
    }
}
