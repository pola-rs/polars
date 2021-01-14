use crate::prelude::*;
use polars_core::prelude::*;

pub fn cov(a: Expr, b: Expr) -> Expr {
    let name = "cov";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::cov(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast::<Float64Type>()?;
                let b = b.cast::<Float64Type>()?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    map_binary(a, b, function, Some(Field::new(name, DataType::Float32))).alias(name)
}

pub fn pearson_corr(a: Expr, b: Expr) -> Expr {
    let name = "pearson_corr";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast::<Float64Type>()?;
                let b = b.cast::<Float64Type>()?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    map_binary(a, b, function, Some(Field::new(name, DataType::Float32))).alias(name)
}
