//! # Functions
//!
//! Functions on expressions that might be useful.
//!
use crate::prelude::*;
use polars_core::prelude::*;

/// Compute the covariance between two columns.
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

/// Compute the pearson correlation between two columns.
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

/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
pub fn argsort_by(by: Vec<Expr>, reverse: &[bool]) -> Expr {
    let reverse = reverse.to_vec();
    let function = NoEq::new(Arc::new(move |by: &mut [Series]| {
        polars_core::functions::argsort_by(by, &reverse).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);

    Expr::Function {
        input: by,
        function,
        output_type: Some(DataType::UInt32),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: false,
        },
    }
}

#[cfg(feature = "concat_str")]
/// Concat string columns in linear time
pub fn concat_str(s: Vec<Expr>, delimiter: &str) -> Expr {
    let delimiter = delimiter.to_string();
    let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
        polars_core::functions::concat_str(s, &delimiter).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);
    Expr::Function {
        input: s,
        function,
        output_type: Some(DataType::Utf8),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
        },
    }
}
