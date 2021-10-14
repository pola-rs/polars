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
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
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
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::pearson_corr(ca_a, ca_b)])
            }
        };
        Ok(s)
    };
    map_binary(a, b, function, Some(Field::new(name, DataType::Float32))).alias(name)
}

/// Compute the spearman rank correlation between two columns.
#[cfg(feature = "rank")]
#[cfg_attr(docsrs, doc(cfg(feature = "rank")))]
pub fn spearman_rank_corr(a: Expr, b: Expr) -> Expr {
    pearson_corr(a.rank(RankMethod::Min), b.rank(RankMethod::Min)).alias("spearman_rank_corr")
}

/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
pub fn argsort_by<E: AsRef<[Expr]>>(by: E, reverse: &[bool]) -> Expr {
    let reverse = reverse.to_vec();
    let function = NoEq::new(Arc::new(move |by: &mut [Series]| {
        polars_core::functions::argsort_by(by, &reverse).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);

    Expr::Function {
        input: by.as_ref().to_vec(),
        function,
        output_type: GetOutput::from_type(DataType::UInt32),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: false,
        },
    }
}

#[cfg(feature = "concat_str")]
#[cfg_attr(docsrs, doc(cfg(feature = "concat_str")))]
/// Concat string columns in linear time
pub fn concat_str(s: Vec<Expr>, sep: &str) -> Expr {
    let sep = sep.to_string();
    let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
        polars_core::functions::concat_str(s, &sep).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);
    Expr::Function {
        input: s,
        function,
        output_type: GetOutput::from_type(DataType::Utf8),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
        },
    }
}

/// Concat lists entries.
#[cfg(feature = "list")]
#[cfg_attr(docsrs, doc(cfg(feature = "list")))]
pub fn concat_lst(s: Vec<Expr>) -> Expr {
    let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
        let first = std::mem::take(&mut s[0]);
        let other = &s[1..];

        let first_ca = first.list()?;
        first_ca.lst_concat(other).map(|ca| ca.into_series())
    }) as Arc<dyn SeriesUdf>);
    Expr::Function {
        input: s,
        function,
        output_type: GetOutput::from_type(DataType::Utf8),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
        },
    }
}

/// Create list entries that are range arrays
/// - if `low` and `high` are a column, every element will expand into an array in a list column.
/// - if `low` and `high` are literals the output will be of `Int64`.
#[cfg(feature = "arange")]
#[cfg_attr(docsrs, doc(cfg(feature = "arange")))]
pub fn arange(low: Expr, high: Expr, step: usize) -> Expr {
    if matches!(low, Expr::Literal(_)) || matches!(high, Expr::Literal(_)) {
        let f = move |sa: Series, sb: Series| {
            let sa = sa.cast(&DataType::Int64)?;
            let sb = sb.cast(&DataType::Int64)?;
            let low = sa
                .i64()?
                .get(0)
                .ok_or_else(|| PolarsError::NoData("no data in `low` evaluation".into()))?;
            let high = sb
                .i64()?
                .get(0)
                .ok_or_else(|| PolarsError::NoData("no data in `high` evaluation".into()))?;

            if step > 1 {
                Ok(Int64Chunked::new_from_iter("arange", (low..high).step_by(step)).into_series())
            } else {
                Ok(Int64Chunked::new_from_iter("arange", low..high).into_series())
            }
        };
        map_binary(low, high, f, Some(Field::new("arange", DataType::Int64)))
    } else {
        let f = move |sa: Series, sb: Series| {
            let sa = sa.cast(&DataType::Int64)?;
            let sb = sb.cast(&DataType::Int64)?;
            let low = sa.i64()?;
            let high = sb.i64()?;
            let mut builder =
                ListPrimitiveChunkedBuilder::<Int64Type>::new("arange", low.len(), low.len() * 3);

            low.into_iter()
                .zip(high.into_iter())
                .for_each(|(opt_l, opt_h)| match (opt_l, opt_h) {
                    (Some(l), Some(r)) => {
                        if step > 1 {
                            builder.append_iter_values((l..r).step_by(step));
                        } else {
                            builder.append_iter_values(l..r);
                        }
                    }
                    _ => builder.append_null(),
                });

            Ok(builder.finish().into_series())
        };
        map_binary(
            low,
            high,
            f,
            Some(Field::new("arange", DataType::List(DataType::Int64.into()))),
        )
    }
}
