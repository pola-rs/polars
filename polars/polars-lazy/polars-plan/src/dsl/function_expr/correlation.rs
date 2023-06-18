#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum CorrelationMethod {
    Pearson,
    #[cfg(all(feature = "rank", feature = "propagate_nans"))]
    SpearmanRank(bool),
    Covariance,
}

impl Display for CorrelationMethod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CorrelationMethod::*;
        let s = match self {
            Pearson => "pearson",
            #[cfg(all(feature = "rank", feature = "propagate_nans"))]
            SpearmanRank(_) => "spearman_rank",
            Covariance => return write!(f, "covariance"),
        };
        write!(f, "{}_correlation", s)
    }
}

pub(super) fn corr(s: &[Series], ddof: u8, method: CorrelationMethod) -> PolarsResult<Series> {
    match method {
        CorrelationMethod::Pearson => pearson_corr(s, ddof),
        #[cfg(all(feature = "rank", feature = "propagate_nans"))]
        CorrelationMethod::SpearmanRank(propagate_nans) => {
            spearman_rank_corr(s, ddof, propagate_nans)
        }
        CorrelationMethod::Covariance => covariance(s),
    }
}

fn covariance(s: &[Series]) -> PolarsResult<Series> {
    let a = &s[0];
    let b = &s[1];
    let name = "cov";

    let s = match a.dtype() {
        DataType::Float32 => {
            let ca_a = a.f32().unwrap();
            let ca_b = b.f32().unwrap();
            Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
        }
        DataType::Float64 => {
            let ca_a = a.f64().unwrap();
            let ca_b = b.f64().unwrap();
            Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
        }
        DataType::Int32 => {
            let ca_a = a.i32().unwrap();
            let ca_b = b.i32().unwrap();
            Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
        }
        DataType::Int64 => {
            let ca_a = a.i64().unwrap();
            let ca_b = b.i64().unwrap();
            Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
        }
        DataType::UInt32 => {
            let ca_a = a.u32().unwrap();
            let ca_b = b.u32().unwrap();
            Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
        }
        DataType::UInt64 => {
            let ca_a = a.u64().unwrap();
            let ca_b = b.u64().unwrap();
            Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
        }
        _ => {
            let a = a.cast(&DataType::Float64)?;
            let b = b.cast(&DataType::Float64)?;
            let ca_a = a.f64().unwrap();
            let ca_b = b.f64().unwrap();
            Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
        }
    };
    Ok(s)
}

fn pearson_corr(s: &[Series], ddof: u8) -> PolarsResult<Series> {
    let a = &s[0];
    let b = &s[1];
    let name = "pearson_corr";

    let s = match a.dtype() {
        DataType::Float32 => {
            let ca_a = a.f32().unwrap();
            let ca_b = b.f32().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
            )
        }
        DataType::Float64 => {
            let ca_a = a.f64().unwrap();
            let ca_b = b.f64().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
            )
        }
        DataType::Int32 => {
            let ca_a = a.i32().unwrap();
            let ca_b = b.i32().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
            )
        }
        DataType::Int64 => {
            let ca_a = a.i64().unwrap();
            let ca_b = b.i64().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
            )
        }
        DataType::UInt32 => {
            let ca_a = a.u32().unwrap();
            let ca_b = b.u32().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
            )
        }
        DataType::UInt64 => {
            let ca_a = a.u64().unwrap();
            let ca_b = b.u64().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
            )
        }
        _ => {
            let a = a.cast(&DataType::Float64)?;
            let b = b.cast(&DataType::Float64)?;
            let ca_a = a.f64().unwrap();
            let ca_b = b.f64().unwrap();
            Series::new(
                name,
                &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
            )
        }
    };
    Ok(s)
}

#[cfg(all(feature = "rank", feature = "propagate_nans"))]
fn spearman_rank_corr(s: &[Series], ddof: u8, propagate_nans: bool) -> PolarsResult<Series> {
    use polars_core::utils::coalesce_nulls_series;
    use polars_ops::chunked_array::nan_propagating_aggregate::nan_max_s;
    let a = &s[0];
    let b = &s[1];

    let (a, b) = coalesce_nulls_series(a, b);

    let name = "spearman_rank_correlation";
    if propagate_nans && a.dtype().is_float() {
        for s in [&a, &b] {
            if nan_max_s(s, "")
                .get(0)
                .unwrap()
                .extract::<f64>()
                .unwrap()
                .is_nan()
            {
                return Ok(Series::new(name, &[f64::NAN]));
            }
        }
    }

    // drop nulls so that they are excluded
    let a = a.drop_nulls();
    let b = b.drop_nulls();

    let a_rank = a.rank(
        RankOptions {
            method: RankMethod::Average,
            ..Default::default()
        },
        None,
    );
    let b_rank = b.rank(
        RankOptions {
            method: RankMethod::Average,
            ..Default::default()
        },
        None,
    );

    pearson_corr(&[a_rank, b_rank], ddof)
}
