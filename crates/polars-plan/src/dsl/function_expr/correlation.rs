#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Debug, Hash)]
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
        },
        CorrelationMethod::Covariance => covariance(s, ddof),
    }
}

fn covariance(s: &[Series], ddof: u8) -> PolarsResult<Series> {
    let a = &s[0];
    let b = &s[1];
    let name = "cov";

    use polars_ops::chunked_array::cov::cov;
    let ret = match a.dtype() {
        DataType::Float32 => {
            let ret = cov(a.f32().unwrap(), b.f32().unwrap(), ddof).map(|v| v as f32);
            return Ok(Series::new(name, &[ret]));
        },
        DataType::Float64 => cov(a.f64().unwrap(), b.f64().unwrap(), ddof),
        DataType::Int32 => cov(a.i32().unwrap(), b.i32().unwrap(), ddof),
        DataType::Int64 => cov(a.i64().unwrap(), b.i64().unwrap(), ddof),
        DataType::UInt32 => cov(a.u32().unwrap(), b.u32().unwrap(), ddof),
        DataType::UInt64 => cov(a.u64().unwrap(), b.u64().unwrap(), ddof),
        _ => {
            let a = a.cast(&DataType::Float64)?;
            let b = b.cast(&DataType::Float64)?;
            cov(a.f64().unwrap(), b.f64().unwrap(), ddof)
        },
    };
    Ok(Series::new(name, &[ret]))
}

fn pearson_corr(s: &[Series], ddof: u8) -> PolarsResult<Series> {
    let a = &s[0];
    let b = &s[1];
    let name = "pearson_corr";

    use polars_ops::chunked_array::cov::pearson_corr;
    let ret = match a.dtype() {
        DataType::Float32 => {
            let ret = pearson_corr(a.f32().unwrap(), b.f32().unwrap(), ddof).map(|v| v as f32);
            return Ok(Series::new(name, &[ret]));
        },
        DataType::Float64 => pearson_corr(a.f64().unwrap(), b.f64().unwrap(), ddof),
        DataType::Int32 => pearson_corr(a.i32().unwrap(), b.i32().unwrap(), ddof),
        DataType::Int64 => pearson_corr(a.i64().unwrap(), b.i64().unwrap(), ddof),
        DataType::UInt32 => pearson_corr(a.u32().unwrap(), b.u32().unwrap(), ddof),
        _ => {
            let a = a.cast(&DataType::Float64)?;
            let b = b.cast(&DataType::Float64)?;
            pearson_corr(a.f64().unwrap(), b.f64().unwrap(), ddof)
        },
    };
    Ok(Series::new(name, &[ret]))
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
