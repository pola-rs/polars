use std::ops::{BitAnd, BitOr};

use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

pub fn sum_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL
        .install(|| {
            s.par_iter()
                .try_fold(
                    || UInt32Chunked::new("", &[0u32]).into_series(),
                    |acc, b| {
                        PolarsResult::Ok(
                            acc.fill_null(FillNullStrategy::Zero)?
                                + b.fill_null(FillNullStrategy::Zero)?,
                        )
                    },
                )
                .try_reduce(
                    || UInt32Chunked::new("", &[0u32]).into_series(),
                    |a, b| {
                        PolarsResult::Ok(
                            a.fill_null(FillNullStrategy::Zero)?
                                + b.fill_null(FillNullStrategy::Zero)?,
                        )
                    },
                )
        })?
        .with_name("sum");
    Ok(out)
}

pub fn any_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL
        .install(|| {
            s.par_iter()
                .try_fold(
                    || BooleanChunked::new("", &[false]),
                    |acc, b| {
                        let b = b.cast(&DataType::Boolean)?;
                        let b = b.bool()?;
                        PolarsResult::Ok((&acc).bitor(b))
                    },
                )
                .try_reduce(|| BooleanChunked::new("", [false]), |a, b| Ok(a.bitor(b)))
        })?
        .with_name("any");
    Ok(out.into_series())
}

pub fn all_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL
        .install(|| {
            s.par_iter()
                .try_fold(
                    || BooleanChunked::new("", &[true]),
                    |acc, b| {
                        let b = b.cast(&DataType::Boolean)?;
                        let b = b.bool()?;
                        PolarsResult::Ok((&acc).bitand(b))
                    },
                )
                .try_reduce(|| BooleanChunked::new("", [true]), |a, b| Ok(a.bitand(b)))
        })?
        .with_name("all");
    Ok(out.into_series())
}

#[cfg(feature = "zip_with")]
pub fn max_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.hmax().map(|opt_s| opt_s.map(|s| s.with_name("max")))
}

#[cfg(feature = "zip_with")]
pub fn min_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.hmin().map(|opt_s| opt_s.map(|s| s.with_name("min")))
}
