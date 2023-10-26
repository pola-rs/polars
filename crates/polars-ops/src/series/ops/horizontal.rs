use std::ops::{BitAnd, BitOr};

use polars_core::frame::NullStrategy;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

pub fn any_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL.install(|| {
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
    })?;
    Ok(out.into_series())
}

pub fn all_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL.install(|| {
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
    })?;
    Ok(out.into_series())
}

#[cfg(feature = "zip_with")]
pub fn max_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.hmax()
}

#[cfg(feature = "zip_with")]
pub fn min_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.hmin()
}

#[cfg(feature = "zip_with")]
pub fn mean_horizontal(s: &[Series], none_strategy: NullStrategy) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.hmean(none_strategy)
}

pub fn sum_horizontal(s: &[Series], none_strategy: NullStrategy) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.hsum(none_strategy)
}
