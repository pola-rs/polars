use std::ops::{BitAnd, BitOr};

use polars_core::frame::NullStrategy;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

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
        .with_name(s[0].name());
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
        .with_name(s[0].name());
    Ok(out.into_series())
}

pub fn max_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.max_horizontal()
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn min_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.min_horizontal()
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn sum_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.sum_horizontal(NullStrategy::Ignore)
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn mean_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.mean_horizontal(NullStrategy::Ignore)
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn coalesce_series(s: &[Series]) -> PolarsResult<Series> {
    // TODO! this can be faster if we have more than two inputs.
    polars_ensure!(!s.is_empty(), NoData: "cannot coalesce empty list");
    let mut out = s[0].clone();
    for s in s {
        if !out.null_count() == 0 {
            return Ok(out);
        } else {
            let mask = out.is_not_null();
            out = out.zip_with_same_type(&mask, s)?;
        }
    }
    Ok(out)
}
