use std::borrow::Cow;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

pub fn sum_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let sum_fn = |acc: &Series, s: &Series| {
        PolarsResult::Ok(
            acc.fill_null(FillNullStrategy::Zero)? + s.fill_null(FillNullStrategy::Zero)?,
        )
    };
    let out = match s.len() {
        0 => Ok(UInt32Chunked::new("", &[0u32]).into_series()),
        1 => Ok(s[0].clone()),
        2 => sum_fn(&s[0], &s[1]),
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over series, not over elements
            POOL.install(|| {
                s.par_iter()
                    .map(|s| Ok(Cow::Borrowed(s)))
                    .try_reduce_with(|l, r| sum_fn(&l, &r).map(Cow::Owned))
                    // we can unwrap the option, because we are certain there is a series
                    .unwrap()
                    .map(|cow| cow.into_owned())
            })
        },
    };
    out.map(|ok| ok.with_name("sum"))
}

#[cfg(feature = "zip_with")]
pub fn max_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.max_horizontal()
        .map(|opt_s| opt_s.map(|s| s.with_name("max")))
}

#[cfg(feature = "zip_with")]
pub fn min_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = DataFrame::new_no_checks(Vec::from(s));
    df.min_horizontal()
        .map(|opt_s| opt_s.map(|s| s.with_name("min")))
}
