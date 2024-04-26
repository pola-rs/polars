use polars_core::frame::NullStrategy;
use polars_core::prelude::*;

pub fn max_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = unsafe { DataFrame::new_no_checks(Vec::from(s)) };
    df.max_horizontal()
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn min_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = unsafe { DataFrame::new_no_checks(Vec::from(s)) };
    df.min_horizontal()
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn sum_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = unsafe { DataFrame::new_no_checks(Vec::from(s)) };
    df.sum_horizontal(NullStrategy::Ignore)
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name())))
}

pub fn mean_horizontal(s: &[Series]) -> PolarsResult<Option<Series>> {
    let df = unsafe { DataFrame::new_no_checks(Vec::from(s)) };
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
