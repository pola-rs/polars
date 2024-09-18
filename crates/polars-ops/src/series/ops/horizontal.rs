use polars_core::frame::NullStrategy;
use polars_core::prelude::*;

pub fn max_horizontal(s: &[Column]) -> PolarsResult<Option<Column>> {
    let df = unsafe { DataFrame::_new_no_checks_impl(Vec::from(s)) };
    df.max_horizontal()
        .map(|s| s.map(Column::from))
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name().clone())))
}

pub fn min_horizontal(s: &[Column]) -> PolarsResult<Option<Column>> {
    let df = unsafe { DataFrame::_new_no_checks_impl(Vec::from(s)) };
    df.min_horizontal()
        .map(|s| s.map(Column::from))
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name().clone())))
}

pub fn sum_horizontal(s: &[Column]) -> PolarsResult<Option<Column>> {
    let df = unsafe { DataFrame::_new_no_checks_impl(Vec::from(s)) };
    df.sum_horizontal(NullStrategy::Ignore)
        .map(|s| s.map(Column::from))
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name().clone())))
}

pub fn mean_horizontal(s: &[Column]) -> PolarsResult<Option<Column>> {
    let df = unsafe { DataFrame::_new_no_checks_impl(Vec::from(s)) };
    df.mean_horizontal(NullStrategy::Ignore)
        .map(|s| s.map(Column::from))
        .map(|opt_s| opt_s.map(|res| res.with_name(s[0].name().clone())))
}

pub fn coalesce_columns(s: &[Column]) -> PolarsResult<Column> {
    // TODO! this can be faster if we have more than two inputs.
    polars_ensure!(!s.is_empty(), NoData: "cannot coalesce empty list");
    let mut out = s[0].clone();
    for s in s {
        if !out.null_count() == 0 {
            return Ok(out);
        } else {
            let mask = out.is_not_null();
            out = out
                .as_materialized_series()
                .zip_with_same_type(&mask, s.as_materialized_series())?
                .into();
        }
    }
    Ok(out)
}
