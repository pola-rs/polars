use polars_core::frame::NullStrategy;
use polars_core::prelude::*;

fn validate_column_lengths(cs: &[Column]) -> PolarsResult<()> {
    let mut length = 1;
    for c in cs {
        let len = c.len();
        if len != 1 && len != length {
            if length == 1 {
                length = len;
            } else {
                polars_bail!(ShapeMismatch: "cannot evaluate two Series of different lengths ({len} and {length})");
            }
        }
    }
    Ok(())
}

pub fn max_horizontal(s: &[Column]) -> PolarsResult<Option<Column>> {
    validate_column_lengths(s)?;
    polars_core::series::arithmetic::horizontal::max_horizontal(s)
        .map(|opt_c| opt_c.map(|res| res.with_name(s[0].name().clone())))
}

pub fn min_horizontal(s: &[Column]) -> PolarsResult<Option<Column>> {
    validate_column_lengths(s)?;
    polars_core::series::arithmetic::horizontal::min_horizontal(s)
        .map(|opt_c| opt_c.map(|res| res.with_name(s[0].name().clone())))
}

pub fn sum_horizontal(s: &[Column], null_strategy: NullStrategy) -> PolarsResult<Option<Column>> {
    validate_column_lengths(s)?;
    polars_core::series::arithmetic::horizontal::sum_horizontal(s, null_strategy)
        .map(|opt_c| opt_c.map(|res| res.with_name(s[0].name().clone())))
}

pub fn mean_horizontal(s: &[Column], null_strategy: NullStrategy) -> PolarsResult<Option<Column>> {
    validate_column_lengths(s)?;
    polars_core::series::arithmetic::horizontal::mean_horizontal(s, null_strategy)
        .map(|opt_c| opt_c.map(|res| res.with_name(s[0].name().clone())))
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
