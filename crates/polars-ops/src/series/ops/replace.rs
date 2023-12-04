use polars_core::prelude::Series;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

pub fn replace(s: &Series, old: &Series, new: &Series) -> PolarsResult<Series> {
    // TODO: Allow 'broadcasting' `new` here for many-to-one replace?
    polars_ensure!(
        old.len() == new.len(),
        ComputeError: "`old` and `new` inputs for `replace` must have the same length"
    );

    match old.len() {
        0 => return Ok(s.clone()),
        1 => (), // dispatch to when/then
        _ => (),
    };

    let s = s.clone();
    Ok(s)
}

pub fn replace_with_default(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
) -> PolarsResult<Series> {
    // TODO: Allow 'broadcasting' `new` here for many-to-one replace?
    polars_ensure!(
        old.len() == new.len(),
        ComputeError: "`old` and `new` inputs for `replace` must have the same length"
    );

    match old.len() {
        0 => return Ok(default.clone()),
        1 => (), // dispatch to when/then
        _ => (),
    };

    let s = s.clone();
    Ok(s)
}
