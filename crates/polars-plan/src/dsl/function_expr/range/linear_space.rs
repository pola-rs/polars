use polars_core::prelude::*;
// use polars_core::with_match_physical_integer_polars_type;
use polars_ops::series::{new_linear_space, ClosedInterval};

use super::utils::ensure_range_bounds_contain_exactly_one_value;

pub(super) fn linear_space(
    s: &[Column],
    num_samples: i64,
    closed: ClosedInterval,
) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    let name = start.name();

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

    let start = start.get(0).unwrap().extract::<f64>().ok_or_else(
        || polars_err!(ComputeError: "Invalid 'start' value supplied to 'linear_space'"),
    )?;
    let end = end.get(0).unwrap().extract::<f64>().ok_or_else(
        || polars_err!(ComputeError: "Invalid 'end' value supplied to 'linear_space'"),
    )?;
    let num_samples = u64::try_from(num_samples).map_err(|v| {
        PolarsError::ComputeError(
            format!("'num_samples' must be nonnegative integer, got {}", v).into(),
        )
    })?;
    new_linear_space(start, end, num_samples, closed, name.clone()).map(Column::from)
}
