use polars_core::prelude::{polars_bail, polars_ensure, PolarsResult};
use polars_core::series::Series;

pub(super) fn temporal_series_to_i64_scalar(s: &Series) -> Option<i64> {
    s.to_physical_repr().get(0).unwrap().extract::<i64>()
}

pub(super) fn ensure_range_bounds_contain_exactly_one_value(
    start: &Series,
    end: &Series,
) -> PolarsResult<()> {
    polars_ensure!(
        start.len() == 1,
        ComputeError: "`start` must contain exactly one value, got {} values", start.len()
    );
    polars_ensure!(
        end.len() == 1,
        ComputeError: "`end` must contain exactly one value, got {} values", end.len()
    );
    Ok(())
}

pub(super) fn broadcast_scalar_inputs(
    start: Series,
    end: Series,
) -> PolarsResult<(Series, Series)> {
    match (start.len(), end.len()) {
        (len1, len2) if len1 == len2 => Ok((start, end)),
        (1, len2) => {
            let start_matched = start.new_from_index(0, len2);
            Ok((start_matched, end))
        },
        (len1, 1) => {
            let end_matched = end.new_from_index(0, len1);
            Ok((start, end_matched))
        },
        (len1, len2) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len1, len2
            )
        },
    }
}
