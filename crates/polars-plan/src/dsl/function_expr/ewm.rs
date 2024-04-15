use super::*;

pub(super) fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    polars_ops::prelude::ewm_mean(s, options)
}

pub(super) fn ewm_mean_by(
    s: &[Series],
    half_life: Duration,
    check_sorted: bool,
) -> PolarsResult<Series> {
    let time_zone = match s[1].dtype() {
        DataType::Datetime(_, Some(time_zone)) => Some(time_zone.as_str()),
        _ => None,
    };
    polars_ensure!(!half_life.negative(), InvalidOperation: "half_life cannot be negative");
    polars_ensure!(half_life.is_constant_duration(time_zone),
        InvalidOperation: "expected `half_life` to be a constant duration \
        (i.e. one independent of differing month durations or of daylight savings time), got {}.\n\
        \n\
        You may want to try:\n\
        - using `'730h'` instead of `'1mo'`\n\
        - using `'24h'` instead of `'1d'` if your series is time-zone-aware", half_life);
    // `half_life` is a constant duration so we can safely use `duration_ns()`.
    let half_life = half_life.duration_ns();
    let values = &s[0];
    let times = &s[1];
    let assume_sorted = !check_sorted || times.is_sorted_flag() == IsSorted::Ascending;
    polars_ops::prelude::ewm_mean_by(values, times, half_life, assume_sorted)
}

pub(super) fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    polars_ops::prelude::ewm_std(s, options)
}

pub(super) fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    polars_ops::prelude::ewm_var(s, options)
}
