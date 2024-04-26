use super::*;

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
    ensure_is_constant_duration(half_life, time_zone, "half_life")?;
    // `half_life` is a constant duration so we can safely use `duration_ns()`.
    let half_life = half_life.duration_ns();
    let values = &s[0];
    let times = &s[1];
    let assume_sorted = !check_sorted || times.is_sorted_flag() == IsSorted::Ascending;
    polars_ops::prelude::ewm_mean_by(values, times, half_life, assume_sorted)
}
