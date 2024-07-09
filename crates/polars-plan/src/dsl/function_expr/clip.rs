use super::*;

pub(super) fn clip(s: &[Series], has_min: bool, has_max: bool) -> PolarsResult<Series> {
    match (has_min, has_max) {
        (true, true) => polars_ops::series::clip(&s[0], &s[1], &s[2]),
        (true, false) => polars_ops::series::clip_min(&s[0], &s[1]),
        (false, true) => polars_ops::series::clip_max(&s[0], &s[1]),
        _ => unreachable!(),
    }
}
