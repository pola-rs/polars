use super::*;

pub(super) fn is_in(s: &mut [Series]) -> PolarsResult<Series> {
    let left = &s[0];
    let other = &s[1];

    left.is_in(other).map(|ca| ca.into_series())
}
