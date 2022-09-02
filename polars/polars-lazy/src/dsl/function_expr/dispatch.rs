use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> Result<Series> {
    Ok(s.shift(periods))
}

pub(super) fn reverse(s: &Series) -> Result<Series> {
    Ok(s.reverse())
}
