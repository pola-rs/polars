use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> Result<Series> {
    Ok(s.shift(periods))
}
