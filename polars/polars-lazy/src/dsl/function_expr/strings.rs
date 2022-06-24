use super::*;

pub(super) fn contains(s: &Series, pat: &str, literal: bool) -> Result<Series> {
    let ca = s.utf8()?;
    if literal {
        ca.contains_literal(pat).map(|ca| ca.into_series())
    } else {
        ca.contains(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn ends_with(s: &Series, sub: &str) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.ends_with(sub).into_series())
}
pub(super) fn starts_with(s: &Series, sub: &str) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.starts_with(sub).into_series())
}
