use super::*;

pub(super) fn entropy(s: &Series, base: f64, normalize: bool) -> PolarsResult<Series> {
    let out = s.entropy(base, normalize)?;
    if matches!(s.dtype(), DataType::Float32) {
        let out = out as f32;
        Ok(Series::new(s.name(), [out]))
    } else {
        Ok(Series::new(s.name(), [out]))
    }
}

pub(super) fn log(s: &Series, base: f64) -> PolarsResult<Series> {
    Ok(s.log(base))
}

pub(super) fn log1p(s: &Series) -> PolarsResult<Series> {
    Ok(s.log1p())
}

pub(super) fn exp(s: &Series) -> PolarsResult<Series> {
    Ok(s.exp())
}
