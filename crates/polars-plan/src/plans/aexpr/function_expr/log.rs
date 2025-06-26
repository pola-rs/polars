use super::*;

pub(super) fn entropy(s: &Column, base: f64, normalize: bool) -> PolarsResult<Column> {
    let out = s.as_materialized_series().entropy(base, normalize)?;
    if matches!(s.dtype(), DataType::Float32) {
        let out = out as f32;
        Ok(Column::new(s.name().clone(), [out]))
    } else {
        Ok(Column::new(s.name().clone(), [out]))
    }
}

pub(super) fn log(s: &Column, base: f64) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().log(base).into())
}

pub(super) fn log1p(s: &Column) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().log1p().into())
}

pub(super) fn exp(s: &Column) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().exp().into())
}
