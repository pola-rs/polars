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

pub(super) fn log(columns: &[Column]) -> PolarsResult<Column> {
    assert_eq!(columns.len(), 2);

    let s = columns[0].as_materialized_series();
    let base = columns[1].as_materialized_series();
    Ok(s.log(base).into())
}

pub(super) fn log1p(s: &Column) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().log1p().into())
}

pub(super) fn exp(s: &Column) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().exp().into())
}
