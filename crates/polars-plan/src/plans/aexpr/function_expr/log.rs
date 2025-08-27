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

    let s = &columns[0].as_materialized_series();
    let base = &columns[1];

    polars_ensure!(
        base.len() == 1,
        ComputeError: "base must be a single value."
    );
    let base = base.strict_cast(&DataType::Float64)?;
    match base.f64()?.get(0) {
        Some(base) => Ok(s.log(base).into()),
        None => polars_bail!(ComputeError: "'n' cannot be None for log"),
    }
}

pub(super) fn log1p(s: &Column) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().log1p().into())
}

pub(super) fn exp(s: &Column) -> PolarsResult<Column> {
    Ok(s.as_materialized_series().exp().into())
}
