use super::*;

pub(super) fn upper_bound(s: &Column) -> PolarsResult<Column> {
    let name = s.name().clone();
    let scalar = s.dtype().to_physical().max()?;
    Ok(Column::new_scalar(name, scalar, 1))
}

pub(super) fn lower_bound(s: &Column) -> PolarsResult<Column> {
    let name = s.name().clone();
    let scalar = s.dtype().to_physical().min()?;
    Ok(Column::new_scalar(name, scalar, 1))
}
