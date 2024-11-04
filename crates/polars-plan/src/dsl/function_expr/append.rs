use polars_core::error::PolarsResult;
use polars_core::prelude::Column;
use polars_core::utils::try_get_supertype;

pub(crate) fn append(cols: &[Column], upcast: bool) -> PolarsResult<Column> {
    assert_eq!(cols.len(), 2);

    let mut lhs = cols[0].clone();
    let rhs = &cols[1];

    // @NOTE: This is only needed for when type_coercion=False, since type_coercion will insert
    // casts and set the upcast flag to false.
    if upcast && lhs.dtype() != rhs.dtype() {
        let supertype = try_get_supertype(lhs.dtype(), rhs.dtype())?;
        lhs = lhs.cast(&supertype)?;
        lhs.append(&rhs.cast(&supertype)?)?;
    } else {
        lhs.append(rhs)?;
    }

    Ok(lhs)
}
