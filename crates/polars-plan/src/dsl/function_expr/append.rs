use polars_core::error::PolarsResult;
use polars_core::prelude::Column;

pub(crate) fn append(cols: &[Column], upcast: bool) -> PolarsResult<Column> {
    assert!(!upcast);
    assert_eq!(cols.len(), 2);

    let mut lhs = cols[0].clone();
    let rhs = &cols[1];

    lhs.append(rhs)?;

    Ok(lhs)
}
