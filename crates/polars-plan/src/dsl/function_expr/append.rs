use polars_core::error::PolarsResult;
use polars_core::prelude::Column;

pub(crate) fn append(cols: &[Column]) -> PolarsResult<Column> {
    assert_eq!(cols.len(), 2);
    let mut out = cols[0].clone();
    out.append(&cols[1])?;
    Ok(out)
}
