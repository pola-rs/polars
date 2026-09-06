use polars_core::prelude::PolarsResult;

use crate::prelude::*;

#[cfg(feature = "serde")]
pub fn serialize(expr: &Expr) -> PolarsResult<Option<Vec<u8>>> {
    let mut buf = vec![];
    expr.serialize_binary_into(&mut buf)?;
    Ok(Some(buf))
}
