use polars_core::prelude::PolarsResult;

use crate::prelude::*;

#[cfg(feature = "serde")]
pub fn serialize(expr: &Expr) -> PolarsResult<Option<Vec<u8>>> {
    let mut buf = vec![];
    polars_utils::pl_serialize::serialize_into_writer::<_, _, true>(&mut buf, expr)?;

    Ok(Some(buf))
}
