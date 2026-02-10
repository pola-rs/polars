use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail};

fn get_byte(bytes: Option<&[u8]>, idx: Option<i64>, null_on_oob: bool) -> PolarsResult<Option<u8>> {
    let (Some(bytes), Some(idx)) = (bytes, idx) else {
        return Ok(None);
    };

    let len = bytes.len() as i64;
    let idx = if idx >= 0 { idx } else { len + idx };
    if idx < 0 || idx >= len {
        if null_on_oob {
            Ok(None)
        } else {
            polars_bail!(ComputeError: "get index is out of bounds")
        }
    } else {
        Ok(Some(bytes[idx as usize]))
    }
}

pub fn bin_get(
    ca: &BinaryChunked,
    index: &Int64Chunked,
    null_on_oob: bool,
) -> PolarsResult<Column> {
    let out: UInt8Chunked =
        broadcast_try_binary_elementwise(ca, index, |b, idx| get_byte(b, idx, null_on_oob))?;
    Ok(out.into_column())
}
