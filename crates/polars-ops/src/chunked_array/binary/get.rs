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

fn collect_result(
    iter: impl Iterator<Item = PolarsResult<Option<u8>>>,
    name: PlSmallStr,
) -> PolarsResult<Column> {
    let mut out: UInt8Chunked = iter.collect::<PolarsResult<_>>()?;
    out.rename(name);
    Ok(out.into_column())
}

pub fn bin_get(
    ca: &BinaryChunked,
    index: &Int64Chunked,
    null_on_oob: bool,
) -> PolarsResult<Column> {
    let name = ca.name().clone();
    match index.len() {
        1 => {
            let idx = index.get(0);
            collect_result(
                ca.into_iter().map(move |b| get_byte(b, idx, null_on_oob)),
                name,
            )
        },
        len if len == ca.len() => collect_result(
            ca.into_iter()
                .zip(index)
                .map(|(b, i)| get_byte(b, i, null_on_oob)),
            name,
        ),
        _ if ca.len() == 1 => {
            let bytes = ca.get(0);
            collect_result(
                index
                    .into_iter()
                    .map(move |i| get_byte(bytes, i, null_on_oob)),
                name,
            )
        },
        len => polars_bail!(
            ComputeError:
            "`bin.get` expression got an index array of length {} while the binary column has {} elements",
            len, ca.len()
        ),
    }
}
