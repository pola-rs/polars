use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::{
    ChunkedArray, DataType, IDX_DTYPE, IdxCa, IdxSize, PolarsIntegerType, Series,
};
use polars_utils::index::ToIdx;

fn convert<T>(ca: &ChunkedArray<T>, target_len: usize) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    T::Native: ToIdx,
{
    let target_len = target_len as u64;
    Ok(unary_elementwise_values(ca, |v| v.to_idx(target_len)))
}

/// Generic OOB handler on an `IdxCa`.
///
/// - `null_on_oob == false`: error if any index >= len
/// - `null_on_oob == true`: set index >= len to null
fn handle_oob(idx: IdxCa, len: usize, null_on_oob: bool) -> PolarsResult<IdxCa> {
    let len_idx = len as IdxSize;

    if !null_on_oob {
        let oob = idx
            .into_iter()
            .any(|opt_v| opt_v.is_some_and(|v| v >= len_idx));
        polars_ensure!(
            !oob,
            OutOfBounds: "gather index is out of bounds"
        );
        Ok(idx)
    } else {
        let out: IdxCa = idx
            .into_iter()
            .map(|opt_v| opt_v.and_then(|v| if v >= len_idx { None } else { Some(v) }))
            .collect();
        Ok(out)
    }
}

pub fn convert_to_unsigned_index(
    s: &Series,
    target_len: usize,
    null_on_oob: bool,
) -> PolarsResult<IdxCa> {
    let dtype = s.dtype();
    polars_ensure!(
        dtype.is_integer(),
        InvalidOperation: "expected integers as index"
    );

    // normalize to IdxCa (may still contain OOB for unsigned)
    let idx: IdxCa = match dtype {
        d if d.is_unsigned_integer() => {
            let nulls_before_cast = s.null_count();
            let out = s.cast(&IDX_DTYPE)?;
            polars_ensure!(
                out.null_count() == nulls_before_cast,
                OutOfBounds: "some integers did not fit polars' index size"
            );
            out.idx().unwrap().clone()
        },

        DataType::Int64 => {
            let ca = s.i64().unwrap();
            convert(ca, target_len)?
        },
        DataType::Int32 => {
            let ca = s.i32().unwrap();
            convert(ca, target_len)?
        },
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            let ca = s.i16().unwrap();
            convert(ca, target_len)?
        },
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => {
            let ca = s.i8().unwrap();
            convert(ca, target_len)?
        },

        _ => unreachable!(),
    };

    handle_oob(idx, target_len, null_on_oob)
}
