use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::{ChunkedArray, DataType, IDX_DTYPE, IdxCa, PolarsIntegerType, Series};
use polars_utils::index::ToIdx;

fn convert<T>(ca: &ChunkedArray<T>, target_len: usize) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    T::Native: ToIdx,
{
    let target_len = target_len as u64;
    Ok(unary_elementwise_values(ca, |v| v.to_idx(target_len)))
}

pub fn convert_to_unsigned_index(s: &Series, target_len: usize) -> PolarsResult<IdxCa> {
    let dtype = s.dtype();
    polars_ensure!(dtype.is_integer(), InvalidOperation: "expected integers as index");
    if dtype.is_unsigned_integer() {
        let nulls_before_cast = s.null_count();
        let out = s.cast(&IDX_DTYPE).unwrap();
        polars_ensure!(out.null_count() == nulls_before_cast, OutOfBounds: "some integers did not fit polars' index size");
        return Ok(out.idx().unwrap().clone());
    }
    match dtype {
        DataType::Int64 => {
            let ca = s.i64().unwrap();
            convert(ca, target_len)
        },
        DataType::Int32 => {
            let ca = s.i32().unwrap();
            convert(ca, target_len)
        },
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            let ca = s.i16().unwrap();
            convert(ca, target_len)
        },
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => {
            let ca = s.i8().unwrap();
            convert(ca, target_len)
        },
        _ => unreachable!(),
    }
}
