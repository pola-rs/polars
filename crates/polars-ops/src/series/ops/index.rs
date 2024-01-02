use std::fmt::Debug;

use polars_core::error::{polars_bail, polars_ensure, PolarsResult};
use polars_core::prelude::{ChunkedArray, DataType, IdxCa, PolarsIntegerType, Series, IDX_DTYPE};
use polars_utils::index::ToIdx;
use polars_utils::IdxSize;

fn convert<T>(ca: &ChunkedArray<T>, target_len: usize) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    IdxSize: TryFrom<T::Native>,
    <IdxSize as TryFrom<T::Native>>::Error: Debug,
    T::Native: ToIdx,
{
    let target_len = target_len as u64;
    Ok(ca.apply_values_generic(|v| v.to_idx(target_len)))
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
