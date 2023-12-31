use std::fmt::Debug;

use polars_core::error::{polars_bail, polars_ensure, PolarsError, PolarsResult};
use polars_core::export::num::{FromPrimitive, Signed, ToPrimitive, Zero};
use polars_core::prelude::{ChunkedArray, DataType, IdxCa, PolarsIntegerType, Series, IDX_DTYPE};
use polars_utils::IdxSize;

fn convert<T>(ca: &ChunkedArray<T>, target_len: usize) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    IdxSize: TryFrom<T::Native>,
    <IdxSize as TryFrom<T::Native>>::Error: Debug,
    T::Native: FromPrimitive + Signed + Zero,
{
    let len =
        i64::from_usize(target_len).ok_or_else(|| PolarsError::ComputeError("overflow".into()))?;

    let zero = T::Native::zero();

    ca.try_apply_values_generic(|v| {
        if v >= zero {
            Ok(IdxSize::try_from(v).unwrap())
        } else {
            IdxSize::from_i64(len + v.to_i64().unwrap()).ok_or_else(|| {
                PolarsError::OutOfBounds(
                    format!(
                        "index {} is out of bounds for series of len {}",
                        v, target_len
                    )
                    .into(),
                )
            })
        }
    })
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
