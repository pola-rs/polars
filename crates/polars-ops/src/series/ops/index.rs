use num_traits::{Signed, Zero};
use polars_core::error::{polars_ensure, PolarsResult};
use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::{ChunkedArray, DataType, IdxCa, PolarsIntegerType, Series, IDX_DTYPE};
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

/// May give false negatives because it ignores the null values.
fn is_positive_idx_uncertain_impl<T>(ca: &ChunkedArray<T>) -> bool
where
    T: PolarsIntegerType,
    T::Native: Signed,
{
    ca.downcast_iter().all(|v| {
        let values = v.values();
        let mut all_positive = true;

        // process chunks to autovec but still have early return
        for chunk in values.chunks(1024) {
            for v in chunk.iter() {
                all_positive &= v.is_positive() | v.is_zero()
            }
            if !all_positive {
                return all_positive;
            }
        }
        all_positive
    })
}

/// May give false negatives because it ignores the null values.
pub fn is_positive_idx_uncertain(s: &Series) -> bool {
    let dtype = s.dtype();
    debug_assert!(dtype.is_integer(), "expected integers as index");
    if dtype.is_unsigned_integer() {
        return true;
    }
    match dtype {
        DataType::Int64 => {
            let ca = s.i64().unwrap();
            is_positive_idx_uncertain_impl(ca)
        },
        DataType::Int32 => {
            let ca = s.i32().unwrap();
            is_positive_idx_uncertain_impl(ca)
        },
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            let ca = s.i16().unwrap();
            is_positive_idx_uncertain_impl(ca)
        },
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => {
            let ca = s.i8().unwrap();
            is_positive_idx_uncertain_impl(ca)
        },
        _ => unreachable!(),
    }
}
