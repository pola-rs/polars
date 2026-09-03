use polars_array::arrow::bridge::{chunk_from_arrow, flat_to_arrow};
use polars_core::chunked_array::ChunkedArray;
use polars_core::chunked_array::ops::arity::unary_elementwise_mut_values_flat;
use polars_core::prelude::DataType;
use polars_core::series::Series;
use polars_core::{with_match_physical_float_polars_type, with_match_physical_integer_polars_type};
use polars_error::{PolarsResult, polars_bail};

use super::*;

// TODO(polars-array-scalar): the bitwise kernels are Arrow ones, so a scalar chunk is written out
// before it reaches them rather than the single value it stands for being counted once.
macro_rules! apply_bitwise_op {
    ($($op:ident),+ $(,)?) => {
        $(
        pub fn $op(s: &Series) -> PolarsResult<Series> {
            match s.dtype() {
                DataType::Boolean => {
                    let ca: &ChunkedArray<BooleanType> = s.as_any().downcast_ref().unwrap();
                    Ok(unary_elementwise_mut_values_flat::<BooleanType, UInt32Type, _, _>(
                        ca,
                        |a| {
                            let a = flat_to_arrow(a);
                            chunk_from_arrow(&polars_compute::bitwise::BitwiseKernel::$op(&a))
                        },
                    ).into_series())
                },
                dt if dt.is_integer() => {
                    with_match_physical_integer_polars_type!(dt, |$T| {
                        let ca: &ChunkedArray<$T> = s.as_any().downcast_ref().unwrap();
                        Ok(unary_elementwise_mut_values_flat::<$T, UInt32Type, _, _>(
                            ca,
                            |a| {
                                let a = flat_to_arrow(a);
                                chunk_from_arrow(&polars_compute::bitwise::BitwiseKernel::$op(&a))
                            },
                        ).into_series())
                    })
                },
                dt if dt.is_float() => {
                    with_match_physical_float_polars_type!(dt, |$T| {
                        let ca: &ChunkedArray<$T> = s.as_any().downcast_ref().unwrap();
                        Ok(unary_elementwise_mut_values_flat::<$T, UInt32Type, _, _>(
                            ca,
                            |a| {
                                let a = flat_to_arrow(a);
                                chunk_from_arrow(&polars_compute::bitwise::BitwiseKernel::$op(&a))
                            },
                        ).into_series())
                    })
                },
                dt => {
                    polars_bail!(InvalidOperation: "dtype {:?} not supported in '{}' operation", dt, stringify!($op))
                },
            }
        }
        )+

    };
}

apply_bitwise_op! {
    count_ones,
    count_zeros,
    leading_ones,
    leading_zeros,
    trailing_ones,
    trailing_zeros,
}
