use arrow::array::{Array, PrimitiveArray};
use arrow::compute::utils::combine_validities_and;
use num::NumCast;
use polars_core::datatypes::PolarsNumericType;
use polars_core::export::num;
use polars_core::prelude::*;
#[cfg(feature = "dtype-struct")]
use polars_core::series::arithmetic::_struct_arithmetic;
use polars_core::with_match_physical_numeric_polars_type;

#[inline]
fn floor_div_element<T: NumericNative>(a: T, b: T) -> T {
    // Safety: the casts of those primitives always succeed
    unsafe {
        let a: f64 = NumCast::from(a).unwrap_unchecked();
        let b: f64 = NumCast::from(b).unwrap_unchecked();

        let out = (a / b).floor();
        let out: T = NumCast::from(out).unwrap_unchecked();
        out
    }
}

fn floor_div_array<T: NumericNative>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> PrimitiveArray<T> {
    assert_eq!(a.len(), b.len());

    if a.null_count() == 0 && b.null_count() == 0 {
        let values = a
            .values()
            .as_slice()
            .iter()
            .copied()
            .zip(b.values().as_slice().iter().copied())
            .map(|(a, b)| floor_div_element(a, b))
            .collect::<Vec<_>>();

        let validity = combine_validities_and(a.validity(), b.validity());

        PrimitiveArray::new(a.data_type().clone(), values.into(), validity)
    } else {
        let iter = a
            .into_iter()
            .zip(b)
            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                (Some(&a), Some(&b)) => Some(floor_div_element(a, b)),
                _ => None,
            });
        PrimitiveArray::from_trusted_len_iter(iter)
    }
}

fn floor_div_ca<T: PolarsNumericType>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> ChunkedArray<T> {
    if a.len() == 1 {
        let name = a.name();
        return if let Some(a) = a.get(0) {
            let mut out = if b.null_count() == 0 {
                b.apply_values(|b| floor_div_element(a, b))
            } else {
                b.apply(|b| b.map(|b| floor_div_element(a, b)))
            };
            out.rename(name);
            out
        } else {
            ChunkedArray::full_null(a.name(), b.len())
        };
    }
    if b.len() == 1 {
        return if let Some(b) = b.get(0) {
            if a.null_count() == 0 {
                a.apply_values(|a| floor_div_element(a, b))
            } else {
                a.apply(|a| a.map(|a| floor_div_element(a, b)))
            }
        } else {
            ChunkedArray::full_null(a.name(), a.len())
        };
    }
    arity::binary(a, b, floor_div_array)
}

pub fn floor_div_series(a: &Series, b: &Series) -> PolarsResult<Series> {
    match (a.dtype(), b.dtype()) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(_), DataType::Struct(_)) => {
            return Ok(_struct_arithmetic(a, b, |a, b| {
                floor_div_series(a, b).unwrap()
            }))
        },
        _ => {},
    }

    let logical_type = a.dtype();

    let a = a.to_physical_repr();
    let b = b.to_physical_repr();

    let out = with_match_physical_numeric_polars_type!(a.dtype(), |$T| {
        let a: &ChunkedArray<$T> = a.as_ref().as_ref().as_ref();
        let b: &ChunkedArray<$T> = b.as_ref().as_ref().as_ref();

        floor_div_ca(a, b).into_series()
    });

    out.cast(logical_type)
}
