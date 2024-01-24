use arrow::array::{Array, PrimitiveArray};
use arrow::compute::utils::combine_validities_and;
use polars_core::datatypes::PolarsNumericType;
use polars_core::prelude::*;
#[cfg(feature = "dtype-struct")]
use polars_core::series::arithmetic::_struct_arithmetic;
use polars_core::with_match_physical_numeric_polars_type;

#[inline]
fn floor_div_element<T: NumericNative>(a: T, b: T) -> PolarsResult<T> {
    polars_ensure!(T::is_float() || b != T::zero(), ComputeError: "integer division by zero");
    Ok(a.floor_div(b))
}

fn floor_div_array<T: NumericNative>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
) -> PolarsResult<PrimitiveArray<T>> {
    assert_eq!(a.len(), b.len());

    if a.null_count() == 0 && b.null_count() == 0 {
        let values = a
            .values()
            .as_slice()
            .iter()
            .copied()
            .zip(b.values().as_slice().iter().copied())
            .map(|(a, b)| floor_div_element(a, b))
            .collect::<PolarsResult<Vec<_>>>()?;

        let validity = combine_validities_and(a.validity(), b.validity());

        Ok(PrimitiveArray::new(
            a.data_type().clone(),
            values.into(),
            validity,
        ))
    } else {
        let iter = a
            .into_iter()
            .zip(b)
            .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
                (Some(&a), Some(&b)) => Ok(Some(floor_div_element(a, b)?)),
                _ => Ok(None),
            });
        PrimitiveArray::try_from_trusted_len_iter(iter)
    }
}

fn floor_div_ca<T: PolarsNumericType>(
    a: &ChunkedArray<T>,
    b: &ChunkedArray<T>,
) -> PolarsResult<ChunkedArray<T>> {
    if a.len() == 1 {
        let name = a.name();
        return if let Some(a) = a.get(0) {
            let out = b.try_apply(|b| floor_div_element(a, b))?;
            Ok(out.with_name(name))
        } else {
            Ok(ChunkedArray::full_null(a.name(), b.len()))
        };
    }
    if b.len() == 1 {
        return if let Some(b) = b.get(0) {
            a.try_apply(|a| floor_div_element(a, b))
        } else {
            Ok(ChunkedArray::full_null(a.name(), a.len()))
        };
    }
    arity::try_binary(a, b, floor_div_array)
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

        floor_div_ca(a, b)?.into_series()
    });

    out.cast(logical_type)
}
