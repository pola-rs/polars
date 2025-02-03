use arrow::array::{BinaryArray, PrimitiveArray};
use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::total_ord::TotalEq;
use row_encode::encode_rows_unordered;

/// Find the index of the value, or ``None`` if it can't be found.
fn index_of_value<'a, DT, AR>(ca: &'a ChunkedArray<DT>, value: AR::ValueT<'a>) -> Option<usize>
where
    DT: PolarsDataType,
    AR: StaticArray,
    AR::ValueT<'a>: TotalEq,
{
    let req_value = &value;
    let mut index = 0;
    for chunk in ca.chunks() {
        let chunk = chunk.as_any().downcast_ref::<AR>().unwrap();
        if chunk.validity().is_some() {
            for maybe_value in chunk.iter() {
                if maybe_value.map(|v| v.tot_eq(req_value)) == Some(true) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        } else {
            // A lack of a validity bitmap means there are no nulls, so we
            // can simplify our logic and use a faster code path:
            for value in chunk.values_iter() {
                if value.tot_eq(req_value) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        }
    }
    None
}

fn index_of_numeric_value<T>(ca: &ChunkedArray<T>, value: T::Native) -> Option<usize>
where
    T: PolarsNumericType,
{
    index_of_value::<_, PrimitiveArray<T::Native>>(ca, value)
}

/// Try casting the value to the correct type, then call
/// index_of_numeric_value().
macro_rules! try_index_of_numeric_ca {
    ($ca:expr, $value:expr) => {{
        let ca = $ca;
        let value = $value;
        // extract() returns None if casting failed, so consider an extract()
        // failure as not finding the value. Nulls should have been handled
        // earlier.
        let value = value.value().extract().unwrap();
        index_of_numeric_value(ca, value)
    }};
}

/// Find the index of a given value (the first and only entry in `value_series`)
/// within the series.
pub fn index_of(series: &Series, needle: Scalar) -> PolarsResult<Option<usize>> {
    polars_ensure!(
        series.dtype() == needle.dtype(),
        InvalidOperation: "Cannot perform index_of with mismatching datatypes: {:?} and {:?}",
        series.dtype(),
        needle.dtype(),
    );

    // Series is null:
    if series.dtype().is_null() {
        if needle.is_null() {
            return Ok((series.len() > 0).then_some(0));
        } else {
            return Ok(None);
        }
    }

    // Series is not null, and the value is null:
    if needle.is_null() {
        let mut index = 0;
        for chunk in series.chunks() {
            let length = chunk.len();
            if let Some(bitmap) = chunk.validity() {
                let leading_ones = bitmap.leading_ones();
                if leading_ones < length {
                    return Ok(Some(index + leading_ones));
                }
            } else {
                index += length;
            }
        }
        return Ok(None);
    }

    if series.dtype().is_primitive_numeric() {
        return Ok(downcast_as_macro_arg_physical!(
            series,
            try_index_of_numeric_ca,
            needle
        ));
    }

    if series.dtype().is_categorical() {
        // See https://github.com/pola-rs/polars/issues/20318
        polars_bail!(InvalidOperation: "index_of() on Categoricals is not supported");
    }

    // For non-numeric dtypes, we convert to row-encoding, which essentially has
    // us searching the physical representation of the data as a series of
    // bytes.
    let value_as_column = Column::new_scalar(PlSmallStr::EMPTY, needle, 1);
    let value_as_row_encoded_ca = encode_rows_unordered(&[value_as_column])?;
    let value = value_as_row_encoded_ca
        .first()
        .expect("Shouldn't have nulls in a row-encoded result");
    let ca = encode_rows_unordered(&[series.clone().into()])?;
    Ok(index_of_value::<_, BinaryArray<i64>>(&ca, value))
}
