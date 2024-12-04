use arrow::array::{Array, BinaryArray, PrimitiveArray};
use arrow::types::NativeType;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::total_ord::TotalEq;
use row_encode::encode_rows_unordered;

/// Annoyingly, iter() and values_iter() on Arrow arrays are not part of the
/// Array trait, so we have to create our own.
trait IterableArray<'a>: Array {
    type Value: TotalEq;

    fn wrapped_iter(&'a self) -> impl Iterator<Item = Option<Self::Value>>;

    fn wrapped_values_iter(&'a self) -> impl Iterator<Item = Self::Value>;
}

impl<'a, T> IterableArray<'a> for PrimitiveArray<T>
where
    T: NativeType,
{
    type Value = &'a T;

    fn wrapped_iter(&'a self) -> impl Iterator<Item = Option<Self::Value>> {
        self.iter()
    }

    fn wrapped_values_iter(&'a self) -> impl Iterator<Item = Self::Value> {
        self.values_iter()
    }
}

impl<'a> IterableArray<'a> for BinaryArray<i64> {
    type Value = &'a [u8];

    fn wrapped_iter(&'a self) -> impl Iterator<Item = Option<Self::Value>> {
        self.iter()
    }

    fn wrapped_values_iter(&'a self) -> impl Iterator<Item = Self::Value> {
        self.values_iter()
    }
}

/// Find the index of the value, or ``None`` if it can't be found.
fn index_of_value<'a, DT, AR>(ca: &'a ChunkedArray<DT>, value: AR::Value) -> Option<usize>
where
    DT: PolarsDataType,
    AR: IterableArray<'a>,
{
    let req_value = &value;
    let mut index = 0;
    for chunk in ca.chunks() {
        let chunk = chunk.as_any().downcast_ref::<AR>().unwrap();
        if chunk.validity().is_some() {
            for maybe_value in chunk.wrapped_iter() {
                if maybe_value.map(|v| v.tot_eq(req_value)) == Some(true) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        } else {
            // A lack of a validity bitmap means there are no nulls, so we
            // can simplify our logic and use a faster code path:
            for value in chunk.wrapped_values_iter() {
                if value.tot_eq(req_value) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        }
    }
    return None;
}

fn index_of_numeric_value<T>(ca: &ChunkedArray<T>, value: T::Native) -> Option<usize>
where
    T: PolarsNumericType,
{
    index_of_value::<_, PrimitiveArray<T::Native>>(ca, &value)
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
        let Some(value) = value.extract() else {
            return Ok(None);
        };
        index_of_numeric_value(ca, value)
    }};
}

/// Find the index of a given value (the first and only entry in `value_series`)
/// within the series.
pub fn index_of(series: &Series, value: &AnyValue<'_>) -> PolarsResult<Option<usize>> {
    // Series is null:
    if series.dtype().is_null() {
        if value.is_null() {
            return Ok((series.len() > 0).then_some(0));
        } else {
            return Ok(None);
        }
    }

    // Series is not null, and the value is null:
    if value.dtype().is_null() {
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

    if series.dtype().is_numeric() {
        return Ok(downcast_as_macro_arg_physical!(
            series,
            try_index_of_numeric_ca,
            value
        ));
    }

    // For non-numeric dtypes, we convert to row-encoding, which essentially has
    // us searching the physical representation of the data as a series of
    // bytes.

    // Arrays currently don't support row encoding (at least in some code
    // paths), so stick to lists if we get an array.
    #[cfg(feature = "dtype-array")]
    let series = if series.dtype().is_array() {
        &series.cast(&DataType::List(Box::new(
            series.dtype().inner_dtype().unwrap().clone(),
        )))?
    } else {
        series
    };

    let value_as_ca =
        encode_rows_unordered(&[
            Series::from_any_values("".into(), &[value.clone()], false)?.cast(series.dtype())?
        ])?;
    let value = value_as_ca
        .first()
        .expect("Shouldn't have nulls in a row-encoded result");
    let ca = encode_rows_unordered(&[series.clone()])?;
    Ok(index_of_value::<_, BinaryArray<i64>>(&ca, value))
}
