use arrow::array::PrimitiveArray;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::float::IsFloat;

/// Search for an item, typically in a ChunkedArray.
trait ChunkSearch<'a, T> {
    /// Return the index of the given value within self, or `None` if not found.
    fn index_of(&'a self, value: Option<T>) -> Option<usize>;
}

/// Find index where predicate is true.
fn index_of_predicate<P, T>(ca: &ChunkedArray<T>, predicate: P) -> Option<usize>
where
    T: PolarsNumericType,
    P: Fn(&T::Native) -> bool,
{
    let mut index = 0;
    for chunk in ca.chunks() {
        let chunk = chunk
            .as_any()
            .downcast_ref::<PrimitiveArray<T::Native>>()
            .unwrap();
        if chunk.validity().is_some() {
            for maybe_value in chunk.iter() {
                if maybe_value.map(&predicate) == Some(true) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        } else {
            // No nulls, so we can simplify:
            for value in chunk.values_iter() {
                if predicate(value) {
                    return Some(index);
                } else {
                    index += 1;
                }
            }
        }
    }
    None
}

impl<'a, T> ChunkSearch<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn index_of(&'a self, value: Option<T::Native>) -> Option<usize> {
        // A NaN is never equal to anything, including itself. But we still want
        // to be able to search for NaNs, so we handle them specially.
        if value.map(|v| v.is_nan()) == Some(true) {
            return index_of_predicate(self, |v| v.is_nan());
        }

        // Searching for an actual value:
        if let Some(value) = value {
            return index_of_predicate(self, |v| *v == value);
        }

        // Searching for null:
        let mut index = 0;
        for chunk in self.chunks() {
            let length = chunk.len();
            if let Some(bitmap) = chunk.validity() {
                let leading_ones = bitmap.leading_ones();
                if leading_ones < length {
                    return Some(index + leading_ones);
                }
            } else {
                index += length;
            }
        }
        None
    }
}

/// Try casting the value to the correct type, then call index_of().
macro_rules! try_index_of_numeric_ca {
    ($ca:expr, $value:expr) => {{
        let ca = $ca;
        let value = $value;
        if value == AnyValue::Null {
            return Ok(ca.index_of(None));
        }
        let cast_value = cast_if_lossless(&value, ca.dtype());
        if cast_value.is_some() {
            ca.index_of(cast_value.map(|v| v.extract().unwrap()))
        } else {
            // We can't cast the searched-for value to a valid data point within
            // the dtype of the Series we're searching, which means we will
            // never find that value.
            None
        }
    }};
}

/// Cast to the dtype, but return ``None`` if the casting changed the value.
pub fn cast_if_lossless<'a>(value: &'a AnyValue, dtype: &'a DataType) -> Option<AnyValue<'a>> {
    let result = value.cast(dtype);
    let value_dtype = value.dtype();
    let roundtrip = result.cast(&value_dtype);
    if &roundtrip == value {
        Some(result)
    } else {
        None
    }
}

/// Find the index of a given value (the first and only entry in `value_series`)
/// within the series.
pub fn index_of(series: &Series, value: &AnyValue<'_>) -> PolarsResult<Option<usize>> {
    let value_dtype = value.dtype();

    let value = match value_dtype {
        dtype if dtype.is_signed_integer() => value.cast(&DataType::Int64),
        dtype if dtype.is_unsigned_integer() => value.cast(&DataType::UInt64),
        dtype if dtype.is_float() => value.cast(&DataType::Float64),
        DataType::Null => AnyValue::Null,
        _ => unimplemented!("index_of() not yet supported for dtype {:?}", value_dtype),
    };

    if *series.dtype() == DataType::Null {
        if value.is_null() {
            return Ok((series.len() > 0).then_some(0));
        } else {
            return Ok(None);
        }
    }

    Ok(downcast_as_macro_arg_physical!(
        series,
        try_index_of_numeric_ca,
        value
    ))
}
