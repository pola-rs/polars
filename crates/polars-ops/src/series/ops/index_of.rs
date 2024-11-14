use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::float::IsFloat;

/// Search for an item, typically in a ChunkedArray.
trait ChunkSearch<'a, T> {
    /// Return the index of the given value within self, or `None` if not found.
    fn index_of(&'a self, value: Option<T>) -> Option<usize>;
}

impl<'a, T> ChunkSearch<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn index_of(&'a self, value: Option<T::Native>) -> Option<usize> {
        // A NaN is never equal to anything, including itself. But we still want
        // to be able to search for NaNs, so we handle them specially.
        if value.map(|v| v.is_nan()) == Some(true) {
            return self
                .iter()
                .position(|opt_val| opt_val.map(|v| v.is_nan()) == Some(true));
        }
        self.iter().position(|opt_val| opt_val == value)
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
        if cast_value == None {
            // We can can't cast the searched-for value to a valid data point
            // within the dtype of the Series we're searching, which means we
            // will never find that value.
            None
        } else {
            ca.index_of(cast_value.map(|v| v.extract().unwrap()))
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

/// Find the index of a given value (the first and only entry in
/// `value_series`), find its index within `series`.
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
