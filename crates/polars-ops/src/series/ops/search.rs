use polars_core::downcast_as_macro_arg_physical;
use polars_core::prelude::*;
use polars_utils::float::IsFloat;

use crate::series::ops::SeriesSealed;

/// Search for an item, typically in a ChunkedArray.
pub trait ChunkSearch<'a, T> {
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
        let cast_value = $value.map(|v| AnyValue::from(v).strict_cast(ca.dtype()));
        if cast_value == Some(None) {
            // We can can't cast the searched-for value to a valid data point
            // within the dtype of the Series we're searching, which means we
            // will never find that value.
            None
        } else {
            let cast_value = cast_value.flatten();
            ca.index_of(cast_value.map(|v| v.extract().unwrap()))
        }
    }};
}

pub trait SearchSeries: SeriesSealed {
    fn index_of(&self, value_series: &Series) -> PolarsResult<Option<usize>> {
        let series = self.as_series();
        let value_series = if value_series.dtype().is_null() {
            // Should be able to cast null dtype to anything, so cast it to dtype of
            // Series we're searching.
            &value_series.cast(series.dtype())?
        } else {
            value_series
        };
        let value_dtype = value_series.dtype();

        if value_dtype.is_signed_integer() {
            let value = value_series.cast(&DataType::Int64)?.i64().unwrap().get(0);
            let result = downcast_as_macro_arg_physical!(series, try_index_of_numeric_ca, value);
            return Ok(result);
        }
        if value_dtype.is_unsigned_integer() {
            let value = value_series.cast(&DataType::UInt64)?.u64().unwrap().get(0);
            return Ok(downcast_as_macro_arg_physical!(series, try_index_of_numeric_ca, value));
        }
        if value_dtype.is_float() {
            let value = value_series.cast(&DataType::Float64)?.f64().unwrap().get(0);
            return Ok(downcast_as_macro_arg_physical!(series, try_index_of_numeric_ca, value));
        }
        // At this point we're done handling integers and floats.
        unimplemented!("TODO")
    }
}

