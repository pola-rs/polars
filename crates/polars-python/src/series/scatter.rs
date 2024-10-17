use polars::export::arrow::array::Array;
use polars::prelude::*;
use polars_core::downcast_as_macro_arg_physical;
use pyo3::prelude::*;

use super::PySeries;
use crate::error::PyPolarsErr;

#[pymethods]
impl PySeries {
    fn scatter(&mut self, py: Python, idx: PySeries, values: PySeries) -> PyResult<()> {
        // we take the value because we want a ref count of 1 so that we can
        // have mutable access cheaply via _get_inner_mut().
        let s = std::mem::take(&mut self.series);
        let result = py.allow_threads(|| scatter(s, &idx.series, &values.series));
        match result {
            Ok(out) => {
                self.series = out;
                Ok(())
            },
            Err((s, e)) => {
                // Restore original series:
                self.series = s;
                Err(PyErr::from(PyPolarsErr::from(e)))
            },
        }
    }
}

fn scatter(mut s: Series, idx: &Series, values: &Series) -> Result<Series, (Series, PolarsError)> {
    let logical_dtype = s.dtype().clone();

    let idx = match polars_ops::prelude::convert_to_unsigned_index(idx, s.len()) {
        Ok(idx) => idx,
        Err(err) => return Err((s, err)),
    };
    let idx = idx.rechunk();
    let idx = idx.downcast_iter().next().unwrap();

    if idx.null_count() > 0 {
        return Err((
            s,
            PolarsError::ComputeError("index values should not be null".into()),
        ));
    }

    let idx = idx.values().as_slice();

    let mut values = match values.to_physical_repr().cast(&s.dtype().to_physical()) {
        Ok(values) => values,
        Err(err) => return Err((s, err)),
    };

    // Broadcast values input
    if values.len() == 1 && idx.len() > 1 {
        values = values.new_from_index(0, idx.len());
    }

    // do not shadow, otherwise s is not dropped immediately
    // and we want to have mutable access
    s = s.to_physical_repr().into_owned();
    let s_mut_ref = &mut s;
    scatter_impl(s_mut_ref, logical_dtype, idx, &values).map_err(|err| (s, err))
}

fn scatter_impl(
    s: &mut Series,
    logical_dtype: DataType,
    idx: &[IdxSize],
    values: &Series,
) -> PolarsResult<Series> {
    let mutable_s = s._get_inner_mut();

    let s = match logical_dtype.to_physical() {
        DataType::Int8 => {
            let ca: &mut ChunkedArray<Int8Type> = mutable_s.as_mut();
            let values = values.i8()?;
            ca.scatter(idx, values)
        },
        DataType::Int16 => {
            let ca: &mut ChunkedArray<Int16Type> = mutable_s.as_mut();
            let values = values.i16()?;
            ca.scatter(idx, values)
        },
        DataType::Int32 => {
            let ca: &mut ChunkedArray<Int32Type> = mutable_s.as_mut();
            let values = values.i32()?;
            ca.scatter(idx, values)
        },
        DataType::Int64 => {
            let ca: &mut ChunkedArray<Int64Type> = mutable_s.as_mut();
            let values = values.i64()?;
            ca.scatter(idx, values)
        },
        DataType::UInt8 => {
            let ca: &mut ChunkedArray<UInt8Type> = mutable_s.as_mut();
            let values = values.u8()?;
            ca.scatter(idx, values)
        },
        DataType::UInt16 => {
            let ca: &mut ChunkedArray<UInt16Type> = mutable_s.as_mut();
            let values = values.u16()?;
            ca.scatter(idx, values)
        },
        DataType::UInt32 => {
            let ca: &mut ChunkedArray<UInt32Type> = mutable_s.as_mut();
            let values = values.u32()?;
            ca.scatter(idx, values)
        },
        DataType::UInt64 => {
            let ca: &mut ChunkedArray<UInt64Type> = mutable_s.as_mut();
            let values = values.u64()?;
            ca.scatter(idx, values)
        },
        DataType::Float32 => {
            let ca: &mut ChunkedArray<Float32Type> = mutable_s.as_mut();
            let values = values.f32()?;
            ca.scatter(idx, values)
        },
        DataType::Float64 => {
            let ca: &mut ChunkedArray<Float64Type> = mutable_s.as_mut();
            let values = values.f64()?;
            ca.scatter(idx, values)
        },
        DataType::Boolean => {
            let ca = s.bool()?;
            let values = values.bool()?;
            ca.scatter(idx, values)
        },
        DataType::String => {
            let ca = s.str()?;
            let values = values.str()?;
            ca.scatter(idx, values)
        },
        _ => {
            return Err(PolarsError::ComputeError(
                format!("not yet implemented for dtype: {logical_dtype}").into(),
            ));
        },
    };

    s.and_then(|s| s.cast(&logical_dtype))
}

#[pymethods]
impl PySeries {
    /// Given a `PySeries` of length 0, find the index of the first value within
    /// self.
    fn index_of(&self, value: PySeries) -> PyResult<Option<usize>> {
        // TODO assert length of value is 1?
        index_of(&self.series, &value.series).map_err(|e| PyErr::from(PyPolarsErr::from(e)))
    }
}

/// Try casting the value to the correct type, then call index_of().
macro_rules! try_index_of {
    ($self:expr, $value:expr) => {{
        let cast_value = $value.map(|v| AnyValue::from(v).strict_cast($self.dtype()));
        if cast_value == Some(None) {
            // We can can't cast the searched-for value to a valid data point
            // within the dtype of the Series we're searching, which means we
            // will never find that value.
            None
        } else {
            let cast_value = cast_value.flatten();
            $self.index_of(cast_value.map(|v| v.extract().unwrap()))
        }
    }};
}

fn index_of(series: &Series, value_series: &Series) -> PolarsResult<Option<usize>> {
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
        return Ok(downcast_as_macro_arg_physical!(series, try_index_of, value));
    }
    if value_dtype.is_unsigned_integer() {
        let value = value_series.cast(&DataType::UInt64)?.u64().unwrap().get(0);
        return Ok(downcast_as_macro_arg_physical!(series, try_index_of, value));
    }
    if value_dtype.is_float() {
        let value = value_series.cast(&DataType::Float64)?.f64().unwrap().get(0);
        return Ok(downcast_as_macro_arg_physical!(series, try_index_of, value));
    }
    // At this point we're done handling integers and floats.
    match value_series.dtype() {
        DataType::List(_) => {
            let value = value_series
                .list()
                .unwrap()
                .get(0)
                .map(|arr| Series::from_arrow("".into(), arr).unwrap());
            Ok(series.list()?.index_of(value.as_ref()))
        },
        #[cfg(feature="dtype-array")]
        DataType::Array(_, _) => {
            let value = value_series
                .array()
                .unwrap()
                .get(0)
                .map(|arr| Series::from_arrow("".into(), arr).unwrap());
            Ok(series.array()?.index_of(value.as_ref()))
        },
        _ => unimplemented!("TODO"),
    }
}
