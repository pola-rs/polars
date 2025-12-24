use arrow::array::Array;
use polars::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::prelude::*;

use super::PySeries;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PySeries {
    fn scatter(&self, py: Python<'_>, idx: PySeries, values: PySeries) -> PyResult<()> {
        py.enter_polars(|| {
            // we take the value because we want a ref count of 1 so that we can
            // have mutable access cheaply via _get_inner_mut().
            let mut lock = self.series.write();
            let s = std::mem::take(&mut *lock);
            let result = scatter(s, &idx.series.into_inner(), &values.series.into_inner());
            match result {
                Ok(out) => {
                    *lock = out;
                    Ok(())
                },
                Err((s, e)) => {
                    // Restore original series:
                    *lock = s;
                    Err(e)
                },
            }
        })
    }
}

fn scatter(mut s: Series, idx: &Series, values: &Series) -> Result<Series, (Series, PolarsError)> {
    let logical_dtype = s.dtype().clone();

    let values = if logical_dtype.is_categorical() || logical_dtype.is_enum() {
        if matches!(
            values.dtype(),
            DataType::Categorical(_, _) | DataType::Enum(_, _) | DataType::String | DataType::Null
        ) {
            match values.strict_cast(&logical_dtype) {
                Ok(values) => values,
                Err(err) => return Err((s, err)),
            }
        } else {
            return Err((
                s,
                polars_err!(InvalidOperation: "invalid values dtype '{}' for scattering into dtype '{}'", values.dtype(), logical_dtype),
            ));
        }
    } else {
        values.clone()
    };

    let idx = match polars_ops::prelude::convert_to_unsigned_index(idx, s.len()) {
        Ok(idx) => idx,
        Err(err) => return Err((s, err)),
    };
    let idx = idx.rechunk();
    let idx = idx.downcast_as_array();

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
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &mut ChunkedArray<$T> = mutable_s.as_mut();
                let values: &ChunkedArray<$T> = values.as_ref().as_ref();
                ca.scatter(idx, values)
            })
        }
        DataType::Boolean => {
            let ca: &mut ChunkedArray<BooleanType> = mutable_s.as_mut();
            let values = values.bool()?;
            ca.scatter(idx, values)
        },
        DataType::Binary => {
            let ca: &mut ChunkedArray<BinaryType> = mutable_s.as_mut();
            let values = values.binary()?;
            ca.scatter(idx, values)
        },
        DataType::String => {
            let ca: &mut ChunkedArray<StringType> = mutable_s.as_mut();
            let values = values.str()?;
            ca.scatter(idx, values)
        },
        _ => {
            return Err(PolarsError::ComputeError(
                format!("not yet implemented for dtype: {logical_dtype}").into(),
            ));
        },
    };

    s.and_then(|s| unsafe { s.from_physical_unchecked(&logical_dtype) })
}
