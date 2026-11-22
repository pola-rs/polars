use polars::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::prelude::*;

use super::PySeries;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PySeries {
    fn scatter(&self, py: Python<'_>, idx: PySeries, values: PySeries) -> PyResult<()> {
        py.enter_polars(|| {
            let mut lock = self.series.write();
            // Prepare while the series is still in place, so an error or a panic leaves it intact.
            let (idx, values) =
                prepare_scatter(&lock, &idx.series.into_inner(), &values.series.into_inner())?;
            // We take the value because we want a ref count of 1 so that we can
            // have mutable access cheaply via _get_inner_mut().
            let s = std::mem::take(&mut *lock);
            match scatter(s, &idx, &values) {
                Ok(out) => {
                    *lock = out;
                    Ok(())
                },
                Err((s, e)) => {
                    *lock = s; // Restore original series.
                    Err(e)
                },
            }
        })
    }
}

/// Converts `idx` to bounded indices and casts `values` to the physical dtype of `s`.
fn prepare_scatter(s: &Series, idx: &Series, values: &Series) -> PolarsResult<(IdxCa, Series)> {
    let logical_dtype = s.dtype();
    let is_categorical = logical_dtype.is_categorical() || logical_dtype.is_enum();
    let valid_values = if is_categorical {
        matches!(
            values.dtype(),
            DataType::Categorical(_, _) | DataType::Enum(_, _) | DataType::String | DataType::Null
        )
    } else if logical_dtype.is_decimal() {
        values.dtype().is_numeric() || values.dtype().is_null()
    } else {
        true
    };
    polars_ensure!(
        valid_values,
        InvalidOperation: "invalid values dtype '{}' for scattering into dtype '{}'", values.dtype(), logical_dtype
    );

    // Cast to the logical dtype, so values keep their meaning (a Decimal its value, a Categorical
    // its string) and values that don't fit raise instead of becoming null.
    let converted_values;
    let values = if values.dtype() != logical_dtype {
        converted_values = values.strict_cast(logical_dtype)?;
        // A Struct cast to a non-struct dtype stays a Struct.
        polars_ensure!(
            converted_values.dtype() == logical_dtype,
            InvalidOperation: "invalid values dtype '{}' for scattering into dtype '{}'", values.dtype(), logical_dtype
        );
        &converted_values
    } else {
        values
    };

    let null_on_oob = false;
    let idx = polars_ops::prelude::convert_and_bound_index(idx, s.len(), null_on_oob)?;
    let idx = idx.rechunk().into_owned();
    polars_ensure!(!idx.has_nulls(), ComputeError: "index values should not be null");

    let mut values = values.to_physical_repr().into_owned();

    // Broadcast values input.
    if values.len() == 1 && idx.len() > 1 {
        values = values.new_from_index(0, idx.len());
    }
    Ok((idx, values))
}

fn scatter(s: Series, idx: &IdxCa, values: &Series) -> Result<Series, (Series, PolarsError)> {
    let logical_dtype = s.dtype().clone();
    let idx = idx.downcast_as_array().values().as_slice();

    let mut phys = s.to_physical_repr().into_owned();
    drop(s); // Reduce refcount to make use of in-place mutation if possible.
    let ret = scatter_impl(&mut phys, &logical_dtype, idx, values);
    match ret {
        Ok(s) => Ok(unsafe { s.from_physical_unchecked(&logical_dtype).unwrap() }),
        Err(e) => Err((
            unsafe { phys.from_physical_unchecked(&logical_dtype).unwrap() },
            e,
        )),
    }
}

fn scatter_impl(
    s: &mut Series,
    logical_dtype: &DataType,
    idx: &[IdxSize],
    values: &Series,
) -> PolarsResult<Series> {
    let mutable_s = s._get_inner_mut();

    match mutable_s.dtype() {
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &mut ChunkedArray<$T> = mutable_s.as_mut();
                let values: &ChunkedArray<$T> = values.as_ref().as_ref();
                ca.scatter(idx, values.iter())
            })
        },
        DataType::Boolean => {
            let ca: &mut ChunkedArray<BooleanType> = mutable_s.as_mut();
            let values = values.bool()?;
            ca.scatter(idx, values.iter())
        },
        DataType::Binary => {
            let ca: &mut ChunkedArray<BinaryType> = mutable_s.as_mut();
            let values = values.binary()?;
            ca.scatter(idx, values.iter())
        },
        DataType::String => {
            let ca: &mut ChunkedArray<StringType> = mutable_s.as_mut();
            let values = values.str()?;
            ca.scatter(idx, values.iter())
        },
        _ => Err(PolarsError::ComputeError(
            format!("not yet implemented for dtype: {logical_dtype}").into(),
        )),
    }
}
