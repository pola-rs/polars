use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::ffi;
use polars_arrow::ffi::{ArrowArray, ArrowArrayStream, ArrowArrayStreamReader, ArrowSchema};
use polars_ffi::version_0::SeriesExport;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyCapsule, PyTuple, PyType};

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::utils::EnterPolarsExt as _;

/// Import `__arrow_c_array__` across Python boundary
pub(crate) fn call_arrow_c_array<'py>(
    ob: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    if !ob.hasattr("__arrow_c_array__")? {
        return Err(PyValueError::new_err(
            "Expected an object with dunder __arrow_c_array__",
        ));
    }

    let tuple = ob.getattr("__arrow_c_array__")?.call0()?;
    if !tuple.is_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(
            "Expected __arrow_c_array__ to return a tuple.",
        ));
    }

    let schema_capsule = tuple.get_item(0)?.cast_into()?;
    let array_capsule = tuple.get_item(1)?.cast_into()?;
    Ok((schema_capsule, array_capsule))
}

pub(crate) fn import_array_pycapsules(
    schema_capsule: &Bound<PyCapsule>,
    array_capsule: &Bound<PyCapsule>,
) -> PyResult<(polars_arrow::datatypes::Field, Box<dyn Array>)> {
    let field = import_schema_pycapsule(schema_capsule)?;

    // # Safety
    // array_capsule holds a valid C ArrowArray pointer, as defined by the Arrow PyCapsule
    // Interface
    unsafe {
        let array_ptr = std::ptr::replace(
            array_capsule
                .pointer_checked(Some(c"arrow_array"))?
                .as_ptr() as _,
            ArrowArray::empty(),
        );
        let array = ffi::import_array_from_c(array_ptr, field.dtype().clone()).unwrap();

        Ok((field, array))
    }
}

pub(crate) fn import_schema_pycapsule(
    schema_capsule: &Bound<PyCapsule>,
) -> PyResult<polars_arrow::datatypes::Field> {
    // # Safety
    // schema_capsule holds a valid C ArrowSchema pointer, as defined by the Arrow PyCapsule
    // Interface
    unsafe {
        let schema_ptr = schema_capsule
            .pointer_checked(Some(c"arrow_schema"))?
            .cast::<ArrowSchema>()
            .as_ref();
        let field = ffi::import_field_from_c(schema_ptr).unwrap();

        Ok(field)
    }
}

/// Import `__arrow_c_stream__` across Python boundary.
pub(crate) fn call_arrow_c_stream<'py>(ob: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyCapsule>> {
    if !ob.hasattr("__arrow_c_stream__")? {
        return Err(PyValueError::new_err(
            "Expected an object with dunder __arrow_c_stream__",
        ));
    }

    let capsule = ob.getattr("__arrow_c_stream__")?.call0()?.cast_into()?;
    Ok(capsule)
}

/// Takes ownership of the `ArrowArrayStream` behind a stream capsule and wraps it
/// for iteration.
///
/// # Safety
/// `capsule` must hold a valid C `ArrowArrayStream` pointer, as defined by the Arrow
/// PyCapsule Interface.
pub(crate) fn open_stream_capsule(
    capsule: &Bound<PyCapsule>,
) -> PyResult<ArrowArrayStreamReader<Box<ArrowArrayStream>>> {
    unsafe {
        let stream_ptr = Box::new(std::ptr::replace(
            capsule
                .pointer_checked(Some(c"arrow_array_stream"))?
                .as_ptr() as _,
            ArrowArrayStream::empty(),
        ));
        ArrowArrayStreamReader::try_new(stream_ptr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

/// Moves an FFI stream reader across a GIL-release boundary.
///
/// # Safety
/// The Arrow C stream interface is a plain C ABI: producers must be callable without
/// the GIL held, and Python-backed producers (pyarrow and friends) re-acquire it
/// internally. The reader is only ever touched by the single thread that owns it.
struct SendStreamReader(ArrowArrayStreamReader<Box<ArrowArrayStream>>);
unsafe impl Send for SendStreamReader {}

pub(crate) fn import_stream_pycapsule(
    py: Python<'_>,
    capsule: &Bound<PyCapsule>,
) -> PyResult<PySeries> {
    let stream = SendStreamReader(open_stream_capsule(capsule)?);

    // Both draining the stream and converting the chunks can be arbitrarily expensive
    // (decoding on the producer side, arrow -> polars casts on ours), so neither may
    // hold the GIL; otherwise concurrent Python threads serialize on this call.
    let s = py.enter_polars(move || {
        let mut stream = stream;

        let mut produced_arrays: Vec<Box<dyn Array>> = vec![];
        while let Some(array) = unsafe { stream.0.next() } {
            produced_arrays.push(array?);
        }

        // Series::try_from fails for an empty vec of chunks
        if produced_arrays.is_empty() {
            let polars_dt = DataType::from_arrow_field(stream.0.field());
            Ok(Series::new_empty(stream.0.field().name.clone(), &polars_dt))
        } else {
            Series::try_from((stream.0.field(), produced_arrays))
        }
    })?;
    Ok(PySeries::new(s))
}
#[pymethods]
impl PySeries {
    #[classmethod]
    pub fn from_arrow_c_array(
        _cls: &Bound<PyType>,
        py: Python<'_>,
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let (schema_capsule, array_capsule) = call_arrow_c_array(ob)?;
        let (field, array) = import_array_pycapsules(&schema_capsule, &array_capsule)?;
        let s = py.enter_polars(|| Series::try_from((&field, array)))?;
        Ok(PySeries::new(s))
    }

    #[classmethod]
    pub fn from_arrow_c_stream(
        _cls: &Bound<PyType>,
        py: Python<'_>,
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let capsule = call_arrow_c_stream(ob)?;
        import_stream_pycapsule(py, &capsule)
    }

    #[classmethod]
    /// Import a series via polars-ffi
    /// Takes ownership of the [`SeriesExport`] at [`location`]
    /// # Safety
    /// [`location`] should be the address of an allocated and initialized [`SeriesExport`]
    pub unsafe fn _import(_cls: &Bound<PyType>, location: usize) -> PyResult<Self> {
        let location = location as *mut SeriesExport;

        // # Safety
        // `location` should be valid for reading
        let series = unsafe {
            let export = location.read();
            polars_ffi::version_0::import_series(export).map_err(PyPolarsErr::from)?
        };
        Ok(PySeries::from(series))
    }

    #[staticmethod]
    pub fn _import_decimal_from_iceberg_binary_repr(
        bytes_list: &Bound<PyAny>, // list[bytes | None]
        precision: usize,
        scale: usize,
    ) -> PyResult<Self> {
        // From iceberg spec:
        // * Decimal(P, S): Stores unscaled value as two’s-complement
        //   big-endian binary, using the minimum number of bytes for the
        //   value.
        let max_abs_decimal_value = 10_i128.pow(u32::try_from(precision).unwrap()) - 1;

        let out: Vec<i128> = bytes_list
            .try_iter()?
            .map(|bytes| {
                let be_bytes: Option<PyBackedBytes> = bytes?.extract()?;

                let mut le_bytes: [u8; 16] = [0; _];

                if let Some(be_bytes) = be_bytes.as_deref() {
                    if be_bytes.len() > le_bytes.len() {
                        return Err(PyValueError::new_err(format!(
                            "iceberg binary data for decimal exceeded 16 bytes: {}",
                            be_bytes.len()
                        )));
                    }

                    for (i, byte) in be_bytes.iter().rev().enumerate() {
                        le_bytes[i] = *byte;
                    }
                }

                let value = i128::from_le_bytes(le_bytes);

                if value.abs() > max_abs_decimal_value {
                    return Err(PyValueError::new_err(format!(
                        "iceberg decoded value for decimal exceeded precision: \
                        value: {value}, precision: {precision}",
                    )));
                }

                Ok(value)
            })
            .collect::<PyResult<_>>()?;

        Ok(PySeries::from(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![PrimitiveArray::<i128>::from_vec(out).boxed()],
                &DataType::Decimal(precision, scale),
            )
        }))
    }
}
