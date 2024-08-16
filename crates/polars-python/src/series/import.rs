use polars::export::arrow;
use polars::export::arrow::array::Array;
use polars::export::arrow::ffi;
use polars::export::arrow::ffi::{
    ArrowArray, ArrowArrayStream, ArrowArrayStreamReader, ArrowSchema,
};
use polars::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple, PyType};

use super::PySeries;

/// Validate PyCapsule has provided name
fn validate_pycapsule_name(capsule: &Bound<PyCapsule>, expected_name: &str) -> PyResult<()> {
    let capsule_name = capsule.name()?;
    if let Some(capsule_name) = capsule_name {
        let capsule_name = capsule_name.to_str()?;
        if capsule_name != expected_name {
            return Err(PyValueError::new_err(format!(
                "Expected name '{}' in PyCapsule, instead got '{}'",
                expected_name, capsule_name
            )));
        }
    } else {
        return Err(PyValueError::new_err(
            "Expected schema PyCapsule to have name set.",
        ));
    }

    Ok(())
}

/// Import `__arrow_c_array__` across Python boundary
pub(crate) fn call_arrow_c_array<'py>(
    ob: &'py Bound<PyAny>,
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

    let schema_capsule = tuple.get_item(0)?.downcast_into()?;
    let array_capsule = tuple.get_item(1)?.downcast_into()?;
    Ok((schema_capsule, array_capsule))
}

pub(crate) fn import_array_pycapsules(
    schema_capsule: &Bound<PyCapsule>,
    array_capsule: &Bound<PyCapsule>,
) -> PyResult<(arrow::datatypes::Field, Box<dyn Array>)> {
    validate_pycapsule_name(schema_capsule, "arrow_schema")?;
    validate_pycapsule_name(array_capsule, "arrow_array")?;

    // # Safety
    // schema_capsule holds a valid C ArrowSchema pointer, as defined by the Arrow PyCapsule
    // Interface
    // array_capsule holds a valid C ArrowArray pointer, as defined by the Arrow PyCapsule
    // Interface
    let (field, array) = unsafe {
        let schema_ptr = schema_capsule.reference::<ArrowSchema>();
        let array_ptr = std::ptr::replace(array_capsule.pointer() as _, ArrowArray::empty());

        let field = ffi::import_field_from_c(schema_ptr).unwrap();
        let array = ffi::import_array_from_c(array_ptr, field.data_type().clone()).unwrap();
        (field, array)
    };

    Ok((field, array))
}

/// Import `__arrow_c_stream__` across Python boundary.
fn call_arrow_c_stream<'py>(ob: &'py Bound<PyAny>) -> PyResult<Bound<'py, PyCapsule>> {
    if !ob.hasattr("__arrow_c_stream__")? {
        return Err(PyValueError::new_err(
            "Expected an object with dunder __arrow_c_stream__",
        ));
    }

    let capsule = ob.getattr("__arrow_c_stream__")?.call0()?.downcast_into()?;
    Ok(capsule)
}

pub(crate) fn import_stream_pycapsule(capsule: &Bound<PyCapsule>) -> PyResult<PySeries> {
    validate_pycapsule_name(capsule, "arrow_array_stream")?;

    // # Safety
    // capsule holds a valid C ArrowArrayStream pointer, as defined by the Arrow PyCapsule
    // Interface
    let mut stream = unsafe {
        // Takes ownership of the pointed to ArrowArrayStream
        // This acts to move the data out of the capsule pointer, setting the release callback to NULL
        let stream_ptr = Box::new(std::ptr::replace(
            capsule.pointer() as _,
            ArrowArrayStream::empty(),
        ));
        ArrowArrayStreamReader::try_new(stream_ptr)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    };

    let mut produced_arrays: Vec<Box<dyn Array>> = vec![];
    while let Some(array) = unsafe { stream.next() } {
        produced_arrays.push(array.unwrap());
    }

    // Series::try_from fails for an empty vec of chunks
    let s = if produced_arrays.is_empty() {
        let polars_dt = DataType::from_arrow(stream.field().data_type(), false);
        Series::new_empty(&stream.field().name, &polars_dt)
    } else {
        Series::try_from((stream.field(), produced_arrays)).unwrap()
    };
    Ok(PySeries::new(s))
}
#[pymethods]
impl PySeries {
    #[classmethod]
    pub fn from_arrow_c_array(_cls: &Bound<PyType>, ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (schema_capsule, array_capsule) = call_arrow_c_array(ob)?;
        let (field, array) = import_array_pycapsules(&schema_capsule, &array_capsule)?;
        let s = Series::try_from((&field, array)).unwrap();
        Ok(PySeries::new(s))
    }

    #[classmethod]
    pub fn from_arrow_c_stream(_cls: &Bound<PyType>, ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capsule = call_arrow_c_stream(ob)?;
        import_stream_pycapsule(&capsule)
    }
}
