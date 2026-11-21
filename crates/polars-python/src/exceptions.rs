//! Define the Polars exception hierarchy.

use pyo3::exceptions::{PyException, PyTypeError, PyWarning};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use pyo3::{PyTypeInfo, create_exception, ffi};

// Errors
create_exception!(polars.exceptions, PolarsError, PyException);

/// `PolarsTypeError(PolarsError, TypeError)`.
///
/// Base class for Polars errors that are also type errors.
/// Defined like this because `create_exception!` only supports a single base class.
/// It does the same thing the macro would do.
/// The macro logic actually would work for a tuple of bases,
/// because the underlying FFI function accepts that.
#[repr(transparent)]
pub struct PolarsTypeError(PyAny);

unsafe impl PyTypeInfo for PolarsTypeError {
    const NAME: &'static str = "PolarsTypeError";
    const MODULE: Option<&'static str> = Some("polars.exceptions");

    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE_OBJECT: PyOnceLock<Py<PyType>> = PyOnceLock::new();

        TYPE_OBJECT
            .get_or_init(py, || {
                let bases = (py.get_type::<PolarsError>(), py.get_type::<PyTypeError>())
                    .into_pyobject(py)
                    .unwrap();
                // The FFI function does accept a tuple of bases.
                unsafe {
                    Bound::from_owned_ptr_or_err(
                        py,
                        ffi::PyErr_NewExceptionWithDoc(
                            c"polars.exceptions.PolarsTypeError".as_ptr(),
                            std::ptr::null(),
                            bases.as_ptr(),
                            std::ptr::null_mut(),
                        ),
                    )
                }
                .and_then(|t| Ok(t.cast_into::<PyType>()?))
                .expect("Failed to initialize new exception type.")
                .unbind()
            })
            .as_ptr()
            .cast()
    }
}

create_exception!(polars.exceptions, ColumnNotFoundError, PolarsError);
create_exception!(polars.exceptions, ComputeError, PolarsError);
create_exception!(polars.exceptions, DuplicateError, PolarsError);
create_exception!(polars.exceptions, InvalidOperationError, PolarsError);
create_exception!(polars.exceptions, NoDataError, PolarsError);
create_exception!(polars.exceptions, OutOfBoundsError, PolarsError);
create_exception!(polars.exceptions, SQLInterfaceError, PolarsError);
create_exception!(polars.exceptions, SQLSyntaxError, PolarsError);
create_exception!(polars.exceptions, SchemaError, PolarsError);
create_exception!(polars.exceptions, SchemaFieldNotFoundError, PolarsError);
create_exception!(polars.exceptions, ShapeError, PolarsError);
create_exception!(polars.exceptions, StringCacheMismatchError, PolarsError);
create_exception!(polars.exceptions, StructFieldNotFoundError, PolarsError);
create_exception!(polars.exceptions, UnsupportedLiteralError, PolarsTypeError);

// Warnings
create_exception!(polars.exceptions, PolarsWarning, PyWarning);
create_exception!(polars.exceptions, PerformanceWarning, PolarsWarning);
create_exception!(
    polars.exceptions,
    CategoricalRemappingWarning,
    PerformanceWarning
);
create_exception!(
    polars.exceptions,
    MapWithoutReturnDtypeWarning,
    PolarsWarning
);
