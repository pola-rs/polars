//! Define the Polars exception hierarchy.

use pyo3::exceptions::{PyException, PyWarning};
use pyo3::{create_exception, panic};

// Errors
create_exception!(polars.exceptions, PolarsBaseError, PyException);
create_exception!(polars.exceptions, ColumnNotFoundError, PolarsBaseError);
create_exception!(polars.exceptions, ComputeError, PolarsBaseError);
create_exception!(polars.exceptions, DuplicateError, PolarsBaseError);
create_exception!(polars.exceptions, InvalidOperationError, PolarsBaseError);
create_exception!(polars.exceptions, NoDataError, PolarsBaseError);
create_exception!(polars.exceptions, OutOfBoundsError, PolarsBaseError);
create_exception!(polars.exceptions, SQLInterfaceError, PolarsBaseError);
create_exception!(polars.exceptions, SQLSyntaxError, PolarsBaseError);
create_exception!(polars.exceptions, SchemaError, PolarsBaseError);
create_exception!(polars.exceptions, SchemaFieldNotFoundError, PolarsBaseError);
create_exception!(polars.exceptions, ShapeError, PolarsBaseError);
create_exception!(polars.exceptions, StringCacheMismatchError, PolarsBaseError);
create_exception!(polars.exceptions, StructFieldNotFoundError, PolarsBaseError);

// Warnings
create_exception!(polars.exceptions, PolarsBaseWarning, PyWarning);
create_exception!(polars.exceptions, PerformanceWarning, PolarsBaseWarning);
create_exception!(
    polars.exceptions,
    CategoricalRemappingWarning,
    PerformanceWarning
);
create_exception!(
    polars.exceptions,
    MapWithoutReturnDtypeWarning,
    PolarsBaseWarning
);

// Panic
create_exception!(polars.exceptions, PanicException, panic::PanicException);
