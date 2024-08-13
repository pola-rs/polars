//! Define the Polars exception hierarchy.

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyWarning};

// Errors
create_exception!(polars.exceptions, PolarsError, PyException);
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
