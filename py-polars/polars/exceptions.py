try:
    from polars.polars import (
        ColumnNotFoundError,
        ComputeError,
        DuplicateError,
        InvalidOperationError,
        NoDataError,
        OutOfBoundsError,
        PolarsPanicError,
        SchemaError,
        SchemaFieldNotFoundError,
        ShapeError,
        StringCacheMismatchError,
        StructFieldNotFoundError,
    )
except ImportError:
    # They are only redefined for documentation purposes
    # when there is no binary yet

    class ColumnNotFoundError(Exception):  # type: ignore[no-redef]
        """Exception raised when a specified column is not found."""

    class ComputeError(Exception):  # type: ignore[no-redef]
        """Exception raised when polars could not finish the computation."""

    class DuplicateError(Exception):  # type: ignore[no-redef]
        """Exception raised when a column name is duplicated."""

    class InvalidOperationError(Exception):  # type: ignore[no-redef]
        """Exception raised when an operation is not allowed on a certain data type."""

    class NoDataError(Exception):  # type: ignore[no-redef]
        """Exception raised when an operation can not be performed on an empty data structure."""  # noqa: W505

    class OutOfBoundsError(Exception):  # type: ignore[no-redef]
        """Exception raised when the given index is out of bounds."""

    class PolarsPanicError(Exception):  # type: ignore[no-redef]
        """Exception raised when an unexpected state causes a panic in the underlying Rust library."""  # noqa: W505

    class SchemaError(Exception):  # type: ignore[no-redef]
        """Exception raised when trying to combine data structures with mismatched schemas."""  # noqa: W505

    class SchemaFieldNotFoundError(Exception):  # type: ignore[no-redef]
        """Exception raised when a specified schema field is not found."""

    class ShapeError(Exception):  # type: ignore[no-redef]
        """Exception raised when trying to combine data structures with incompatible shapes."""  # noqa: W505

    class StringCacheMismatchError(Exception):  # type: ignore[no-redef]
        """Exception raised when string caches come from different sources."""

    class StructFieldNotFoundError(Exception):  # type: ignore[no-redef]
        """Exception raised when a specified schema field is not found."""


class ChronoFormatWarning(Warning):
    """
    Warning raised when a chrono format string contains dubious patterns.

    Polars uses Rust's chrono crate to convert between string data and temporal data.
    The patterns used by chrono differ slightly from Python's built-in datetime module.
    Refer to the `chrono strftime documentation
    <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_ for the full
    specification.
    """


class InvalidAssert(Exception):
    """Exception raised when an unsupported testing assert is made."""


class RowsError(Exception):
    """Exception raised when the number of returned rows does not match expectation."""


class NoRowsReturnedError(RowsError):
    """Exception raised when no rows are returned, but at least one row is expected."""


class ParameterCollisionError(RuntimeError):
    """Exception raised when the same parameter occurs multiple times."""


class PolarsInefficientMapWarning(Warning):
    """Warning raised when a potentially slow `apply` operation is performed."""


class TooManyRowsReturnedError(RowsError):
    """Exception raised when more rows than expected are returned."""


class TimeZoneAwareConstructorWarning(Warning):
    """Warning raised when constructing Series from non-UTC time-zone-aware inputs."""


class UnsuitableSQLError(ValueError):
    """Exception raised when unsuitable SQL is given to a database method."""


class ArrowError(Exception):
    """deprecated will be removed."""


__all__ = [
    "ArrowError",
    "ColumnNotFoundError",
    "ComputeError",
    "ChronoFormatWarning",
    "DuplicateError",
    "InvalidOperationError",
    "NoDataError",
    "NoRowsReturnedError",
    "OutOfBoundsError",
    "PolarsInefficientMapWarning",
    "PolarsPanicError",
    "RowsError",
    "SchemaError",
    "SchemaFieldNotFoundError",
    "ShapeError",
    "StringCacheMismatchError",
    "StructFieldNotFoundError",
    "TooManyRowsReturnedError",
]
