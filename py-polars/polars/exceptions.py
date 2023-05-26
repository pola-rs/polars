try:
    from polars.polars import (
        ArrowError,
        ColumnNotFoundError,
        ComputeError,
        DuplicateError,
        InvalidOperationError,
        NoDataError,
        PolarsPanicError,
        SchemaError,
        SchemaFieldNotFoundError,
        ShapeError,
        StructFieldNotFoundError,
    )
except ImportError:
    # They are only redefined for documentation purposes
    # when there is no binary yet

    class ArrowError(Exception):  # type: ignore[no-redef]
        """Exception raised the underlying Arrow library encounters an error."""

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

    class SchemaError(Exception):  # type: ignore[no-redef]
        """Exception raised when trying to combine data structures with mismatched schemas."""  # noqa: W505

    class SchemaFieldNotFoundError(Exception):  # type: ignore[no-redef]
        """Exception raised when a specified schema field is not found."""

    class ShapeError(Exception):  # type: ignore[no-redef]
        """Exception raised when trying to combine data structures with incompatible shapes."""  # noqa: W505

    class StructFieldNotFoundError(Exception):  # type: ignore[no-redef]
        """Exception raised when a specified schema field is not found."""

    class PolarsPanicError(Exception):  # type: ignore[no-redef]
        """Exception raised when an unexpected state causes a panic in the underlying Rust library."""  # noqa: W505


class InvalidAssert(Exception):
    """Exception raised when an unsupported testing assert is made."""


class RowsError(Exception):
    """Exception raised when the number of returned rows does not match expectation."""


class NoRowsReturnedError(RowsError):
    """Exception raised when no rows are returned, but at least one row is expected."""


class TooManyRowsReturnedError(RowsError):
    """Exception raised when more rows than expected are returned."""


class TimeZoneAwareConstructorWarning(Warning):
    """Warning raised when constructing Series from non-UTC time-zone-aware inputs."""


class ChronoFormatWarning(Warning):
    """
    Warning raised when a chrono format string contains dubious patterns.

    Polars uses Rust's chrono crate to convert between string data and temporal data.
    The patterns used by chrono differ slightly from Python's built-in datetime module.
    Refer to the `chrono strftime documentation
    <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_ for the full
    specification.
    """


__all__ = [
    "ArrowError",
    "ColumnNotFoundError",
    "ComputeError",
    "ChronoFormatWarning",
    "DuplicateError",
    "InvalidOperationError",
    "NoDataError",
    "NoRowsReturnedError",
    "PolarsPanicError",
    "RowsError",
    "SchemaError",
    "SchemaFieldNotFoundError",
    "ShapeError",
    "StructFieldNotFoundError",
    "TooManyRowsReturnedError",
]
