try:
    from polars.polars import (
        ArrowError,
        ColumnNotFoundError,
        ComputeError,
        DuplicateError,
        InvalidOperationError,
        NoDataError,
        PanicException,
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

    class PanicException(Exception):  # type: ignore[no-redef]
        """Exception raised when an unexpected state causes a panic in the underlying Rust library."""  # noqa: W505


class InvalidAssert(Exception):
    """Exception raised when an unsupported testing assert is made."""


class RowsException(Exception):
    """Exception raised when the number of returned rows does not match expectation."""


class NoRowsReturned(RowsException):
    """Exception raised when no rows are returned, but at least one row is expected."""


class TooManyRowsReturned(RowsException):
    """Exception raised when more rows than expected are returned."""


__all__ = [
    "ArrowError",
    "ColumnNotFoundError",
    "ComputeError",
    "DuplicateError",
    "InvalidOperationError",
    "NoDataError",
    "NoRowsReturned",
    "PanicException",
    "RowsException",
    "SchemaError",
    "SchemaFieldNotFoundError",
    "ShapeError",
    "StructFieldNotFoundError",
    "TooManyRowsReturned",
]
