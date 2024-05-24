try:
    from polars.polars import (
        CategoricalRemappingWarning,
        ColumnNotFoundError,
        ComputeError,
        DuplicateError,
        InvalidOperationError,
        MapWithoutReturnDtypeWarning,
        NoDataError,
        OutOfBoundsError,
        PolarsError,
        PolarsPanicError,
        PolarsWarning,
        SchemaError,
        SchemaFieldNotFoundError,
        ShapeError,
        StringCacheMismatchError,
        StructFieldNotFoundError,
    )
except ImportError:
    # redefined for documentation purposes when there is no binary

    class PolarsError(Exception):  # type: ignore[no-redef]
        """Base class for all Polars errors."""

    class ColumnNotFoundError(PolarsError):  # type: ignore[no-redef, misc]
        """
        Exception raised when a specified column is not found.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select("b")
        polars.exceptions.ColumnNotFoundError: b
        """

    class ComputeError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when Polars could not perform an underlying computation."""

    class DuplicateError(PolarsError):  # type: ignore[no-redef, misc]
        """
        Exception raised when a column name is duplicated.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 1]})
        >>> pl.concat([df, df], how="horizontal")
        polars.exceptions.DuplicateError: unable to hstack, column with name "a" already exists
        """  # noqa: W505

    class InvalidOperationError(PolarsError):  # type: ignore[no-redef, misc]
        """
        Exception raised when an operation is not allowed (or possible) against a given object or data structure.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.is_in(["x", "y"])
        polars.exceptions.InvalidOperationError: `is_in` cannot check for String values in Int64 data
        """  # noqa: W505

    class NoDataError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when an operation cannot be performed on an empty data structure."""  # noqa: W505

    class OutOfBoundsError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when the given index is out of bounds."""

    class PolarsPanicError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when an unexpected state causes a panic in the underlying Rust library."""  # noqa: W505

    class SchemaError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when an unexpected schema mismatch causes an error."""

    class SchemaFieldNotFoundError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when a specified schema field is not found."""

    class ShapeError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when trying to perform operations on data structures with incompatible shapes."""  # noqa: W505

    class StringCacheMismatchError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when string caches come from different sources."""

    class StructFieldNotFoundError(PolarsError):  # type: ignore[no-redef, misc]
        """Exception raised when a specified Struct field is not found."""

    class PolarsWarning(Exception):  # type: ignore[no-redef]
        """Base class for all Polars warnings."""

    class CategoricalRemappingWarning(PolarsWarning):  # type: ignore[no-redef, misc]
        """Warning raised when a categorical needs to be remapped to be compatible with another categorical."""  # noqa: W505

    class MapWithoutReturnDtypeWarning(PolarsWarning):  # type: ignore[no-redef, misc]
        """Warning raised when `map_elements` is performed without specifying the return dtype."""  # noqa: W505


class InvalidAssert(PolarsError):  # type: ignore[misc]
    """Exception raised when an unsupported testing assert is made."""


class RowsError(PolarsError):  # type: ignore[misc]
    """Exception raised when the number of returned rows does not match expectation."""


class NoRowsReturnedError(RowsError):
    """Exception raised when no rows are returned, but at least one row is expected."""


class TooManyRowsReturnedError(RowsError):
    """Exception raised when more rows than expected are returned."""


class ModuleUpgradeRequired(ModuleNotFoundError):
    """Exception raised when a module is installed but needs to be upgraded."""


class ParameterCollisionError(PolarsError):  # type: ignore[misc]
    """Exception raised when the same parameter occurs multiple times."""


class UnsuitableSQLError(PolarsError):  # type: ignore[misc]
    """Exception raised when unsuitable SQL is given to a database method."""


class ChronoFormatWarning(PolarsWarning):  # type: ignore[misc]
    """
    Warning issued when a chrono format string contains dubious patterns.

    Polars uses Rust's chrono crate to convert between string data and temporal data.
    The patterns used by chrono differ slightly from Python's built-in datetime module.
    Refer to the `chrono strftime documentation
    <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_ for the full
    specification.
    """


class PolarsInefficientMapWarning(PolarsWarning):  # type: ignore[misc]
    """Warning issued when a potentially slow `map_*` operation is performed."""


class TimeZoneAwareConstructorWarning(PolarsWarning):  # type: ignore[misc]
    """Warning issued when constructing Series from non-UTC time-zone-aware inputs."""


class UnstableWarning(PolarsWarning):  # type: ignore[misc]
    """Warning issued when unstable functionality is used."""


class ArrowError(Exception):
    """Deprecated: will be removed."""


class CustomUFuncWarning(PolarsWarning):  # type: ignore[misc]
    """Warning issued when a custom ufunc is handled differently than numpy ufunc would."""  # noqa: W505


__all__ = [
    "ArrowError",
    "ColumnNotFoundError",
    "ComputeError",
    "ChronoFormatWarning",
    "DuplicateError",
    "InvalidOperationError",
    "MapWithoutReturnDtypeWarning",
    "ModuleUpgradeRequired",
    "NoDataError",
    "NoRowsReturnedError",
    "OutOfBoundsError",
    "PolarsInefficientMapWarning",
    "CategoricalRemappingWarning",
    "PolarsError",
    "PolarsPanicError",
    "PolarsWarning",
    "RowsError",
    "SchemaError",
    "SchemaFieldNotFoundError",
    "ShapeError",
    "StringCacheMismatchError",
    "StructFieldNotFoundError",
    "TooManyRowsReturnedError",
]
