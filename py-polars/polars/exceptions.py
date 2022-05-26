try:
    from polars.polars import (
        ArrowError,
        ComputeError,
        DuplicateError,
        NoDataError,
        NotFoundError,
        PanicException,
        SchemaError,
        ShapeError,
    )
except ImportError:  # pragma: no cover
    # They are only redefined for documentation purposes
    # when there is no binary yet

    class ArrowError(Exception):  # type: ignore
        """Exception raised the underlying Arrow library encounters an error"""

        pass

    class ComputeError(Exception):  # type: ignore
        """Exception raised when we couldn't finish the computation"""

        pass

    class NoDataError(Exception):  # type: ignore
        """Exception raised when an operation can not be performed on an empty data structure"""

        pass

    class NotFoundError(Exception):  # type: ignore
        """Exception raised when a specified column is not found"""

        pass

    class SchemaError(Exception):  # type: ignore
        """Exception raised when trying to combine data structures with mismatched schemas"""

        pass

    class ShapeError(Exception):  # type: ignore
        """Exception raised when trying to combine data structures with incompatible shapes"""

        pass

    class DuplicateError(Exception):  # type: ignore
        """Exception raised when a column name is duplicated"""

        pass

    class PanicException(Exception):  # type: ignore
        """Exception raised when an unexpected state causes a panic in the underlying Rust library"""

        pass


__all__ = [
    "ArrowError",
    "ComputeError",
    "NoDataError",
    "NotFoundError",
    "SchemaError",
    "ShapeError",
    "DuplicateError",
    "PanicException",
]
