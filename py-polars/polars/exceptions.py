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
        pass

    class ComputeError(Exception):  # type: ignore
        pass

    class NoDataError(Exception):  # type: ignore
        pass

    class NotFoundError(Exception):  # type: ignore
        pass

    class SchemaError(Exception):  # type: ignore
        pass

    class ShapeError(Exception):  # type: ignore
        pass

    class DuplicateError(Exception):  # type: ignore
        pass

    class PanicException(Exception):  # type: ignore
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
