try:
    from polars.polars import (
        ArrowError,
        ComputeError,
        NoDataError,
        NotFoundError,
        SchemaError,
        ShapeError,
    )
except ImportError:  # pragma: no cover
    # They are only redefined for documentation purposes
    # when there is no binary yet

    class ArrowError:  # type: ignore
        pass

    class ComputeError:  # type: ignore
        pass

    class NoDataError:  # type: ignore
        pass

    class NotFoundError:  # type: ignore
        pass

    class SchemaError:  # type: ignore
        pass

    class ShapeError:  # type: ignore
        pass


__all__ = [
    "ArrowError",
    "ComputeError",
    "NoDataError",
    "NotFoundError",
    "SchemaError",
    "ShapeError",
]
