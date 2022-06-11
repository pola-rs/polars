from types import TracebackType
from typing import Optional, Type

try:
    from polars.polars import toggle_string_cache as pytoggle_string_cache

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


class StringCache:
    """
    Context manager that allows data sources to share the same categorical features.
    This will temporarily cache the string categories until the context manager is finished.

    >>> df = pl.DataFrame(
    ...     {
    ...         "a_col": ["red", "green", "blue"],
    ...         "b_col": ["yellow", "orange", "black"],
    ...     }
    ... )
    >>> with pl.StringCache():
    ...     df = df.with_columns(
    ...         [
    ...             pl.col("a_col").cast(pl.Categorical).alias("a_col"),
    ...             pl.col("b_col").cast(pl.Categorical).alias("b_col"),
    ...         ]
    ...     )
    ...
    """

    def __init__(self) -> None:
        pass

    def __enter__(self) -> "StringCache":
        pytoggle_string_cache(True)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pytoggle_string_cache(False)


def toggle_string_cache(toggle: bool) -> None:
    """
    Turn on/off the global string cache. This ensures that casts to Categorical types have the categories when string
    values are equal.
    """
    pytoggle_string_cache(toggle)
