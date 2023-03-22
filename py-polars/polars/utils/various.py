from __future__ import annotations

import os
import re
import sys
from collections.abc import MappingView, Sized
from typing import TYPE_CHECKING, Any, Generator, Iterable, Sequence, TypeVar

from polars import functions as F
from polars.datatypes import Int64, is_polars_dtype

if TYPE_CHECKING:
    from polars.series import Series


# note: reversed views don't match as instances of MappingView
if sys.version_info >= (3, 11):
    _views: list[Reversible[Any]] = [{}.keys(), {}.values(), {}.items()]
    _reverse_mapping_views = tuple(type(reversed(view)) for view in _views)

if TYPE_CHECKING:
    from collections.abc import Reversible
    from pathlib import Path

    from polars.type_aliases import PolarsDataType, SizeUnit

    if sys.version_info >= (3, 10):
        from typing import ParamSpec, TypeGuard
    else:
        from typing_extensions import ParamSpec, TypeGuard

    P = ParamSpec("P")
    T = TypeVar("T")


def _process_null_values(
    null_values: None | str | Sequence[str] | dict[str, str] = None,
) -> None | str | Sequence[str] | list[tuple[str, str]]:
    if isinstance(null_values, dict):
        return list(null_values.items())
    else:
        return null_values


def _is_generator(val: object) -> bool:
    return (
        (isinstance(val, (Generator, Iterable)) and not isinstance(val, Sized))
        or isinstance(val, MappingView)
        or (sys.version_info >= (3, 11) and isinstance(val, _reverse_mapping_views))
    )


def _is_iterable_of(val: Iterable[object], eltype: type | tuple[type, ...]) -> bool:
    """Check whether the given iterable is of the given type(s)."""
    return all(isinstance(x, eltype) for x in val)


def is_bool_sequence(val: object) -> TypeGuard[Sequence[bool]]:
    """Check whether the given sequence is a sequence of booleans."""
    return isinstance(val, Sequence) and _is_iterable_of(val, bool)


def is_dtype_sequence(val: object) -> TypeGuard[Sequence[PolarsDataType]]:
    """Check whether the given object is a sequence of polars DataTypes."""
    return isinstance(val, Sequence) and all(is_polars_dtype(x) for x in val)


def is_int_sequence(val: object) -> TypeGuard[Sequence[int]]:
    """Check whether the given sequence is a sequence of integers."""
    return isinstance(val, Sequence) and _is_iterable_of(val, int)


def is_str_sequence(
    val: object, *, allow_str: bool = False
) -> TypeGuard[Sequence[str]]:
    """
    Check that `val` is a sequence of strings.

    Note that a single string is a sequence of strings by definition, use
    `allow_str=False` to return False on a single string.
    """
    if allow_str is False and isinstance(val, str):
        return False
    return isinstance(val, Sequence) and _is_iterable_of(val, str)


def range_to_series(
    name: str, rng: range, dtype: PolarsDataType | None = Int64
) -> Series:
    """Fast conversion of the given range to a Series."""
    return F.arange(
        low=rng.start,
        high=rng.stop,
        step=rng.step,
        eager=True,
        dtype=dtype,
    ).rename(name, in_place=True)


def range_to_slice(rng: range) -> slice:
    """Return the given range as an equivalent slice."""
    return slice(rng.start, rng.stop, rng.step)


def handle_projection_columns(
    columns: Sequence[str] | Sequence[int] | str | None,
) -> tuple[list[int] | None, Sequence[str] | None]:
    """Disambiguates between columns specified as integers vs. strings."""
    projection: list[int] | None = None
    new_columns: Sequence[str] | None = None
    if columns is not None:
        if isinstance(columns, str):
            new_columns = [columns]
        elif is_int_sequence(columns):
            projection = list(columns)
        elif not is_str_sequence(columns):
            raise ValueError(
                "'columns' arg should contain a list of all integers or all strings"
                " values."
            )
        else:
            new_columns = columns
        if columns and len(set(columns)) != len(columns):
            raise ValueError(
                f"'columns' arg should only have unique values. Got '{columns}'."
            )
        if projection and len(set(projection)) != len(projection):
            raise ValueError(
                f"'columns' arg should only have unique values. Got '{projection}'."
            )
    return projection, new_columns


def _prepare_row_count_args(
    row_count_name: str | None = None,
    row_count_offset: int = 0,
) -> tuple[str, int] | None:
    if row_count_name is not None:
        return (row_count_name, row_count_offset)
    else:
        return None


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def arrlen(obj: Any) -> int | None:
    """Return length of (non-string) sequence object; returns None for non-sequences."""
    try:
        return None if isinstance(obj, str) else len(obj)
    except TypeError:
        return None


def normalise_filepath(path: str | Path, check_not_directory: bool = True) -> str:
    """Create a string path, expanding the home directory if present."""
    path = os.path.expanduser(path)
    if check_not_directory and os.path.exists(path) and os.path.isdir(path):
        raise IsADirectoryError(f"Expected a file path; {path!r} is a directory")
    return path


def parse_version(version: Sequence[str | int]) -> tuple[int, ...]:
    """Simple version parser; split into a tuple of ints for comparison."""
    if isinstance(version, str):
        version = version.split(".")
    return tuple(int(re.sub(r"\D", "", str(v))) for v in version)


def scale_bytes(sz: int, unit: SizeUnit) -> int | float:
    """Scale size in bytes to other size units (eg: "kb", "mb", "gb", "tb")."""
    if unit in {"b", "bytes"}:
        return sz
    elif unit in {"kb", "kilobytes"}:
        return sz / 1024
    elif unit in {"mb", "megabytes"}:
        return sz / 1024**2
    elif unit in {"gb", "gigabytes"}:
        return sz / 1024**3
    elif unit in {"tb", "terabytes"}:
        return sz / 1024**4
    else:
        raise ValueError(
            f"unit must be one of {{'b', 'kb', 'mb', 'gb', 'tb'}}, got {unit!r}"
        )


# when building docs (with Sphinx) we need access to the functions
# associated with the namespaces from the class, as we don't have
# an instance; @sphinx_accessor is a @property that allows this.
NS = TypeVar("NS")


class sphinx_accessor(property):
    def __get__(  # type: ignore[override]
        self,
        instance: Any,
        cls: type[NS],
    ) -> NS:
        try:
            return self.fget(  # type: ignore[misc]
                instance if isinstance(instance, cls) else cls
            )
        except AttributeError:
            return None  # type: ignore[return-value]
