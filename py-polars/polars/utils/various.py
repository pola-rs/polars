from __future__ import annotations

import inspect
import os
import re
import sys
from collections.abc import MappingView, Sized
from enum import Enum
from typing import TYPE_CHECKING, Any, Generator, Iterable, Sequence, TypeVar

import polars as pl
from polars import functions as F
from polars.datatypes import (
    Boolean,
    Date,
    Datetime,
    Duration,
    Int64,
    Time,
    Utf8,
    is_polars_dtype,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from collections.abc import Reversible
    from pathlib import Path

    from polars import DataFrame, Series
    from polars.type_aliases import PolarsDataType, SizeUnit

    if sys.version_info >= (3, 10):
        from typing import ParamSpec, TypeGuard
    else:
        from typing_extensions import ParamSpec, TypeGuard

    P = ParamSpec("P")
    T = TypeVar("T")

# note: reversed views don't match as instances of MappingView
if sys.version_info >= (3, 11):
    _views: list[Reversible[Any]] = [{}.keys(), {}.values(), {}.items()]
    _reverse_mapping_views = tuple(type(reversed(view)) for view in _views)


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
    name: str, rng: range, dtype: PolarsDataType | None = None
) -> Series:
    """Fast conversion of the given range to a Series."""
    return F.arange(
        start=rng.start,
        end=rng.stop,
        step=rng.step,
        dtype=dtype,
        eager=True,
    ).alias(name)


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


def ordered_unique(values: Sequence[Any]) -> list[Any]:
    """Return unique list of sequence values, maintaining their order of appearance."""
    seen: set[Any] = set()
    add_ = seen.add
    return [v for v in values if not (v in seen or add_(v))]


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


def _cast_repr_strings_with_schema(
    df: DataFrame, schema: dict[str, PolarsDataType | None]
) -> DataFrame:
    """
    Utility function to cast table repr/string values into frame-native types.

    Parameters
    ----------
    df
        Dataframe containing string-repr column data.
    schema
        DataFrame schema containing the desired end-state types.

    Notes
    -----
    Table repr strings are less strict (or different) than equivalent CSV data, so need
    special handling; as this function is only used for reprs, parsing is flexible.

    """
    tp: PolarsDataType | None
    if not df.is_empty():
        for tp in df.schema.values():
            if tp != Utf8:
                raise TypeError(
                    f"DataFrame should contain only Utf8 string repr data; found {tp}"
                )

    # duration string scaling
    ns_sec = 1_000_000_000
    duration_scaling = {
        "ns": 1,
        "us": 1_000,
        "µs": 1_000,
        "ms": 1_000_000,
        "s": ns_sec,
        "m": ns_sec * 60,
        "h": ns_sec * 60 * 60,
        "d": ns_sec * 3_600 * 24,
        "w": ns_sec * 3_600 * 24 * 7,
    }

    # identify duration units and convert to nanoseconds
    def str_duration_(td: str | None) -> int | None:
        return (
            None
            if td is None
            else sum(
                int(value) * duration_scaling[unit.strip()]
                for value, unit in re.findall(r"(\d+)(\D+)", td)
            )
        )

    cast_cols = {}
    for c, tp in schema.items():
        if tp is not None:
            if tp.base_type() == Datetime:
                tp_base = Datetime(tp.time_unit)  # type: ignore[union-attr]
                d = F.col(c).str.replace(r"[A-Z ]+$", "")
                cast_cols[c] = (
                    F.when(d.str.lengths() == 19)
                    .then(d + ".000000000")
                    .otherwise(d + "000000000")
                    .str.slice(0, 29)
                    .str.strptime(tp_base, "%Y-%m-%d %H:%M:%S.%9f")
                )
                if getattr(tp, "time_zone", None) is not None:
                    cast_cols[c] = cast_cols[c].dt.replace_time_zone(tp.time_zone)  # type: ignore[union-attr]
            elif tp == Date:
                cast_cols[c] = F.col(c).str.strptime(tp, "%Y-%m-%d")  # type: ignore[arg-type]
            elif tp == Time:
                cast_cols[c] = (
                    F.when(F.col(c).str.lengths() == 8)
                    .then(F.col(c) + ".000000000")
                    .otherwise(F.col(c) + "000000000")
                    .str.slice(0, 18)
                    .str.strptime(tp, "%H:%M:%S.%9f")  # type: ignore[arg-type]
                )
            elif tp == Duration:
                cast_cols[c] = (
                    F.col(c)
                    .apply(str_duration_, return_dtype=Int64)
                    .cast(Duration("ns"))
                    .cast(tp)
                )
            elif tp == Boolean:
                cast_cols[c] = F.col(c).map_dict(
                    {"true": True, "false": False}, return_dtype=Boolean
                )
            elif tp != df.schema[c]:
                cast_cols[c] = F.col(c).cast(tp)

    return df.with_columns(**cast_cols) if cast_cols else df


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


class _NoDefault(Enum):
    # "borrowed" from
    # https://github.com/pandas-dev/pandas/blob/e7859983a814b1823cf26e3b491ae2fa3be47c53/pandas/_libs/lib.pyx#L2736-L2748
    no_default = "NO_DEFAULT"

    def __repr__(self) -> str:
        return "<no_default>"


# 'NoDefault' is a sentinel indicating that no default value has been set; note that
# this should typically be used only when one of the valid parameter values is also
# None, as otherwise we cannot determine if the caller has explicitly set that value.
no_default = _NoDefault.no_default
NoDefault = Literal[_NoDefault.no_default]


def find_stacklevel() -> int:
    """
    Find the first place in the stack that is not inside polars (tests notwithstanding).

    Taken from:
    https://github.com/pandas-dev/pandas/blob/ab89c53f48df67709a533b6a95ce3d911871a0a8/pandas/util/_exceptions.py#L30-L51
    """
    pkg_dir = os.path.dirname(pl.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and not fname.startswith(test_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def _get_stack_locals(
    of_type: type | tuple[type, ...] | None = None, n_objects: int | None = None
) -> dict[str, Any]:
    """
    Retrieve f_locals from all stack frames (starting from the current frame).

    Parameters
    ----------
    of_type
        Only return objects of this type.
    n_objects
        If specified, return only the most recent ``n`` matching objects.

    """
    objects = {}
    stack_frame = getattr(inspect.currentframe(), "f_back", None)
    while stack_frame:
        local_items = list(stack_frame.f_locals.items())
        for nm, obj in reversed(local_items):
            if nm not in objects and (not of_type or isinstance(obj, of_type)):
                objects[nm] = obj
                if n_objects is not None and len(objects) >= n_objects:
                    return objects
        stack_frame = stack_frame.f_back
    return objects
