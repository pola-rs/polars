from __future__ import annotations

import contextlib
import math
import os
from datetime import date, datetime, time, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Generator,
    Iterable,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
    Union,
    overload,
)

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import (
    Array,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    Object,
    String,
    Time,
    UInt32,
    UInt64,
    Unknown,
    dtype_to_ctype,
    is_polars_dtype,
    maybe_cast,
    numpy_char_code_to_dtype,
    py_type_to_dtype,
    supported_numpy_char_code,
)
from polars.dependencies import (
    _HVPLOT_AVAILABLE,
    _PYARROW_AVAILABLE,
    _check_for_numpy,
    _check_for_pandas,
    _check_for_pyarrow,
    dataframe_api_compat,
    hvplot,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import ModuleUpgradeRequired, ShapeError
from polars.series.array import ArrayNameSpace
from polars.series.binary import BinaryNameSpace
from polars.series.categorical import CatNameSpace
from polars.series.datetime import DateTimeNameSpace
from polars.series.list import ListNameSpace
from polars.series.string import StringNameSpace
from polars.series.struct import StructNameSpace
from polars.series.utils import expr_dispatch, get_ffi_func
from polars.slice import PolarsSlice
from polars.utils._construction import (
    arrow_to_pyseries,
    iterable_to_pyseries,
    numpy_to_idxs,
    numpy_to_pyseries,
    pandas_to_pyseries,
    sequence_to_pyseries,
    series_to_pyseries,
)
from polars.utils._wrap import wrap_df
from polars.utils.convert import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
)
from polars.utils.deprecation import (
    deprecate_function,
    deprecate_nonkeyword_arguments,
    deprecate_renamed_function,
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)
from polars.utils.meta import get_index_type
from polars.utils.various import (
    _is_generator,
    _warn_null_comparison,
    no_default,
    parse_percentiles,
    parse_version,
    range_to_series,
    range_to_slice,
    scale_bytes,
    sphinx_accessor,
)

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PyDataFrame, PySeries


if TYPE_CHECKING:
    import sys

    from polars import DataFrame, DataType, Expr
    from polars.series._numpy import SeriesView
    from polars.type_aliases import (
        BufferInfo,
        ClosedInterval,
        ComparisonOperator,
        FillNullStrategy,
        InterpolationMethod,
        IntoExpr,
        IntoExprColumn,
        NullBehavior,
        NumericLiteral,
        OneOrMoreDataTypes,
        PolarsDataType,
        PythonLiteral,
        RankMethod,
        RollingInterpolationMethod,
        SearchSortedSide,
        SizeUnit,
        TemporalLiteral,
    )
    from polars.utils.various import (
        NoDefault,
    )

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
elif os.getenv("BUILDING_SPHINX_DOCS"):
    property = sphinx_accessor

ArrayLike = Union[
    Sequence[Any],
    "Series",
    "pa.Array",
    "pa.ChunkedArray",
    "np.ndarray[Any, Any]",
    "pd.Series[Any]",
    "pd.DatetimeIndex",
]


@expr_dispatch
class Series:
    """
    A one-dimensional column of data. Each column of a `DataFrame` is a `Series`.

    Parameters
    ----------
    name : str, default None
        The name of the `Series`. Will be used as a column name when used in a
        `DataFrame`. When not specified, defaults to the empty string.
    values : ArrayLike, default None
        One-dimensional data in various forms. Supported are: `Sequence`, `Series`,
        :class:`pyarrow.Array`, and :class:`numpy.ndarray`.
    dtype : DataType, default None
        The polars dtype of the `Series` data. If not specified, the dtype is
        inferred.
    strict
        Whether to throw an error on numeric overflow.
    nan_to_null
        Whether to set floating-point `NaN` values to `null`, if `values` is a
        :class:`numpy.ndarray`. This is a no-op for all other input data.
    dtype_if_empty : DataType, default None
        If no dtype is specified and `values` is `None`, an empty list, or a list with
        only `None` values, use this Polars dtype for the `Series`. Defaults to
        :class:`Float32`.

    Examples
    --------
    Constructing a `Series` by specifying name and values positionally:

    >>> s = pl.Series("a", [1, 2, 3])
    >>> s
    shape: (3,)
    Series: 'a' [i64]
    [
            1
            2
            3
    ]

    Notice that the dtype is automatically inferred as a polars :class:`Int64`:

    >>> s.dtype
    Int64

    Constructing a `Series` with a specific dtype:

    >>> s2 = pl.Series("a", [1, 2, 3], dtype=pl.Float32)
    >>> s2
    shape: (3,)
    Series: 'a' [f32]
    [
        1.0
        2.0
        3.0
    ]

    It is possible to construct a `Series` with values as the first positional argument.
    This syntax considered an anti-pattern, but it can be useful in certain
    scenarios. You must specify any other arguments through keywords.

    >>> s3 = pl.Series([1, 2, 3])
    >>> s3
    shape: (3,)
    Series: '' [i64]
    [
            1
            2
            3
    ]

    """

    _s: PySeries = None
    _accessors: ClassVar[set[str]] = {
        "arr",
        "cat",
        "dt",
        "list",
        "str",
        "bin",
        "struct",
        "plot",
    }

    def __init__(
        self,
        name: str | ArrayLike | None = None,
        values: ArrayLike | None = None,
        dtype: PolarsDataType | None = None,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
        dtype_if_empty: PolarsDataType = Null,
    ):
        # If 'Unknown' treat as None to attempt inference
        if dtype == Unknown:
            dtype = None

        # Raise early error on invalid dtype
        if (
            dtype is not None
            and not is_polars_dtype(dtype)
            and py_type_to_dtype(dtype, raise_unmatched=False) is None
        ):
            raise ValueError(
                f"given dtype: {dtype!r} is not a valid Polars data type and cannot be converted into one"
            )

        # Handle case where values are passed as the first argument
        original_name: str | None = None
        if name is None:
            name = ""
        elif isinstance(name, str):
            original_name = name
        else:
            if values is None:
                values = name
                name = ""
            else:
                raise TypeError("Series name must be a string")

        if values is None:
            self._s = sequence_to_pyseries(
                name, [], dtype=dtype, dtype_if_empty=dtype_if_empty
            )

        elif isinstance(values, range):
            self._s = range_to_series(name, values, dtype=dtype)._s

        elif isinstance(values, Series):
            name = values.name if original_name is None else name
            self._s = series_to_pyseries(name, values, dtype=dtype, strict=strict)

        elif isinstance(values, Sequence):
            self._s = sequence_to_pyseries(
                name,
                values,
                dtype=dtype,
                strict=strict,
                dtype_if_empty=dtype_if_empty,
                nan_to_null=nan_to_null,
            )

        elif _check_for_numpy(values) and isinstance(values, np.ndarray):
            self._s = numpy_to_pyseries(
                name, values, strict=strict, nan_to_null=nan_to_null
            )
            if values.dtype.type in [np.datetime64, np.timedelta64]:
                # cast to appropriate dtype, handling NaT values
                dtype = _resolve_temporal_dtype(dtype, values.dtype)
                if dtype is not None:
                    self._s = (
                        self.cast(dtype)
                        .scatter(np.argwhere(np.isnat(values)).flatten(), None)
                        ._s
                    )
                    return

            if dtype is not None:
                self._s = self.cast(dtype, strict=True)._s

        elif _check_for_pyarrow(values) and isinstance(
            values, (pa.Array, pa.ChunkedArray)
        ):
            self._s = arrow_to_pyseries(name, values)

        elif _check_for_pandas(values) and isinstance(
            values, (pd.Series, pd.DatetimeIndex)
        ):
            self._s = pandas_to_pyseries(name, values)

        elif _is_generator(values):
            self._s = iterable_to_pyseries(
                name,
                values,
                dtype=dtype,
                dtype_if_empty=dtype_if_empty,
                strict=strict,
            )

        elif isinstance(values, pl.DataFrame):
            to_struct = values.width > 1
            name = (
                values.columns[0] if (original_name is None and not to_struct) else name
            )
            s = values.to_struct(name) if to_struct else values.to_series().rename(name)
            if dtype is not None and dtype != s.dtype:
                s = s.cast(dtype)
            self._s = s._s

        else:
            raise TypeError(
                f"Series constructor called with unsupported type {type(values).__name__!r}"
                " for the `values` parameter"
            )

    @classmethod
    def _from_pyseries(cls, pyseries: PySeries) -> Self:
        series = cls.__new__(cls)
        series._s = pyseries
        return series

    @classmethod
    def _from_arrow(cls, name: str, values: pa.Array, *, rechunk: bool = True) -> Self:
        """Construct a Series from an Arrow Array."""
        return cls._from_pyseries(arrow_to_pyseries(name, values, rechunk=rechunk))

    @classmethod
    def _from_pandas(
        cls,
        name: str,
        values: pd.Series[Any] | pd.DatetimeIndex,
        *,
        nan_to_null: bool = True,
    ) -> Self:
        """Construct a `Series` from a pandas Series or DatetimeIndex."""
        return cls._from_pyseries(
            pandas_to_pyseries(name, values, nan_to_null=nan_to_null)
        )

    def _get_buffer_info(self) -> BufferInfo:
        """
        Return pointer, offset, and length information about the underlying buffer.

        Returns
        -------
        tuple of ints
            A tuple of the form `(pointer, offset, length)`.

        Raises
        ------
        ComputeError
            If the `Series` contains multiple chunks.
        """
        return self._s._get_buffer_info()

    @overload
    def _get_buffer(self, index: Literal[0]) -> Self:
        ...

    @overload
    def _get_buffer(self, index: Literal[1, 2]) -> Self | None:
        ...

    def _get_buffer(self, index: Literal[0, 1, 2]) -> Self | None:
        """
        Return the underlying data, validity, or offsets buffer as a `Series`.

        The data buffer always exists.
        The validity buffer may not exist if the column contains no null values.
        The offsets buffer only exists for `Series` of data type :class:`String`
        and :class:`List`.

        Parameters
        ----------
        index
            An index indicating the buffer to return:

            - `0` -> data buffer
            - `1` -> validity buffer
            - `2` -> offsets buffer

        Returns
        -------
        Series or None
            A `Series` if the specified buffer exists, `None` otherwise.

        Raises
        ------
        ComputeError
            If the `Series` contains multiple chunks.
        """
        buffer = self._s._get_buffer(index)
        if buffer is None:
            return None
        return self._from_pyseries(buffer)

    @classmethod
    def _from_buffer(
        self, dtype: PolarsDataType, buffer_info: BufferInfo, owner: Any
    ) -> Self:
        """
        Construct a `Series` from information about its underlying buffer.

        Parameters
        ----------
        dtype
            The data type of the buffer.
        buffer_info
            Tuple containing buffer information in the form `(pointer, offset, length)`.
        owner
            The object owning the buffer.

        Returns
        -------
        Series
        """
        return self._from_pyseries(PySeries._from_buffer(dtype, buffer_info, owner))

    @classmethod
    def _from_buffers(
        self,
        dtype: PolarsDataType,
        data: Series | Sequence[Series],
        validity: Series | None = None,
    ) -> Self:
        """
        Construct a `Series` from information about its underlying buffers.

        Parameters
        ----------
        dtype
            The data type of the resulting `Series`.
        data
            Buffers describing the data. For most data types, this is a single
            `Series` of the physical data type of `dtype`. Some data types
            require multiple buffers:

            - `String`: A data buffer of type :class:`UInt8` and an offsets buffer
                        of type :class:`Int64`.
        validity
            Validity buffer. If specified, must be a :class:`Boolean` `Series`.

        Returns
        -------
        Series
        """
        if isinstance(data, Series):
            data = [data._s]
        else:
            data = [s._s for s in data]
        if validity is not None:
            validity = validity._s
        return self._from_pyseries(PySeries._from_buffers(dtype, data, validity))

    @property
    def dtype(self) -> DataType:
        """
        The data type of this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.dtype
        Int64

        """
        return self._s.dtype()

    @property
    def flags(self) -> dict[str, bool]:
        """
        The flags that are set on this `Series`.

        Returns
        -------
        dict
            A mapping from flag names to values.

        """
        out = {
            "SORTED_ASC": self._s.is_sorted_ascending_flag(),
            "SORTED_DESC": self._s.is_sorted_descending_flag(),
        }
        if self.dtype == List:
            out["FAST_EXPLODE"] = self._s.can_fast_explode_flag()
        return out

    @property
    def inner_dtype(self) -> DataType | None:
        """
        Get the inner dtype of a :class:`List` `Series`.

        Not available for non-:class:`List` `Series`.

        .. deprecated:: 0.19.14
            Use `Series.dtype.inner` instead.

        Returns
        -------
        DataType

        """
        issue_deprecation_warning(
            "`Series.inner_dtype` is deprecated. Use `Series.dtype.inner` instead.",
            version="0.19.14",
        )
        try:
            return self.dtype.inner  # type: ignore[attr-defined]
        except AttributeError:
            return None

    @property
    def name(self) -> str:
        """The name of this `Series`."""
        return self._s.name()

    @property
    def shape(self) -> tuple[int]:
        """The shape of this `Series` as a tuple, i.e. `(length,)`."""
        return (self._s.len(),)

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "the truth value of a Series is ambiguous"
            "\n\nHint: use '&' or '|' to chain Series boolean results together, not and/or."
            " To check if a Series contains any values, use `is_empty()`."
        )

    def __getstate__(self) -> bytes:
        return self._s.__getstate__()

    def __setstate__(self, state: bytes) -> None:
        self._s = Series()._s  # Initialize with a dummy
        self._s.__setstate__(state)

    def __str__(self) -> str:
        s_repr: str = self._s.as_str()
        return s_repr.replace("Series", f"{self.__class__.__name__}", 1)

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.len()

    def __and__(self, other: Series) -> Self:
        if not isinstance(other, Series):
            other = Series([other])
        return self._from_pyseries(self._s.bitand(other._s))

    def __rand__(self, other: Series) -> Series:
        if not isinstance(other, Series):
            other = Series([other])
        return other & self

    def __or__(self, other: Series) -> Self:
        if not isinstance(other, Series):
            other = Series([other])
        return self._from_pyseries(self._s.bitor(other._s))

    def __ror__(self, other: Series) -> Series:
        if not isinstance(other, Series):
            other = Series([other])
        return other | self

    def __xor__(self, other: Series) -> Self:
        if not isinstance(other, Series):
            other = Series([other])
        return self._from_pyseries(self._s.bitxor(other._s))

    def __rxor__(self, other: Series) -> Series:
        if not isinstance(other, Series):
            other = Series([other])
        return other ^ self

    def _comp(self, other: Any, op: ComparisonOperator) -> Series:
        # special edge-case; boolean broadcast series (eq/neq) is its own result
        if self.dtype == Boolean and isinstance(other, bool) and op in ("eq", "neq"):
            if (other is True and op == "eq") or (other is False and op == "neq"):
                return self.clone()
            elif (other is False and op == "eq") or (other is True and op == "neq"):
                return ~self

        elif isinstance(other, float) and self.dtype.is_integer():
            # require upcast when comparing int series to float value
            self = self.cast(Float64)
            f = get_ffi_func(op + "_<>", Float64, self._s)
            assert f is not None
            return self._from_pyseries(f(other))

        elif isinstance(other, datetime):
            if self.dtype == Date:
                # require upcast when comparing date series to datetime
                self = self.cast(Datetime("us"))
                time_unit = "us"
            elif self.dtype == Datetime:
                # Use local time zone info
                time_zone = self.dtype.time_zone  # type: ignore[attr-defined]
                if str(other.tzinfo) != str(time_zone):
                    raise TypeError(
                        f"Datetime time zone {other.tzinfo!r} does not match Series timezone {time_zone!r}"
                    )
                time_unit = self.dtype.time_unit  # type: ignore[attr-defined]
            else:
                raise ValueError(
                    f"cannot compare datetime.datetime to Series of type {self.dtype}"
                )
            ts = _datetime_to_pl_timestamp(other, time_unit)  # type: ignore[arg-type]
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return self._from_pyseries(f(ts))

        elif isinstance(other, time) and self.dtype == Time:
            d = _time_to_pl_time(other)
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return self._from_pyseries(f(d))

        elif isinstance(other, timedelta) and self.dtype == Duration:
            time_unit = self.dtype.time_unit  # type: ignore[attr-defined]
            td = _timedelta_to_pl_timedelta(other, time_unit)  # type: ignore[arg-type]
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return self._from_pyseries(f(td))

        elif self.dtype in [Categorical, Enum] and not isinstance(other, Series):
            other = Series([other])

        elif isinstance(other, date) and self.dtype == Date:
            d = _date_to_pl_date(other)
            f = get_ffi_func(op + "_<>", Int32, self._s)
            assert f is not None
            return self._from_pyseries(f(d))

        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, dtype_if_empty=self.dtype)
        if isinstance(other, Series):
            return self._from_pyseries(getattr(self._s, op)(other._s))

        if other is not None:
            other = maybe_cast(other, self.dtype)
        f = get_ffi_func(op + "_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented

        return self._from_pyseries(f(other))

    @overload  # type: ignore[override]
    def __eq__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __eq__(self, other: Any) -> Series:
        ...

    def __eq__(self, other: Any) -> Series | Expr:
        _warn_null_comparison(other)
        if isinstance(other, pl.Expr):
            return F.lit(self).__eq__(other)
        return self._comp(other, "eq")

    @overload  # type: ignore[override]
    def __ne__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __ne__(self, other: Any) -> Series:
        ...

    def __ne__(self, other: Any) -> Series | Expr:
        _warn_null_comparison(other)
        if isinstance(other, pl.Expr):
            return F.lit(self).__ne__(other)
        return self._comp(other, "neq")

    @overload
    def __gt__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __gt__(self, other: Any) -> Series:
        ...

    def __gt__(self, other: Any) -> Series | Expr:
        _warn_null_comparison(other)
        if isinstance(other, pl.Expr):
            return F.lit(self).__gt__(other)
        return self._comp(other, "gt")

    @overload
    def __lt__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __lt__(self, other: Any) -> Series:
        ...

    def __lt__(self, other: Any) -> Series | Expr:
        _warn_null_comparison(other)
        if isinstance(other, pl.Expr):
            return F.lit(self).__lt__(other)
        return self._comp(other, "lt")

    @overload
    def __ge__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __ge__(self, other: Any) -> Series:
        ...

    def __ge__(self, other: Any) -> Series | Expr:
        _warn_null_comparison(other)
        if isinstance(other, pl.Expr):
            return F.lit(self).__ge__(other)
        return self._comp(other, "gt_eq")

    @overload
    def __le__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __le__(self, other: Any) -> Series:
        ...

    def __le__(self, other: Any) -> Series | Expr:
        _warn_null_comparison(other)
        if isinstance(other, pl.Expr):
            return F.lit(self).__le__(other)
        return self._comp(other, "lt_eq")

    def le(self, other: Any) -> Self | Expr:
        """The method equivalent of `series <= other`."""
        return self.__le__(other)

    def lt(self, other: Any) -> Self | Expr:
        """The method equivalent of `series < other`."""
        return self.__lt__(other)

    def eq(self, other: Any) -> Self | Expr:
        """The method equivalent of `series == other`."""
        return self.__eq__(other)

    @overload
    def eq_missing(self, other: Any) -> Self:
        ...

    @overload
    def eq_missing(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    def eq_missing(self, other: Any) -> Self | Expr:
        """
        The method equivalent of `series == other`, but where `null == null`.

        This differs from the standard :func:`eq` where `null` values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        See Also
        --------
        ne_missing
        eq

        Examples
        --------
        >>> s1 = pl.Series("a", [333, 200, None])
        >>> s2 = pl.Series("a", [100, 200, None])
        >>> s1.eq(s2)
        shape: (3,)
        Series: 'a' [bool]
        [
            false
            true
            null
        ]
        >>> s1.eq_missing(s2)
        shape: (3,)
        Series: 'a' [bool]
        [
            false
            true
            true
        ]

        """

    def ne(self, other: Any) -> Self | Expr:
        """The method equivalent of `series != other`."""
        return self.__ne__(other)

    @overload
    def ne_missing(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def ne_missing(self, other: Any) -> Self:
        ...

    def ne_missing(self, other: Any) -> Self | Expr:
        """
        The method equivalent of `series != other`, but where `null == null`.

        This differs from the standard :func:`ne` where `null` values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        See Also
        --------
        eq_missing
        ne

        Examples
        --------
        >>> s1 = pl.Series("a", [333, 200, None])
        >>> s2 = pl.Series("a", [100, 200, None])
        >>> s1.ne(s2)
        shape: (3,)
        Series: 'a' [bool]
        [
            true
            false
            null
        ]
        >>> s1.ne_missing(s2)
        shape: (3,)
        Series: 'a' [bool]
        [
            true
            false
            false
        ]

        """

    def ge(self, other: Any) -> Self | Expr:
        """The method equivalent of `series >= other`."""
        return self.__ge__(other)

    def gt(self, other: Any) -> Self | Expr:
        """The method equivalent of `series > other`."""
        return self.__gt__(other)

    def _arithmetic(self, other: Any, op_s: str, op_ffi: str) -> Self:
        if isinstance(other, pl.Expr):
            # expand pl.lit, pl.datetime, pl.duration Exprs to compatible Series
            other = self.to_frame().select_seq(other).to_series()
        if isinstance(other, Series):
            return self._from_pyseries(getattr(self._s, op_s)(other._s))
        if _check_for_numpy(other) and isinstance(other, np.ndarray):
            return self._from_pyseries(getattr(self._s, op_s)(Series(other)._s))
        if (
            isinstance(other, (float, date, datetime, timedelta, str))
            and not self.dtype.is_float()
        ):
            _s = sequence_to_pyseries(self.name, [other])
            if "rhs" in op_ffi:
                return self._from_pyseries(getattr(_s, op_s)(self._s))
            else:
                return self._from_pyseries(getattr(self._s, op_s)(_s))
        else:
            other = maybe_cast(other, self.dtype)
            f = get_ffi_func(op_ffi, self.dtype, self._s)
        if f is None:
            raise TypeError(
                f"cannot do arithmetic with Series of dtype: {self.dtype!r} and argument"
                f" of type: {type(other).__name__!r}"
            )
        return self._from_pyseries(f(other))

    @overload
    def __add__(self, other: DataFrame) -> DataFrame:  # type: ignore[overload-overlap]
        ...

    @overload
    def __add__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __add__(self, other: Any) -> Self:
        ...

    def __add__(self, other: Any) -> Self | DataFrame | Expr:
        if isinstance(other, str):
            other = Series("", [other])
        elif isinstance(other, pl.DataFrame):
            return other + self
        elif isinstance(other, pl.Expr):
            return F.lit(self) + other
        return self._arithmetic(other, "add", "add_<>")

    @overload
    def __sub__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __sub__(self, other: Any) -> Self:
        ...

    def __sub__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) - other
        return self._arithmetic(other, "sub", "sub_<>")

    @overload
    def __truediv__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __truediv__(self, other: Any) -> Series:
        ...

    def __truediv__(self, other: Any) -> Series | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) / other
        if self.dtype.is_temporal():
            raise TypeError("first cast to integer before dividing datelike dtypes")

        # this branch is exactly the floordiv function without rounding the floats
        if self.dtype.is_float() or self.dtype == Decimal:
            return self._arithmetic(other, "div", "div_<>")

        return self.cast(Float64) / other

    @overload
    def __floordiv__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __floordiv__(self, other: Any) -> Series:
        ...

    def __floordiv__(self, other: Any) -> Series | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) // other
        if self.dtype.is_temporal():
            raise TypeError("first cast to integer before dividing datelike dtypes")

        if not isinstance(other, pl.Expr):
            other = F.lit(other)
        return self.to_frame().select_seq(F.col(self.name) // other).to_series()

    def __invert__(self) -> Series:
        return self.not_()

    @overload
    def __mul__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __mul__(self, other: DataFrame) -> DataFrame:  # type: ignore[overload-overlap]
        ...

    @overload
    def __mul__(self, other: Any) -> Series:
        ...

    def __mul__(self, other: Any) -> Series | DataFrame | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) * other
        if self.dtype.is_temporal():
            raise TypeError("first cast to integer before multiplying datelike dtypes")
        elif isinstance(other, pl.DataFrame):
            return other * self
        else:
            return self._arithmetic(other, "mul", "mul_<>")

    @overload
    def __mod__(self, other: Expr) -> Expr:  # type: ignore[overload-overlap]
        ...

    @overload
    def __mod__(self, other: Any) -> Series:
        ...

    def __mod__(self, other: Any) -> Series | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__mod__(other)
        if self.dtype.is_temporal():
            raise TypeError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>")

    def __rmod__(self, other: Any) -> Series:
        if self.dtype.is_temporal():
            raise TypeError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>_rhs")

    def __radd__(self, other: Any) -> Series:
        if isinstance(other, str):
            return (other + self.to_frame()).to_series()
        return self._arithmetic(other, "add", "add_<>_rhs")

    def __rsub__(self, other: Any) -> Series:
        return self._arithmetic(other, "sub", "sub_<>_rhs")

    def __rtruediv__(self, other: Any) -> Series:
        if self.dtype.is_temporal():
            raise TypeError("first cast to integer before dividing datelike dtypes")
        if self.dtype.is_float():
            self.__rfloordiv__(other)

        if isinstance(other, int):
            other = float(other)
        return self.cast(Float64).__rfloordiv__(other)

    def __rfloordiv__(self, other: Any) -> Series:
        if self.dtype.is_temporal():
            raise TypeError("first cast to integer before dividing datelike dtypes")
        return self._arithmetic(other, "div", "div_<>_rhs")

    def __rmul__(self, other: Any) -> Series:
        if self.dtype.is_temporal():
            raise TypeError("first cast to integer before multiplying datelike dtypes")
        return self._arithmetic(other, "mul", "mul_<>")

    def __pow__(self, exponent: int | float | None | Series) -> Series:
        return self.pow(exponent)

    def __rpow__(self, other: Any) -> Series:
        if self.dtype.is_temporal():
            raise TypeError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        return self.to_frame().select_seq(other ** F.col(self.name)).to_series()

    def __matmul__(self, other: Any) -> float | Series | None:
        if isinstance(other, Sequence) or (
            _check_for_numpy(other) and isinstance(other, np.ndarray)
        ):
            other = Series(other)
        # elif isinstance(other, pl.DataFrame):
        #     return other.__rmatmul__(self)  # type: ignore[return-value]
        return self.dot(other)

    def __rmatmul__(self, other: Any) -> float | Series | None:
        if isinstance(other, Sequence) or (
            _check_for_numpy(other) and isinstance(other, np.ndarray)
        ):
            other = Series(other)
        return other.dot(self)

    def __neg__(self) -> Series:
        return 0 - self

    def __pos__(self) -> Series:
        return 0 + self

    def __abs__(self) -> Series:
        return self.abs()

    def __copy__(self) -> Self:
        return self.clone()

    def __deepcopy__(self, memo: None = None) -> Self:
        return self.clone()

    def __contains__(self, item: Any) -> bool:
        if item is None:
            return self.null_count() > 0
        return self.implode().list.contains(item).item()

    def __iter__(self) -> Generator[Any, None, None]:
        if self.dtype in (List, Array):
            # TODO: either make a change and return py-native list data here, or find
            #  a faster way to return nested/List series; sequential 'get_index' calls
            #  make this path a lot slower (~10x) than it needs to be.
            get_index = self._s.get_index
            for idx in range(self.len()):
                yield get_index(idx)
        else:
            buffer_size = 25_000
            for offset in range(0, self.len(), buffer_size):
                yield from self.slice(offset, buffer_size).to_list()

    def _pos_idxs(self, size: int) -> Series:
        # Unsigned or signed `Series` (ordered from fastest to slowest).
        #   - :class:`UInt32` (polars) or :class:`UInt64` (polars_u64_idx) `Series`
        #     indexes.
        #   - Other unsigned `Series` indexes are converted to :class:`UInt32`
        #     (polars) or :class:`UInt64` (polars_u64_idx).
        #   - Signed `Series` indexes are converted :class:`UInt32` (polars) or
        #     :class:`UInt64` (polars_u64_idx) after negative indexes are converted
        #     to absolute indexes.

        # pl.UInt32 (polars) or pl.UInt64 (polars_u64_idx).
        idx_type = get_index_type()

        if self.dtype == idx_type:
            return self

        if not self.dtype.is_integer():
            raise NotImplementedError("unsupported idxs datatype")

        if self.len() == 0:
            return Series(self.name, [], dtype=idx_type)

        if idx_type == UInt32:
            if self.dtype in {Int64, UInt64}:
                if self.max() >= 2**32:  # type: ignore[operator]
                    raise ValueError("index positions should be smaller than 2^32")
            if self.dtype == Int64:
                if self.min() < -(2**32):  # type: ignore[operator]
                    raise ValueError("index positions should be bigger than -2^32 + 1")

        if self.dtype.is_signed_integer():
            if self.min() < 0:  # type: ignore[operator]
                if idx_type == UInt32:
                    idxs = self.cast(Int32) if self.dtype in {Int8, Int16} else self
                else:
                    idxs = (
                        self.cast(Int64) if self.dtype in {Int8, Int16, Int32} else self
                    )

                # Update negative indexes to absolute indexes.
                return (
                    idxs.to_frame()
                    .select(
                        F.when(F.col(idxs.name) < 0)
                        .then(size + F.col(idxs.name))
                        .otherwise(F.col(idxs.name))
                        .cast(idx_type)
                    )
                    .to_series(0)
                )

        return self.cast(idx_type)

    def _take_with_series(self, s: Series) -> Series:
        return self._from_pyseries(self._s.take_with_series(s._s))

    @overload
    def __getitem__(self, item: int) -> Any:
        ...

    @overload
    def __getitem__(
        self,
        item: Series | range | slice | np.ndarray[Any, Any] | list[int],
    ) -> Series:
        ...

    def __getitem__(
        self,
        item: (int | Series | range | slice | np.ndarray[Any, Any] | list[int]),
    ) -> Any:
        if isinstance(item, Series) and item.dtype.is_integer():
            return self._take_with_series(item._pos_idxs(self.len()))

        elif _check_for_numpy(item) and isinstance(item, np.ndarray):
            return self._take_with_series(numpy_to_idxs(item, self.len()))

        # Integer
        elif isinstance(item, int):
            return self._s.get_index_signed(item)

        # Slice
        elif isinstance(item, slice):
            return PolarsSlice(self).apply(item)

        # Range
        elif isinstance(item, range):
            return self[range_to_slice(item)]

        # Sequence of integers (also triggers on empty sequence)
        elif isinstance(item, Sequence) and (
            not item or (isinstance(item[0], int) and not isinstance(item[0], bool))  # type: ignore[redundant-expr]
        ):
            idx_series = Series("", item, dtype=Int64)._pos_idxs(self.len())
            if idx_series.has_validity():
                raise ValueError(
                    "cannot use `__getitem__` with index values containing nulls"
                )
            return self._take_with_series(idx_series)

        raise TypeError(
            f"cannot use `__getitem__` on Series of dtype {self.dtype!r}"
            f" with argument {item!r} of type {type(item).__name__!r}"
        )

    def __setitem__(
        self,
        key: int | Series | np.ndarray[Any, Any] | Sequence[object] | tuple[object],
        value: Any,
    ) -> None:
        # do the single idx as first branch as those are likely in a tight loop
        if isinstance(key, int) and not isinstance(key, bool):
            self.scatter(key, value)
            return None
        elif isinstance(value, Sequence) and not isinstance(value, str):
            if self.dtype.is_numeric() or self.dtype.is_temporal():
                self.scatter(key, value)  # type: ignore[arg-type]
                return None
            raise TypeError(
                f"cannot set Series of dtype: {self.dtype!r} with list/tuple as value;"
                " use a scalar value"
            )
        if isinstance(key, Series):
            if key.dtype == Boolean:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.scatter(key.cast(UInt32), value)._s
            elif key.dtype == UInt32:
                self._s = self.scatter(key, value)._s

        # TODO: implement for these types without casting to series
        elif _check_for_numpy(key) and isinstance(key, np.ndarray):
            if key.dtype == np.bool_:
                # boolean numpy mask
                self._s = self.scatter(np.argwhere(key)[:, 0], value)._s
            else:
                s = self._from_pyseries(
                    PySeries.new_u32("", np.array(key, np.uint32), _strict=True)
                )
                self.__setitem__(s, value)
        elif isinstance(key, (list, tuple)):
            s = self._from_pyseries(sequence_to_pyseries("", key, dtype=UInt32))
            self.__setitem__(s, value)
        else:
            raise TypeError(f'cannot use "{key!r}" for indexing')

    def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
        """
        Numpy __array__ interface protocol.

        Ensures that `np.asarray(pl.Series(..))` works as expected, see
        https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method.
        """
        if not dtype and self.dtype == String and not self.null_count():
            dtype = np.dtype("U")
        if dtype:
            return self.to_numpy().__array__(dtype)
        else:
            return self.to_numpy().__array__()

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Series:
        """Numpy universal functions."""
        if self._s.n_chunks() > 1:
            self._s.rechunk(in_place=True)

        s = self._s

        if method == "__call__":
            if not ufunc.nout == 1:
                raise NotImplementedError(
                    "only ufuncs that return one 1D array are supported"
                )

            args: list[int | float | np.ndarray[Any, Any]] = []

            validity_mask = self.is_not_null()
            for arg in inputs:
                if isinstance(arg, (int, float, np.ndarray)):
                    args.append(arg)
                elif isinstance(arg, Series):
                    validity_mask &= arg.is_not_null()
                    args.append(arg._view(ignore_nulls=True))
                else:
                    raise TypeError(
                        f"unsupported type {type(arg).__name__!r} for {arg!r}"
                    )

            # Get minimum dtype needed to be able to cast all input arguments to the
            # same dtype.
            dtype_char_minimum = np.result_type(*args).char

            # Get all possible output dtypes for ufunc.
            # Input dtypes and output dtypes seem to always match for ufunc.types,
            # so pick all the different output dtypes.
            dtypes_ufunc = [
                input_output_type[-1]
                for input_output_type in ufunc.types
                if supported_numpy_char_code(input_output_type[-1])
            ]

            # Get the first ufunc dtype from all possible ufunc dtypes for which
            # the input arguments can be safely cast to that ufunc dtype.
            for dtype_ufunc in dtypes_ufunc:
                if np.can_cast(dtype_char_minimum, dtype_ufunc):
                    dtype_char_minimum = dtype_ufunc
                    break

            # Override minimum dtype if requested.
            dtype_char = (
                np.dtype(kwargs.pop("dtype")).char
                if "dtype" in kwargs
                else dtype_char_minimum
            )

            f = get_ffi_func("apply_ufunc_<>", numpy_char_code_to_dtype(dtype_char), s)

            if f is None:
                raise NotImplementedError(
                    "could not find "
                    f"`apply_ufunc_{numpy_char_code_to_dtype(dtype_char)}`"
                )

            series = f(lambda out: ufunc(*args, out=out, dtype=dtype_char, **kwargs))
            return (
                self._from_pyseries(series)
                .to_frame()
                .select(F.when(validity_mask).then(F.col(self.name)))
                .to_series(0)
            )
        else:
            raise NotImplementedError(
                "only `__call__` is implemented for numpy ufuncs on a Series, got "
                f"`{method!r}`"
            )

    def __column_consortium_standard__(self, *, api_version: str | None = None) -> Any:
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of polars.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_column(
                self, api_version=api_version
            )
        )

    def _repr_html_(self) -> str:
        """Format output data in HTML for display in Jupyter Notebooks."""
        return self.to_frame()._repr_html_(from_series=True)

    @deprecate_renamed_parameter("row", "index", version="0.19.3")
    def item(self, index: int | None = None) -> Any:
        """
        Convert a length-1 `Series`, or the element at `index`, to a scalar.

        If no `index` is provided, this is equivalent to `s[0]`, with a check that the
        shape is `(1,)`. With an `index`, this is equivalent to `s[index]`.

        Examples
        --------
        >>> s1 = pl.Series("a", [1])
        >>> s1.item()
        1
        >>> s2 = pl.Series("a", [9, 8, 7])
        >>> s2.cum_sum().item(-1)
        24

        """
        if index is None:
            if len(self) != 1:
                raise ValueError(
                    "can only call '.item()' if the Series is of length 1,"
                    f" or an explicit index is provided (Series is of length {len(self)})"
                )
            return self._s.get_index(0)

        return self._s.get_index_signed(index)

    def estimated_size(self, unit: SizeUnit = "b") -> int | float:
        """
        Estimate the total (heap) allocated size of the `Series`.

        The estimated size is given in the specified unit (bytes by default).

        This estimation is the sum of the size of its buffers, validity, including
        nested arrays. Multiple arrays may share buffers and bitmaps. Therefore, the
        size of 2 arrays is not the sum of the sizes computed from this function. In
        particular, `StructArray
        <https://arrow.apache.org/docs/python/generated/pyarrow.StructArray.html>`_'s
        size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer
        is unchanged. However, this function will yield a smaller number. This is
        because this function returns the visible size of the buffer, not its total
        capacity.

        Foreign Function Interface (FFI) buffers are included in this estimation.

        Parameters
        ----------
        unit : {'b', 'kb', 'mb', 'gb', 'tb'}
            The unit to return the estimated size in.

        Examples
        --------
        >>> s = pl.Series("values", list(range(1_000_000)), dtype=pl.UInt32)
        >>> s.estimated_size()
        4000000
        >>> s.estimated_size("mb")
        3.814697265625

        """
        sz = self._s.estimated_size()
        return scale_bytes(sz, unit)

    def sqrt(self) -> Series:
        """
        Compute the square root of each element.

        Syntactic sugar for:

        >>> pl.Series([1, 2]) ** 0.5
        shape: (2,)
        Series: '' [f64]
        [
            1.0
            1.414214
        ]

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.sqrt()
        shape: (3,)
        Series: '' [f64]
        [
            1.0
            1.414214
            1.732051
        ]

        """

    def cbrt(self) -> Series:
        """
        Compute the cube root of each element.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.cbrt()
        shape: (3,)
        Series: '' [f64]
        [
            1.0
            1.259921
            1.44225
        ]

        """

    @overload
    def any(self, *, ignore_nulls: Literal[True] = ...) -> bool:
        ...

    @overload
    def any(self, *, ignore_nulls: bool) -> bool | None:
        ...

    @deprecate_renamed_parameter("drop_nulls", "ignore_nulls", version="0.19.0")
    def any(self, *, ignore_nulls: bool = True) -> bool | None:
        """
        Return whether this `Series` has any `True` elements.

        Only works on :class:`Boolean` `Series`.

        Parameters
        ----------
        ignore_nulls
            Whether to ignore `null` values.

            If `ignore_nulls=False`, `Kleene logic`_ is used to deal with `null` values:
            if the column contains any `null` values and no `True` values,
            the output is `null`.

            .. _Kleene logic: https://en.wikipedia.org/wiki/Three-valued_logic

        Returns
        -------
        bool or None

        Examples
        --------
        >>> pl.Series([True, False]).any()
        True
        >>> pl.Series([False, False]).any()
        False
        >>> pl.Series([None, False]).any()
        False

        Enable Kleene logic by setting `ignore_nulls=False`:

        >>> pl.Series([None, False]).any(ignore_nulls=False)  # Returns None

        """
        return self._s.any(ignore_nulls=ignore_nulls)

    @overload
    def all(self, *, ignore_nulls: Literal[True] = ...) -> bool:
        ...

    @overload
    def all(self, *, ignore_nulls: bool) -> bool | None:
        ...

    @deprecate_renamed_parameter("drop_nulls", "ignore_nulls", version="0.19.0")
    def all(self, *, ignore_nulls: bool = True) -> bool | None:
        """
        Return whether all values in the column are `True`.

        Only works on columns of data type :class:`Boolean`.

        Parameters
        ----------
        ignore_nulls
            Whether to ignore `null` values.

            If `ignore_nulls=False`, `Kleene logic`_ is used to deal with `null` values:
            if the column contains any `null` values and no `True` values,
            the output is `null`.

            .. _Kleene logic: https://en.wikipedia.org/wiki/Three-valued_logic

        Returns
        -------
        bool or None

        Examples
        --------
        >>> pl.Series([True, True]).all()
        True
        >>> pl.Series([False, True]).all()
        False
        >>> pl.Series([None, True]).all()
        True

        Enable Kleene logic by setting `ignore_nulls=False`:

        >>> pl.Series([None, True]).all(ignore_nulls=False)  # Returns None

        """
        return self._s.all(ignore_nulls=ignore_nulls)

    def log(self, base: float = math.e) -> Series:
        """
        Compute the logarithm of each element.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.log()
        shape: (3,)
        Series: '' [f64]
        [
            0.0
            0.693147
            1.098612
        ]
        """

    def log1p(self) -> Series:
        """
        Compute the natural logarithm of each element plus one.

        This computes `log(1 + x)` but is more numerically stable for `x` close to zero.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.log1p()
        shape: (3,)
        Series: '' [f64]
        [
            0.693147
            1.098612
            1.386294
        ]
        """

    def log10(self) -> Series:
        """
        Compute the base-10 logarithm of each element.

        Examples
        --------
        >>> s = pl.Series([10, 100, 1000])
        >>> s.log10()
        shape: (3,)
        Series: '' [f64]
        [
            1.0
            2.0
            3.0
        ]
        """

    def exp(self) -> Series:
        """
        Compute the exponential of each element.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.exp()
        shape: (3,)
        Series: '' [f64]
        [
            2.718282
            7.389056
            20.085537
        ]
        """

    def drop_nulls(self) -> Series:
        """
        Remove all `null` values.

        The original order of the remaining elements is preserved.

        See Also
        --------
        drop_nans

        Notes
        -----
        A `null` value is not the same as a `NaN` value.
        To drop `NaN` values, use :func:`drop_nans`.

        Examples
        --------
        >>> s = pl.Series([1.0, None, 3.0, float("nan")])
        >>> s.drop_nulls()
        shape: (3,)
        Series: '' [f64]
        [
                1.0
                3.0
                NaN
        ]

        """

    def drop_nans(self) -> Series:
        """
        Remove all floating-point `NaN` values.

        The original order of the remaining elements is preserved.

        See Also
        --------
        drop_nulls

        Notes
        -----
        A `NaN` value is not the same as a `null` value.
        To drop `null` values, use :func:`drop_nulls`.

        Examples
        --------
        >>> s = pl.Series([1.0, None, 3.0, float("nan")])
        >>> s.drop_nans()
        shape: (3,)
        Series: '' [f64]
        [
                1.0
                null
                3.0
        ]

        """

    def to_frame(self, name: str | None = None) -> DataFrame:
        """
        Cast this `Series` to a `DataFrame`.

        Parameters
        ----------
        name
            optionally name/rename the `Series` column in the new
            `DataFrame`.

        Examples
        --------
        >>> s = pl.Series("a", [123, 456])
        >>> df = s.to_frame()
        >>> df
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 123 │
        │ 456 │
        └─────┘

        >>> df = s.to_frame("xyz")
        >>> df
        shape: (2, 1)
        ┌─────┐
        │ xyz │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 123 │
        │ 456 │
        └─────┘

        """
        if isinstance(name, str):
            return wrap_df(PyDataFrame([self.rename(name)._s]))
        return wrap_df(PyDataFrame([self._s]))

    def describe(
        self, percentiles: Sequence[float] | float | None = (0.25, 0.50, 0.75)
    ) -> DataFrame:
        """
        Tabulates summary statistics for this `Series`.

        A `Series` with mixed datatypes will return summary statistics for the
        datatype of the first value.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics (if the
            `Series` is numeric). All values must be in the range `[0, 1]`.

        Returns
        -------
        DataFrame
            A `DataFrame` of summary statistics of the `Series`.

        Warnings
        --------
        The output of describe is not guaranteed to be consistent between polars
        versions. It will show statistics that we deem informative and may be updated in
        the future.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3, 4, 5])
        >>> s.describe()
        shape: (9, 2)
        ┌────────────┬──────────┐
        │ statistic  ┆ value    │
        │ ---        ┆ ---      │
        │ str        ┆ f64      │
        ╞════════════╪══════════╡
        │ count      ┆ 5.0      │
        │ null_count ┆ 0.0      │
        │ mean       ┆ 3.0      │
        │ std        ┆ 1.581139 │
        │ min        ┆ 1.0      │
        │ 25%        ┆ 2.0      │
        │ 50%        ┆ 3.0      │
        │ 75%        ┆ 4.0      │
        │ max        ┆ 5.0      │
        └────────────┴──────────┘

        Non-numeric data types may not have all statistics available:

        >>> s = pl.Series(["a", "a", None, "b", "c"])
        >>> s.describe()
        shape: (3, 2)
        ┌────────────┬───────┐
        │ statistic  ┆ value │
        │ ---        ┆ ---   │
        │ str        ┆ i64   │
        ╞════════════╪═══════╡
        │ count      ┆ 4     │
        │ null_count ┆ 1     │
        │ unique     ┆ 4     │
        └────────────┴───────┘

        """
        stats: dict[str, PythonLiteral | None]
        stats_dtype: PolarsDataType

        if self.dtype.is_numeric():
            stats_dtype = Float64
            stats = {
                "count": self.count(),
                "null_count": self.null_count(),
                "mean": self.mean(),
                "std": self.std(),
                "min": self.min(),
            }
            for p in parse_percentiles(percentiles):
                stats[f"{p:.0%}"] = self.quantile(p)
            stats["max"] = self.max()

        elif self.dtype == Boolean:
            stats_dtype = Int64
            stats = {
                "count": self.count(),
                "null_count": self.null_count(),
                "sum": self.sum(),
            }
        elif self.dtype == String:
            stats_dtype = Int64
            stats = {
                "count": self.count(),
                "null_count": self.null_count(),
                "unique": self.n_unique(),
            }
        elif self.dtype.is_temporal():
            # we coerce all to string, because a polars column
            # only has a single dtype and dates: datetime and count: int don't match
            stats_dtype = String
            stats = {
                "count": str(self.count()),
                "null_count": str(self.null_count()),
                "min": str(self.dt.min()),
                "50%": str(self.dt.median()),
                "max": str(self.dt.max()),
            }
        else:
            raise TypeError(f"cannot describe Series of data type {self.dtype}")

        return pl.DataFrame(
            {"statistic": stats.keys(), "value": stats.values()},
            schema={"statistic": String, "value": stats_dtype},
        )

    def sum(self) -> int | float:
        """
        Get the sum of the values in this `Series`.

        Notes
        -----
        Dtypes in {:class:`Int8`, :class:`UInt8`, :class:`Int16`, :class:`UInt16`} are
        cast to :class:`Int64` before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.sum()
        6

        """
        return self._s.sum()

    def mean(self) -> int | float | None:
        """
        Get the mean of the values in this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.mean()
        2.0

        """
        return self._s.mean()

    def product(self) -> int | float:
        """
        Get the product of the values in this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.product()
        6
        """
        return self._s.product()

    def pow(self, exponent: int | float | None | Series) -> Series:
        """
        Raise each element to the power of the given exponent.

        Parameters
        ----------
        exponent
            The exponent. Accepts Series input.

        Examples
        --------
        >>> s = pl.Series("foo", [1, 2, 3, 4])
        >>> s.pow(3)
        shape: (4,)
        Series: 'foo' [f64]
        [
                1.0
                8.0
                27.0
                64.0
        ]

        """
        if self.dtype.is_temporal():
            raise TypeError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        if _check_for_numpy(exponent) and isinstance(exponent, np.ndarray):
            exponent = Series(exponent)
        return self.to_frame().select_seq(F.col(self.name).pow(exponent)).to_series()

    def min(self) -> PythonLiteral | None:
        """
        Get the minimum value of the elements of this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        1

        """
        return self._s.min()

    def max(self) -> PythonLiteral | None:
        """
        Get the maximum value of the elements of this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.max()
        3

        """
        return self._s.max()

    def nan_max(self) -> int | float | date | datetime | timedelta | str:
        """
        Get the maximum value, but propagate/poison encountered `NaN` values.

        This differs from `numpy.nanmax
        <https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html>`_
        as numpy defaults to propagating NaN values, whereas polars defaults to
        ignoring them.

        """
        return self.to_frame().select_seq(F.col(self.name).nan_max()).item()

    def nan_min(self) -> int | float | date | datetime | timedelta | str:
        """
        Get the minimum value, but propagate/poison encountered `NaN` values.

        This differs from `numpy.nanmin
        <https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html>`_
        as numpy defaults to propagating NaN values, whereas polars defaults to
        ignoring them.

        """
        return self.to_frame().select_seq(F.col(self.name).nan_min()).item()

    def std(self, ddof: int = 1) -> float | None:
        """
        Get the standard deviation of the elements of this `Series`.

        Parameters
        ----------
        ddof
            "Delta Degrees of Freedom": the divisor used in the calculation is
            `N - ddof`, where `N` represents the number of elements.
            By default, `ddof` is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.std()
        1.0

        """
        if not self.dtype.is_numeric():
            return None
        return self._s.std(ddof)

    def var(self, ddof: int = 1) -> float | None:
        """
        Get the variance of the elements of this `Series`.

        Parameters
        ----------
        ddof
            "Delta Degrees of Freedom": the divisor used in the calculation is
            `N - ddof`, where `N` represents the number of elements.
            By default, `ddof` is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.var()
        1.0

        """
        if not self.dtype.is_numeric():
            return None
        return self._s.var(ddof)

    def median(self) -> float | None:
        """
        Get the median of the elements of this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.median()
        2.0

        """
        return self._s.median()

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> float | None:
        """
        Get the specified `quantile` of the elements of this `Series`.

        Parameters
        ----------
        quantile
            A quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            The interpolation method to use when the specified quantile falls between
            two values.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.quantile(0.5)
        2.0

        """
        return self._s.quantile(quantile, interpolation)

    def to_dummies(self, separator: str = "_") -> DataFrame:
        """
        "One-hot encode" this `Series` into a `DataFrame` of dummy/indicator variables.

        The `Series` will be converted into a `DataFrame` with `n` binary :class:`UInt8`
        columns, where `n` is the number of unique values in the `Series`.

        Parameters
        ----------
        separator
            Separator/delimiter used when generating column names.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_dummies()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a_1 ┆ a_2 ┆ a_3 │
        │ --- ┆ --- ┆ --- │
        │ u8  ┆ u8  ┆ u8  │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 0   ┆ 0   │
        │ 0   ┆ 1   ┆ 0   │
        │ 0   ┆ 0   ┆ 1   │
        └─────┴─────┴─────┘

        """
        return wrap_df(self._s.to_dummies(separator))

    @overload
    def cut(
        self,
        breaks: Sequence[float],
        labels: Sequence[str] | None = ...,
        break_point_label: str = ...,
        category_label: str = ...,
        *,
        left_closed: bool = ...,
        include_breaks: bool = ...,
        as_series: Literal[True] = ...,
    ) -> Series:
        ...

    @overload
    def cut(
        self,
        breaks: Sequence[float],
        labels: Sequence[str] | None = ...,
        break_point_label: str = ...,
        category_label: str = ...,
        *,
        left_closed: bool = ...,
        include_breaks: bool = ...,
        as_series: Literal[False],
    ) -> DataFrame:
        ...

    @overload
    def cut(
        self,
        breaks: Sequence[float],
        labels: Sequence[str] | None = ...,
        break_point_label: str = ...,
        category_label: str = ...,
        *,
        left_closed: bool = ...,
        include_breaks: bool = ...,
        as_series: bool,
    ) -> Series | DataFrame:
        ...

    @deprecate_nonkeyword_arguments(["self", "breaks"], version="0.19.0")
    @deprecate_renamed_parameter("series", "as_series", version="0.19.0")
    def cut(
        self,
        breaks: Sequence[float],
        labels: Sequence[str] | None = None,
        break_point_label: str = "break_point",
        category_label: str = "category",
        *,
        left_closed: bool = False,
        include_breaks: bool = False,
        as_series: bool = True,
    ) -> Series | DataFrame:
        """
        Bin continuous values into discrete categories.

        Parameters
        ----------
        breaks
            A list of unique cut points.
        labels
            The names of the bins. The number of labels must be equal to the number of
            cut points plus one.
        break_point_label
            The name of the breakpoint column. Only used when `include_breaks=True`.

            .. deprecated:: 0.19.0
                This parameter will be removed. Use `Series.struct.rename_fields` to
                rename the field instead.
        category_label
            The name of the category column. Only used when `include_breaks=True`.

            .. deprecated:: 0.19.0
                This parameter will be removed. Use `Series.struct.rename_fields` to
                rename the field instead.
        left_closed
            Whether to set the intervals to be left-closed instead of right-closed.
        include_breaks
            Whether to include a column with the right endpoint of the bin each
            observation falls in. This will change the data type of the output from a
            :class:`Categorical` to a :class:`Struct`.
        as_series
            If set to `False`, return a `DataFrame` containing the original
            values, the breakpoints, and the categories.

            .. deprecated:: 0.19.0
                This parameter will be removed. The same behavior can be achieved by
                setting `include_breaks=True`, unnesting the resulting :class:`Struct`
                `Series`, and adding the result to the original `Series`.

        Returns
        -------
        Series
            A `Series` of data type :class:`Categorical` if `include_breaks=False` (the
            default), or :class:`Struct` if `include_breaks=True`.

        See Also
        --------
        qcut

        Examples
        --------
        Divide the column into three categories.

        >>> s = pl.Series("foo", [-2, -1, 0, 1, 2])
        >>> s.cut([-1, 1], labels=["a", "b", "c"])
        shape: (5,)
        Series: 'foo' [cat]
        [
                "a"
                "a"
                "b"
                "b"
                "c"
        ]

        Create a `DataFrame` with the breakpoint and category for each value.

        >>> cut = s.cut([-1, 1], include_breaks=True).alias("cut")
        >>> s.to_frame().with_columns(cut).unnest("cut")
        shape: (5, 3)
        ┌─────┬─────────────┬────────────┐
        │ foo ┆ break_point ┆ category   │
        │ --- ┆ ---         ┆ ---        │
        │ i64 ┆ f64         ┆ cat        │
        ╞═════╪═════════════╪════════════╡
        │ -2  ┆ -1.0        ┆ (-inf, -1] │
        │ -1  ┆ -1.0        ┆ (-inf, -1] │
        │ 0   ┆ 1.0         ┆ (-1, 1]    │
        │ 1   ┆ 1.0         ┆ (-1, 1]    │
        │ 2   ┆ inf         ┆ (1, inf]   │
        └─────┴─────────────┴────────────┘

        """
        if break_point_label != "break_point":
            issue_deprecation_warning(
                "The `break_point_label` parameter for `Series.cut` will be removed."
                " Use `Series.struct.rename_fields` to rename the field instead.",
                version="0.19.0",
            )
        if category_label != "category":
            issue_deprecation_warning(
                "The `category_label` parameter for `Series.cut` will be removed."
                " Use `Series.struct.rename_fields` to rename the field instead.",
                version="0.19.0",
            )
        if not as_series:
            issue_deprecation_warning(
                "The `as_series` parameter for `Series.cut` will be removed."
                " The same behavior can be achieved by setting `include_breaks=True`,"
                " unnesting the resulting struct Series,"
                " and adding the result to the original Series.",
                version="0.19.0",
            )
            temp_name = self.name + "_bin"
            return (
                self.to_frame()
                .with_columns(
                    F.col(self.name)
                    .cut(
                        breaks,
                        labels=labels,
                        left_closed=left_closed,
                        include_breaks=True,  # always include breaks
                    )
                    .alias(temp_name)
                )
                .unnest(temp_name)
                .rename({"brk": break_point_label, temp_name: category_label})
            )

        result = (
            self.to_frame()
            .select_seq(
                F.col(self.name).cut(
                    breaks,
                    labels=labels,
                    left_closed=left_closed,
                    include_breaks=include_breaks,
                )
            )
            .to_series()
        )

        if include_breaks:
            result = result.struct.rename_fields([break_point_label, category_label])

        return result

    @overload
    def qcut(
        self,
        quantiles: Sequence[float] | int,
        *,
        labels: Sequence[str] | None = ...,
        left_closed: bool = ...,
        allow_duplicates: bool = ...,
        include_breaks: bool = ...,
        break_point_label: str = ...,
        category_label: str = ...,
        as_series: Literal[True] = ...,
    ) -> Series:
        ...

    @overload
    def qcut(
        self,
        quantiles: Sequence[float] | int,
        *,
        labels: Sequence[str] | None = ...,
        left_closed: bool = ...,
        allow_duplicates: bool = ...,
        include_breaks: bool = ...,
        break_point_label: str = ...,
        category_label: str = ...,
        as_series: Literal[False],
    ) -> DataFrame:
        ...

    @overload
    def qcut(
        self,
        quantiles: Sequence[float] | int,
        *,
        labels: Sequence[str] | None = ...,
        left_closed: bool = ...,
        allow_duplicates: bool = ...,
        include_breaks: bool = ...,
        break_point_label: str = ...,
        category_label: str = ...,
        as_series: bool,
    ) -> Series | DataFrame:
        ...

    def qcut(
        self,
        quantiles: Sequence[float] | int,
        *,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        include_breaks: bool = False,
        break_point_label: str = "break_point",
        category_label: str = "category",
        as_series: bool = True,
    ) -> Series | DataFrame:
        """
        Bin continuous values into discrete categories based on their quantiles.

        Parameters
        ----------
        quantiles
            Either a list of quantile probabilities between 0 and 1, or a positive
            integer determining the number of bins with uniform probability.
        labels
            The names of the bins. The number of labels must be equal to the number of
            cut points plus one.
        left_closed
            Whether to set the intervals to be left-closed instead of right-closed.
        allow_duplicates
            Whether to drop duplicates in the resulting quantiles rather than raising a
            :class:`DuplicateError`. Duplicates can happen even with unique
            probabilities, depending on the data.
        include_breaks
            Whether to include a column with the right endpoint of the bin each
            observation falls in. This will change the data type of the output from a
            :class:`Categorical` to a :class:`Struct`.
        break_point_label
            The name of the breakpoint column. Only used when `include_breaks=True`.

            .. deprecated:: 0.19.0
                This parameter will be removed. Use `Series.struct.rename_fields` to
                rename the field instead.
        category_label
            The name of the category column. Only used when `include_breaks=True`.

            .. deprecated:: 0.19.0
                This parameter will be removed. Use `Series.struct.rename_fields` to
                rename the field instead.
        as_series
            If set to `False`, return a `DataFrame` containing the original
            values, the breakpoints, and the categories.

            .. deprecated:: 0.19.0
                This parameter will be removed. The same behavior can be achieved by
                setting `include_breaks=True`, unnesting the resulting :class:`Struct`
                `Series`, and adding the result to the original `Series`.

        Returns
        -------
        Series
            A `Series` of data type :class:`Categorical` if `include_breaks=False` (the
            default), or :class:`Struct` if `include_breaks=True`.

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        See Also
        --------
        cut

        Examples
        --------
        Divide a column into three categories according to pre-defined quantile
        probabilities.

        >>> s = pl.Series("foo", [-2, -1, 0, 1, 2])
        >>> s.qcut([0.25, 0.75], labels=["a", "b", "c"])
        shape: (5,)
        Series: 'foo' [cat]
        [
                "a"
                "a"
                "b"
                "b"
                "c"
        ]

        Divide a column into two categories using uniform quantile probabilities.

        >>> s.qcut(2, labels=["low", "high"], left_closed=True)
        shape: (5,)
        Series: 'foo' [cat]
        [
                "low"
                "low"
                "high"
                "high"
                "high"
        ]

        Create a DataFrame with the breakpoint and category for each value.

        >>> cut = s.qcut([0.25, 0.75], include_breaks=True).alias("cut")
        >>> s.to_frame().with_columns(cut).unnest("cut")
        shape: (5, 3)
        ┌─────┬─────────────┬────────────┐
        │ foo ┆ break_point ┆ category   │
        │ --- ┆ ---         ┆ ---        │
        │ i64 ┆ f64         ┆ cat        │
        ╞═════╪═════════════╪════════════╡
        │ -2  ┆ -1.0        ┆ (-inf, -1] │
        │ -1  ┆ -1.0        ┆ (-inf, -1] │
        │ 0   ┆ 1.0         ┆ (-1, 1]    │
        │ 1   ┆ 1.0         ┆ (-1, 1]    │
        │ 2   ┆ inf         ┆ (1, inf]   │
        └─────┴─────────────┴────────────┘

        """
        if break_point_label != "break_point":
            issue_deprecation_warning(
                "The `break_point_label` parameter for `Series.cut` will be removed."
                " Use `Series.struct.rename_fields` to rename the field instead.",
                version="0.19.0",
            )
        if category_label != "category":
            issue_deprecation_warning(
                "The `category_label` parameter for `Series.cut` will be removed."
                " Use `Series.struct.rename_fields` to rename the field instead.",
                version="0.19.0",
            )
        if not as_series:
            issue_deprecation_warning(
                "the `as_series` parameter for `Series.qcut` will be removed."
                " The same behavior can be achieved by setting `include_breaks=True`,"
                " unnesting the resulting struct Series,"
                " and adding the result to the original Series.",
                version="0.19.0",
            )
            temp_name = self.name + "_bin"
            return (
                self.to_frame()
                .with_columns(
                    F.col(self.name)
                    .qcut(
                        quantiles,
                        labels=labels,
                        left_closed=left_closed,
                        allow_duplicates=allow_duplicates,
                        include_breaks=True,  # always include breaks
                    )
                    .alias(temp_name)
                )
                .unnest(temp_name)
                .rename({"brk": break_point_label, temp_name: category_label})
            )

        result = (
            self.to_frame()
            .select(
                F.col(self.name).qcut(
                    quantiles,
                    labels=labels,
                    left_closed=left_closed,
                    allow_duplicates=allow_duplicates,
                    include_breaks=include_breaks,
                )
            )
            .to_series()
        )

        if include_breaks:
            result = result.struct.rename_fields([break_point_label, category_label])

        return result

    def rle(self) -> Series:
        """
        Get the lengths of runs of identical values.

        Returns
        -------
        Series
            A :class:`Struct` `Series` with `"lengths"` and `"values"` fields.

        Examples
        --------
        >>> s = pl.Series("s", [1, 1, 2, 1, None, 1, 3, 3])
        >>> s.rle().struct.unnest()
        shape: (6, 2)
        ┌─────────┬────────┐
        │ lengths ┆ values │
        │ ---     ┆ ---    │
        │ i32     ┆ i64    │
        ╞═════════╪════════╡
        │ 2       ┆ 1      │
        │ 1       ┆ 2      │
        │ 1       ┆ 1      │
        │ 1       ┆ null   │
        │ 1       ┆ 1      │
        │ 2       ┆ 3      │
        └─────────┴────────┘
        """

    def rle_id(self) -> Series:
        """
        Map values to run IDs.

        Similar to RLE, but it maps each value to an ID corresponding to the run into
        which it falls. This is especially useful when you want to define groups by
        runs of identical values rather than the values themselves.

        Returns
        -------
        Series

        See Also
        --------
        rle

        Examples
        --------
        >>> s = pl.Series("s", [1, 1, 2, 1, None, 1, 3, 3])
        >>> s.rle_id()
        shape: (8,)
        Series: 's' [u32]
        [
            0
            0
            1
            2
            3
            4
            5
            5
        ]
        """

    def hist(
        self,
        bins: list[float] | None = None,
        *,
        bin_count: int | None = None,
        include_category: bool = True,
        include_breakpoint: bool = True,
    ) -> DataFrame:
        """
        Bin the values in this `Series` into buckets and count their occurrences.

        Parameters
        ----------
        bins
            A Python list or :class:`List` column of the bin boundaries.
            If `None`, determine the boundaries based on the data.
        bin_count
            If `bins` is `None`, partition the data with this many bin boundaries.
            The number of bins will be `bin_count + 1`.
        include_breakpoint
            Whether to include a column that indicates the upper boundary of each bin.
        include_category
            Whether to include a column that indicates the lower and upper boundary of
            each bin, as a :class:`Categorical` column.

        Returns
        -------
        DataFrame

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Examples
        --------
        >>> a = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
        >>> a.hist(bin_count=4)
        shape: (5, 3)
        ┌─────────────┬─────────────┬───────┐
        │ break_point ┆ category    ┆ count │
        │ ---         ┆ ---         ┆ ---   │
        │ f64         ┆ cat         ┆ u32   │
        ╞═════════════╪═════════════╪═══════╡
        │ 0.0         ┆ (-inf, 0.0] ┆ 0     │
        │ 2.25        ┆ (0.0, 2.25] ┆ 3     │
        │ 4.5         ┆ (2.25, 4.5] ┆ 2     │
        │ 6.75        ┆ (4.5, 6.75] ┆ 0     │
        │ inf         ┆ (6.75, inf] ┆ 2     │
        └─────────────┴─────────────┴───────┘

        """
        out = (
            self.to_frame()
            .select_seq(
                F.col(self.name).hist(
                    bins=bins,
                    bin_count=bin_count,
                    include_category=include_category,
                    include_breakpoint=include_breakpoint,
                )
            )
            .to_series()
        )
        if not include_breakpoint and not include_category:
            return out.to_frame()
        else:
            return out.struct.unnest()

    def value_counts(self, *, sort: bool = False, parallel: bool = False) -> DataFrame:
        """
        Compute the unique values of this `Series` and how many times they occur.

        If you just need the counts and not the values themselves, use
        :func:`unique_counts`.

        Parameters
        ----------
        sort
            Whether to sort the output in descending order of number of occurrences.
            If `sort=False` (the default), the order of the output is random.
        parallel
            Whether to execute the computation in parallel.

            .. note::
                This option should likely not be enabled in an aggregation context,
                as the computation is already parallelized per group.

        Returns
        -------
        DataFrame
            A :class:`DataFrame` mapping unique values to their counts. The
            :class:`DataFrame` has two columns: the first is the name of the original
            column, containing the values, and the second is `"count"`, containing the
            counts.

        Examples
        --------
        >>> s = pl.Series("color", ["red", "blue", "red", "green", "blue", "blue"])
        >>> s.value_counts()  # doctest: +IGNORE_RESULT
        shape: (3, 2)
        ┌───────┬───────┐
        │ color ┆ count │
        │ ---   ┆ ---   │
        │ str   ┆ u32   │
        ╞═══════╪═══════╡
        │ red   ┆ 2     │
        │ green ┆ 1     │
        │ blue  ┆ 3     │
        └───────┴───────┘

        Sort the output by count.

        >>> s.value_counts(sort=True)
        shape: (3, 2)
        ┌───────┬───────┐
        │ color ┆ count │
        │ ---   ┆ ---   │
        │ str   ┆ u32   │
        ╞═══════╪═══════╡
        │ blue  ┆ 3     │
        │ red   ┆ 2     │
        │ green ┆ 1     │
        └───────┴───────┘
        """
        return pl.DataFrame._from_pydf(
            self._s.value_counts(sort=sort, parallel=parallel)
        )

    def unique_counts(self) -> Series:
        """
        Count how many times each unique value occurs, in order of first appearance.

        This method differs from :func:`value_counts` in that it does not return the
        values, only the counts, and might be faster.

        Examples
        --------
        >>> s = pl.Series("id", ["a", "b", "b", "c", "c", "c"])
        >>> s.unique_counts()
        shape: (3,)
        Series: 'id' [u32]
        [
            1
            2
            3
        ]

        """

    def entropy(self, base: float = math.e, *, normalize: bool = False) -> float | None:
        """
        Compute the entropy of a `Series`.

        Uses the formula `-sum(pk * log(pk)`, where `pk` are discrete probabilities.

        Parameters
        ----------
        base
            The base of the logarithm to use in the entropy calculation, defaults to
            `e`.
        normalize
            Normalize `pk` if it doesn't sum to 1.

        Examples
        --------
        >>> a = pl.Series([0.99, 0.005, 0.005])
        >>> a.entropy(normalize=True)
        0.06293300616044681
        >>> b = pl.Series([0.65, 0.10, 0.25])
        >>> b.entropy(normalize=True)
        0.8568409950394724

        """
        return (
            self.to_frame()
            .select_seq(F.col(self.name).entropy(base, normalize=normalize))
            .to_series()
            .item()
        )

    def cumulative_eval(
        self, expr: Expr, min_periods: int = 1, *, parallel: bool = False
    ) -> Series:
        """
        Run an expression over a sliding window that slides one element at a time.

        Parameters
        ----------
        expr
            The expression to evaluate.
        min_periods
            The number of non-`null` values there should be in the window before
            the expression is evaluated.
        parallel
            Whether to run in parallel. Don't do this in a group by or another
            operation that already has substantial parallelization.

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        This can be really slow as it can have `O(n^2)` complexity. Don't use this
        for operations that visit all elements.

        Examples
        --------
        >>> s = pl.Series("values", [1, 2, 3, 4, 5])
        >>> s.cumulative_eval(pl.element().first() - pl.element().last() ** 2)
        shape: (5,)
        Series: 'values' [f64]
        [
            0.0
            -3.0
            -8.0
            -15.0
            -24.0
        ]

        """

    def alias(self, name: str) -> Series:
        """
        Rename this `Series`.

        Parameters
        ----------
        name
            The new name.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.alias("b")
        shape: (3,)
        Series: 'b' [i64]
        [
                1
                2
                3
        ]

        """
        s = self.clone()
        s._s.rename(name)
        return s

    def rename(self, name: str) -> Series:
        """
        Rename this `Series`.

        Alias for :func:`Series.alias`.

        Parameters
        ----------
        name
            The new name.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.rename("b")
        shape: (3,)
        Series: 'b' [i64]
        [
                1
                2
                3
        ]

        """
        return self.alias(name)

    def chunk_lengths(self) -> list[int]:
        """
        Get the length of each chunk of this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate `Series` with `rechunk=True`:

        >>> pl.concat([s, s2]).chunk_lengths()
        [6]

        Concatenate `Series` with `rechunk=False`:

        >>> pl.concat([s, s2], rechunk=False).chunk_lengths()
        [3, 3]

        """
        return self._s.chunk_lengths()

    def n_chunks(self) -> int:
        """
        Get the number of chunks that this `Series` contains.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.n_chunks()
        1
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate `Series` with `rechunk=True`:

        >>> pl.concat([s, s2]).n_chunks()
        1

        Concatenate `Series` with `rechunk=False`:

        >>> pl.concat([s, s2], rechunk=False).n_chunks()
        2

        """
        return self._s.n_chunks()

    def cum_max(self, *, reverse: bool = False) -> Series:
        """
        Get the cumulative maximum of the elements of this `Series`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).

        Examples
        --------
        >>> s = pl.Series("s", [3, 5, 1])
        >>> s.cum_max()
        shape: (3,)
        Series: 's' [i64]
        [
            3
            5
            5
        ]

        """

    def cum_min(self, *, reverse: bool = False) -> Series:
        """
        Get the cumulative minimum of the elements of this `Series`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).

        Examples
        --------
        >>> s = pl.Series("s", [1, 2, 3])
        >>> s.cum_min()
        shape: (3,)
        Series: 's' [i64]
        [
            1
            1
            1
        ]

        """

    def cum_prod(self, *, reverse: bool = False) -> Series:
        """
        Get the cumulative product of the elements of this `Series`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).

        Notes
        -----
        Dtypes in {:class:`Int8`, :class:`UInt8`, :class:`Int16`, :class:`UInt16`} are
        cast to `Int64` before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cum_prod()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            6
        ]

        """

    def cum_sum(self, *, reverse: bool = False) -> Series:
        """
        Get the cumulative sum of the elements of this `Series`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).

        Notes
        -----
        Dtypes in {:class:`Int8`, :class:`UInt8`, :class:`Int16`, :class:`UInt16`} are
        cast to :class:`Int64` before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cum_sum()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            3
            6
        ]

        """

    def slice(self, offset: int, length: int | None = None) -> Series:
        """
        Get a contiguous set of rows from this `Series`.

        Parameters
        ----------
        offset
            The start index. Negative indexing is supported.
        length
            The length of the slice. If `length=None`, all rows starting from the
            `offset` will be selected.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.slice(1, 2)
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                3
        ]

        """
        return self._from_pyseries(self._s.slice(offset=offset, length=length))

    def append(self, other: Series) -> Self:
        """
        Append a `Series` to this one.

        The resulting `Series` will consist of multiple chunks.

        Parameters
        ----------
        other
           The  `Series` to append.

        Warnings
        --------
        This method modifies the `Series` in-place. The `Series` is returned for
        convenience only.

        See Also
        --------
        extend

        Examples
        --------
        >>> a = pl.Series("a", [1, 2, 3])
        >>> b = pl.Series("b", [4, 5])
        >>> a.append(b)
        shape: (5,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
        ]

        The resulting `Series` will consist of multiple chunks:

        >>> a.n_chunks()
        2

        """
        try:
            self._s.append(other._s)
        except RuntimeError as exc:
            if str(exc) == "Already mutably borrowed":
                self._s.append(other._s.clone())
            else:
                raise
        return self

    def extend(self, other: Series) -> Self:
        """
        Extend the memory backed by this `DataFrame` with the values from `other`.

        Different from :func:`vstack` which adds the chunks from `other` to the chunks
        of this `DataFrame`, `extend` appends the data from `other` to the underlying
        memory locations and thus may cause a reallocation.

        The resulting data structure will not have any extra chunks and thus will yield
        faster queries.

        Prefer `extend` over :func:`vstack` when you want to do a query after a single
        append. For instance, during online operations where you add `n` rows and rerun
        a query.

        Prefer :func:`vstack` over `extend` when you want to append many times before
        doing a query. For instance, when you read in multiple files and want to store
        them in a single `DataFrame`. In the latter case, finish the sequence of
        :func:`vstack` operations with a :func:`rechunk`.

        Parameters
        ----------
        other
            `Series` to extend `self` with.

        Warnings
        --------
        This method modifies the `Series` in-place. The `Series` is returned for
        convenience only.

        See Also
        --------
        append

        Examples
        --------
        >>> a = pl.Series("a", [1, 2, 3])
        >>> b = pl.Series("b", [4, 5])
        >>> a.extend(b)
        shape: (5,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
        ]

        The resulting `Series` will consist of a single chunk:

        >>> a.n_chunks()
        1

        """
        try:
            self._s.extend(other._s)
        except RuntimeError as exc:
            if str(exc) == "Already mutably borrowed":
                self._s.extend(other._s.clone())
            else:
                raise
        return self

    def filter(self, predicate: Series | list[bool]) -> Self:
        """
        Filter this `Series` to rows where a predicate is `True`.

        The original order of the remaining elements is preserved.

        Parameters
        ----------
        predicate
            A :class:`Boolean` `Series` or list of booleans.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> mask = pl.Series("", [True, False, True])
        >>> s.filter(mask)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                3
        ]

        """
        if isinstance(predicate, list):
            predicate = Series("", predicate)
        return self._from_pyseries(self._s.filter(predicate._s))

    def head(self, n: int = 10) -> Series:
        """
        Get the first `n` elements.

        Parameters
        ----------
        n
            The number of elements to return.
            If `n` is negative, return all elements except the last `abs(n)`.

        See Also
        --------
        tail, slice

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.head(3)
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                2
                3
        ]

        Pass a negative value to get all rows `except` the last `abs(n)`:

        >>> s.head(-3)
        shape: (2,)
        Series: 'a' [i64]
        [
                1
                2
        ]

        """
        if n < 0:
            n = max(0, self.len() + n)
        return self._from_pyseries(self._s.head(n))

    def tail(self, n: int = 10) -> Series:
        """
        Get the last `n` elements.

        Parameters
        ----------
        n
            The number of elements to return.
            If `n` is negative, return all elements except the last `abs(n)`:

        See Also
        --------
        head, slice

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.tail(3)
        shape: (3,)
        Series: 'a' [i64]
        [
                3
                4
                5
        ]

        Pass a negative value to get all rows `except` the first `abs(n)`:

        >>> s.tail(-3)
        shape: (2,)
        Series: 'a' [i64]
        [
                4
                5
        ]

        """
        if n < 0:
            n = max(0, self.len() + n)
        return self._from_pyseries(self._s.tail(n))

    def limit(self, n: int = 10) -> Series:
        """
        Get the first `n` elements.

        Alias for :func:`Series.head`.

        Parameters
        ----------
        n
            The number of elements to return.
            If `n` is negative, return all elements except the last `abs(n)`.

        See Also
        --------
        head

        """
        return self.head(n)

    def gather_every(self, n: int, offset: int = 0) -> Series:
        """
        Get every nth element of this `Series`.

        `s.gather_every(n, offset)` is equivalent to `s[offset::n]`.

        Parameters
        ----------
        n
            The spacing between the rows to be gathered.
        offset
            The index of the first row to be gathered.

        See Also
        --------
        gather : Get multiple elements by index.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.gather_every(2)
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            3
        ]
        >>> s.gather_every(2, offset=1)
        shape: (2,)
        Series: 'a' [i64]
        [
            2
            4
        ]

        """

    def sort(self, *, descending: bool = False, in_place: bool = False) -> Self:
        """
        Sort this `Series`.

        Parameters
        ----------
        descending
            Whether to sort in descending instead of ascending order.
        in_place
            Whether to sort in-place.

        Examples
        --------
        >>> s = pl.Series("a", [1, 3, 4, 2])
        >>> s.sort()
        shape: (4,)
        Series: 'a' [i64]
        [
                1
                2
                3
                4
        ]
        >>> s.sort(descending=True)
        shape: (4,)
        Series: 'a' [i64]
        [
                4
                3
                2
                1
        ]

        """
        if in_place:
            self._s = self._s.sort(descending)
            return self
        else:
            return self._from_pyseries(self._s.sort(descending))

    def top_k(self, k: int | IntoExprColumn = 5) -> Series:
        r"""
        Return the `k` largest elements of this `Series`.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            The number of elements to return.

        See Also
        --------
        bottom_k

        Examples
        --------
        >>> s = pl.Series("a", [2, 5, 1, 4, 3])
        >>> s.top_k(3)
        shape: (3,)
        Series: 'a' [i64]
        [
            5
            4
            3
        ]

        """

    def bottom_k(self, k: int | IntoExprColumn = 5) -> Series:
        r"""
        Return the `k` smallest elements of this `Series`.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            The number of smallest elements to return.

        See Also
        --------
        top_k

        Examples
        --------
        >>> s = pl.Series("a", [2, 5, 1, 4, 3])
        >>> s.bottom_k(3)
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            3
        ]

        """

    def arg_sort(self, *, descending: bool = False, nulls_last: bool = False) -> Series:
        """
        Get the index values that would sort this `Series`.

        Parameters
        ----------
        descending
            Whether to sort in descending instead of ascending order.
        nulls_last
            Whether to place `null` values last instead of first.

        Examples
        --------
        >>> s = pl.Series("a", [5, 3, 4, 1, 2])
        >>> s.arg_sort()
        shape: (5,)
        Series: 'a' [u32]
        [
            3
            4
            1
            2
            0
        ]

        """

    def arg_unique(self) -> Series:
        """
        Get the indices of the first occurrence of each value in this `Series`.

        Preserves `null` values as `null`.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.arg_unique()
        shape: (3,)
        Series: 'a' [u32]
        [
                0
                1
                3
        ]

        """

    def arg_min(self) -> int | None:
        """
        Get the index of the minimum value in this `Series`.

        Returns
        -------
        int

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_min()
        2

        """
        return self._s.arg_min()

    def arg_max(self) -> int | None:
        """
        Get the index of the maximum value in this `Series`.

        Returns
        -------
        int

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_max()
        0

        """
        return self._s.arg_max()

    @overload
    def search_sorted(self, element: int | float, side: SearchSortedSide = ...) -> int:
        ...

    @overload
    def search_sorted(
        self,
        element: Series | np.ndarray[Any, Any] | list[int] | list[float],
        side: SearchSortedSide = ...,
    ) -> Series:
        ...

    def search_sorted(
        self,
        element: int | float | Series | np.ndarray[Any, Any] | list[int] | list[float],
        side: SearchSortedSide = "any",
    ) -> int | Series:
        """
        Find indices where elements should be inserted to maintain order.

        Assumes (but does not check) that this `Series` is sorted.

        If `element` is a scalar, returns a length-1 `Series` for each column containing
        an index where, if you inserted `element` immediately before that index, this
        `Series` would remain sorted.

        If `element` is a sequence of elements, returns a `Series` of length
        `len(element)`, containing such an index for *each* element in `element`.

        If there are no duplicate values in this `Series`, there is only one such index
        for a given element. However, if this `Series` does have duplicates, there may
        be multiple such indices. Set `side='left'` to get the left-most of these
        indices, `side='right'` to get the right-most, and `side='any'` for speed if you
        don't care whether you get the left-most index, the right-most index, or
        something in between.

        Parameters
        ----------
        element
            An expression or scalar value.
        side : {'any', 'left', 'right'}
            If `"left"`, give the index `i` of the left-most suitable location found, so
            that `self[i-1] < element <= self[i]`.
            If `"right"`, give the index of the right-most suitable location found, so
            that `self[i-1] <= element < self[i]`.
            If `"any"`, give the index of the first suitable location found, so that
            `self[i-1] <= element <= self[i]`. This is fastest.

        """
        if isinstance(element, (int, float)):
            return F.select(F.lit(self).search_sorted(element, side)).item()
        element = Series(element)
        return F.select(F.lit(self).search_sorted(element, side)).to_series()

    def unique(self, *, maintain_order: bool = False) -> Series:
        """
        Get the unique elements of this `Series`.

        Parameters
        ----------
        maintain_order
            Whether to get the unique values in the same order that they appeared
            in the original `Series`. This is slower.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.unique().sort()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            3
        ]

        """

    def gather(
        self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]
    ) -> Series:
        """
        Get multiple elements from this `Series` by index.

        `s.gather(indices)` is equivalent to `s[indices]`.

        Parameters
        ----------
        indices
            The indices of the elements to get.

        See Also
        --------
        gather_every : Get every nth element.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.gather([1, 3])
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                4
        ]

        """

    def null_count(self) -> int:
        """
        Get the number of `null` values in this `Series`.

        Examples
        --------
        >>> s = pl.Series([1, None, None])
        >>> s.null_count()
        2
        """
        return self._s.null_count()

    def has_validity(self) -> bool:
        """
        Return `True` if the `Series` has a validity bitmask.

        If there is no mask, it means that there are no `null` values.

        Notes
        -----
        While the *absence* of a validity bitmask guarantees that a `Series`
        does not have `null` values, the converse is not true, e.g. the *presence* of a
        bitmask does not mean that there are `null` values, as every value of the
        bitmask could be `False`.

        To confirm that a column has `null` values, use :func:`null_count`.

        """
        return self._s.has_validity()

    def is_empty(self) -> bool:
        """
        Check if this `Series` is empty.

        Examples
        --------
        >>> s = pl.Series("a", [], dtype=pl.Float32)
        >>> s.is_empty()
        True

        """
        return self.len() == 0

    def is_sorted(self, *, descending: bool = False) -> bool:
        """
        Check if this `Series` is sorted.

        Parameters
        ----------
        descending
            Check if the `Series` is sorted in descending rather than ascending order.

        Examples
        --------
        >>> s = pl.Series([1, 3, 2])
        >>> s.is_sorted()
        False

        >>> s = pl.Series([3, 2, 1])
        >>> s.is_sorted(descending=True)
        True

        """
        return self._s.is_sorted(descending)

    def not_(self) -> Series:
        """
        Negate a :class:`Boolean` `Series`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, False])
        >>> s.not_()
        shape: (3,)
        Series: 'a' [bool]
        [
            false
            true
            true
        ]

        """

    def is_null(self) -> Series:
        """
        Get a :class:`Boolean` `Series` of which elements are `null`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_null()
        shape: (4,)
        Series: 'a' [bool]
        [
            false
            false
            false
            true
        ]

        """

    def is_not_null(self) -> Series:
        """
        Get a :class:`Boolean` `Series` of which elements are not `null`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_not_null()
        shape: (4,)
        Series: 'a' [bool]
        [
            true
            true
            true
            false
        ]

        """

    def is_finite(self) -> Series:
        """
        Get a :class:`Boolean` `Series` of which elements are finite.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, np.inf])
        >>> s.is_finite()
        shape: (3,)
        Series: 'a' [bool]
        [
                true
                true
                false
        ]

        """

    def is_infinite(self) -> Series:
        """
        Get a :class:`Boolean` `Series` of which elements are infinite.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, np.inf])
        >>> s.is_infinite()
        shape: (3,)
        Series: 'a' [bool]
        [
                false
                false
                true
        ]

        """

    def is_nan(self) -> Series:
        """
        Get a :class:`Boolean` `Series` of which values are not `NaN`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.nan])
        >>> s.is_nan()
        shape: (4,)
        Series: 'a' [bool]
        [
                false
                false
                false
                true
        ]

        """

    def is_not_nan(self) -> Series:
        """
        Get a :class:`Boolean` `Series` of which values are not `NaN`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.nan])
        >>> s.is_not_nan()
        shape: (4,)
        Series: 'a' [bool]
        [
                true
                true
                true
                false
        ]

        """

    def is_in(self, other: Series | Collection[Any]) -> Series:
        """
        Get a :class:`Boolean` mask of which elements in `self` are present in `other`.

        Alternately, when `other` is a :class:`List` `Series`, get a mask of which
        elements in `self` are present in the corresponding list element of `other`.

        `self` must not be a :class:`List` or :class:`Array` `Series`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [2, 4])
        >>> s2.is_in(s)
        shape: (2,)
        Series: 'b' [bool]
        [
                true
                false
        ]

        Check if each element in `elements` is present in the corresponding list element
        of `sets`:

        >>> sets = pl.Series("sets", [[1, 2, 3], [1, 2], [9, 10]])
        >>> elements = pl.Series("elements", [1, 2, 3])
        >>> print(sets)
        shape: (3,)
        Series: 'sets' [list[i64]]
        [
            [1, 2, 3]
            [1, 2]
            [9, 10]
        ]
        >>> print(elements)
        shape: (3,)
        Series: 'elements' [i64]
        [
            1
            2
            3
        ]
        >>> elements.is_in(sets)
        shape: (3,)
        Series: 'elements' [bool]
        [
            true
            true
            false
        ]

        """

    def arg_true(self) -> Series:
        """
        Get the indices of the `True` elements of this `Series`.

        `self` must be :class:`Boolean`.

        Returns
        -------
        Series
            A :class:`UInt32` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> (s == 2).arg_true()
        shape: (1,)
        Series: 'a' [u32]
        [
                1
        ]

        """
        return F.arg_where(self, eager=True)

    def is_unique(self) -> Series:
        """
        Get a :class:`Boolean` mask of which elements of this `Series` are unique.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.is_unique()
        shape: (4,)
        Series: 'a' [bool]
        [
                true
                false
                false
                true
        ]

        """

    def is_first_distinct(self) -> Series:
        """
        Get a :class:`Boolean` mask of the first occurrence of each distinct value.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series([1, 1, 2, 3, 2])
        >>> s.is_first_distinct()
        shape: (5,)
        Series: '' [bool]
        [
                true
                false
                true
                true
                false
        ]

        """

    def is_last_distinct(self) -> Series:
        """
        Get a :class:`Boolean` mask of the last occurrence of each distinct value.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series([1, 1, 2, 3, 2])
        >>> s.is_last_distinct()
        shape: (5,)
        Series: '' [bool]
        [
                false
                true
                false
                true
                true
        ]

        """

    def is_duplicated(self) -> Series:
        """
        Get a :class:`Boolean` mask of which values appear more than once.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.is_duplicated()
        shape: (4,)
        Series: 'a' [bool]
        [
                false
                true
                true
                false
        ]

        """

    def explode(self) -> Series:
        """
        Put every element of every list of a :class:`List` `Series` on its own row.

        Returns
        -------
        Series
            A `Series` with the same data type as the inner data type of the list
            elements.

        See Also
        --------
        Series.list.explode : Explode a :class:`List` `Series`.
        Series.str.explode : Explode a :class:`String` `Series`.

        """

    def equals(
        self, other: Series, *, null_equal: bool = True, strict: bool = False
    ) -> bool:
        """
        Check whether this `Series` is equal to another `Series`.

        Parameters
        ----------
        other
            The `Series` to compare with.
        null_equal
            Whether to consider `null` values as equal. If `null_equal=False`,
            a `Series` containing any `null` values will always compare as `False`,
            even to itself.
        strict
            Don't allow different numerical dtypes, e.g. comparing :class:`UInt32` with
            :class:`Int64` will return `False`.

        See Also
        --------
        assert_series_equal

        Examples
        --------
        >>> s1 = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s1.equals(s1)
        True
        >>> s1.equals(s2)
        False
        """
        return self._s.equals(other._s, null_equal, strict)

    def cast(
        self,
        dtype: (PolarsDataType | type[int] | type[float] | type[str] | type[bool]),
        *,
        strict: bool = True,
    ) -> Self:
        """
        Cast this `Series` to another data type.

        Parameters
        ----------
        dtype
            The data type to cast to.
        strict
            Whether to raise an error if a cast could not be done (for instance, due to
            an overflow).

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s
        shape: (3,)
        Series: 'a' [bool]
        [
            true
            false
            true
        ]

        >>> s.cast(pl.UInt32)
        shape: (3,)
        Series: 'a' [u32]
        [
            1
            0
            1
        ]

        """
        # Do not dispatch cast as it is slow and used in other functions.
        dtype = py_type_to_dtype(dtype)
        return self._from_pyseries(self._s.cast(dtype, strict))

    def to_physical(self) -> Series:
        """
        Cast this `Series` to its underlying numeric representation.

        - :class:`Date` -> :class:`Int32`
        - :class:`Datetime` -> :class:`Int64`
        - :class:`Time` -> :class:`Int64`
        - :class:`Duration` -> :class:`Int64`
        - :class:`Categorical` -> :class:`UInt32`
        - `List(inner)` -> `List(physical of inner)`

        Other data types will be left unchanged.

        Examples
        --------
        We can use `to_physical` to replicate the `pandas.factorize
        <https://pandas.pydata.org/docs/reference/api/pandas.factorize.html>`_
        function:

        >>> s = pl.Series("values", ["a", None, "x", "a"])
        >>> s.cast(pl.Categorical).to_physical()
        shape: (4,)
        Series: 'values' [u32]
        [
            0
            null
            1
            0
        ]

        """

    def to_list(self, *, use_pyarrow: bool | None = None) -> list[Any]:
        """
        Convert this `Series` to a Python list. This operation clones data.

        Parameters
        ----------
        use_pyarrow
            Whether to use :mod:`pyarrow` for the conversion.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]
        >>> type(s.to_list())
        <class 'list'>

        """
        if use_pyarrow is not None:
            issue_deprecation_warning(
                "The parameter `use_pyarrow` for `Series.to_list` is deprecated."
                " Call the method without `use_pyarrow` to silence this warning.",
                version="0.19.9",
            )
            if use_pyarrow:
                return self.to_arrow().to_pylist()

        return self._s.to_list()

    def rechunk(self, *, in_place: bool = False) -> Self:
        """
        Move this `Series` to a single chunk of memory, if in multiple chunks.

        This will make sure all subsequent operations have optimal and predictable
        performance.

        Parameters
        ----------
        in_place
            Whether to rechunk in-place.

        """
        opt_s = self._s.rechunk(in_place)
        return self if in_place else self._from_pyseries(opt_s)

    def reverse(self) -> Series:
        """
        Reverse the order of the elements of this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3], dtype=pl.Int8)
        >>> s.reverse()
        shape: (3,)
        Series: 'a' [i8]
        [
            3
            2
            1
        ]

        """

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Series:
        """
        Determine which values are between `lower_bound` and `upper_bound`.

        Parameters
        ----------
        lower_bound
            The lower bound. Accepts expression input. Strings are parsed as column
            names, other non-expression inputs are parsed as literals.
        upper_bound
            The upper bound. Accepts expression input. Strings are parsed as column
            names, other non-expression inputs are parsed as literals.
        closed : {'both', 'left', 'right', 'none'}
            Whether to include both endpoints, only the left or right endpoint, or
            neither endpoint.

        Returns
        -------
        Series
            A :class:`Boolean` `Series` of which values are between `lower_bound`
            and `upper_bound`.

        Examples
        --------
        >>> s = pl.Series("num", [1, 2, 3, 4, 5])
        >>> s.is_between(2, 4)
        shape: (5,)
        Series: 'num' [bool]
        [
            false
            true
            true
            true
            false
        ]

        Use the `closed` argument to include or exclude the values at the bounds:

        >>> s.is_between(2, 4, closed="left")
        shape: (5,)
        Series: 'num' [bool]
        [
            false
            true
            true
            false
            false
        ]

        You can also use strings as well as numeric/temporal values:

        >>> s = pl.Series("s", ["a", "b", "c", "d", "e"])
        >>> s.is_between("b", "d", closed="both")
        shape: (5,)
        Series: 's' [bool]
        [
            false
            true
            true
            true
            false
        ]

        """
        if closed == "none":
            out = (self > lower_bound) & (self < upper_bound)
        elif closed == "both":
            out = (self >= lower_bound) & (self <= upper_bound)
        elif closed == "right":
            out = (self > lower_bound) & (self <= upper_bound)
        elif closed == "left":
            out = (self >= lower_bound) & (self < upper_bound)

        if isinstance(out, pl.Expr):
            out = F.select(out).to_series()

        return out

    def to_numpy(
        self,
        *args: Any,
        zero_copy_only: bool = False,
        writable: bool = False,
        use_pyarrow: bool = True,
    ) -> np.ndarray[Any, Any]:
        """
        Convert this `Series` to a :class:`numpy.ndarray`.

        This operation may clone data but is completely safe. Note that:

        - data which is purely numeric AND without `null` values is not cloned;
        - floating point `NaN` values can be zero-copied;
        - :class:`Boolean` `Series` can't be zero-copied.

        To ensure that no data is cloned, set `zero_copy_only=True`.

        Parameters
        ----------
        *args
            Positional arguments to be passed to `pyarrow.Array.to_numpy
            <https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy>`_.
            Only used when `use_pyarrow=True`.
        zero_copy_only
            Wheher to raise an exception if the conversion to a NumPy array would
            require copying the underlying data (e.g. in the presence of `null` values,
            or for non-primitive types).
        writable
            For NumPy arrays created with zero copy (view on the Arrow data), the
            resulting array is not writable (since Arrow data is immutable). By setting
            `writable=True`, a copy of the array is made to ensure it is writable.
        use_pyarrow
            Whether to use `pyarrow.Array.to_numpy
            <https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy>`_.
            for the conversion to NumPy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> arr = s.to_numpy()
        >>> arr  # doctest: +IGNORE_RESULT
        array([1, 2, 3], dtype=int64)
        >>> type(arr)
        <class 'numpy.ndarray'>

        """

        def convert_to_date(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            if self.dtype == Date:
                tp = "datetime64[D]"
            elif self.dtype == Duration:
                tp = f"timedelta64[{self.dtype.time_unit}]"  # type: ignore[attr-defined]
            else:
                tp = f"datetime64[{self.dtype.time_unit}]"  # type: ignore[attr-defined]
            return arr.astype(tp)

        def raise_no_zero_copy() -> None:
            if zero_copy_only:
                raise ValueError("cannot return a zero-copy array")

        if self.dtype == Array:
            np_array = self.explode().to_numpy(
                zero_copy_only=zero_copy_only,
                writable=writable,
                use_pyarrow=use_pyarrow,
            )
            np_array.shape = (self.len(), self.dtype.width)  # type: ignore[attr-defined]
            return np_array

        if (
            use_pyarrow
            and _PYARROW_AVAILABLE
            and self.dtype != Object
            and (self.dtype == Time or not self.dtype.is_temporal())
        ):
            return self.to_arrow().to_numpy(
                *args, zero_copy_only=zero_copy_only, writable=writable
            )

        elif self.dtype in (Time, Decimal):
            raise_no_zero_copy()
            # note: there are no native numpy "time" or "decimal" dtypes
            return np.array(self.to_list(), dtype="object")
        else:
            if not self.null_count():
                if self.dtype.is_temporal():
                    np_array = convert_to_date(self._view(ignore_nulls=True))
                elif self.dtype.is_numeric():
                    np_array = self._view(ignore_nulls=True)
                else:
                    raise_no_zero_copy()
                    np_array = self._s.to_numpy()

            elif self.dtype.is_temporal():
                np_array = convert_to_date(self.to_physical()._s.to_numpy())
            else:
                raise_no_zero_copy()
                np_array = self._s.to_numpy()

            if writable and not np_array.flags.writeable:
                raise_no_zero_copy()
                return np_array.copy()
            else:
                return np_array

    def _view(self, *, ignore_nulls: bool = False) -> SeriesView:
        """
        Get a view into this `Series` data with a :class:`np.ndarray`.

        This operation doesn't clone data, but does not include missing values.

        Returns
        -------
        SeriesView

        Parameters
        ----------
        ignore_nulls
            If `True`, `null` values are converted to 0.
            If `False`, an exception is raised if `null` values are present.

        Examples
        --------
        >>> s = pl.Series("a", [1, None])
        >>> s._view(ignore_nulls=True)
        SeriesView([1, 0])

        """
        if not ignore_nulls:
            assert not self.null_count()

        from polars.series._numpy import SeriesView, _ptr_to_numpy

        ptr_type = dtype_to_ctype(self.dtype)
        ptr = self._s.as_single_ptr()
        array = _ptr_to_numpy(ptr, self.len(), ptr_type)
        array.setflags(write=False)
        return SeriesView(array, self)

    def to_arrow(self) -> pa.Array:
        """
        Get the :class:`pyarrow.Array` underlying this `Series`.

        If the `Series` contains only a single chunk, this operation is zero-copy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s = s.to_arrow()
        >>> s  # doctest: +ELLIPSIS
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          3
        ]

        """
        return self._s.to_arrow()

    def to_pandas(  # noqa: D417
        self, *args: Any, use_pyarrow_extension_array: bool = False, **kwargs: Any
    ) -> pd.Series[Any]:
        """
        Convert this `Series` to a :class:`pandas.Series`.

        This requires that :mod:`pandas` and :mod:`pyarrow` are installed.
        This operation clones data, unless `use_pyarrow_extension_array=True`.

        Parameters
        ----------
        use_pyarrow_extension_array
            Whether to use a :mod:`pyarrow`-backed extension array instead of a
            :class:`numpy.ndarray` as the underlying representation of the pandas
            `Series`. This allows zero-copy operations and preservation of `null`
            values. Further operations on this pandas `Series` might still trigger
            conversion to NumPy arrays if that operation is not supported by pandas's
            :mod:`pyarrow` compute functions.
        kwargs
            Keyword arguments to be passed to :meth:`pyarrow.Array.to_pandas`.

        Examples
        --------
        >>> s1 = pl.Series("a", [1, 2, 3])
        >>> s1.to_pandas()
        0    1
        1    2
        2    3
        Name: a, dtype: int64
        >>> s1.to_pandas(use_pyarrow_extension_array=True)  # doctest: +SKIP
        0    1
        1    2
        2    3
        Name: a, dtype: int64[pyarrow]
        >>> s2 = pl.Series("b", [1, 2, None, 4])
        >>> s2.to_pandas()
        0    1.0
        1    2.0
        2    NaN
        3    4.0
        Name: b, dtype: float64
        >>> s2.to_pandas(use_pyarrow_extension_array=True)  # doctest: +SKIP
        0       1
        1       2
        2    <NA>
        3       4
        Name: b, dtype: int64[pyarrow]

        """
        if use_pyarrow_extension_array:
            if parse_version(pd.__version__) < (1, 5):
                raise ModuleUpgradeRequired(
                    f'pandas>=1.5.0 is required for `to_pandas("use_pyarrow_extension_array=True")`, found Pandas {pd.__version__}'
                )
            if not _PYARROW_AVAILABLE or parse_version(pa.__version__) < (8, 0):
                raise ModuleUpgradeRequired(
                    f'pyarrow>=8.0.0 is required for `to_pandas("use_pyarrow_extension_array=True")`'
                    f", found pyarrow {pa.__version__!r}"
                    if _PYARROW_AVAILABLE
                    else ""
                )

        pd_series = (
            self.to_arrow().to_pandas(
                self_destruct=True,
                split_blocks=True,
                types_mapper=lambda pa_dtype: pd.ArrowDtype(pa_dtype),
                **kwargs,
            )
            if use_pyarrow_extension_array
            else self.to_arrow().to_pandas(**kwargs)
        )
        pd_series.name = self.name
        return pd_series

    def to_init_repr(self, n: int = 1000) -> str:
        """
        Convert this `Series` to an instantiatable string representation.

        Parameters
        ----------
        n
            Only use the first `n` elements.

        See Also
        --------
        polars.Series.to_init_repr
        polars.from_repr

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None, 4], dtype=pl.Int16)
        >>> print(s.to_init_repr())
        pl.Series("a", [1, 2, None, 4], dtype=pl.Int16)
        >>> s_from_str_repr = eval(s.to_init_repr())
        >>> s_from_str_repr
        shape: (4,)
        Series: 'a' [i16]
        [
            1
            2
            null
            4
        ]

        """
        return (
            f'pl.Series("{self.name}", {self.head(n).to_list()}, dtype=pl.{self.dtype})'
        )

    def count(self) -> int:
        """
        Get the number of non-`null` elements in this `Series`.

        See Also
        --------
        len

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None])
        >>> s.count()
        2
        """
        return self.len() - self.null_count()

    def len(self) -> int:
        """
        Get the number of elements in this `Series`, including `null` elements.

        See Also
        --------
        count

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None])
        >>> s.len()
        3
        """
        return self._s.len()

    def set(self, filter: Series, value: int | float | str | bool | None) -> Series:
        """
        Replace elements of this `Series` where a :class:`Boolean` mask is `True`.

        Parameters
        ----------
        filter
            A :class:`Boolean` mask.
        value
            A scalar value that will replace elements where `filter` is `True`.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc.). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set(s == 2, 10)
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().select(
        ...     pl.when(pl.col("a") == 2).then(10).otherwise(pl.col("a"))
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ 1       │
        │ 10      │
        │ 3       │
        └─────────┘

        """
        f = get_ffi_func("set_with_mask_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented
        return self._from_pyseries(f(filter._s, value))

    def scatter(
        self,
        indices: Series | np.ndarray[Any, Any] | Sequence[int] | int,
        values: (
            int
            | float
            | str
            | bool
            | date
            | datetime
            | Sequence[int]
            | Sequence[float]
            | Sequence[bool]
            | Sequence[str]
            | Sequence[date]
            | Sequence[datetime]
            | Series
            | None
        ),
    ) -> Series:
        """
        Replace elements of this `Series` at the specified indices.

        Parameters
        ----------
        indices
            The integer indices of elements to be replaced.
        values
            The values that will replace the elements at these indices. May be a scalar
            or a sequence of the same length as `indices`. If `values` is a sequence
            shorter than `indices`, only the first `len(values)` indices are replaced.
            If `values` is a sequence longer than `indices`, the extra values are
            ignored.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimization (predicate pushdown, etc.). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.scatter(1, 10)
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().with_row_count("row_nr").select(
        ...     pl.when(pl.col("row_nr") == 1).then(10).otherwise(pl.col("a"))
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ 1       │
        │ 10      │
        │ 3       │
        └─────────┘

        """
        if isinstance(indices, int):
            indices = [indices]
        if len(indices) == 0:
            return self

        indices = Series("", indices)
        if isinstance(values, (int, float, bool, str)) or (values is None):
            values = Series("", [values])

            # if we need to set more than a single value, we extend it
            if len(indices) > 0:
                values = values.extend_constant(values[0], len(indices) - 1)
        elif not isinstance(values, Series):
            values = Series("", values)
        self._s.scatter(indices._s, values._s)
        return self

    def clear(self, n: int = 0) -> Series:
        """
        Create an all-`null` `Series` of length `n` with the same name and dtype.

        With the default `n=0`, equivalent to
        `pl.Series(name=self.name, dtype=self.dtype)`.

        `n` can be greater than the current number of rows in `self`.

        Parameters
        ----------
        n
            The number of rows in the returned `Series`.

        See Also
        --------
        clone : A cheap deepcopy/clone.

        Examples
        --------
        >>> s = pl.Series("a", [None, True, False])
        >>> s.clear()
        shape: (0,)
        Series: 'a' [bool]
        [
        ]

        >>> s.clear(n=2)
        shape: (2,)
        Series: 'a' [bool]
        [
            null
            null
        ]

        """
        if n == 0:
            return self._from_pyseries(self._s.clear())
        s = (
            self.__class__(name=self.name, values=[], dtype=self.dtype)
            if len(self) > 0
            else self.clone()
        )
        return s.extend_constant(None, n=n) if n > 0 else s

    def clone(self) -> Self:
        """
        Create a copy of this `Series`.

        This is a cheap operation that does not copy the underlying data.

        See Also
        --------
        clear : Create an all-`null` `Series` of length `n` with the same name and
                dtype.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.clone()
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                2
                3
        ]

        """
        return self._from_pyseries(self._s.clone())

    def fill_nan(self, value: int | float | Expr | None) -> Series:
        """
        Fill floating-point `NaN` values with the specified `value`.

        Parameters
        ----------
        value
            The value to replace `NaN` values with.

        Returns
        -------
        Series
            A `Series` with `NaN` values replaced by the given value.

        Warnings
        --------
        Note that floating point `NaN` (Not a Number) is not a missing value!
        To replace missing values, use :func:`fill_null`.

        See Also
        --------
        fill_null

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, float("nan")])
        >>> s.fill_nan(0)
        shape: (4,)
        Series: 'a' [f64]
        [
                1.0
                2.0
                3.0
                0.0
        ]

        """

    def fill_null(
        self,
        value: Any | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Series:
        """
        Fill `null` values using the specified `value` or `strategy`.

        To fill `null` values via interpolation, see :func:`interpolate`.

        Parameters
        ----------
        value
            The value used to fill `null` values. Mutually exclusive with `strategy`.
        strategy : {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}
            The strategy used to fill `null` values. Mutually exclusive with `value`.
        limit
            The number of consecutive `null` values to fill when using the `"forward"`
            or `"backward"` strategy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, None])
        >>> s.fill_null(strategy="forward")
        shape: (4,)
        Series: 'a' [i64]
        [
            1
            2
            3
            3
        ]
        >>> s.fill_null(strategy="min")
        shape: (4,)
        Series: 'a' [i64]
        [
            1
            2
            3
            1
        ]
        >>> s = pl.Series("b", ["x", None, "z"])
        >>> s.fill_null(pl.lit(""))
        shape: (3,)
        Series: 'b' [str]
        [
            "x"
            ""
            "z"
        ]

        """

    def floor(self) -> Series:
        """
        Round each element down to the nearest integer.

        Only supported for :class:`Float32` and :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.floor()
        shape: (3,)
        Series: 'a' [f64]
        [
                1.0
                2.0
                3.0
        ]

        """

    def ceil(self) -> Series:
        """
        Round each element up to the nearest integer.

        Only supported for :class:`Float32` and :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.ceil()
        shape: (3,)
        Series: 'a' [f64]
        [
                2.0
                3.0
                4.0
        ]

        """

    def round(self, decimals: int = 0) -> Series:
        """
        Round each element to `decimals` decimal places.

        Only supported for :class:`Float32` and :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.round(2)
        shape: (3,)
        Series: 'a' [f64]
        [
                1.12
                2.57
                3.9
        ]

        Parameters
        ----------
        decimals
            The number of decimal places to round to.

        """

    def round_sig_figs(self, digits: int) -> Series:
        """
        Round each element to `digits` significant digits/figures.

        Only supported for :class:`Float32` and :class:`Float64` columns.

        Parameters
        ----------
        digits
            The number of significant figures to round to.

        Examples
        --------
        >>> s = pl.Series([0.01234, 3.333, 1234.0])
        >>> s.round_sig_figs(2)
        shape: (3,)
        Series: '' [f64]
        [
                0.012
                3.3
                1200.0
        ]

        """

    def dot(self, other: Series | ArrayLike) -> float | None:
        """
        Compute the dot/inner product with another `Series`.

        Parameters
        ----------
        other
            The `Series` (or array) to compute the dot product with.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4.0, 5.0, 6.0])
        >>> s.dot(s2)
        32.0

        """
        if not isinstance(other, Series):
            other = Series(other)
        if len(self) != len(other):
            n, m = len(self), len(other)
            raise ShapeError(f"Series length mismatch: expected {n!r}, found {m!r}")
        return self._s.dot(other._s)

    def mode(self) -> Series:
        """
        Compute the most commonly occurring value(s) in this `Series`.

        Can return multiple values.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.mode()
        shape: (1,)
        Series: 'a' [i64]
        [
                2
        ]

        """

    def sign(self) -> Series:
        """
        Indicate the sign of each element of this `Series`.

        The returned values can be `-1`, `0`, `1`, or `null`:

        * `-1` if the element is negative.
        * `0` if the element is zero.
        * `1` if the element is positive.
        * `null` if the element is `null`.

        Examples
        --------
        >>> s = pl.Series("a", [-9.0, -0.0, 0.0, 4.0, None])
        >>> s.sign()
        shape: (5,)
        Series: 'a' [i64]
        [
                -1
                0
                0
                1
                null
        ]

        """

    def sin(self) -> Series:
        """
        Compute the sine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.sin()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.0
            1.2246e-16
        ]

        """

    def cos(self) -> Series:
        """
        Compute the cosine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.cos()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.0
            6.1232e-17
            -1.0
        ]

        """

    def tan(self) -> Series:
        """
        Compute the tangent of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.tan()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.6331e16
            -1.2246e-16
        ]

        """

    def cot(self) -> Series:
        """
        Compute the cotangent of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.cot()
        shape: (3,)
        Series: 'a' [f64]
        [
            inf
            6.1232e-17
            -8.1656e15
        ]

        """

    def arcsin(self) -> Series:
        """
        Compute the inverse sine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arcsin()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.570796
            0.0
            -1.570796
        ]

        """

    def arccos(self) -> Series:
        """
        Compute the inverse cosine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arccos()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            1.570796
            3.141593
        ]

        """

    def arctan(self) -> Series:
        """
        Compute the inverse tangent of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arctan()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.785398
            0.0
            -0.785398
        ]

        """

    def arcsinh(self) -> Series:
        """
        Compute the inverse hyperbolic sine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arcsinh()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.881374
            0.0
            -0.881374
        ]

        """

    def arccosh(self) -> Series:
        """
        Compute the inverse hyperbolic cosine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [5.0, 1.0, 0.0, -1.0])
        >>> s.arccosh()
        shape: (4,)
        Series: 'a' [f64]
        [
            2.292432
            0.0
            NaN
            NaN
        ]

        """

    def arctanh(self) -> Series:
        """
        Compute the inverse hyperbolic tangent of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.1])
        >>> s.arctanh()
        shape: (7,)
        Series: 'a' [f64]
        [
            NaN
            inf
            0.549306
            0.0
            -0.549306
            -inf
            NaN
        ]

        """

    def sinh(self) -> Series:
        """
        Compute the hyperbolic sine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.sinh()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.175201
            0.0
            -1.175201
        ]

        """

    def cosh(self) -> Series:
        """
        Compute the hyperbolic cosine of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.cosh()
        shape: (3,)
        Series: 'a' [f64]
        [
            1.543081
            1.0
            1.543081
        ]

        """

    def tanh(self) -> Series:
        """
        Compute the hyperbolic tangent of each element of this `Series`.

        Returns
        -------
        Expr
            A :class:`Float64` `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.tanh()
        shape: (3,)
        Series: 'a' [f64]
        [
            0.761594
            0.0
            -0.761594
        ]

        """

    def map_elements(
        self,
        function: Callable[[Any], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        skip_nulls: bool = True,
    ) -> Self:
        """
        Apply a custom Python function to each element of this `Series`.

        The custom function must take and return a `Series` element.

        .. warning::
            This method is much slower than the native expressions API.
            Only use it if you cannot implement your logic otherwise.

        If the function returns a different data type, consider setting the
        `return_dtype` argument to avoid errors.

        Implementing logic using a Python function is almost always *significantly*
        slower and more memory intensive than implementing the same logic using
        the native expression API because:

        - The native expression engine runs in Rust; UDFs run in Python.
        - Use of Python UDFs forces the DataFrame to be materialized in memory.
        - Polars-native expressions can be parallelised (UDFs typically cannot).
        - Polars-native expressions can be logically optimised (UDFs cannot).

        Wherever possible you should strongly prefer the native expression API
        to achieve the best performance.

        Parameters
        ----------
        function
            The function or `Callable` to apply; must take and return a `Series`
            element.
        return_dtype
            The data type of the output `Series`. If not set, will be auto-inferred;
            this may lead to unexpected results.
        skip_nulls
            Whether to skip mapping the function over `null` values (this is faster).

        Notes
        -----
        If your function is slow and you don't want it to be called more than once
        for a given input, consider decorating it with `@lru_cache
        <https://docs.python.org/3/library/functools.html#functools.lru_cache>`_.
        If your data is suitable, you may achieve *significant* speedups.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.map_elements(lambda x: x + 10)  # doctest: +SKIP
        shape: (3,)
        Series: 'a' [i64]
        [
                11
                12
                13
        ]

        Returns
        -------
        Series

        """
        from polars.utils.udfs import warn_on_inefficient_map

        if return_dtype is None:
            pl_return_dtype = None
        else:
            pl_return_dtype = py_type_to_dtype(return_dtype)

        warn_on_inefficient_map(function, columns=[self.name], map_target="series")
        return self._from_pyseries(
            self._s.apply_lambda(function, pl_return_dtype, skip_nulls)
        )

    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Series:
        """
        Shift elements by the given number of indices.

        Parameters
        ----------
        n
            The number of indices to shift forward by. If negative, elements are shifted
            backward instead.
        fill_value
            Fill the resulting `null` values with this value. Accepts expression input.
            Non-expression inputs, including strings, are treated as literals.

        Notes
        -----
        This method is similar to the `LAG` operation in SQL when the value for `n`
        is positive. With a negative value for `n`, it is similar to `LEAD`.

        Examples
        --------
        By default, elements are shifted forward by one index:

        >>> s = pl.Series([1, 2, 3, 4])
        >>> s.shift()
        shape: (4,)
        Series: '' [i64]
        [
                null
                1
                2
                3
        ]

        Pass a negative value to shift backwards instead:

        >>> s.shift(-2)
        shape: (4,)
        Series: '' [i64]
        [
                3
                4
                null
                null
        ]

        Specify `fill_value` to fill the resulting `null` values:

        >>> s.shift(-2, fill_value=100)
        shape: (4,)
        Series: '' [i64]
        [
                3
                4
                100
                100
        ]

        """

    def zip_with(self, mask: Series, other: Series) -> Self:
        """
        Take values from `self` or `other` based on the given mask.

        Where `mask` is `True`, take values from self. Where mask is `False`,
        take values from `other`.

        Parameters
        ----------
        mask
            A :class:`Boolean` `Series`.
        other
            A `Series` of same type as `self`.

        Returns
        -------
        Series

        Examples
        --------
        >>> s1 = pl.Series([1, 2, 3, 4, 5])
        >>> s2 = pl.Series([5, 4, 3, 2, 1])
        >>> s1.zip_with(s1 < s2, s2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                2
                1
        ]
        >>> mask = pl.Series([True, False, True, False, True])
        >>> s1.zip_with(mask, s2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                4
                3
                2
                5
        ]

        """
        return self._from_pyseries(self._s.zip_with(mask._s, other._s))

    def rolling_min(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Get the rolling (moving) minimum of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to their minimum.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_min(window_size=3)
        shape: (5,)
        Series: 'a' [i64]
        [
            null
            null
            100
            200
            300
        ]

        """
        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_min(
                    window_size, weights, min_periods, center=center
                )
            )
            .to_series()
        )

    def rolling_max(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Get the rolling (moving) maximum of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to their maximum.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_max(window_size=2)
        shape: (5,)
        Series: 'a' [i64]
        [
            null
            200
            300
            400
            500
        ]

        """
        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_max(
                    window_size, weights, min_periods, center=center
                )
            )
            .to_series()
        )

    def rolling_mean(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Get the rolling (moving) mean of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to their mean.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_mean(window_size=2)
        shape: (5,)
        Series: 'a' [f64]
        [
            null
            150.0
            250.0
            350.0
            450.0
        ]

        """
        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_mean(
                    window_size, weights, min_periods, center=center
                )
            )
            .to_series()
        )

    def rolling_sum(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Get the rolling (moving) sum of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to their sum.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.rolling_sum(window_size=2)
        shape: (5,)
        Series: 'a' [i64]
        [
                null
                3
                5
                7
                9
        ]

        """
        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_sum(
                    window_size, weights, min_periods, center=center
                )
            )
            .to_series()
        )

    def rolling_std(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> Series:
        """
        Get the rolling (moving) standard deviation of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to their standard
        deviation.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.
        ddof
            "Delta Degrees of Freedom": the divisor for a length-`N` window is
            `N - ddof`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_std(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.0
                1.0
                1.527525
                2.0
        ]

        """
        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_std(
                    window_size, weights, min_periods, center=center, ddof=ddof
                )
            )
            .to_series()
        )

    def rolling_var(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> Series:
        """
        Get the rolling (moving) variance of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to their variance.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.
        ddof
            "Delta Degrees of Freedom": the divisor for a length-`N` window is
            `N - ddof`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_var(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.0
                1.0
                2.333333
                4.0
        ]

        """
        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_var(
                    window_size, weights, min_periods, center=center, ddof=ddof
                )
            )
            .to_series()
        )

    def rolling_map(
        self,
        function: Callable[[Series], Any],
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Apply a custom rolling (moving) aggregation to the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to a single value
        via the custom function.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        .. warning::
            Computing custom functions is extremely slow. Use specialized rolling
            functions such as :func:`Series.rolling_sum` if at all possible.

        Parameters
        ----------
        function
            A custom aggregation function.
        window_size
            The size of the rolling window. The window at a given row will include the
            row itself and the `window_size - 1` elements before it.
        weights
            A list of weights with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> from numpy import nansum
        >>> s = pl.Series([11.0, 2.0, 9.0, float("nan"), 8.0])
        >>> s.rolling_map(nansum, window_size=3)
        shape: (5,)
        Series: '' [f64]
        [
                null
                null
                22.0
                11.0
                17.0
        ]

        """

    def rolling_median(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Get the rolling (moving) median of the elements of this `Series`.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their median.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_median(window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                2.0
                3.0
                4.0
                6.0
        ]

        """
        if min_periods is None:
            min_periods = window_size

        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_median(
                    window_size, weights, min_periods, center=center
                )
            )
            .to_series()
        )

    def rolling_quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = "nearest",
        window_size: int = 2,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Get the specified rolling (moving) quantile of the values in this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will (optionally) be multiplied by the weights given by the
        `weights` vector. The resulting values will be aggregated to the specified
        quantile.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        quantile
            A quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            The interpolation method to use when the specified quantile falls between
            two values.
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_quantile(quantile=0.33, window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.0
                2.0
                3.0
                4.0
        ]
        >>> s.rolling_quantile(quantile=0.33, interpolation="linear", window_size=3)
        shape: (6,)
        Series: 'a' [f64]
        [
                null
                null
                1.66
                2.66
                3.66
                5.32
        ]

        """
        if min_periods is None:
            min_periods = window_size

        return (
            self.to_frame()
            .select(
                F.col(self.name).rolling_quantile(
                    quantile,
                    interpolation,
                    window_size,
                    weights,
                    min_periods,
                    center=center,
                )
            )
            .to_series()
        )

    def rolling_skew(self, window_size: int, *, bias: bool = True) -> Series:
        """
        Get the rolling (moving) skew (skewness) of the elements of this `Series`.

        A window of length `window_size` will traverse the `Series`. The values that
        fill this window will be aggregated to their skew.

        The window corresponding to a given element of the output will include the
        corresponding element of the input and the `window_size - 1` elements before it.
        This means that the first `window_size - 1` elements of the output will be
        `null`.

        Parameters
        ----------
        window_size
            The size of the rolling window.
        bias
            Whether to correct the skew calculation for statistical bias.

        Examples
        --------
        >>> pl.Series([1, 4, 2, 9]).rolling_skew(3)
        shape: (4,)
        Series: '' [f64]
        [
            null
            null
            0.381802
            0.47033
        ]

        Note how the values match

        >>> pl.Series([1, 4, 2]).skew(), pl.Series([4, 2, 9]).skew()
        (0.38180177416060584, 0.47033046033698594)

        """

    def sample(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Series:
        """
        Randomly sample elements from this `Series`.

        Parameters
        ----------
        n
            The number of elements to return. Cannot be used with `fraction`. Defaults
            to `1` if `fraction` is `None`.
        fraction
            The fraction of elements to return. Cannot be used with `n`.
        with_replacement
            Whether to allow elements to be sampled more than once.
        shuffle
            Whether to shuffle the order of the sampled elements. If `shuffle=False`
            (the default), the order will be neither stable nor fully random.
        seed
            The seed for the random number generator. If `seed=None` (the default), a
            random seed is generated anew for each `sample` operation. Set to an integer
            (e.g. `seed=0`) for fully reproducible results.

        Warnings
        --------
        `sample(fraction=1)` returns the expression as-is! To properly shuffle the
        values, use :func:`shuffle` (or add `shuffle=True`).

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.sample(2, seed=0)  # doctest: +IGNORE_RESULT
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            5
        ]

        """

    def peak_max(self) -> Self:
        """
        Get a :class:`Boolean` mask of which elements of this `Series` are local maxima.

        An element is a local maximum if it is larger than both the element immediately
        before it, and the element immediately after it.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.peak_max()
        shape: (5,)
        Series: 'a' [bool]
        [
                false
                false
                false
                false
                true
        ]

        """

    def peak_min(self) -> Self:
        """
        Get a :class:`Boolean` mask of which elements of this `Series` are local minima.

        An element is a local minimum if it is smaller than both the element immediately
        before it, and the element immediately after it.

        Examples
        --------
        >>> s = pl.Series("a", [4, 1, 3, 2, 5])
        >>> s.peak_min()
        shape: (5,)
        Series: 'a' [bool]
        [
            false
            true
            false
            true
            false
        ]

        """

    def n_unique(self) -> int:
        """
        Get the number of unique values in this `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.n_unique()
        3

        """
        return self._s.n_unique()

    def shrink_to_fit(self, *, in_place: bool = False) -> Series:
        """
        Reduce the memory usage of this `Series`.

        Shrinks the underlying array capacity of the `Series` to the exact amount
        needed to hold the data.

        (Note that this function does not change the data type of the `Series`.)

        """
        if in_place:
            self._s.shrink_to_fit()
            return self
        else:
            series = self.clone()
            series._s.shrink_to_fit()
            return series

    def hash(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> Series:
        """
        Hash the elements of this `Series`. The hash value is of type :class:`UInt64`.

        Parameters
        ----------
        seed
            Random seed parameter. Defaults to `0`.
        seed_1
            Random seed parameter. Defaults to `seed` if not set.
        seed_2
            Random seed parameter. Defaults to `seed` if not set.
        seed_3
            Random seed parameter. Defaults to `seed` if not set.

        Notes
        -----
        This implementation of `hash` does not guarantee stable results across
        Polars versions. Its stability is only guaranteed within a single version.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.hash(seed=42)  # doctest: +IGNORE_RESULT
        shape: (3,)
        Series: 'a' [u64]
        [
            10734580197236529959
            3022416320763508302
            13756996518000038261
        ]

        """

    def reinterpret(self, *, signed: bool = True) -> Series:
        """
        Reinterpret the underlying bits as a signed/unsigned integer.

        This operation is only allowed for 64-bit integers. For smaller integer dtypes,
        you can safely use the :func:`cast` operation.

        Parameters
        ----------
        signed
            Whether to reinterpret as :class:`Int64` rather than :class:`UInt64`.
        """

    def interpolate(self, method: InterpolationMethod = "linear") -> Series:
        """
        Fill `null` values using linear or nearest-neighbors interpolation.

        Parameters
        ----------
        method : {'linear', 'nearest'}
            The interpolation method to use.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None, None, 5])
        >>> s.interpolate()
        shape: (5,)
        Series: 'a' [f64]
        [
            1.0
            2.0
            3.0
            4.0
            5.0
        ]

        """

    def abs(self) -> Series:
        """
        Get the absolute value of each element.

        Same as `abs(self)`.

        Examples
        --------
        >>> s = pl.Series([1, -2, -3])
        >>> s.abs()
        shape: (3,)
        Series: '' [i64]
        [
            1
            2
            3
        ]
        """

    def rank(
        self,
        method: RankMethod = "average",
        *,
        descending: bool = False,
        seed: int | None = None,
    ) -> Series:
        """
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'dense', 'ordinal', 'random'}
            The method used to assign ranks to tied elements.
            The following methods are available (the default is `"average"`):

            - `"average"` : The average of the ranks that would have been assigned to
              all the tied values is assigned to each value.
            - `"min"` : The minimum of the ranks that would have been assigned to all
              the tied values is assigned to each value. (This is also referred to
              as "competition" ranking.)
            - `"max"` : The maximum of the ranks that would have been assigned to all
              the tied values is assigned to each value.
            - `"dense"` : Like `"min"`, but the rank of the next highest element is
              assigned the rank immediately after those assigned to the tied
              elements.
            - `"ordinal"` : All values are given a distinct rank, corresponding to
              the order that the values occur.
            - `"random"` : Like `"ordinal"`, but the rank for ties is not dependent
              on the order that the values occur.
        descending
            Whether to rank in descending instead of ascending order.
        seed
            If `method="random"`, use this as the random seed.

        Examples
        --------
        The `"average"` method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank()
        shape: (5,)
        Series: 'a' [f64]
        [
            3.0
            4.5
            1.5
            1.5
            4.5
        ]

        The `"ordinal"` method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank("ordinal")
        shape: (5,)
        Series: 'a' [u32]
        [
            3
            4
            1
            2
            5
        ]

        """

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Series:
        """
        Get the first discrete difference between shifted items.

        Parameters
        ----------
        n
            The number of items to shift by when calculating the difference.
        null_behavior : {'ignore', 'drop'}
            How to handle `null` values.

        Examples
        --------
        >>> s = pl.Series("s", values=[20, 10, 30, 25, 35], dtype=pl.Int8)
        >>> s.diff()
        shape: (5,)
        Series: 's' [i8]
        [
            null
            -10
            20
            -5
            10
        ]

        >>> s.diff(n=2)
        shape: (5,)
        Series: 's' [i8]
        [
            null
            null
            10
            15
            5
        ]

        >>> s.diff(n=2, null_behavior="drop")
        shape: (3,)
        Series: 's' [i8]
        [
            10
            15
            5
        ]

        """

    def pct_change(self, n: int | IntoExprColumn = 1) -> Series:
        """
        Get the percentage change between values `n` elements apart.

        Specifically, gets the percentage change (as a fraction) between the current
        element and the most-recent non-`null` element at least `n` period(s) before
        the current element.

        Gets the change from the previous element by default.

        Parameters
        ----------
        n
            The number of elements to shift by when calculating the percent change.

        Examples
        --------
        >>> pl.Series(range(10)).pct_change()
        shape: (10,)
        Series: '' [f64]
        [
            null
            inf
            1.0
            0.5
            0.333333
            0.25
            0.2
            0.166667
            0.142857
            0.125
        ]

        >>> pl.Series([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]).pct_change(2)
        shape: (10,)
        Series: '' [f64]
        [
            null
            null
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
        ]

        """

    def skew(self, *, bias: bool = True) -> float | None:
        r"""
        Get the skew (skewness) of the elements in this `Series`.

        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `scipy.stats.skewtest
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html>`_
        can be used to determine if the skewness value is close enough to zero,
        statistically speaking.

        See `scipy.stats.skew
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html>`_
        for more information.

        Parameters
        ----------
        bias : bool, optional
            Whether to correct the skew calculation for statistical bias.

        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.

        .. math:: g_1=\frac{m_3}{m_2^{3/2}}

        where

        .. math:: m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If `bias` is `False`, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math::
            G_1 = \frac{k_3}{k_2^{3/2}} = \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}

        Examples
        --------
        >>> s = pl.Series([1, 2, 2, 4, 5])
        >>> s.skew()
        0.34776706224699483

        """
        return self._s.skew(bias)

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> float | None:
        """
        Compute the kurtosis (Fisher or Pearson) of the elements in this `Series`.

        Kurtosis is the fourth central moment divided by the square of the
        variance. If Fisher's definition is used, then `3.0` is subtracted from
        the result to give a kurtosis of `0.0` for a standard normal distribution.
        If `bias=False`, the kurtosis is computed using k statistics to eliminate bias
        coming from biased moment estimators.

        See `scipy.stats.kurtosis
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html>`_
        for more information.

        Parameters
        ----------
        fisher : bool, optional
            Whether to use Fisher's definition of kurtosis (where a standard normal
            distribution has a kurtosis of `0.0`) instead of Pearson's definition
            (where a standard normal distribution has a kurtosis of `3.0`).
        bias : bool, optional
            Whether to correct the kurtosis calculation for statistical bias.

        """
        return self._s.kurtosis(fisher, bias)

    def clip(
        self,
        lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None = None,
        upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None = None,
    ) -> Series:
        """
        Set values outside the given boundaries to the boundary value.

        Parameters
        ----------
        lower_bound
            Lower bound. Accepts expression input.
            Non-expression inputs are parsed as literals.
            If `lower_bound=None` (the default), no lower bound is applied.
        upper_bound
            Upper bound. Accepts expression input.
            Non-expression inputs are parsed as literals.
            If `upper_bound=None` (the default), no upper bound is applied.

        See Also
        --------
        when

        Notes
        -----
        This method only works for numeric and temporal columns. To clip other data
        types, consider writing a `when-then-otherwise` expression. See :func:`when`.

        Examples
        --------
        Specifying both a lower and upper bound:

        >>> s = pl.Series([-50, 5, 50, None])
        >>> s.clip(1, 10)
        shape: (4,)
        Series: '' [i64]
        [
                1
                5
                10
                null
        ]

        Specifying only a single bound:

        >>> s.clip(upper_bound=10)
        shape: (4,)
        Series: '' [i64]
        [
                -50
                5
                10
                null
        ]

        """

    def lower_bound(self) -> Self:
        """
        Compute the lower bound of the dtype of this `Series`.

        Returns a length-1 `Series` containing the minimum value representable by the
        dtype of this `Series`.

        See Also
        --------
        upper_bound : compute the upper bound of the dtype of a `Series`.

        Examples
        --------
        >>> s = pl.Series("s", [-1, 0, 1], dtype=pl.Int32)
        >>> s.lower_bound()
        shape: (1,)
        Series: 's' [i32]
        [
            -2147483648
        ]

        >>> s = pl.Series("s", [1.0, 2.5, 3.0], dtype=pl.Float32)
        >>> s.lower_bound()
        shape: (1,)
        Series: 's' [f32]
        [
            -inf
        ]

        """

    def upper_bound(self) -> Self:
        """
        Compute the upper bound of the dtype of this `Series`.

        Returns a length-1 `Series` containing the maximum value representable by the
        dtype of this `Series`.

        See Also
        --------
        lower_bound : compute the lower  bound of the dtype of a `Series`.

        Examples
        --------
        >>> s = pl.Series("s", [-1, 0, 1], dtype=pl.Int8)
        >>> s.upper_bound()
        shape: (1,)
        Series: 's' [i8]
        [
            127
        ]

        >>> s = pl.Series("s", [1.0, 2.5, 3.0], dtype=pl.Float64)
        >>> s.upper_bound()
        shape: (1,)
        Series: 's' [f64]
        [
            inf
        ]

        """

    def replace(
        self,
        old: IntoExpr | Sequence[Any] | Mapping[Any, Any],
        new: IntoExpr | Sequence[Any] | NoDefault = no_default,
        *,
        default: IntoExpr | NoDefault = no_default,
        return_dtype: PolarsDataType | None = None,
    ) -> Self:
        """
        Replace values with other values.

        Parameters
        ----------
        old
            A value or sequence of values to replace with the values in `new`.
            Also accepts a mapping of values to their replacement as syntactic sugar for
            `replace(new=pl.Series(mapping.keys()), old=pl.Series(mapping.values()))`.
        new
            A value or sequence of values to replace by the values in `old` with.
            It must match the length of `old` or have length 1.
        default
            Set values that were not replaced to this value. Defaults to keeping the
            original value. Accepts expression input. Non-expression inputs are parsed
            as literals.
        return_dtype
            The data type of the output `Series`. If not set, will be auto-inferred.

        See Also
        --------
        str.replace

        Notes
        -----
        The global string cache must be enabled when replacing categorical values.

        Examples
        --------
        Replace a single value by another value. Values that were not replaced remain
        unchanged.

        >>> s = pl.Series([1, 2, 2, 3])
        >>> s.replace(2, 100)
        shape: (4,)
        Series: '' [i64]
        [
                1
                100
                100
                3
        ]

        Replace multiple values by passing sequences to the `old` and `new` parameters.

        >>> s.replace([2, 3], [100, 200])
        shape: (4,)
        Series: '' [i64]
        [
                1
                100
                100
                200
        ]

        Passing a mapping with replacements is also supported as syntactic sugar.
        Specify a default to set all values that were not matched.

        >>> mapping = {2: 100, 3: 200}
        >>> s.replace(mapping, default=-1)
        shape: (4,)
        Series: '' [i64]
        [
                -1
                100
                100
                200
        ]


        The default can be another `Series`.

        >>> default = pl.Series([2.5, 5.0, 7.5, 10.0])
        >>> s.replace(2, 100, default=default)
        shape: (4,)
        Series: '' [f64]
        [
                2.5
                100.0
                100.0
                10.0
        ]

        Replacing by values of a different data type sets the return type based on
        a combination of the `new` data type and either the `old` data type or the
        `default` data type if it was set.

        >>> s = pl.Series(["x", "y", "z"])
        >>> mapping = {"x": 1, "y": 2, "z": 3}
        >>> s.replace(mapping)
        shape: (3,)
        Series: '' [str]
        [
                "1"
                "2"
                "3"
        ]
        >>> s.replace(mapping, default=None)
        shape: (3,)
        Series: '' [i64]
        [
                1
                2
                3
        ]

        Set the `return_dtype` parameter to control the resulting data type directly.

        >>> s.replace(mapping, return_dtype=pl.UInt8)
        shape: (3,)
        Series: '' [u8]
        [
                1
                2
                3
        ]
        """

    def reshape(self, dimensions: tuple[int, ...]) -> Series:
        """
        Reshape to a flat `Series` or a :class:`List` `Series`.

        Parameters
        ----------
        dimensions
            A length-1 or length-2 tuple of the dimensions of each column:

            - Specify a length-1 tuple to explode :class:`List` columns to columns of
              length `dimensions[0]`. Non-:class:`List` columns will be left unchanged.
            - Specify a length-2 tuple to reshape either :class:`List` or
              non-:class:`List` columns to :class:`List` columns of length
              `dimensions[0]`, with `dimensions[1]` items per list.

            If a `-1` is given for one dimension, that dimension will be inferred.

        See Also
        --------
        Series.list.explode : Explode a :class:`List` column.

        Examples
        --------
        >>> s = pl.Series("foo", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> s.reshape((3, 3))
        shape: (3,)
        Series: 'foo' [list[i64]]
        [
                [1, 2, 3]
                [4, 5, 6]
                [7, 8, 9]
        ]

        """

    def shuffle(self, seed: int | None = None) -> Series:
        """
        Randomly shuffle the values in this `Series`.

        Parameters
        ----------
        seed
            The seed for the random number generator. If `seed=None` (the default), a
            random seed is generated anew for each `shuffle` operation. Set to an
            integer (e.g. `seed=0`) for fully reproducible results.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.shuffle(seed=1)
        shape: (3,)
        Series: 'a' [i64]
        [
                2
                1
                3
        ]

        """

    @deprecate_nonkeyword_arguments(version="0.19.10")
    def ewm_mean(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Series:
        r"""
        Get an exponentially-weighted moving average along this `Series`.

        Parameters
        ----------
        com
            Specify the exponential decay factor :math:`\alpha` in terms of the center
            of mass :math:`\gamma`:

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify the exponential decay factor :math:`\alpha` in terms of the span
            :math:`\theta`:

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify the exponential decay factor :math:`\alpha` in terms of the
            half-life :math:`\lambda`:

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify the exponential decay factor `alpha` directly:
            :math:`0 < \alpha \leq 1`.
        adjust
            Whether to divide by a decaying adjustment factor in the initial few periods
            to account for the imbalance in their relative weightings:

                - When `adjust=True` the exponential weighting function is computed
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When `adjust=False` the exponential weighting function is computed
                  recursively via:

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        min_periods
            The minimum number of observations in a window required to assign it a value
            (otherwise the result will be `null`).
        ignore_nulls
            Whether to ignore missing values when calculating weights.

                - When `ignore_nulls=False` (the default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if `adjust=True`, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if `adjust=False`.

                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if `adjust=True`,
                  and :math:`1-\alpha` and :math:`\alpha` if `adjust=False`.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.ewm_mean(com=1)
        shape: (3,)
        Series: '' [f64]
        [
                1.0
                1.666667
                2.428571
        ]

        """

    @deprecate_nonkeyword_arguments(version="0.19.10")
    def ewm_std(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Series:
        r"""
        Get an exponentially-weighted moving standard deviation along this `Series`.

        Parameters
        ----------
        com
            Specify the exponential decay factor :math:`\alpha` in terms of the center
            of mass :math:`\gamma`:

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify the exponential decay factor :math:`\alpha` in terms of the span
            :math:`\theta`:

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify the exponential decay factor :math:`\alpha` in terms of the
            half-life :math:`\lambda`:

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify the exponential decay factor `alpha` directly:
            :math:`0 < \alpha \leq 1`.
        adjust
            Whether to divide by a decaying adjustment factor in the initial few periods
            to account for the imbalance in their relative weightings:

                - When `adjust=True` the exponential weighting function is computed
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When `adjust=False` the exponential weighting function is computed
                  recursively via:

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            Whether to correct the standard deviation calculation for statistical bias.
        min_periods
            The minimum number of observations in a window required to assign it a value
            (otherwise the result will be `null`).
        ignore_nulls
            Whether to ignore missing values when calculating weights.

                - When `ignore_nulls=False` (the default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if `adjust=True`, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if `adjust=False`.

                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if `adjust=True`,
                  and :math:`1-\alpha` and :math:`\alpha` if `adjust=False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_std(com=1)
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            0.707107
            0.963624
        ]

        """

    @deprecate_nonkeyword_arguments(version="0.19.10")
    def ewm_var(
        self,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        bias: bool = False,
        min_periods: int = 1,
        ignore_nulls: bool = True,
    ) -> Series:
        r"""
        Get an exponentially-weighted moving variance along this `Series`.

        Parameters
        ----------
        com
            Specify the exponential decay factor :math:`\alpha` in terms of the center
            of mass :math:`\gamma`:

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify the exponential decay factor :math:`\alpha` in terms of the span
            :math:`\theta`:

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify the exponential decay factor :math:`\alpha` in terms of the
            half-life :math:`\lambda`:

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify the exponential decay factor `alpha` directly:
            :math:`0 < \alpha \leq 1`.
        adjust
            Whether to divide by a decaying adjustment factor in the initial few periods
            to account for the imbalance in their relative weightings:

                - When `adjust=True` the exponential weighting function is computed
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When `adjust=False` the exponential weighting function is computed
                  recursively via:

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            Whether to correct the standard deviation calculation for statistical bias.
        min_periods
            The minimum number of observations in a window required to assign it a value
            (otherwise the result will be `null`).
        ignore_nulls
            Whether to ignore missing values when calculating weights.

                - When `ignore_nulls=False` (the default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if `adjust=True`, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if `adjust=False`.

                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if `adjust=True`,
                  and :math:`1-\alpha` and :math:`\alpha` if `adjust=False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_var(com=1)
        shape: (3,)
        Series: 'a' [f64]
        [
            0.0
            0.5
            0.928571
        ]

        """

    def extend_constant(self, value: PythonLiteral | None, n: int) -> Series:
        """
        An extremely fast method for extending a `Series` with `n` copies of a value.

        Parameters
        ----------
        value
            A constant literal value (not an expression) with which to extend
            the `Series`; can pass `None` to extend with `null` values.
        n
            The number of additional values that will be added.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.extend_constant(99, n=2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                99
                99
        ]

        """

    def set_sorted(self, *, descending: bool = False) -> Self:
        """
        Flags the `Series` as sorted.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        descending
            Whether the columns are sorted in descending instead of ascending order.

        Warnings
        --------
        This can lead to incorrect results if this `Series` is not sorted.
        Use with care!

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_sorted().max()
        3

        """
        return self._from_pyseries(self._s.set_sorted_flag(descending))

    def new_from_index(self, index: int, length: int) -> Self:
        """
        Create a new `Series` by repeating the value at `self[index]` `length` times.

        Parameters
        ----------
        index
            The index of `self` containing the value to be repeated.
        length
            The number of times to repeat the value at `self[index]`.
        """
        return self._from_pyseries(self._s.new_from_index(index, length))

    def shrink_dtype(self) -> Series:
        """
        Shrink numeric columns to the smallest dtype able to represent them.

        This can be used to reduce memory pressure.
        """

    def get_chunks(self) -> list[Series]:
        """Get the chunks of this `Series` as a list of `Series`."""
        return self._s.get_chunks()

    def implode(self) -> Self:
        """Aggregate values into a :class:`List`."""

    @deprecate_renamed_function("map_elements", version="0.19.0")
    def apply(
        self,
        function: Callable[[Any], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        skip_nulls: bool = True,
    ) -> Self:
        """
        Apply a custom/user-defined function (UDF) over this `Series`' elements.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`Series.map_elements`.

        Parameters
        ----------
        function
            The function or `Callable` to apply; must take and return a `Series`
            element.
        return_dtype
            The data type of the output `Series`. If not set, will be auto-inferred.
        skip_nulls
            Whether to skip mapping the function over `null` values (this is faster).

        """
        return self.map_elements(function, return_dtype, skip_nulls=skip_nulls)

    @deprecate_renamed_function("rolling_map", version="0.19.0")
    def rolling_apply(
        self,
        function: Callable[[Series], Any],
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Series:
        """
        Apply a custom rolling window function.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`Series.rolling_map`.

        Parameters
        ----------
        function
            A custom aggregation function.
        window_size
            The size of the rolling window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-`null` before
            computing a result. If `None`, it will be set equal to:
            - `window_size`, if `window_size` is a fixed integer
            - `1`, if `window_size` is a dynamic temporal size
        center
            Whether to set the labels at the center of the window.

        """

    @deprecate_renamed_function("is_first_distinct", version="0.19.3")
    def is_first(self) -> Series:
        """
        Return a :class:`Boolean` mask of the first occurrence of each distinct value.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`Series.is_first_distinct`.

        Returns
        -------
        Series
            `Series` of data type :class:`Boolean`.

        """

    @deprecate_renamed_function("is_last_distinct", version="0.19.3")
    def is_last(self) -> Series:
        """
        Return a :class:`Boolean` mask of the last occurrence of each distinct value.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`Series.is_last_distinct`.

        Returns
        -------
        Series
            `Series` of data type :class:`Boolean`.

        """

    @deprecate_function("Use `clip` instead.", version="0.19.12")
    def clip_min(
        self, lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn
    ) -> Series:
        """
        Clip (limit) the values in an array to a `min` boundary.

        .. deprecated:: 0.19.12
            Use :func:`clip` instead.

        Parameters
        ----------
        lower_bound
            Lower bound.

        """

    @deprecate_function("Use `clip` instead.", version="0.19.12")
    def clip_max(
        self, upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn
    ) -> Series:
        """
        Clip (limit) the values in an array to a `max` boundary.

        .. deprecated:: 0.19.12
            Use :func:`clip` instead.

        Parameters
        ----------
        upper_bound
            Upper bound.

        """

    @deprecate_function("Use `shift` instead.", version="0.19.12")
    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift_and_fill(
        self,
        fill_value: int | Expr,
        *,
        n: int = 1,
    ) -> Series:
        """
        Shift values by `n` places, filling the resulting `null`s with `fill_value`.

        .. deprecated:: 0.19.12
            Use :func:`shift` instead.

        Parameters
        ----------
        fill_value
            Fill `null` values with the result of this expression.
        n
            Number of places to shift (may be negative).

        """

    @deprecate_function("Use `Series.dtype.is_float()` instead.", version="0.19.13")
    def is_float(self) -> bool:
        """
        Check if this `Series` has floating-point numbers.

        .. deprecated:: 0.19.13
            Use `Series.dtype.is_float()` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0])
        >>> s.is_float()  # doctest: +SKIP
        True

        """
        return self.dtype.is_float()

    @deprecate_function(
        "Use `Series.dtype.is_integer()` instead."
        " For signed/unsigned variants, use `Series.dtype.is_signed_integer()`"
        " or `Series.dtype.is_unsigned_integer()`.",
        version="0.19.13",
    )
    def is_integer(self, signed: bool | None = None) -> bool:
        """
        Check if this Series datatype is an integer (signed or unsigned).

        .. deprecated:: 0.19.13
            Use `Series.dtype.is_integer()` instead.
            For signed/unsigned variants, use `Series.dtype.is_signed_integer()`
            or `Series.dtype.is_unsigned_integer()`.

        Parameters
        ----------
        signed
            * if `None`, both signed and unsigned integer dtypes will match.
            * if `True`, only signed integer dtypes will be considered a match.
            * if `False`, only unsigned integer dtypes will be considered a match.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3], dtype=pl.UInt32)
        >>> s.is_integer()  # doctest: +SKIP
        True
        >>> s.is_integer(signed=False)  # doctest: +SKIP
        True
        >>> s.is_integer(signed=True)  # doctest: +SKIP
        False

        """
        if signed is None:
            return self.dtype.is_integer()
        elif signed is True:
            return self.dtype.is_signed_integer()
        elif signed is False:
            return self.dtype.is_unsigned_integer()

        raise ValueError(f"`signed` must be None, True or False; got {signed!r}")

    @deprecate_function("Use `Series.dtype.is_numeric()` instead.", version="0.19.13")
    def is_numeric(self) -> bool:
        """
        Check if this `Series`' datatype is numeric.

        .. deprecated:: 0.19.13
            Use `Series.dtype.is_numeric()` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.is_numeric()  # doctest: +SKIP
        True

        """
        return self.dtype.is_numeric()

    @deprecate_function("Use `Series.dtype.is_temporal()` instead.", version="0.19.13")
    def is_temporal(self, excluding: OneOrMoreDataTypes | None = None) -> bool:
        """
        Check if this `Series`' datatype is temporal.

        .. deprecated:: 0.19.13
            Use `Series.dtype.is_temporal()` instead.

        Parameters
        ----------
        excluding
            Optionally exclude one or more temporal dtypes from matching.

        Examples
        --------
        >>> from datetime import date
        >>> s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
        >>> s.is_temporal()  # doctest: +SKIP
        True
        >>> s.is_temporal(excluding=[pl.Date])  # doctest: +SKIP
        False

        """
        if excluding is not None:
            if not isinstance(excluding, Iterable):
                excluding = [excluding]
            if self.dtype in excluding:
                return False

        return self.dtype.is_temporal()

    @deprecate_function("Use `Series.dtype == pl.Boolean` instead.", version="0.19.14")
    def is_boolean(self) -> bool:
        """
        Check if this `Series`' datatype is :class:`Boolean`.

        .. deprecated:: 0.19.14
            Use `Series.dtype == pl.Boolean` instead.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s.is_boolean()  # doctest: +SKIP
        True

        """
        return self.dtype == Boolean

    @deprecate_function("Use `Series.dtype == pl.String` instead.", version="0.19.14")
    def is_utf8(self) -> bool:
        """
        Check if this `Series`' datatype is :class:`String`.

        .. deprecated:: 0.19.14
            Use `Series.dtype == pl.String` instead.

        Examples
        --------
        >>> s = pl.Series("x", ["a", "b", "c"])
        >>> s.is_utf8()  # doctest: +SKIP
        True

        """
        return self.dtype == String

    @deprecate_renamed_function("gather_every", version="0.19.14")
    def take_every(self, n: int, offset: int = 0) -> Series:
        """
        Return every nth value in the `Series` as a new `Series`.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`gather_every`.

        Parameters
        ----------
        n
            Gather every `n`-th row.
        offset
            Starting index.
        """
        return self.gather_every(n, offset)

    @deprecate_renamed_function("gather", version="0.19.14")
    def take(
        self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]
    ) -> Series:
        """
        Take values by index.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`gather`.

        Parameters
        ----------
        indices
            Index location used for selection.
        """
        return self.gather(indices)

    @deprecate_renamed_function("scatter", version="0.19.14")
    @deprecate_renamed_parameter("idx", "indices", version="0.19.14")
    @deprecate_renamed_parameter("value", "values", version="0.19.14")
    def set_at_idx(
        self,
        indices: Series | np.ndarray[Any, Any] | Sequence[int] | int,
        values: (
            int
            | float
            | str
            | bool
            | date
            | datetime
            | Sequence[int]
            | Sequence[float]
            | Sequence[bool]
            | Sequence[str]
            | Sequence[date]
            | Sequence[datetime]
            | Series
            | None
        ),
    ) -> Series:
        """
        Set values at the index locations.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`scatter`.

        Parameters
        ----------
        indices
            Integers representing the index locations.
        values
            Replacement values.
        """
        return self.scatter(indices, values)

    @deprecate_renamed_function("cum_sum", version="0.19.14")
    def cumsum(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative sum computed at every element.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`cum_sum`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).

        """
        return self.cum_sum(reverse=reverse)

    @deprecate_renamed_function("cum_max", version="0.19.14")
    def cummax(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative max computed at every element.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`cum_max`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).
        """
        return self.cum_max(reverse=reverse)

    @deprecate_renamed_function("cum_min", version="0.19.14")
    def cummin(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative min computed at every element.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`cum_min`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).
        """
        return self.cum_min(reverse=reverse)

    @deprecate_renamed_function("cum_prod", version="0.19.14")
    def cumprod(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative product computed at every element.

        .. deprecated:: 0.19.14
            This method has been renamed to :meth:`cum_prod`.

        Parameters
        ----------
        reverse
            Whether to accumulate from the top (if `False`) or bottom (if `True`).
        """
        return self.cum_prod(reverse=reverse)

    @deprecate_function(
        "Use `Series.to_numpy(zero_copy_only=True) instead.", version="0.19.14"
    )
    def view(self, *, ignore_nulls: bool = False) -> SeriesView:
        """
        Get a view into this Series data with a numpy array.

        .. deprecated:: 0.19.14
            This method will be removed in a future version.

        This operation doesn't clone data, but does not include missing values.
        Don't use this unless you know what you are doing.

        Parameters
        ----------
        ignore_nulls
            If `True`, `null` values are converted to 0.
            If `False`, an exception is raised if `null` values are present.

        """
        return self._view(ignore_nulls=ignore_nulls)

    @deprecate_function(
        "It has been renamed to `replace`."
        " The default behavior has changed to keep any values not present in the mapping unchanged."
        " Pass `default=None` to keep existing behavior.",
        version="0.19.16",
    )
    @deprecate_renamed_parameter("remapping", "mapping", version="0.19.16")
    def map_dict(
        self,
        mapping: dict[Any, Any],
        *,
        default: Any = None,
        return_dtype: PolarsDataType | None = None,
    ) -> Self:
        """
        Replace values in the Series using a remapping dictionary.

        .. deprecated:: 0.19.16
            This method has been renamed to :meth:`replace`. The default behavior
            has changed to keep any values not present in the mapping unchanged.
            Pass `default=None` to keep existing behavior.

        Parameters
        ----------
        mapping
            Dictionary containing the before/after values to map.
        default
            Value to use when the remapping dict does not contain the lookup value.
            Use `pl.first()`, to keep the original value.
        return_dtype
            The data type of the output `Series`. If not set, will be auto-inferred.
        """
        return self.replace(mapping, default=default, return_dtype=return_dtype)

    @deprecate_renamed_function("equals", version="0.19.16")
    def series_equal(
        self, other: Series, *, null_equal: bool = True, strict: bool = False
    ) -> bool:
        """
        Check whether the Series is equal to another Series.

        .. deprecated:: 0.19.16
            This method has been renamed to :meth:`equals`.

        Parameters
        ----------
        other
            Series to compare with.
        null_equal
            Consider null values as equal.
        strict
            Don't allow different numerical dtypes, e.g. comparing `pl.UInt32` with a
            :class:`Int64` will return `False`.
        """
        return self.equals(other, null_equal=null_equal, strict=strict)

    # Keep the `list` and `str` properties below at the end of the definition of Series,
    # as to not confuse mypy with the type annotation `str` and `list`

    @property
    def bin(self) -> BinaryNameSpace:
        """Create an object namespace of all binary-related methods."""
        return BinaryNameSpace(self)

    @property
    def cat(self) -> CatNameSpace:
        """Create an object namespace of all categorical-related methods."""
        return CatNameSpace(self)

    @property
    def dt(self) -> DateTimeNameSpace:
        """Create an object namespace of all datetime-related methods."""
        return DateTimeNameSpace(self)

    @property
    def list(self) -> ListNameSpace:
        """Create an object namespace of all list-related methods."""
        return ListNameSpace(self)

    @property
    def arr(self) -> ArrayNameSpace:
        """Create an object namespace of all array-related methods."""
        return ArrayNameSpace(self)

    @property
    def str(self) -> StringNameSpace:
        """Create an object namespace of all string-related methods."""
        return StringNameSpace(self)

    @property
    def struct(self) -> StructNameSpace:
        """Create an object namespace of all struct-related methods."""
        return StructNameSpace(self)

    @property
    def plot(self) -> Any:
        """
        Create a plot namespace.

        Polars does not implement plotting logic itself, but instead defers to
        hvplot. Please see the `hvplot reference gallery <https://hvplot.holoviz.org/reference/index.html>`_
        for more information and documentation.

        Examples
        --------
        Histogram:

        >>> s = pl.Series([1, 4, 2])
        >>> s.plot.hist()  # doctest: +SKIP

        KDE plot (note: in addition to ``hvplot``, this one also requires ``scipy``):

        >>> s.plot.kde()  # doctest: +SKIP

        For more info on what you can pass, you can use ``hvplot.help``:

        >>> import hvplot  # doctest: +SKIP
        >>> hvplot.help("hist")  # doctest: +SKIP
        """
        if not _HVPLOT_AVAILABLE or parse_version(hvplot.__version__) < parse_version(
            "0.9.1"
        ):
            raise ModuleUpgradeRequired("hvplot>=0.9.1 is required for `.plot`")
        hvplot.post_patch()
        return hvplot.plotting.core.hvPlotTabularPolars(self)


def _resolve_temporal_dtype(
    dtype: PolarsDataType | None,
    ndtype: np.dtype[np.datetime64] | np.dtype[np.timedelta64],
) -> PolarsDataType | None:
    """Given polars/numpy temporal dtypes, resolve to an explicit unit."""
    PolarsType = Duration if ndtype.type == np.timedelta64 else Datetime
    if dtype is None or (dtype == Datetime and not getattr(dtype, "time_unit", None)):
        time_unit = getattr(dtype, "time_unit", None) or np.datetime_data(ndtype)[0]
        # explicit formulation is verbose, but keeps mypy happy
        # (and avoids unsupported timeunits such as "s")
        if time_unit == "ns":
            dtype = PolarsType("ns")
        elif time_unit == "us":
            dtype = PolarsType("us")
        elif time_unit == "ms":
            dtype = PolarsType("ms")
        elif time_unit == "D" and ndtype.type == np.datetime64:
            dtype = Date
    return dtype
