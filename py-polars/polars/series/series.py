from __future__ import annotations

import contextlib
import math
import os
import typing
import warnings
from datetime import date, datetime, time, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Generator,
    Iterable,
    NoReturn,
    Sequence,
    Union,
    overload,
)

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    SIGNED_INTEGER_DTYPES,
    TEMPORAL_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    Array,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Object,
    Time,
    UInt32,
    UInt64,
    Unknown,
    Utf8,
    dtype_to_ctype,
    is_polars_dtype,
    maybe_cast,
    numpy_char_code_to_dtype,
    py_type_to_dtype,
    supported_numpy_char_code,
)
from polars.dependencies import (
    _PYARROW_AVAILABLE,
    _check_for_numpy,
    _check_for_pandas,
    _check_for_pyarrow,
)
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import ShapeError
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
)
from polars.utils.decorators import deprecated_alias
from polars.utils.meta import get_index_type
from polars.utils.various import (
    _is_generator,
    find_stacklevel,
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

    from polars import DataFrame, Expr
    from polars.series._numpy import SeriesView
    from polars.type_aliases import (
        ClosedInterval,
        ComparisonOperator,
        FillNullStrategy,
        InterpolationMethod,
        IntoExpr,
        NullBehavior,
        OneOrMoreDataTypes,
        PolarsDataType,
        PythonLiteral,
        RankMethod,
        RollingInterpolationMethod,
        SearchSortedSide,
        SizeUnit,
        TimeUnit,
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
    "np.ndarray",
    "pd.Series",
    "pd.DatetimeIndex",
]


@expr_dispatch
class Series:
    """
    A Series represents a single column in a polars DataFrame.

    Parameters
    ----------
    name : str, default None
        Name of the series. Will be used as a column name when used in a DataFrame.
        When not specified, name is set to an empty string.
    values : ArrayLike, default None
        One-dimensional data in various forms. Supported are: Sequence, Series,
        pyarrow Array, and numpy ndarray.
    dtype : DataType, default None
        Polars dtype of the Series data. If not specified, the dtype is inferred.
    strict
        Throw error on numeric overflow.
    nan_to_null
        In case a numpy array is used to create this Series, indicate how to deal
        with np.nan values. (This parameter is a no-op on non-numpy data).
    dtype_if_empty=dtype_if_empty : DataType, default None
        If no dtype is specified and values contains None, an empty list, or a
        list with only None values, set the Polars dtype of the Series data.
        If not specified, Float32 is used in those cases.

    Examples
    --------
    Constructing a Series by specifying name and values positionally:

    >>> s = pl.Series("a", [1, 2, 3])
    >>> s
    shape: (3,)
    Series: 'a' [i64]
    [
            1
            2
            3
    ]

    Notice that the dtype is automatically inferred as a polars Int64:

    >>> s.dtype
    Int64

    Constructing a Series with a specific dtype:

    >>> s2 = pl.Series("a", [1, 2, 3], dtype=pl.Float32)
    >>> s2
    shape: (3,)
    Series: 'a' [f32]
    [
        1.0
        2.0
        3.0
    ]

    It is possible to construct a Series with values as the first positional argument.
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
    _accessors: set[str] = {"arr", "cat", "dt", "list", "str", "bin", "struct"}

    def __init__(
        self,
        name: str | ArrayLike | None = None,
        values: ArrayLike | None = None,
        dtype: PolarsDataType | None = None,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
        dtype_if_empty: PolarsDataType | None = None,
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
                f"Given dtype: '{dtype}' is not a valid Polars data type and cannot be converted into one."
            )

        # Handle case where values are passed as the first argument
        if name is None:
            name = ""
        elif not isinstance(name, str):
            if values is None:
                values = name
                name = ""
            else:
                raise ValueError("Series name must be a string.")

        if values is None:
            self._s = sequence_to_pyseries(
                name, [], dtype=dtype, dtype_if_empty=dtype_if_empty
            )
        elif isinstance(values, Series):
            self._s = series_to_pyseries(name, values)

        elif isinstance(values, range):
            self._s = range_to_series(name, values, dtype=dtype)._s

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
            self._s = numpy_to_pyseries(name, values, strict, nan_to_null)
            if values.dtype.type == np.datetime64:
                # cast to appropriate dtype, handling NaT values
                dtype = _resolve_datetime_dtype(dtype, values.dtype)
                if dtype is not None:
                    self._s = (
                        self.cast(dtype)
                        .set_at_idx(np.argwhere(np.isnat(values)).flatten(), None)
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
                strict=strict,
                dtype_if_empty=dtype_if_empty,
            )
        else:
            raise ValueError(
                f"Series constructor called with unsupported type; got {type(values)}"
            )

    @classmethod
    def _from_pyseries(cls, pyseries: PySeries) -> Self:
        series = cls.__new__(cls)
        series._s = pyseries
        return series

    @classmethod
    def _from_arrow(cls, name: str, values: pa.Array, *, rechunk: bool = True) -> Self:
        """Construct a Series from an Arrow Array."""
        return cls._from_pyseries(arrow_to_pyseries(name, values, rechunk))

    @classmethod
    def _from_pandas(
        cls,
        name: str,
        values: pd.Series | pd.DatetimeIndex,
        *,
        nan_to_null: bool = True,
    ) -> Self:
        """Construct a Series from a pandas Series or DatetimeIndex."""
        return cls._from_pyseries(
            pandas_to_pyseries(name, values, nan_to_null=nan_to_null)
        )

    def _get_ptr(self) -> int:
        """
        Get a pointer to the start of the values buffer of a numeric Series.

        This will raise an error if the
        ``Series`` contains multiple chunks

        """
        return self._s.get_ptr()

    @property
    def dtype(self) -> PolarsDataType:
        """
        Get the data type of this Series.

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
        Get flags that are set on the Series.

        Returns
        -------
        Dictionary containing the flag name and the value

        """
        out = {
            "SORTED_ASC": self._s.is_sorted_ascending_flag(),
            "SORTED_DESC": self._s.is_sorted_descending_flag(),
        }
        if self.dtype == List:
            out["FAST_EXPLODE"] = self._s.can_fast_explode_flag()
        return out

    @property
    def inner_dtype(self) -> PolarsDataType | None:
        """
        Get the inner dtype in of a List typed Series.

        Returns
        -------
        DataType

        """
        return self._s.inner_dtype()

    @property
    def name(self) -> str:
        """Get the name of this Series."""
        return self._s.name()

    @property
    def shape(self) -> tuple[int]:
        """Shape of this Series."""
        return (self._s.len(),)

    @property
    def time_unit(self) -> TimeUnit | None:
        """Get the time unit of underlying Datetime Series as {"ns", "us", "ms"}."""
        warnings.warn(
            "`Series.time_unit` is deprecated and will be removed in a future version,"
            " please use `Series.dtype.time_unit` instead",
            category=DeprecationWarning,
            stacklevel=find_stacklevel(),
        )
        return self._s.time_unit()

    def __bool__(self) -> NoReturn:
        raise ValueError(
            "The truth value of a Series is ambiguous. Hint: use '&' or '|' to chain "
            "Series boolean results together, not and/or; to check if a Series "
            "contains any values, use 'is_empty()'"
        )

    def __getstate__(self) -> Any:
        return self._s.__getstate__()

    def __setstate__(self, state: Any) -> None:
        self._s = sequence_to_pyseries("", [], Float32)
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
        return self.__and__(other)

    def __or__(self, other: Series) -> Self:
        if not isinstance(other, Series):
            other = Series([other])
        return self._from_pyseries(self._s.bitor(other._s))

    def __ror__(self, other: Series) -> Self:
        return self.__or__(other)

    def __xor__(self, other: Series) -> Self:
        if not isinstance(other, Series):
            other = Series([other])
        return self._from_pyseries(self._s.bitxor(other._s))

    def __rxor__(self, other: Series) -> Series:
        return self.__xor__(other)

    def _comp(self, other: Any, op: ComparisonOperator) -> Self:
        # special edge-case; boolean broadcast series (eq/neq) is its own result
        if self.dtype == Boolean and isinstance(other, bool) and op in ("eq", "neq"):
            if (other is True and op == "eq") or (other is False and op == "neq"):
                return self.clone()
            elif (other is False and op == "eq") or (other is True and op == "neq"):
                return ~self

        if isinstance(other, datetime) and self.dtype == Datetime:
            ts = _datetime_to_pl_timestamp(other, self._s.time_unit())
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return self._from_pyseries(f(ts))
        elif isinstance(other, time) and self.dtype == Time:
            d = _time_to_pl_time(other)
            f = get_ffi_func(op + "_<>", Int64, self._s)
            assert f is not None
            return self._from_pyseries(f(d))
        elif isinstance(other, date) and self.dtype == Date:
            d = _date_to_pl_date(other)
            f = get_ffi_func(op + "_<>", Int32, self._s)
            assert f is not None
            return self._from_pyseries(f(d))
        elif self.dtype == Categorical and not isinstance(other, Series):
            other = Series([other])

        if isinstance(other, Sequence) and not isinstance(other, str):
            other = Series("", other, dtype_if_empty=self.dtype)
        if isinstance(other, Series):
            return self._from_pyseries(getattr(self._s, op)(other._s))

        if other is not None:
            other = maybe_cast(other, self.dtype, self._s.time_unit())
        f = get_ffi_func(op + "_<>", self.dtype, self._s)
        if f is None:
            return NotImplemented

        return self._from_pyseries(f(other))

    @overload  # type: ignore[override]
    def __eq__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __eq__(self, other: Any) -> Self:
        ...

    def __eq__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__eq__(other)
        return self._comp(other, "eq")

    @overload  # type: ignore[override]
    def __ne__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __ne__(self, other: Any) -> Self:
        ...

    def __ne__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__ne__(other)
        return self._comp(other, "neq")

    @overload
    def __gt__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __gt__(self, other: Any) -> Self:
        ...

    def __gt__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__gt__(other)
        return self._comp(other, "gt")

    @overload
    def __lt__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __lt__(self, other: Any) -> Self:
        ...

    def __lt__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__lt__(other)
        return self._comp(other, "lt")

    @overload
    def __ge__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __ge__(self, other: Any) -> Self:
        ...

    def __ge__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__ge__(other)
        return self._comp(other, "gt_eq")

    @overload
    def __le__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __le__(self, other: Any) -> Self:
        ...

    def __le__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__le__(other)
        return self._comp(other, "lt_eq")

    def le(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series <= other``."""
        return self.__le__(other)

    def lt(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series < other``."""
        return self.__lt__(other)

    def eq(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series == other``."""
        return self.__eq__(other)

    @overload
    def eq_missing(self, other: Any) -> Self:
        ...

    @overload
    def eq_missing(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    def eq_missing(self, other: Any) -> Self | Expr:
        """
        Method equivalent of equality operator ``expr == other`` where `None` == None`.

        This differs from default ``ne`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        """

    def ne(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series != other``."""
        return self.__ne__(other)

    @overload
    def ne_missing(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def ne_missing(self, other: Any) -> Self:
        ...

    def ne_missing(self, other: Any) -> Self | Expr:
        """
        Method equivalent of equality operator ``expr != other`` where `None` == None`.

        This differs from default ``ne`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        """

    def ge(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series >= other``."""
        return self.__ge__(other)

    def gt(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series > other``."""
        return self.__gt__(other)

    def _arithmetic(self, other: Any, op_s: str, op_ffi: str) -> Self:
        if isinstance(other, pl.Expr):
            # expand pl.lit, pl.datetime, pl.duration Exprs to compatible Series
            other = self.to_frame().select(other).to_series()
        if isinstance(other, Series):
            return self._from_pyseries(getattr(self._s, op_s)(other._s))
        if _check_for_numpy(other) and isinstance(other, np.ndarray):
            return self._from_pyseries(getattr(self._s, op_s)(Series(other)._s))
        if (
            isinstance(other, (float, date, datetime, timedelta, str))
            and not self.is_float()
        ):
            _s = sequence_to_pyseries(self.name, [other])
            if "rhs" in op_ffi:
                return self._from_pyseries(getattr(_s, op_s)(self._s))
            else:
                return self._from_pyseries(getattr(self._s, op_s)(_s))
        else:
            other = maybe_cast(other, self.dtype, self._s.time_unit())
            f = get_ffi_func(op_ffi, self.dtype, self._s)
        if f is None:
            raise ValueError(
                f"cannot do arithmetic with series of dtype: {self.dtype} and argument"
                f" of type: {type(other)}"
            )
        return self._from_pyseries(f(other))

    @overload
    def __add__(self, other: DataFrame) -> DataFrame:  # type: ignore[misc]
        ...

    @overload
    def __add__(self, other: Expr) -> Expr:  # type: ignore[misc]
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
    def __sub__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __sub__(self, other: Any) -> Self:
        ...

    def __sub__(self, other: Any) -> Self | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) - other
        return self._arithmetic(other, "sub", "sub_<>")

    @overload
    def __truediv__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __truediv__(self, other: Any) -> Series:
        ...

    def __truediv__(self, other: Any) -> Series | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) / other
        if self.is_temporal():
            raise ValueError("first cast to integer before dividing datelike dtypes")

        # this branch is exactly the floordiv function without rounding the floats
        if self.is_float() or self.dtype == Decimal:
            return self._arithmetic(other, "div", "div_<>")

        return self.cast(Float64) / other

    # python 3.7 is not happy. Remove this when we finally ditch that
    @typing.no_type_check
    def __floordiv__(self, other: Any) -> Series:
        if isinstance(other, pl.Expr):
            return F.lit(self).__floordiv__(other)
        if self.is_temporal():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        if not isinstance(other, pl.Expr):
            other = F.lit(other)
        return self.to_frame().select(F.col(self.name) // other).to_series()

    def __invert__(self) -> Self:
        if self.dtype == Boolean:
            return self._from_pyseries(self._s._not())
        return NotImplemented

    @overload
    def __mul__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __mul__(self, other: DataFrame) -> DataFrame:  # type: ignore[misc]
        ...

    @overload
    def __mul__(self, other: Any) -> Series:
        ...

    def __mul__(self, other: Any) -> Series | DataFrame | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self) * other
        if self.is_temporal():
            raise ValueError("first cast to integer before multiplying datelike dtypes")
        elif isinstance(other, pl.DataFrame):
            return other * self
        else:
            return self._arithmetic(other, "mul", "mul_<>")

    @overload
    def __mod__(self, other: Expr) -> Expr:  # type: ignore[misc]
        ...

    @overload
    def __mod__(self, other: Any) -> Series:
        ...

    def __mod__(self, other: Any) -> Series | Expr:
        if isinstance(other, pl.Expr):
            return F.lit(self).__mod__(other)
        if self.is_temporal():
            raise ValueError(
                "first cast to integer before applying modulo on datelike dtypes"
            )
        return self._arithmetic(other, "rem", "rem_<>")

    def __rmod__(self, other: Any) -> Series:
        if self.is_temporal():
            raise ValueError(
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
        if self.is_temporal():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        if self.is_float():
            self.__rfloordiv__(other)

        if isinstance(other, int):
            other = float(other)
        return self.cast(Float64).__rfloordiv__(other)

    def __rfloordiv__(self, other: Any) -> Series:
        if self.is_temporal():
            raise ValueError("first cast to integer before dividing datelike dtypes")
        return self._arithmetic(other, "div", "div_<>_rhs")

    def __rmul__(self, other: Any) -> Series:
        if self.is_temporal():
            raise ValueError("first cast to integer before multiplying datelike dtypes")
        return self._arithmetic(other, "mul", "mul_<>")

    def __pow__(self, exponent: int | float | Series) -> Series:
        return self.pow(exponent)

    def __rpow__(self, other: Any) -> Series:
        if self.is_temporal():
            raise ValueError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        return self.to_frame().select(other ** F.col(self.name)).to_series()

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

    def __iter__(self) -> Generator[Any, None, None]:
        if self.dtype == List:
            # TODO: either make a change and return py-native list data here, or find
            #  a faster way to return nested/List series; sequential 'get_idx' calls
            #  make this path a lot slower (~10x) than it needs to be.
            get_idx = self._s.get_idx
            for idx in range(0, self.len()):
                yield get_idx(idx)
        else:
            buffer_size = 25_000
            for offset in range(0, self.len(), buffer_size):
                yield from self.slice(offset, buffer_size).to_list()

    def _pos_idxs(self, size: int) -> Series:
        # Unsigned or signed Series (ordered from fastest to slowest).
        #   - pl.UInt32 (polars) or pl.UInt64 (polars_u64_idx) Series indexes.
        #   - Other unsigned Series indexes are converted to pl.UInt32 (polars)
        #     or pl.UInt64 (polars_u64_idx).
        #   - Signed Series indexes are converted pl.UInt32 (polars) or
        #     pl.UInt64 (polars_u64_idx) after negative indexes are converted
        #     to absolute indexes.

        # pl.UInt32 (polars) or pl.UInt64 (polars_u64_idx).
        idx_type = get_index_type()

        if self.dtype == idx_type:
            return self

        if self.dtype not in INTEGER_DTYPES:
            raise NotImplementedError("Unsupported idxs datatype.")

        if self.len() == 0:
            return Series(self.name, [], dtype=idx_type)

        if idx_type == UInt32:
            if self.dtype in {Int64, UInt64}:
                if self.max() >= 2**32:  # type: ignore[operator]
                    raise ValueError("Index positions should be smaller than 2^32.")
            if self.dtype == Int64:
                if self.min() < -(2**32):  # type: ignore[operator]
                    raise ValueError("Index positions should be bigger than -2^32 + 1.")

        if self.dtype in SIGNED_INTEGER_DTYPES:
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
        if isinstance(item, Series) and item.dtype in INTEGER_DTYPES:
            return self._take_with_series(item._pos_idxs(self.len()))

        elif _check_for_numpy(item) and isinstance(item, np.ndarray):
            return self._take_with_series(numpy_to_idxs(item, self.len()))

        # Integer.
        elif isinstance(item, int):
            if item < 0:
                item = self.len() + item
            return self._s.get_idx(item)

        # Slice.
        elif isinstance(item, slice):
            return PolarsSlice(self).apply(item)

        # Range.
        elif isinstance(item, range):
            return self[range_to_slice(item)]

        # Sequence of integers (also triggers on empty sequence)
        elif isinstance(item, Sequence) and (
            not item or (isinstance(item[0], int) and not isinstance(item[0], bool))  # type: ignore[redundant-expr]
        ):
            idx_series = Series("", item, dtype=Int64)._pos_idxs(self.len())
            if idx_series.has_validity():
                raise ValueError(
                    "Cannot __getitem__ with index values containing nulls"
                )
            return self._take_with_series(idx_series)

        raise ValueError(
            f"Cannot __getitem__ on Series of dtype: '{self.dtype}' "
            f"with argument: '{item}' of type: '{type(item)}'."
        )

    def __setitem__(
        self,
        key: int | Series | np.ndarray[Any, Any] | Sequence[object] | tuple[object],
        value: Any,
    ) -> None:
        # do the single idx as first branch as those are likely in a tight loop
        if isinstance(key, int) and not isinstance(key, bool):
            self.set_at_idx(key, value)
            return None
        elif isinstance(value, Sequence) and not isinstance(value, str):
            if self.is_numeric() or self.is_temporal():
                self.set_at_idx(key, value)  # type: ignore[arg-type]
                return None
            raise ValueError(
                f"cannot set Series of dtype: {self.dtype} with list/tuple as value;"
                " use a scalar value"
            )
        if isinstance(key, Series):
            if key.dtype == Boolean:
                self._s = self.set(key, value)._s
            elif key.dtype == UInt64:
                self._s = self.set_at_idx(key.cast(UInt32), value)._s
            elif key.dtype == UInt32:
                self._s = self.set_at_idx(key, value)._s

        # TODO: implement for these types without casting to series
        elif _check_for_numpy(key) and isinstance(key, np.ndarray):
            if key.dtype == np.bool_:
                # boolean numpy mask
                self._s = self.set_at_idx(np.argwhere(key)[:, 0], value)._s
            else:
                s = self._from_pyseries(
                    PySeries.new_u32("", np.array(key, np.uint32), True)
                )
                self.__setitem__(s, value)
        elif isinstance(key, (list, tuple)):
            s = self._from_pyseries(sequence_to_pyseries("", key, dtype=UInt32))
            self.__setitem__(s, value)
        else:
            raise ValueError(f'cannot use "{key}" for indexing')

    def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
        """
        Numpy __array__ interface protocol.

        Ensures that `np.asarray(pl.Series(..))` works as expected, see
        https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method.
        """
        if not dtype and self.dtype == Utf8 and not self.has_validity():
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
                    "Only ufuncs that return one 1D array, are supported."
                )

            args: list[int | float | np.ndarray[Any, Any]] = []

            for arg in inputs:
                if isinstance(arg, (int, float, np.ndarray)):
                    args.append(arg)
                elif isinstance(arg, Series):
                    args.append(arg.view(ignore_nulls=True))
                else:
                    raise ValueError(f"Unsupported type {type(arg)} for {arg}.")

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
                    "Could not find "
                    f"`apply_ufunc_{numpy_char_code_to_dtype(dtype_char)}`."
                )

            series = f(lambda out: ufunc(*args, out=out, dtype=dtype_char, **kwargs))
            return self._from_pyseries(series)
        else:
            raise NotImplementedError(
                "Only `__call__` is implemented for numpy ufuncs on a Series, got"
                f" `{method}`."
            )

    def _repr_html_(self) -> str:
        """Format output data in HTML for display in Jupyter Notebooks."""
        return self.to_frame()._repr_html_(from_series=True)

    def item(self, row: int | None = None) -> Any:
        """
        Return the series as a scalar, or return the element at the given row index.

        If no row index is provided, this is equivalent to ``s[0]``, with a check
        that the shape is (1,). With a row index, this is equivalent to ``s[row]``.

        Examples
        --------
        >>> s1 = pl.Series("a", [1])
        >>> s1.item()
        1
        >>> s2 = pl.Series("a", [9, 8, 7])
        >>> s2.cumsum().item(-1)
        24

        """
        if row is None and len(self) != 1:
            raise ValueError(
                f"Can only call '.item()' if the series is of length 1, or an "
                f"explicit row index is provided (series is of length {len(self)})"
            )
        return self[row or 0]

    def estimated_size(self, unit: SizeUnit = "b") -> int | float:
        """
        Return an estimation of the total (heap) allocated size of the Series.

        Estimated size is given in the specified unit (bytes by default).

        This estimation is the sum of the size of its buffers, validity, including
        nested arrays. Multiple arrays may share buffers and bitmaps. Therefore, the
        size of 2 arrays is not the sum of the sizes computed from this function. In
        particular, [`StructArray`]'s size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer
        unchanged. However, this function will yield a smaller number. This is because
        this function returns the visible size of the buffer, not its total capacity.

        FFI buffers are included in this estimation.

        Parameters
        ----------
        unit : {'b', 'kb', 'mb', 'gb', 'tb'}
            Scale the returned size to the given unit.

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
        Compute the square root of the elements.

        Syntactic sugar for

        >>> pl.Series([1, 2]) ** 0.5
        shape: (2,)
        Series: '' [f64]
        [
            1.0
            1.414214
        ]

        """

    def any(self) -> bool:
        """
        Check if any boolean value in the column is `True`.

        Returns
        -------
        Boolean literal

        """
        return self.to_frame().select(F.col(self.name).any()).to_series()[0]

    def all(self) -> bool:
        """
        Check if all boolean values in the column are `True`.

        Returns
        -------
        Boolean literal

        """
        return self.to_frame().select(F.col(self.name).all()).to_series()[0]

    def log(self, base: float = math.e) -> Series:
        """Compute the logarithm to a given base."""

    def log1p(self) -> Series:
        """Compute the natural logarithm of the input array plus one, element-wise."""

    def log10(self) -> Series:
        """Compute the base 10 logarithm of the input array, element-wise."""

    def exp(self) -> Series:
        """Compute the exponential, element-wise."""

    def drop_nulls(self) -> Series:
        """
        Drop all null values.

        Creates a new Series that copies data from this Series without null values.
        """

    def drop_nans(self) -> Series:
        """Drop NaN values."""

    def to_frame(self, name: str | None = None) -> DataFrame:
        """
        Cast this Series to a DataFrame.

        Parameters
        ----------
        name
            optionally name/rename the Series column in the new DataFrame.

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
        self, percentiles: Sequence[float] | float | None = (0.25, 0.75)
    ) -> DataFrame:
        """
        Quick summary statistics of a series.

        Series with mixed datatypes will return summary statistics for the datatype of
        the first value.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics (if the
            series has a numeric dtype). All values must be in the range `[0, 1]`.

        Returns
        -------
        Dictionary with summary statistics of a Series.

        Examples
        --------
        >>> series_num = pl.Series([1, 2, 3, 4, 5])
        >>> series_num.describe()
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
        │ max        ┆ 5.0      │
        │ median     ┆ 3.0      │
        │ 25%        ┆ 2.0      │
        │ 75%        ┆ 4.0      │
        └────────────┴──────────┘

        >>> series_str = pl.Series(["a", "a", None, "b", "c"])
        >>> series_str.describe()
        shape: (3, 2)
        ┌────────────┬───────┐
        │ statistic  ┆ value │
        │ ---        ┆ ---   │
        │ str        ┆ i64   │
        ╞════════════╪═══════╡
        │ count      ┆ 5     │
        │ null_count ┆ 1     │
        │ unique     ┆ 4     │
        └────────────┴───────┘

        """
        if isinstance(percentiles, float):
            percentiles = [percentiles]
        if percentiles and not all((0 <= p <= 1) for p in percentiles):
            raise ValueError("Percentiles must all be in the range [0, 1].")

        stats: dict[str, PythonLiteral | None]

        if self.len() == 0:
            raise ValueError("Series must contain at least one value")

        elif self.is_numeric():
            s = self.cast(Float64)
            stats = {
                "count": s.len(),
                "null_count": s.null_count(),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "median": s.median(),
            }
            if percentiles:
                stats.update({f"{p:.0%}": s.quantile(p) for p in percentiles})

        elif self.is_boolean():
            stats = {
                "count": self.len(),
                "null_count": self.null_count(),
                "sum": self.sum(),
            }
        elif self.is_utf8():
            stats = {
                "count": self.len(),
                "null_count": self.null_count(),
                "unique": len(self.unique()),
            }
        elif self.is_temporal():
            # we coerce all to string, because a polars column
            # only has a single dtype and dates: datetime and count: int don't match
            stats = {
                "count": str(self.len()),
                "null_count": str(self.null_count()),
                "min": str(self.dt.min()),
                "max": str(self.dt.max()),
                "median": str(self.dt.median()),
            }
        else:
            raise TypeError("This type is not supported")

        return pl.DataFrame({"statistic": stats.keys(), "value": stats.values()})

    def sum(self) -> int | float:
        """
        Reduce this Series to the sum value.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.sum()
        6

        """
        return self._s.sum()

    def mean(self) -> int | float | None:
        """
        Reduce this Series to the mean value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.mean()
        2.0

        """
        return self._s.mean()

    def product(self) -> int | float:
        """Reduce this Series to the product value."""
        return self.to_frame().select(F.col(self.name).product()).to_series().item()

    def pow(self, exponent: int | float | Series) -> Series:
        """
        Raise to the power of the given exponent.

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
        if self.is_temporal():
            raise ValueError(
                "first cast to integer before raising datelike dtypes to a power"
            )
        if _check_for_numpy(exponent) and isinstance(exponent, np.ndarray):
            exponent = Series(exponent)
        return self.to_frame().select(F.col(self.name).pow(exponent)).to_series()

    def min(self) -> PythonLiteral | None:
        """
        Get the minimal value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        1

        """
        return self._s.min()

    def max(self) -> PythonLiteral | None:
        """
        Get the maximum value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.max()
        3

        """
        return self._s.max()

    def nan_max(self) -> int | float | date | datetime | timedelta | str:
        """
        Get maximum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        """
        return self.to_frame().select(F.col(self.name).nan_max()).item()

    def nan_min(self) -> int | float | date | datetime | timedelta | str:
        """
        Get minimum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        """
        return self.to_frame().select(F.col(self.name).nan_min()).item()

    def std(self, ddof: int = 1) -> float | None:
        """
        Get the standard deviation of this Series.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.std()
        1.0

        """
        if not self.is_numeric():
            return None
        return self.to_frame().select(F.col(self.name).std(ddof)).to_series()[0]

    def var(self, ddof: int = 1) -> float | None:
        """
        Get variance of this Series.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.var()
        1.0

        """
        if not self.is_numeric():
            return None
        return self.to_frame().select(F.col(self.name).var(ddof)).to_series()[0]

    def median(self) -> float | None:
        """
        Get the median of this Series.

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
        Get the quantile value of this Series.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.quantile(0.5)
        2.0

        """
        return self._s.quantile(quantile, interpolation)

    def to_dummies(self, separator: str = "_") -> DataFrame:
        """
        Get dummy/indicator variables.

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

    def cut(
        self,
        bins: list[float],
        labels: list[str] | None = None,
        break_point_label: str = "break_point",
        category_label: str = "category",
        *,
        maintain_order: bool = False,
    ) -> DataFrame:
        """
        Bin values into discrete values.

        Parameters
        ----------
        bins
            Bins to create.
        labels
            Labels to assign to the bins. If given the length of labels must be
            len(bins) + 1.
        break_point_label
            Name given to the breakpoint column.
        category_label
            Name given to the category column.
        maintain_order
            Keep the order of the original `Series`.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
        >>> a.cut([-1, 1])
        shape: (12, 3)
        ┌──────┬─────────────┬──────────────┐
        │ a    ┆ break_point ┆ category     │
        │ ---  ┆ ---         ┆ ---          │
        │ f64  ┆ f64         ┆ cat          │
        ╞══════╪═════════════╪══════════════╡
        │ -3.0 ┆ -1.0        ┆ (-inf, -1.0] │
        │ -2.5 ┆ -1.0        ┆ (-inf, -1.0] │
        │ -2.0 ┆ -1.0        ┆ (-inf, -1.0] │
        │ -1.5 ┆ -1.0        ┆ (-inf, -1.0] │
        │ …    ┆ …           ┆ …            │
        │ 1.0  ┆ 1.0         ┆ (-1.0, 1.0]  │
        │ 1.5  ┆ inf         ┆ (1.0, inf]   │
        │ 2.0  ┆ inf         ┆ (1.0, inf]   │
        │ 2.5  ┆ inf         ┆ (1.0, inf]   │
        └──────┴─────────────┴──────────────┘

        """
        return wrap_df(
            self._s.cut(
                Series(break_point_label, bins, dtype=Float64)._s,
                labels,
                break_point_label,
                category_label,
                maintain_order,
            )
        )

    def qcut(
        self,
        quantiles: list[float],
        *,
        labels: list[str] | None = None,
        break_point_label: str = "break_point",
        category_label: str = "category",
        maintain_order: bool = False,
    ) -> DataFrame:
        """
        Bin values into discrete values based on their quantiles.

        Parameters
        ----------
        quantiles
            List of quantiles to create.
            We expect quantiles ``0.0 <= quantile <= 1``
        labels
            Labels to assign to the quantiles. If given the length of labels must be
            len(bins) + 1.
        break_point_label
            Name given to the breakpoint column.
        category_label
            Name given to the category column.
        maintain_order
            Keep the order of the original `Series`.

        Returns
        -------
        DataFrame

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Examples
        --------
        >>> a = pl.Series("a", range(-5, 3))
        >>> a.qcut([0.0, 0.25, 0.75])
        shape: (8, 3)
        ┌──────┬─────────────┬───────────────┐
        │ a    ┆ break_point ┆ category      │
        │ ---  ┆ ---         ┆ ---           │
        │ f64  ┆ f64         ┆ cat           │
        ╞══════╪═════════════╪═══════════════╡
        │ -5.0 ┆ -5.0        ┆ (-inf, -5.0]  │
        │ -4.0 ┆ -3.25       ┆ (-5.0, -3.25] │
        │ -3.0 ┆ 0.25        ┆ (-3.25, 0.25] │
        │ -2.0 ┆ 0.25        ┆ (-3.25, 0.25] │
        │ -1.0 ┆ 0.25        ┆ (-3.25, 0.25] │
        │ 0.0  ┆ 0.25        ┆ (-3.25, 0.25] │
        │ 1.0  ┆ inf         ┆ (0.25, inf]   │
        │ 2.0  ┆ inf         ┆ (0.25, inf]   │
        └──────┴─────────────┴───────────────┘

        """
        return wrap_df(
            self._s.qcut(
                Series(quantiles, dtype=Float64)._s,
                labels,
                break_point_label,
                category_label,
                maintain_order,
            )
        )

    def hist(
        self,
        bins: list[float] | None = None,
        *,
        bin_count: int | None = None,
    ) -> DataFrame:
        """
        Bin values into buckets and count their occurrences.

        Parameters
        ----------
        bins
            Discretizations to make.
            If None given, we determine the boundaries based on the data.
        bin_count
            If no bins provided, this will be used to determine
            the distance of the bins

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
        ┌─────────────┬─────────────┬─────────┐
        │ break_point ┆ category    ┆ a_count │
        │ ---         ┆ ---         ┆ ---     │
        │ f64         ┆ cat         ┆ u32     │
        ╞═════════════╪═════════════╪═════════╡
        │ 0.0         ┆ (-inf, 0.0] ┆ 0       │
        │ 2.25        ┆ (0.0, 2.25] ┆ 3       │
        │ 4.5         ┆ (2.25, 4.5] ┆ 2       │
        │ 6.75        ┆ (4.5, 6.75] ┆ 0       │
        │ inf         ┆ (6.75, inf] ┆ 2       │
        └─────────────┴─────────────┴─────────┘

        """
        if bins:
            bins = Series(bins, dtype=Float64)._s
        return wrap_df(self._s.hist(bins, bin_count))

    def value_counts(self, *, sort: bool = False) -> DataFrame:
        """
        Count the unique values in a Series.

        Parameters
        ----------
        sort
            Ensure the output is sorted from most values to least.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.value_counts().sort(by="a")
        shape: (3, 2)
        ┌─────┬────────┐
        │ a   ┆ counts │
        │ --- ┆ ---    │
        │ i64 ┆ u32    │
        ╞═════╪════════╡
        │ 1   ┆ 1      │
        │ 2   ┆ 2      │
        │ 3   ┆ 1      │
        └─────┴────────┘

        """
        return wrap_df(self._s.value_counts(sort))

    def unique_counts(self) -> Series:
        """
        Return a count of the unique values in the order of appearance.

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
        Computes the entropy.

        Uses the formula ``-sum(pk * log(pk)`` where ``pk`` are discrete probabilities.

        Parameters
        ----------
        base
            Given base, defaults to `e`
        normalize
            Normalize pk if it doesn't sum to 1.

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
            .select(F.col(self.name).entropy(base, normalize=normalize))
            .to_series()[0]
        )

    def cumulative_eval(
        self, expr: Expr, min_periods: int = 1, *, parallel: bool = False
    ) -> Series:
        """
        Run an expression over a sliding window that increases `1` slot every iteration.

        Parameters
        ----------
        expr
            Expression to evaluate
        min_periods
            Number of valid values there should be in the window before the expression
            is evaluated. valid values = `length - null_count`
        parallel
            Run in parallel. Don't do this in a groupby or another operation that
            already has much parallelization.

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
        Return a copy of the Series with a new alias/name.

        Parameters
        ----------
        name
            New name.

        Examples
        --------
        >>> srs = pl.Series("x", [1, 2, 3])
        >>> new_aliased_srs = srs.alias("y")

        """
        s = self.clone()
        s._s.rename(name)
        return s

    def rename(self, name: str, *, in_place: bool | None = None) -> Series:
        """
        Rename this Series.

        Parameters
        ----------
        name
            New name.
        in_place
            Modify the Series in-place.

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
        if in_place is not None:
            # if 'in_place' is not None, this indicates that the parameter was
            # explicitly set by the caller, and we should warn against it (use
            # of NoDefault only applies when one of the valid values is None).
            warnings.warn(
                "the `in_place` parameter is deprecated and will be removed in a future"
                " version; note that renaming is a shallow-copy operation with"
                " essentially zero cost.",
                category=DeprecationWarning,
                stacklevel=find_stacklevel(),
            )
        if in_place:
            self._s.rename(name)
            return self
        else:
            return self.alias(name)

    def chunk_lengths(self) -> list[int]:
        """
        Get the length of each individual chunk.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2]).chunk_lengths()
        [6]

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).chunk_lengths()
        [3, 3]

        """
        return self._s.chunk_lengths()

    def n_chunks(self) -> int:
        """
        Get the number of chunks that this Series contains.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.n_chunks()
        1
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2]).n_chunks()
        1

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).n_chunks()
        2

        """
        return self._s.n_chunks()

    def cummax(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("s", [3, 5, 1])
        >>> s.cummax()
        shape: (3,)
        Series: 's' [i64]
        [
            3
            5
            5
        ]

        """

    def cummin(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("s", [1, 2, 3])
        >>> s.cummin()
        shape: (3,)
        Series: 's' [i64]
        [
            1
            1
            1
        ]

        """

    def cumprod(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative product computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumprod()
        shape: (3,)
        Series: 'a' [i64]
        [
            1
            2
            6
        ]

        """

    def cumsum(self, *, reverse: bool = False) -> Series:
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumsum()
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
        Get a slice of this Series.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None``, all rows starting at the offset
            will be selected.

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

    def append(self, other: Series, *, append_chunks: bool = True) -> Series:
        """
        Append a Series to this one.

        Parameters
        ----------
        other
            Series to append.
        append_chunks
            If set to `True` the append operation will add the chunks from `other` to
            self. This is super cheap.

            If set to `False` the append operation will do the same as
            `DataFrame.extend` which extends the memory backed by this `Series` with
            the values from `other`.

            Different from `append chunks`, `extend` appends the data from `other` to
            the underlying memory locations and thus may cause a reallocation (which are
            expensive).

            If this does not cause a reallocation, the resulting data structure will not
            have any extra chunks and thus will yield faster queries.

            Prefer `extend` over `append_chunks` when you want to do a query after a
            single append. For instance during online operations where you add `n` rows
            and rerun a query.

            Prefer `append_chunks` over `extend` when you want to append many times
            before doing a query. For instance when you read in multiple files and when
            to store them in a single `Series`. In the latter case, finish the sequence
            of `append_chunks` operations with a `rechunk`.


        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s.append(s2)
        shape: (6,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
            6
        ]

        """
        try:
            if append_chunks:
                self._s.append(other._s)
            else:
                self._s.extend(other._s)
        except RuntimeError as exc:
            if str(exc) == "Already mutably borrowed":
                self.append(other.clone(), append_chunks=append_chunks)
            else:
                raise exc
        return self

    def filter(self, predicate: Series | list[bool]) -> Self:
        """
        Filter elements by a boolean mask.

        Parameters
        ----------
        predicate
            Boolean mask.

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
            Number of elements to return. If a negative value is passed, return all
            elements except the last ``abs(n)``.

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

        Pass a negative value to get all rows `except` the last ``abs(n)``.

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
        return self.to_frame().select(F.col(self.name).head(n)).to_series()

    def tail(self, n: int = 10) -> Series:
        """
        Get the last `n` elements.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the first ``abs(n)``.

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

        Pass a negative value to get all rows `except` the first ``abs(n)``.

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
        return self.to_frame().select(F.col(self.name).tail(n)).to_series()

    def limit(self, n: int = 10) -> Series:
        """
        Get the first `n` elements.

        Alias for :func:`Series.head`.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the last ``abs(n)``.

        See Also
        --------
        head

        """
        return self.head(n)

    def take_every(self, n: int) -> Series:
        """
        Take every nth value in the Series and return as new Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take_every(2)
        shape: (2,)
        Series: 'a' [i64]
        [
            1
            3
        ]

        """

    def sort(self, *, descending: bool = False, in_place: bool = False) -> Self:
        """
        Sort this Series.

        Parameters
        ----------
        descending
            Sort in descending order.
        in_place
            Sort in-place.

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

    def top_k(self, k: int = 5) -> Series:
        r"""
        Return the `k` largest elements.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.

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

    def bottom_k(self, k: int = 5) -> Series:
        r"""
        Return the `k` smallest elements.

        This has time complexity:

        .. math:: O(n + k \\log{}n - \frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.

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
        Get the index values that would sort this Series.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last instead of first.

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
        Get unique index as Series.

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
        Get the index of the minimal value.

        Returns
        -------
        Integer

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_min()
        2

        """
        return self._s.arg_min()

    def arg_max(self) -> int | None:
        """
        Get the index of the maximal value.

        Returns
        -------
        Integer

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

        .. math:: a[i-1] < v <= a[i]

        Parameters
        ----------
        element
            Expression or scalar value.
        side : {'any', 'left', 'right'}
            If 'any', the index of the first suitable location found is given.
            If 'left', the index of the leftmost suitable location found is given.
            If 'right', return the rightmost suitable location found is given.

        """
        if isinstance(element, (int, float)):
            return F.select(F.lit(self).search_sorted(element, side)).item()
        element = Series(element)
        return F.select(F.lit(self).search_sorted(element, side)).to_series()

    def unique(self, *, maintain_order: bool = False) -> Series:
        """
        Get unique elements in series.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

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

    def take(
        self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]
    ) -> Series:
        """
        Take values by index.

        Parameters
        ----------
        indices
            Index location used for selection.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take([1, 3])
        shape: (2,)
        Series: 'a' [i64]
        [
                2
                4
        ]

        """
        return self.to_frame().select(F.col(self.name).take(indices)).to_series()

    def null_count(self) -> int:
        """Count the null values in this Series."""
        return self._s.null_count()

    def has_validity(self) -> bool:
        """
        Return True if the Series has a validity bitmask.

        If there is none, it means that there are no null values.
        Use this to swiftly assert a Series does not have null values.

        """
        return self._s.has_validity()

    def is_empty(self) -> bool:
        """
        Check if the Series is empty.

        Examples
        --------
        >>> s = pl.Series("a", [], dtype=pl.Float32)
        >>> s.is_empty()
        True

        """
        return self.len() == 0

    def is_sorted(self, *, descending: bool = False) -> bool:
        """
        Check if the Series is sorted.

        Parameters
        ----------
        descending
            Check if the Series is sorted in descending order

        """
        return self._s.is_sorted(descending)

    def is_null(self) -> Series:
        """
        Returns a boolean Series indicating which values are null.

        Returns
        -------
        Boolean Series

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
        Returns a boolean Series indicating which values are not null.

        Returns
        -------
        Boolean Series

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
        Returns a boolean Series indicating which values are finite.

        Returns
        -------
        Boolean Series

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
        Returns a boolean Series indicating which values are infinite.

        Returns
        -------
        Boolean Series

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
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
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
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
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
        Check if elements of this Series are in the other Series.

        Returns
        -------
        Boolean Series

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

        >>> # check if some values are a member of sublists
        >>> sets = pl.Series("sets", [[1, 2, 3], [1, 2], [9, 10]])
        >>> optional_members = pl.Series("optional_members", [1, 2, 3])
        >>> print(sets)
        shape: (3,)
        Series: 'sets' [list[i64]]
        [
            [1, 2, 3]
            [1, 2]
            [9, 10]
        ]
        >>> print(optional_members)
        shape: (3,)
        Series: 'optional_members' [i64]
        [
            1
            2
            3
        ]
        >>> optional_members.is_in(sets)
        shape: (3,)
        Series: 'optional_members' [bool]
        [
            true
            true
            false
        ]

        """

    def arg_true(self) -> Series:
        """
        Get index values where Boolean Series evaluate True.

        Returns
        -------
        UInt32 Series

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
        Get mask of all unique values.

        Returns
        -------
        Boolean Series

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

    def is_first(self) -> Series:
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series

        """

    def is_duplicated(self) -> Series:
        """
        Get mask of all duplicated values.

        Returns
        -------
        Boolean Series

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
        Explode a list Series.

        This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype

        See Also
        --------
        ListNameSpace.explode : Explode a list column.
        StringNameSpace.explode : Explode a string column.

        """

    def series_equal(
        self, other: Series, *, null_equal: bool = True, strict: bool = False
    ) -> bool:
        """
        Check if series is equal with another Series.

        Parameters
        ----------
        other
            Series to compare with.
        null_equal
            Consider null values as equal.
        strict
            Don't allow different numerical dtypes, e.g. comparing `pl.UInt32` with a
            `pl.Int64` will return `False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s.series_equal(s)
        True
        >>> s.series_equal(s2)
        False

        """
        return self._s.series_equal(other._s, null_equal, strict)

    def len(self) -> int:
        """
        Length of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.len()
        3

        """
        return self._s.len()

    def cast(
        self,
        dtype: (PolarsDataType | type[int] | type[float] | type[str] | type[bool]),
        *,
        strict: bool = True,
    ) -> Self:
        """
        Cast between data types.

        Parameters
        ----------
        dtype
            DataType to cast to.
        strict
            Throw an error if a cast could not be done for instance due to an overflow.

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
        # Do not dispatch cast as it is expensive and used in other functions.
        dtype = py_type_to_dtype(dtype)
        return self._from_pyseries(self._s.cast(dtype, strict))

    def to_physical(self) -> Series:
        """
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - ``List(inner)`` -> ``List(physical of inner)``
        - Other data types will be left unchanged.

        Examples
        --------
        Replicating the pandas
        `pd.Series.factorize
        <https://pandas.pydata.org/docs/reference/api/pandas.Series.factorize.html>`_
        method.

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

    def to_list(self, *, use_pyarrow: bool = False) -> list[Any]:
        """
        Convert this Series to a Python List. This operation clones data.

        Parameters
        ----------
        use_pyarrow
            Use pyarrow for the conversion.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]
        >>> type(s.to_list())
        <class 'list'>

        """
        if use_pyarrow:
            return self.to_arrow().to_pylist()
        return self._s.to_list()

    def rechunk(self, *, in_place: bool = False) -> Self:
        """
        Create a single chunk of memory for this Series.

        Parameters
        ----------
        in_place
            In place or not.

        """
        opt_s = self._s.rechunk(in_place)
        return self if in_place else self._from_pyseries(opt_s)

    def reverse(self) -> Series:
        """
        Return Series in reverse order.

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
        Get a boolean mask of the values that fall between the given start/end values.

        Parameters
        ----------
        lower_bound
            Lower bound value. Accepts expression input. Non-expression inputs
            (including strings) are parsed as literals.
        upper_bound
            Upper bound value. Accepts expression input. Non-expression inputs
            (including strings) are parsed as literals.
        closed : {'both', 'left', 'right', 'none'}
            Define which sides of the interval are closed (inclusive).

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

        Use the ``closed`` argument to include or exclude the values at the bounds:

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
        if isinstance(lower_bound, str):
            lower_bound = F.lit(lower_bound)
        if isinstance(upper_bound, str):
            upper_bound = F.lit(upper_bound)

        return (
            self.to_frame()
            .select(F.col(self.name).is_between(lower_bound, upper_bound, closed))
            .to_series()
        )

    def is_numeric(self) -> bool:
        """
        Check if this Series datatype is numeric.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.is_numeric()
        True

        """
        return self.dtype in NUMERIC_DTYPES

    def is_integer(self, signed: bool | None = None) -> bool:
        """
        Check if this Series datatype is an integer (signed or unsigned).

        Parameters
        ----------
        signed
            * if `None`, both signed and unsigned integer dtypes will match.
            * if `True`, only signed integer dtypes will be considered a match.
            * if `False`, only unsigned integer dtypes will be considered a match.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3], dtype=pl.UInt32)
        >>> s.is_integer()
        True
        >>> s.is_integer(signed=False)
        True
        >>> s.is_integer(signed=True)
        False

        """
        if signed is None:
            return self.dtype in INTEGER_DTYPES
        elif signed is True:
            return self.dtype in SIGNED_INTEGER_DTYPES
        elif signed is False:
            return self.dtype in UNSIGNED_INTEGER_DTYPES

        raise ValueError(f"'signed' must be None, True or False; given {signed!r}")

    def is_temporal(self, excluding: OneOrMoreDataTypes | None = None) -> bool:
        """
        Check if this Series datatype is temporal.

        Parameters
        ----------
        excluding
            Optionally exclude one or more temporal dtypes from matching.

        Examples
        --------
        >>> from datetime import date
        >>> s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
        >>> s.is_temporal()
        True
        >>> s.is_temporal(excluding=[pl.Date])
        False

        """
        if excluding is not None:
            if not isinstance(excluding, Iterable):
                excluding = [excluding]
            if self.dtype in excluding:
                return False

        return self.dtype in TEMPORAL_DTYPES

    def is_float(self) -> bool:
        """
        Check if this Series has floating point numbers.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0])
        >>> s.is_float()
        True

        """
        return self.dtype in FLOAT_DTYPES

    def is_boolean(self) -> bool:
        """
        Check if this Series is a Boolean.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s.is_boolean()
        True

        """
        return self.dtype is Boolean

    def is_utf8(self) -> bool:
        """
        Check if this Series datatype is a Utf8.

        Examples
        --------
        >>> s = pl.Series("x", ["a", "b", "c"])
        >>> s.is_utf8()
        True

        """
        return self.dtype is Utf8

    def view(self, *, ignore_nulls: bool = False) -> SeriesView:
        """
        Get a view into this Series data with a numpy array.

        This operation doesn't clone data, but does not include missing values.
        Don't use this unless you know what you are doing.

        Parameters
        ----------
        ignore_nulls
            If True then nulls are converted to 0.
            If False then an Exception is raised if nulls are present.

        Examples
        --------
        >>> s = pl.Series("a", [1, None])
        >>> s.view(ignore_nulls=True)
        SeriesView([1, 0])

        """
        if not ignore_nulls:
            assert not self.has_validity()

        from polars.series._numpy import SeriesView, _ptr_to_numpy

        ptr_type = dtype_to_ctype(self.dtype)
        ptr = self._s.as_single_ptr()
        array = _ptr_to_numpy(ptr, self.len(), ptr_type)
        array.setflags(write=False)
        return SeriesView(array, self)

    def to_numpy(
        self,
        *args: Any,
        zero_copy_only: bool = False,
        writable: bool = False,
        use_pyarrow: bool = True,
    ) -> np.ndarray[Any, Any]:
        """
        Convert this Series to numpy. This operation clones data but is completely safe.

        If you want a zero-copy view and know what you are doing, use `.view()`.

        Parameters
        ----------
        *args
            args will be sent to pyarrow.Array.to_numpy.
        zero_copy_only
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        writable
            For numpy arrays created with zero copy (view on the Arrow data),
            the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure
            it is writable.
        use_pyarrow
            Use `pyarrow.Array.to_numpy
            <https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy>`_

            for the conversion to numpy.

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
                tp = f"timedelta64[{self._s.time_unit()}]"
            else:
                tp = f"datetime64[{self._s.time_unit()}]"
            return arr.astype(tp)

        def raise_no_zero_copy() -> None:
            if zero_copy_only:
                raise ValueError("Cannot return a zero-copy array")

        if self.dtype == Array:
            np_array = self.explode().to_numpy(
                zero_copy_only=zero_copy_only,
                writable=writable,
                use_pyarrow=use_pyarrow,
            )
            np_array.shape = (self.len(), self.dtype.width)  # type: ignore[union-attr]
            return np_array

        if (
            use_pyarrow
            and _PYARROW_AVAILABLE
            and self.dtype != Object
            and not self.is_temporal(excluding=Time)
        ):
            return self.to_arrow().to_numpy(
                *args, zero_copy_only=zero_copy_only, writable=writable
            )

        elif self.dtype == Time:
            raise_no_zero_copy()
            # note: there is no native numpy "time" dtype
            return np.array(self.to_list(), dtype="object")
        else:
            if not self.has_validity():
                if self.is_temporal():
                    np_array = convert_to_date(self.view(ignore_nulls=True))
                elif self.is_numeric():
                    np_array = self.view(ignore_nulls=True)
                else:
                    raise_no_zero_copy()
                    np_array = self._s.to_numpy()

            elif self.is_temporal():
                np_array = convert_to_date(self.to_physical()._s.to_numpy())
            else:
                raise_no_zero_copy()
                np_array = self._s.to_numpy()

            if writable and not np_array.flags.writeable:
                raise_no_zero_copy()
                return np_array.copy()
            else:
                return np_array

    def to_arrow(self) -> pa.Array:
        """
        Get the underlying Arrow Array.

        If the Series contains only a single chunk this operation is zero copy.

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
    ) -> pd.Series:
        """
        Convert this Series to a pandas Series.

        This requires that :mod:`pandas` and :mod:`pyarrow` are installed.
        This operation clones data, unless `use_pyarrow_extension_array=True`.

        Parameters
        ----------
        use_pyarrow_extension_array
            Further operations on this Pandas series, might trigger conversion to numpy.
            Use PyArrow backed-extension array instead of numpy array for pandas
            Series. This allows zero copy operations and preservation of nulls
            values.
            Further operations on this pandas Series, might trigger conversion
            to NumPy arrays if that operation is not supported by pyarrow compute
            functions.
        kwargs
            Arguments will be sent to :meth:`pyarrow.Table.to_pandas`.

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
            if parse_version(pd.__version__) < parse_version("1.5"):
                raise ModuleNotFoundError(
                    f'pandas>=1.5.0 is required for `to_pandas("use_pyarrow_extension_array=True")`, found Pandas {pd.__version__}.'
                )
            if not _PYARROW_AVAILABLE or parse_version(pa.__version__) < parse_version(
                "8"
            ):
                raise ModuleNotFoundError(
                    f'pyarrow>=8.0.0 is required for `to_pandas("use_pyarrow_extension_array=True")`'
                    f", found pyarrow {pa.__version__}."
                    if _PYARROW_AVAILABLE
                    else "."
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
        Convert Series to instantiatable string representation.

        Parameters
        ----------
        n
            Only use first n elements.

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

    def set(self, filter: Series, value: int | float | str) -> Series:
        """
        Set masked values.

        Parameters
        ----------
        filter
            Boolean mask.
        value
            Value with which to replace the masked values.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc). Consider using
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

    def set_at_idx(
        self,
        idx: Series | np.ndarray[Any, Any] | Sequence[int] | int,
        value: (
            int
            | float
            | str
            | bool
            | Sequence[int]
            | Sequence[float]
            | Sequence[bool]
            | Sequence[str]
            | Sequence[date]
            | Sequence[datetime]
            | date
            | datetime
            | Series
            | None
        ),
    ) -> Series:
        """
        Set values at the index locations.

        Parameters
        ----------
        idx
            Integers representing the index locations.
        value
            replacement values.

        Returns
        -------
        the series mutated

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_at_idx(1, 10)
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
        if isinstance(idx, int):
            idx = [idx]
        if len(idx) == 0:
            return self

        idx = Series("", idx)
        if isinstance(value, (int, float, bool, str)) or (value is None):
            value = Series("", [value])

            # if we need to set more than a single value, we extend it
            if len(idx) > 0:
                value = value.extend_constant(value[0], len(idx) - 1)
        elif not isinstance(value, Series):
            value = Series("", value)
        self._s.set_at_idx(idx._s, value._s)
        return self

    def clear(self, n: int = 0) -> Series:
        """
        Create an empty copy of the current Series, with zero to 'n' elements.

        The copy has an identical name/dtype, but no data.

        Parameters
        ----------
        n
            Number of (empty) elements to return in the cleared frame.

        See Also
        --------
        clone : Cheap deepcopy/clone.

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
        Very cheap deepcopy/clone.

        See Also
        --------
        clear : Create an empty copy of the current Series, with identical
            schema but no data.

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
        Fill floating point NaN value with a fill value.

        Parameters
        ----------
        value
            Value used to fill NaN values.

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
        Fill null values using the specified value or strategy.

        Parameters
        ----------
        value
            Value used to fill null values.
        strategy : {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}
            Strategy used to fill null values.
        limit
            Number of consecutive null values to fill when using the 'forward' or
            'backward' strategy.

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
        Rounds down to the nearest integer value.

        Only works on floating point Series.

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
        Rounds up to the nearest integer value.

        Only works on floating point Series.

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
        Round underlying floating point data by `decimals` digits.

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
            number of decimals to round by.

        """

    def dot(self, other: Series | ArrayLike) -> float | None:
        """
        Compute the dot/inner product between two Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4.0, 5.0, 6.0])
        >>> s.dot(s2)
        32.0

        Parameters
        ----------
        other
            Series (or array) to compute dot product with.

        """
        if not isinstance(other, Series):
            other = Series(other)
        if len(self) != len(other):
            n, m = len(self), len(other)
            raise ShapeError(f"Series length mismatch: expected {n}, found {m}")
        return self._s.dot(other._s)

    def mode(self) -> Series:
        """
        Compute the most occurring value(s).

        Can return multiple Values.

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
        Compute the element-wise indication of the sign.

        The returned values can be -1, 0, or 1:

        * -1 if x  < 0.
        *  0 if x == 0.
        *  1 if x  > 0.

        (null values are preserved as-is).

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
        Compute the element-wise value for the sine.

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
        Compute the element-wise value for the cosine.

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
        Compute the element-wise value for the tangent.

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

    def arcsin(self) -> Series:
        """
        Compute the element-wise value for the inverse sine.

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
        Compute the element-wise value for the inverse cosine.

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
        Compute the element-wise value for the inverse tangent.

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
        Compute the element-wise value for the inverse hyperbolic sine.

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
        Compute the element-wise value for the inverse hyperbolic cosine.

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
        Compute the element-wise value for the inverse hyperbolic tangent.

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
        Compute the element-wise value for the hyperbolic sine.

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
        Compute the element-wise value for the hyperbolic cosine.

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
        Compute the element-wise value for the hyperbolic tangent.

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

    def apply(
        self,
        function: Callable[[Any], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        skip_nulls: bool = True,
    ) -> Self:
        """
        Apply a custom/user-defined function (UDF) over elements in this Series.

        If the function returns a different datatype, the return_dtype arg should
        be set, otherwise the method will fail.

        Implementing logic using a Python function is almost always _significantly_
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
            Custom function or lambda.
        return_dtype
            Output datatype. If none is given, the same datatype as this Series will be
            used.
        skip_nulls
            Nulls will be skipped and not passed to the python function.
            This is faster because python can be skipped and because we call
            more specialized functions.

        Warnings
        --------
        If ``return_dtype`` is not provided, this may lead to unexpected results.
        We allow this, but it is considered a bug in the user's query.

        Notes
        -----
        If your function is expensive and you don't want it to be called more than
        once for a given input, consider applying an ``@lru_cache`` decorator to it.
        With suitable data you may achieve order-of-magnitude speedups (or more).

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.apply(lambda x: x + 10)
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
        if return_dtype is None:
            pl_return_dtype = None
        else:
            pl_return_dtype = py_type_to_dtype(return_dtype)
        return self._from_pyseries(
            self._s.apply_lambda(function, pl_return_dtype, skip_nulls)
        )

    def shift(self, periods: int = 1) -> Series:
        """
        Shift the values by a given period.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.shift(periods=1)
        shape: (3,)
        Series: 'a' [i64]
        [
                null
                1
                2
        ]
        >>> s.shift(periods=-1)
        shape: (3,)
        Series: 'a' [i64]
        [
                2
                3
                null
        ]

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        """

    def shift_and_fill(
        self,
        fill_value: int | Expr,
        *,
        periods: int = 1,
    ) -> Series:
        """
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        fill_value
            Fill None values with the result of this expression.
        periods
            Number of places to shift (may be negative).

        """

    def zip_with(self, mask: Series, other: Series) -> Self:
        """
        Take values from self or other based on the given mask.

        Where mask evaluates true, take values from self. Where mask evaluates false,
        take values from other.

        Parameters
        ----------
        mask
            Boolean Series.
        other
            Series of same type.

        Returns
        -------
        New Series

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
        Apply a rolling min (moving min) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

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
        Apply a rolling max (moving max) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

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
        Apply a rolling mean (moving mean) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

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
        Apply a rolling sum (moving sum) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

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
        Compute a rolling std dev.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        ddof
            "Delta Degrees of Freedom": The divisor for a length N window is N - ddof

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
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        ddof
            "Delta Degrees of Freedom": The divisor for a length N window is N - ddof

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

        Prefer the specific rolling window functions over this one, as they are faster:

            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        function
            Aggregation function
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("A", [11.0, 2.0, 9.0, float("nan"), 8.0])
        >>> print(s.rolling_apply(function=np.nanstd, window_size=3))
        shape: (5,)
        Series: 'A' [f64]
        [
            null
            null
            3.858612
            3.5
            0.5
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
        Compute a rolling median.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

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
        Compute a rolling quantile.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

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
        Compute a rolling skew.

        The window at a given row includes the row itself and the
        `window_size - 1` elements before it.

        Parameters
        ----------
        window_size
            Integer size of the rolling window.
        bias
            If False, the calculations are corrected for statistical bias.

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

    @deprecated_alias(frac="fraction")
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
        Sample from this Series.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `fraction`. Defaults to 1 if
            `fraction` is None.
        fraction
            Fraction of items to return. Cannot be used with `n`.
        with_replacement
            Allow values to be sampled more than once.
        shuffle
            Shuffle the order of sampled data points.
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

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
        return (
            self.to_frame()
            .select(
                F.col(self.name).sample(
                    n,
                    fraction=fraction,
                    with_replacement=with_replacement,
                    shuffle=shuffle,
                    seed=seed,
                )
            )
            .to_series()
        )

    def peak_max(self) -> Self:
        """
        Get a boolean mask of the local maximum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.peak_max()
        shape: (5,)
        Series: '' [bool]
        [
                false
                false
                false
                false
                true
        ]

        """
        return self._from_pyseries(self._s.peak_max())

    def peak_min(self) -> Self:
        """
        Get a boolean mask of the local minimum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [4, 1, 3, 2, 5])
        >>> s.peak_min()
        shape: (5,)
        Series: '' [bool]
        [
            false
            true
            false
            true
            false
        ]

        """
        return self._from_pyseries(self._s.peak_min())

    def n_unique(self) -> int:
        """
        Count the number of unique values in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.n_unique()
        3

        """
        return self._s.n_unique()

    def shrink_to_fit(self, *, in_place: bool = False) -> Series:
        """
        Shrink Series memory usage.

        Shrinks the underlying array capacity to exactly fit the actual data.
        (Note that this function does not change the Series data type).

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
        Hash the Series.

        The hash value is of type `UInt64`.

        Parameters
        ----------
        seed
            Random seed parameter. Defaults to 0.
        seed_1
            Random seed parameter. Defaults to `seed` if not set.
        seed_2
            Random seed parameter. Defaults to `seed` if not set.
        seed_3
            Random seed parameter. Defaults to `seed` if not set.

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

        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            If True, reinterpret as `pl.Int64`. Otherwise, reinterpret as `pl.UInt64`.

        """

    def interpolate(self, method: InterpolationMethod = "linear") -> Series:
        """
        Interpolate intermediate values. The interpolation method is linear.

        Parameters
        ----------
        method : {'linear', 'nearest'}
            Interpolation method

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None, None, 5])
        >>> s.interpolate()
        shape: (5,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
        ]

        """

    def abs(self) -> Series:
        """
        Compute absolute values.

        Same as `abs(series)`.
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
            The following methods are available (default is 'average'):

            - 'average' : The average of the ranks that would have been assigned to
              all the tied values is assigned to each value.
            - 'min' : The minimum of the ranks that would have been assigned to all
              the tied values is assigned to each value. (This is also referred to
              as "competition" ranking.)
            - 'max' : The maximum of the ranks that would have been assigned to all
              the tied values is assigned to each value.
            - 'dense' : Like 'min', but the rank of the next highest element is
              assigned the rank immediately after those assigned to the tied
              elements.
            - 'ordinal' : All values are given a distinct rank, corresponding to
              the order that the values occur in the Series.
            - 'random' : Like 'ordinal', but the rank for ties is not dependent
              on the order that the values occur in the Series.
        descending
            Rank in descending order.
        seed
            If `method="random"`, use this as seed.

        Examples
        --------
        The 'average' method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank()
        shape: (5,)
        Series: 'a' [f32]
        [
            3.0
            4.5
            1.5
            1.5
            4.5
        ]

        The 'ordinal' method:

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
        return (
            self.to_frame()
            .select(
                F.col(self._s.name()).rank(
                    method=method, descending=descending, seed=seed
                )
            )
            .to_series()
        )

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Series:
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            Number of slots to shift.
        null_behavior : {'ignore', 'drop'}
            How to handle null values.

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

    def pct_change(self, n: int = 1) -> Series:
        """
        Computes percentage change between values.

        Percentage change (as fraction) between current element and most-recent
        non-null element at least ``n`` period(s) before the current element.

        Computes the change from the previous row by default.

        Parameters
        ----------
        n
            periods to shift for forming percent change.

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
        Compute the sample skewness of a data set.

        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.


        See scipy.stats for more information.

        Parameters
        ----------
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.

        .. math:: g_1=\frac{m_3}{m_2^{3/2}}

        where

        .. math:: m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i

        is the biased sample :math:`i\texttt{th}` central moment, and
        :math:`\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math::
            G_1 = \frac{k_3}{k_2^{3/2}} = \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}

        """
        return self._s.skew(bias)

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> float | None:
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset.

        Kurtosis is the fourth central moment divided by the square of the
        variance. If Fisher's definition is used, then 3.0 is subtracted from
        the result to give 0.0 for a normal distribution.
        If bias is False then the kurtosis is calculated using k statistics to
        eliminate bias coming from biased moment estimators

        See scipy.stats for more information

        Parameters
        ----------
        fisher : bool, optional
            If True, Fisher's definition is used (normal ==> 0.0). If False,
            Pearson's definition is used (normal ==> 3.0).
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        """
        return self._s.kurtosis(fisher, bias)

    def clip(self, lower_bound: int | float, upper_bound: int | float) -> Series:
        """
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Minimum value.
        upper_bound
            Maximum value.

        Examples
        --------
        >>> s = pl.Series("foo", [-50, 5, None, 50])
        >>> s.clip(1, 10)
        shape: (4,)
        Series: 'foo' [i64]
        [
            1
            5
            null
            10
        ]

        """

    def clip_min(self, lower_bound: int | float) -> Series:
        """
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Lower bound.

        """

    def clip_max(self, upper_bound: int | float) -> Series:
        """
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        upper_bound
            Upper bound.

        """

    def lower_bound(self) -> Self:
        """
        Return the lower bound of this Series' dtype as a unit Series.

        See Also
        --------
        upper_bound : return the upper bound of the given Series' dtype.

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
        Return the upper bound of this Series' dtype as a unit Series.

        See Also
        --------
        lower_bound : return the lower bound of the given Series' dtype.

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

    def map_dict(
        self,
        remapping: dict[Any, Any],
        *,
        default: Any = None,
        return_dtype: PolarsDataType | None = None,
    ) -> Self:
        """
        Replace values in the Series using a remapping dictionary.

        Parameters
        ----------
        remapping
            Dictionary containing the before/after values to map.
        default
            Value to use when the remapping dict does not contain the lookup value.
            Use ``pl.first()``, to keep the original value.
        return_dtype
            Set return dtype to override automatic return dtype determination.

        Examples
        --------
        >>> s = pl.Series("iso3166", ["TUR", "???", "JPN", "NLD"])
        >>> country_lookup = {
        ...     "JPN": "Japan",
        ...     "TUR": "Türkiye",
        ...     "NLD": "Netherlands",
        ... }

        Remap, setting a default for unrecognised values...

        >>> s.map_dict(country_lookup, default="Unspecified").alias("country_name")
        shape: (4,)
        Series: 'country_name' [str]
        [
            "Türkiye"
            "Unspecified"
            "Japan"
            "Netherlands"
        ]

        ...or keep the original value, by making use of ``pl.first()``:

        >>> s.map_dict(country_lookup, default=pl.first()).alias("country_name")
        shape: (4,)
        Series: 'country_name' [str]
        [
            "Türkiye"
            "???"
            "Japan"
            "Netherlands"
        ]

        ...or keep the original value, by assigning the input series:

        >>> s.map_dict(country_lookup, default=s).alias("country_name")
        shape: (4,)
        Series: 'country_name' [str]
        [
            "Türkiye"
            "???"
            "Japan"
            "Netherlands"
        ]

        Override return dtype:

        >>> s = pl.Series("int8", [5, 2, 3], dtype=pl.Int8)
        >>> s.map_dict({2: 7}, default=pl.first(), return_dtype=pl.Int16)
        shape: (3,)
        Series: 'int8' [i16]
        [
            5
            7
            3
        ]

        """

    def reshape(self, dimensions: tuple[int, ...]) -> Series:
        """
        Reshape this Series to a flat Series or a Series of Lists.

        Parameters
        ----------
        dimensions
            Tuple of the dimension sizes. If a -1 is used in any of the dimensions, that
            dimension is inferred.

        Returns
        -------
        Series
            If a single dimension is given, results in a flat Series of shape (len,).
            If a multiple dimensions are given, results in a Series of Lists with shape
            (rows, cols).

        See Also
        --------
        ListNameSpace.explode : Explode a list column.

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
        Shuffle the contents of this Series.

        Parameters
        ----------
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

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
        Exponentially-weighted moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

        """

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
        Exponentially-weighted moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

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
        Exponentially-weighted moving variance.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\gamma`, with

                .. math::
                    \alpha = \frac{1}{1 + \gamma} \; \forall \; \gamma \geq 0
        span
            Specify decay in terms of span, :math:`\theta`, with

                .. math::
                    \alpha = \frac{2}{\theta + 1} \; \forall \; \theta \geq 1
        half_life
            Specify decay in terms of half-life, :math:`\lambda`, with

                .. math::
                    \alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \lambda } \right\} \;
                    \forall \; \lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \alpha \leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\
                    y_t &= (1 - \alpha)y_{t - 1} + \alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

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
        Extremely fast method for extending the Series with 'n' copies of a value.

        Parameters
        ----------
        value
            A constant literal value (not an expression) with which to extend
            the Series; can pass None to extend with nulls.
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
        Flags the Series as 'sorted'.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        descending
            If the `Series` order is descending.

        Warnings
        --------
        This can lead to incorrect results if this `Series` is not sorted!!
        Use with care!

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_sorted().max()
        3

        """
        return self._from_pyseries(self._s.set_sorted_flag(descending))

    def new_from_index(self, index: int, length: int) -> Self:
        """Create a new Series filled with values from the given index."""
        return self._from_pyseries(self._s.new_from_index(index, length))

    def shrink_dtype(self) -> Series:
        """
        Shrink numeric columns to the minimal required datatype.

        Shrink to the dtype needed to fit the extrema of this [`Series`].
        This can be used to reduce memory pressure.
        """

    def get_chunks(self) -> list[Series]:
        """Get the chunks of this Series as a list of Series."""
        return self._s.get_chunks()

    def implode(self) -> Self:
        """Aggregate values into a list."""

    # Keep the `list` and `str` properties below at the end of the definition of Series,
    # as to not confuse mypy with the type annotation `str` and `list`

    @property
    def bin(self) -> BinaryNameSpace:
        """Create an object namespace of all binary related methods."""
        return BinaryNameSpace(self)

    @property
    def cat(self) -> CatNameSpace:
        """Create an object namespace of all categorical related methods."""
        return CatNameSpace(self)

    @property
    def dt(self) -> DateTimeNameSpace:
        """Create an object namespace of all datetime related methods."""
        return DateTimeNameSpace(self)

    @property
    def list(self) -> ListNameSpace:
        """Create an object namespace of all list related methods."""
        return ListNameSpace(self)

    @property
    def arr(self) -> ArrayNameSpace:
        """Create an object namespace of all array related methods."""
        return ArrayNameSpace(self)

    @property
    def str(self) -> StringNameSpace:
        """Create an object namespace of all string related methods."""
        return StringNameSpace(self)

    @property
    def struct(self) -> StructNameSpace:
        """Create an object namespace of all struct related methods."""
        return StructNameSpace(self)


def _resolve_datetime_dtype(
    dtype: PolarsDataType | None, ndtype: np.datetime64
) -> PolarsDataType | None:
    """Given polars/numpy datetime dtypes, resolve to an explicit unit."""
    if dtype is None or (dtype == Datetime and not getattr(dtype, "time_unit", None)):
        time_unit = getattr(dtype, "time_unit", None) or np.datetime_data(ndtype)[0]
        # explicit formulation is verbose, but keeps mypy happy
        # (and avoids unsupported timeunits such as "s")
        if time_unit == "ns":
            dtype = Datetime("ns")
        elif time_unit == "us":
            dtype = Datetime("us")
        elif time_unit == "ms":
            dtype = Datetime("ms")
    return dtype
