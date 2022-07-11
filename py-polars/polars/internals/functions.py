from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Sequence, Union, overload

from polars import internals as pli
from polars.datatypes import Date
from polars.utils import (
    _datetime_to_pl_timestamp,
    _timedelta_to_pl_duration,
    in_nanoseconds_window,
)

try:
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_lf as _concat_lf
    from polars.polars import concat_series as _concat_series
    from polars.polars import py_date_range as _py_date_range
    from polars.polars import py_diag_concat_df as _diag_concat_df
    from polars.polars import py_hor_concat_df as _hor_concat_df

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


def get_dummies(df: pli.DataFrame) -> pli.DataFrame:
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    """
    return df.to_dummies()


@overload
def concat(
    items: Sequence[pli.DataFrame],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.DataFrame:
    ...


@overload
def concat(
    items: Sequence[pli.Series],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.Series:
    ...


@overload
def concat(
    items: Sequence[pli.LazyFrame],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.LazyFrame:
    ...


@overload
def concat(
    items: Sequence[pli.Expr],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.Expr:
    ...


def concat(
    items: (
        Sequence[pli.DataFrame]
        | Sequence[pli.Series]
        | Sequence[pli.LazyFrame]
        | Sequence[pli.Expr]
    ),
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.DataFrame | pli.Series | pli.LazyFrame | pli.Expr:
    """
    Aggregate all the Dataframes/Series in a List of DataFrames/Series to a single DataFrame/Series.

    Parameters
    ----------
    items
        DataFrames/Series/LazyFrames to concatenate.
    rechunk
        rechunk the final DataFrame/Series.
    how
        Only used if the items are DataFrames.

        One of {"vertical", "diagonal", "horizontal"}.

        - Vertical: Applies multiple `vstack` operations.
        - Diagonal: Finds a union between the column schemas and fills missing column values with null.
        - Horizontal: Stacks Series horizontally and fills with nulls if the lengths don't match.

    Examples
    --------
    >>> df1 = pl.DataFrame({"a": [1], "b": [3]})
    >>> df2 = pl.DataFrame({"a": [2], "b": [4]})
    >>> pl.concat([df1, df2])
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 4   │
    └─────┴─────┘

    """
    if not len(items) > 0:
        raise ValueError("cannot concat empty list")

    out: pli.Series | pli.DataFrame | pli.LazyFrame | pli.Expr
    first = items[0]
    if isinstance(first, pli.DataFrame):
        if how == "vertical":
            out = pli.wrap_df(_concat_df(items))
        elif how == "diagonal":
            out = pli.wrap_df(_diag_concat_df(items))
        elif how == "horizontal":
            out = pli.wrap_df(_hor_concat_df(items))
        else:
            raise ValueError(
                f"how should be one of {'vertical', 'diagonal'}, got {how}"
            )
    elif isinstance(first, pli.LazyFrame):
        return pli.wrap_ldf(_concat_lf(items, rechunk))
    elif isinstance(first, pli.Series):
        out = pli.wrap_s(_concat_series(items))
    elif isinstance(first, pli.Expr):
        out = first
        for e in items[1:]:
            out = out.append(e)  # type: ignore
    else:
        raise ValueError(f"did not expect type: {type(first)} in 'pl.concat'.")

    if rechunk:
        return out.rechunk()
    return out


def _ensure_datetime(value: date | datetime) -> tuple[datetime, bool]:
    is_date_type = False
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
        is_date_type = True
    return value, is_date_type


def _interval_granularity(interval: str) -> str:
    return interval[-2:].lstrip("0123456789")


def date_range(
    low: date | datetime,
    high: date | datetime,
    interval: str | timedelta,
    closed: str | None = "both",
    name: str | None = None,
    time_unit: str | None = None,
) -> pli.Series:
    """
    Create a range of type `Datetime` (or `Date`).

    Parameters
    ----------
    low
        Lower bound of the date range.
    high
        Upper bound of the date range.
    interval
        Interval periods. It can be a python timedelta object, like ``timedelta(days=10)``,
        or a polars duration string, such as ``3d12h4m25s`` representing 3 days, 12 hours,
        4 minutes, and 25 seconds.
    closed : {None, 'left', 'right', 'both', 'none'}
        Make the interval closed to the 'left', 'right', 'none' or 'both' sides.
    name
        Name of the output Series.
    time_unit : {'ns', 'us', 'ms'}
        Set the time unit.

    Notes
    -----
    If both ``low`` and ``high`` are passed as date types (not datetime), and the
    interval granularity is no finer than 1d, the returned range is also of
    type date. All other permutations return a datetime Series.

    Returns
    -------
    A Series of type `Datetime` or `Date`.

    Examples
    --------
    Using polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", name="drange")
    shape: (3,)
    Series: 'drange' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]

    Using `timedelta` object to specify the interval:

    >>> from datetime import datetime, timedelta
    >>> pl.date_range(
    ...     datetime(1985, 1, 1),
    ...     datetime(1985, 1, 10),
    ...     timedelta(days=1, hours=12),
    ...     time_unit="ms",
    ... )
    shape: (7,)
    Series: '' [datetime[ms]]
    [
        1985-01-01 00:00:00
        1985-01-02 12:00:00
        1985-01-04 00:00:00
        1985-01-05 12:00:00
        1985-01-07 00:00:00
        1985-01-08 12:00:00
        1985-01-10 00:00:00
    ]

    """
    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)

    low, low_is_date = _ensure_datetime(low)
    high, high_is_date = _ensure_datetime(high)

    if in_nanoseconds_window(low) and in_nanoseconds_window(high) and time_unit is None:
        tu = "ns"
    elif time_unit is not None:
        tu = time_unit
    else:
        tu = "ms"

    start = _datetime_to_pl_timestamp(low, tu)
    stop = _datetime_to_pl_timestamp(high, tu)
    if name is None:
        name = ""

    dt_range = pli.wrap_s(_py_date_range(start, stop, interval, closed, name, tu))
    if (
        low_is_date
        and high_is_date
        and not _interval_granularity(interval).endswith(("h", "m", "s"))
    ):
        dt_range = dt_range.cast(Date)

    return dt_range


FrameOrSeries = Union["pli.DataFrame", "pli.Series"]


class PolarsSlice:
    """
    Apply python slice object to Polars DataFrame or Series,
    with full support for negative indexing and/or stride.
    """

    stop: int
    start: int
    stride: int
    slice_length: int
    is_unbounded: bool
    obj: FrameOrSeries

    def __init__(self, obj: FrameOrSeries):
        self.obj = obj

    @staticmethod
    def _as_original(lazy: "pli.LazyFrame", original: FrameOrSeries) -> FrameOrSeries:
        """
        Return lazy variant back to its original type.
        """
        frame = lazy.collect()
        return frame if isinstance(original, pli.DataFrame) else frame.to_series()

    @staticmethod
    def _lazify(obj: FrameOrSeries) -> "pli.LazyFrame":
        """
        Make lazy to ensure efficent/consistent handling.
        """
        return obj.lazy() if isinstance(obj, pli.DataFrame) else obj.to_frame().lazy()

    def _slice_positive(self, obj: "pli.LazyFrame") -> "pli.LazyFrame":
        """
        Logic for slices with positive stride.
        """
        # note: at this point stride is guaranteed to be > 1
        return obj.slice(self.start, self.slice_length).take_every(self.stride)

    def _slice_negative(self, obj: "pli.LazyFrame") -> "pli.LazyFrame":
        """
        Logic for slices with negative stride.
        """
        stride = abs(self.stride)
        lazyslice = obj.slice(self.stop + 1, self.slice_length).reverse()
        return lazyslice.take_every(stride) if (stride > 1) else lazyslice

    def _slice_setup(self, s: slice) -> None:
        """
        Normalise slice bounds, identify unbounded and/or zero-length slices.
        """
        # can normalise slice indices as we know object size
        obj_len = len(self.obj)
        start, stop, stride = slice(s.start, s.stop, s.step).indices(obj_len)

        # check if slice is actually unbounded
        if stride >= 1:
            self.is_unbounded = (start <= 0) and (stop >= obj_len)
        else:
            self.is_unbounded = (stop == -1) and (start >= obj_len - 1)

        # determine slice length
        if self.obj.is_empty():
            self.slice_length = 0
        elif self.is_unbounded:
            self.slice_length = obj_len
        else:
            self.slice_length = (
                0
                if (
                    (start == stop)
                    or (stride > 0 and start > stop)
                    or (stride < 0 and start < stop)
                )
                else abs(stop - start)
            )
        self.start, self.stop, self.stride = start, stop, stride

    def apply(self, s: slice) -> FrameOrSeries:
        """
        Apply a slice operation, taking advantage of any potential fast paths.
        """
        # normalise slice
        self._slice_setup(s)

        # check for fast-paths / single-operation calls
        if self.slice_length == 0:
            return self.obj.cleared()

        elif self.is_unbounded and self.stride in (-1, 1):
            return self.obj.reverse() if (self.stride < 0) else self.obj.clone()

        elif self.start >= 0 and self.stop >= 0 and self.stride == 1:
            return self.obj.slice(self.start, self.slice_length)

        elif self.stride < 0 and self.slice_length == 1:
            return self.obj.slice(self.stop + 1, 1)
        else:
            # multi-operation calls; make lazy
            lazyobj = self._lazify(self.obj)
            sliced = (
                self._slice_positive(lazyobj)
                if self.stride > 0
                else self._slice_negative(lazyobj)
            )
            return self._as_original(sliced, self.obj)


class LazyPolarsSlice:
    """
    Apply python slice object to Polars LazyFrame. Only slices with efficient
    computation paths mapping directly to existing lazy methods are supported.
    """

    obj: "pli.LazyFrame"

    def __init__(self, obj: "pli.LazyFrame"):
        self.obj = obj

    def apply(self, s: slice) -> "pli.LazyFrame":
        """
        Apply a slice operation. Note that LazyFrame is designed primarily for efficient
        computation and does not know its own length so, unlike DataFrame, certain slice
        patterns (such as those requiring negative stop/step) may not be supported.
        """
        start = s.start or 0
        step = s.step or 1

        # fail on operations that require length to do efficiently
        if s.stop and s.stop < 0:
            raise ValueError("Negative stop is not supported for lazy slices")
        if step < 0 and (start > 0 or s.stop is not None) and (start != s.stop):
            if not (start > 0 > step and s.stop is None):
                raise ValueError(
                    "Negative stride is not supported in conjunction with start+stop"
                )

        # ---------------------------------------
        # empty slice patterns.
        # ---------------------------------------
        # [:0]
        # [i:<=i]
        # [i:>=i:-k]
        if step > 0 and (s.stop is not None and start >= s.stop):
            return self.obj.cleared()
        elif step < 0 and (s.stop is not None and s.stop >= s.start >= 0):
            return self.obj.cleared()

        # ---------------------------------------
        # straight-though mappings for "reverse"
        # and/or "take_every"
        # ---------------------------------------
        # [:]    => clone()
        # [::k]  => take_every(k),
        # [::-1] => reverse(),
        # [::-k] => reverse().take_every(abs(k))
        elif start == 0 and s.stop is None:
            if step == 1:
                return self.obj.clone()
            elif step > 1:
                return self.obj.take_every(step)
            elif step == -1:
                return self.obj.reverse()
            elif step < -1:
                return self.obj.reverse().take_every(abs(step))

        elif start > 0 > step and s.stop is None:
            obj = self.obj.head(s.start + 1).reverse()
            return obj if (abs(step) == 1) else obj.take_every(abs(step))

        # ---------------------------------------
        # straight-though mappings for "head"
        # ---------------------------------------
        # [:j]    => head(j)
        # [:j:k]  => head(j).take_every(k)
        elif start == 0 and (s.stop or 0) >= 1:
            obj = self.obj.head(s.stop)
            return obj if (step == 1) else obj.take_every(step)

        # ---------------------------------------
        # straight-though mappings for "tail"
        # ---------------------------------------
        # [-i:]    => tail(abs(i))
        # [-i::k]  => tail(abs(i)).take_every(k)
        elif start < 0 and s.stop is None:
            obj = self.obj.tail(abs(start))
            return obj if (step == 1) else obj.take_every(step)

        # ---------------------------------------
        # straight-though mappings for "slice"
        # ---------------------------------------
        # [i:]     => slice(i)
        # [i:j]    => slice(i,j-i)
        # [i:j:k]  => slice(i,j-i).take_every(k)
        elif start > 0 and (s.stop is None or s.stop >= 0):
            slice_length = None if (s.stop is None) else (s.stop - start)
            obj = self.obj.slice(start, slice_length)
            return obj if (step == 1) else obj.take_every(step)

        raise ValueError(
            f"The given slice {s} is not supported by lazy computation; consider a "
            f"more efficient approach, or construct explicitly with other methods"
        )
