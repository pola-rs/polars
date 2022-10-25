from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any, Sequence

from hypothesis import settings
from hypothesis.errors import InvalidArgument, NonInteractiveExampleWarning
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    booleans,
    composite,
    dates,
    datetimes,
    floats,
    from_type,
    integers,
    lists,
    sampled_from,
    text,
    timedeltas,
    times,
)
from hypothesis.strategies._internal.utils import defines_strategy

import polars.internals as pli
from polars.datatypes import (
    Boolean,
    Categorical,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    PolarsDataType,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    is_polars_dtype,
    py_type_to_dtype,
)
from polars.testing.asserts import is_categorical_dtype

# Default profile (eg: running locally)
common_settings = {"print_blob": True, "deadline": None}
settings.register_profile(
    name="polars.default",
    max_examples=100,
    **common_settings,  # type: ignore[arg-type]
)
# CI 'max' profile (10x the number of iterations).
# this is expensive, and not actually enabled in
# our usual CI pipeline; requires explicit opt-in.
settings.register_profile(
    name="polars.ci",
    max_examples=1000,
    **common_settings,  # type: ignore[arg-type]
)
if os.getenv("CI_MAX"):
    settings.load_profile("polars.ci")
else:
    settings.load_profile("polars.default")


MAX_DATA_SIZE = 10
MAX_COLS = 8

# =====================================================================
# Polars-specific 'hypothesis' strategies and helper functions
# See: https://hypothesis.readthedocs.io/
# =====================================================================

dtype_strategy_mapping: dict[PolarsDataType, Any] = {
    Boolean: booleans(),
    Float32: floats(width=32),
    Float64: floats(width=64),
    Int8: integers(min_value=-(2**7), max_value=(2**7) - 1),
    Int16: integers(min_value=-(2**15), max_value=(2**15) - 1),
    Int32: integers(min_value=-(2**31), max_value=(2**31) - 1),
    Int64: integers(min_value=-(2**63), max_value=(2**63) - 1),
    UInt8: integers(min_value=0, max_value=(2**8) - 1),
    UInt16: integers(min_value=0, max_value=(2**16) - 1),
    UInt32: integers(min_value=0, max_value=(2**32) - 1),
    UInt64: integers(min_value=0, max_value=(2**64) - 1),
    # TODO: when generating text for categorical, ensure there are repeats -
    #  don't want all to be unique.
    Categorical: text(max_size=10),
    Utf8: text(max_size=10),
    # TODO: generate arrow temporal types with different resolution (32/64) to
    #  validate compatibility.
    Time: times(),
    Date: dates(),
    Duration: timedeltas(
        min_value=timedelta(microseconds=-(2**63)),
        max_value=timedelta(microseconds=(2**63) - 1),
    ),
    # TODO: confirm datetime min/max limits with different timeunit granularity.
    # TODO: specific strategies for temporal dtypes with timeunits.
    Datetime: datetimes(min_value=datetime(1970, 1, 1)),
    # Datetime("ms")
    # Datetime("us")
    # Datetime("ns")
    # Duration("ms")
    # Duration("us")
    # Duration("ns")
    # TODO: strategies for non-scalar/structured dtypes.
    # List
    # Struct
    # Object
}

strategy_dtypes = list(dtype_strategy_mapping)


def between(draw: DrawFn, type_: type, min_: Any, max_: Any) -> Any:
    """Draw a value in a given range from a type-inferred strategy."""
    strategy_init = from_type(type_).function  # type: ignore[attr-defined]
    return draw(strategy_init(min_, max_))


@dataclass
class column:
    """
    Define a column for use with the @dataframes strategy.

    Parameters
    ----------
    name : str
        string column name.
    dtype : dtype
        a recognised polars dtype.
    strategy : strategy, optional
        supports overriding the default strategy for the given dtype.
    null_probability : float, optional
        percentage chance (expressed between 0.0 => 1.0) that a generated value
        is None. this is applied in addition to any None values output by the
        given/inferred strategy for the column.
    unique : bool, optional
        flag indicating that all values generated for the column should be unique.

    Examples
    --------
    >>> from hypothesis.strategies import sampled_from
    >>> from polars.testing.parametric import column
    >>> column(name="unique_small_ints", dtype=pl.UInt8, unique=True)
    column(name='unique_small_ints', dtype=<class 'polars.datatypes.UInt8'>, strategy=None, null_probability=None, unique=True)
    >>> column(name="ccy", strategy=sampled_from(["GBP", "EUR", "JPY"]))
    column(name='ccy', dtype=<class 'polars.datatypes.Utf8'>, strategy=sampled_from(['GBP', 'EUR', 'JPY']), null_probability=None, unique=False)

    """  # noqa: E501

    name: str
    dtype: PolarsDataType | None = None
    strategy: SearchStrategy[pli.Series | int] | None = None
    null_probability: float | None = None
    unique: bool = False

    def __post_init__(self) -> None:
        if (self.null_probability is not None) and (
            self.null_probability < 0 or self.null_probability > 1
        ):
            raise InvalidArgument(
                "null_probability should be between 0.0 and 1.0, or None; found"
                f" {self.null_probability}"
            )
        if self.dtype is None and self.strategy is None:
            self.dtype = random.choice(strategy_dtypes)
        elif self.dtype not in dtype_strategy_mapping:
            if self.dtype is not None:
                raise InvalidArgument(
                    f"No strategy (currently) available for {self.dtype} type"
                )
            else:
                # given a custom strategy, but no explicit dtype. infer one
                # from the first non-None value that the strategy produces.
                with warnings.catch_warnings():
                    # note: usually you should not call "example()" outside of an
                    # interactive shell, hence the warning. however, here it is
                    # reasonable to do so, so we catch and ignore it
                    warnings.simplefilter("ignore", NonInteractiveExampleWarning)
                    sample_value_iter = (
                        self.strategy.example()  # type: ignore[union-attr]
                        for _ in range(100)
                    )
                    try:
                        sample_value_type = type(
                            next(e for e in sample_value_iter if e is not None)
                        )
                    except StopIteration:
                        raise InvalidArgument(
                            "Unable to determine dtype for strategy"
                        ) from None
                if sample_value_type is not None:
                    self.dtype = py_type_to_dtype(sample_value_type)


def columns(
    cols: int | Sequence[str] | None = None,
    *,
    dtype: PolarsDataType | Sequence[PolarsDataType] | None = None,
    min_cols: int | None = 0,
    max_cols: int | None = MAX_COLS,
    unique: bool = False,
) -> list[column]:
    """
    Define multiple columns for use with the @dataframes strategy.

    Generate a fixed sequence of `column` objects suitable for passing to the
    @dataframes strategy, or using standalone (note that this function is not itself
    a strategy).

    Notes
    -----
    Additional control is available by creating a sequence of columns explicitly,
    using the `column` class (an especially useful option is to override the default
    data-generating strategy for a given col/dtype).

    Parameters
    ----------
    cols : {int, [str]}, optional
        integer number of cols to create, or explicit list of column names. if
        omitted a random number of columns (between mincol and max_cols) are
        created.
    dtype : dtype, optional
        a single dtype for all cols, or list of dtypes (the same length as `cols`).
        if omitted, each generated column is assigned a random dtype.
    min_cols : int, optional
        if not passing an exact size, can set a minimum here (defaults to 0).
    max_cols : int, optional
        if not passing an exact size, can set a maximum value here (defaults to
        MAX_COLS).
    unique : bool, optional
        indicate if the values generated for these columns should be unique
        (per-column).

    Examples
    --------
    >>> from polars.testing.parametric import columns
    >>> from string import punctuation
    >>>
    >>> def test_special_char_colname_init() -> None:
    ...     cols = [(c.name, c.dtype) for c in columns(punctuation)]
    ...     df = pl.DataFrame(columns=cols)
    ...     assert len(cols) == len(df.columns)
    ...     assert 0 == len(df.rows())
    ...
    >>> from polars.testing.parametric import columns
    >>> from hypothesis import given
    >>>
    >>> @given(dataframes(columns(["x", "y", "z"], unique=True)))
    ... def test_unique_xyz(df: pl.DataFrame) -> None:
    ...     assert_something(df)

    """
    # create/assign named columns
    if cols is None:
        cols = random.randint(
            a=min_cols or 0,
            b=max_cols or MAX_COLS,
        )
    if isinstance(cols, int):
        names: list[str] = [f"col{n}" for n in range(cols)]
    else:
        names = list(cols)

    if isinstance(dtype, Sequence):
        if len(dtype) != len(names):
            raise InvalidArgument(f"Given {len(dtype)} dtypes for {len(names)} names")
        dtypes = list(dtype)
    elif dtype is None:
        dtypes = [random.choice(strategy_dtypes) for _ in range(len(names))]
    elif is_polars_dtype(dtype):
        dtypes = [dtype] * len(names)
    else:
        raise InvalidArgument(f"{dtype} is not a valid polars datatype")

    # init list of named/typed columns
    return [column(name=nm, dtype=tp, unique=unique) for nm, tp in zip(names, dtypes)]


@defines_strategy()
def series(
    *,
    name: str | SearchStrategy[str] | None = None,
    dtype: PolarsDataType | None = None,
    size: int | None = None,
    min_size: int | None = 0,
    max_size: int | None = MAX_DATA_SIZE,
    strategy: SearchStrategy[object] | None = None,
    null_probability: float = 0.0,
    allow_infinities: bool = True,
    unique: bool = False,
    chunked: bool | None = None,
    allowed_dtypes: Sequence[PolarsDataType] | None = None,
    excluded_dtypes: Sequence[PolarsDataType] | None = None,
) -> SearchStrategy[pli.Series]:
    """
    Strategy for producing a polars Series.

    Parameters
    ----------
    name : {str, strategy}, optional
        literal string or a strategy for strings (or None), passed to the Series
        constructor name-param.
    dtype : dtype, optional
        a valid polars DataType for the resulting series.
    size : int, optional
        if set, creates a Series of exactly this size (ignoring min/max params).
    min_size : int, optional
        if not passing an exact size, can set a minimum here (defaults to 0).
        no-op if `size` is set.
    max_size : int, optional
        if not passing an exact size, can set a maximum value here (defaults to
        MAX_DATA_SIZE). no-op if `size` is set.
    strategy : strategy, optional
        supports overriding the default strategy for the given dtype.
    null_probability : float, optional
        percentage chance (expressed between 0.0 => 1.0) that a generated value is
        None. this is applied independently of any None values generated by the
        underlying strategy.
    allow_infinities : bool, optional
        optionally disallow generation of +/-inf values for floating-point dtypes.
    unique : bool, optional
        indicate whether Series values should all be distinct.
    chunked : bool, optional
        ensure that Series with more than one element have ``n_chunks`` > 1.
        if omitted, chunking is applied at random.
    allowed_dtypes : {list,set}, optional
        when automatically generating Series data, allow only these dtypes.
    excluded_dtypes : {list,set}, optional
        when automatically generating Series data, exclude these dtypes.

    Notes
    -----
    In actual usage this is deployed as a unit test decorator, providing a strategy
    that generates multiple Series with the given dtype/size characteristics for the
    unit test. While developing a strategy/test, it can also be useful to call
    `.example()` directly on a given strategy to see concrete instances of the
    generated data.

    Examples
    --------
    >>> from polars.testing.parametric import series
    >>> from hypothesis import given
    >>>
    >>> @given(df=series())
    ... def test_repr(s: pl.Series) -> None:
    ...     assert isinstance(repr(s), str)
    >>>
    >>> s = series(dtype=pl.Int32, max_size=5)
    >>> s.example()  # doctest: +SKIP
    shape: (4,)
    Series: '' [i64]
    [
        54666
        -35
        6414
        -63290
    ]

    """
    selectable_dtypes = [
        dtype
        for dtype in (allowed_dtypes or strategy_dtypes)
        if dtype not in (excluded_dtypes or ())
    ]
    if null_probability and (null_probability < 0 or null_probability > 1):
        raise InvalidArgument(
            "null_probability should be between 0.0 and 1.0; found"
            f" {null_probability}"
        )
    null_probability = float(null_probability or 0.0)

    @composite
    def draw_series(draw: DrawFn) -> pli.Series:
        # create/assign series dtype and retrieve matching strategy
        series_dtype = draw(sampled_from(selectable_dtypes)) if dtype is None else dtype
        if strategy is None:
            dtype_strategy = dtype_strategy_mapping[series_dtype]
        else:
            dtype_strategy = strategy

        if series_dtype in (Float32, Float64) and not allow_infinities:
            dtype_strategy = dtype_strategy.filter(
                lambda x: not isinstance(x, float) or isfinite(x)
            )

        # create/assign series size
        series_size = (
            between(draw, int, min_=(min_size or 0), max_=(max_size or MAX_DATA_SIZE))
            if size is None
            else size
        )
        # assign series name
        series_name = name if isinstance(name, str) or name is None else draw(name)

        # create series using dtype-specific strategy to generate values
        if series_size == 0:
            series_values = []
        elif null_probability == 1:
            series_values = [None] * series_size
        else:
            series_values = draw(
                lists(
                    dtype_strategy,
                    min_size=series_size,
                    max_size=series_size,
                    unique=unique,
                )
            )

        # apply null values (custom frequency)
        if null_probability and null_probability != 1:
            for idx in range(series_size):
                if random.random() < null_probability:
                    series_values[idx] = None

        # init series with strategy-generated data
        s = pli.Series(
            name=series_name,
            dtype=series_dtype,
            values=series_values,
        )
        if is_categorical_dtype(dtype):
            s = s.cast(Categorical)
        if series_size:
            if chunked or (chunked is None and draw(booleans())):
                split_at = series_size // 2
                s = s[:split_at].append(s[split_at:], append_chunks=True)
        return s

    return draw_series()


@defines_strategy()
def dataframes(
    cols: int | column | Sequence[column] | None = None,
    lazy: bool = False,
    *,
    min_cols: int | None = 0,
    max_cols: int | None = MAX_COLS,
    size: int | None = None,
    min_size: int | None = 0,
    max_size: int | None = MAX_DATA_SIZE,
    chunked: bool | None = None,
    include_cols: Sequence[column] | None = None,
    null_probability: float | dict[str, float] = 0.0,
    allow_infinities: bool = True,
    allowed_dtypes: Sequence[PolarsDataType] | None = None,
    excluded_dtypes: Sequence[PolarsDataType] | None = None,
) -> SearchStrategy[pli.DataFrame | pli.LazyFrame]:
    """
    Provides a strategy for producing a DataFrame or LazyFrame.

    Parameters
    ----------
    cols : {int, columns}, optional
        integer number of columns to create, or a sequence of `column` objects
        that describe the desired DataFrame column data.
    lazy : bool, optional
        produce a LazyFrame instead of a DataFrame.
    min_cols : int, optional
        if not passing an exact size, can set a minimum here (defaults to 0).
    max_cols : int, optional
        if not passing an exact size, can set a maximum value here (defaults to
        MAX_COLS).
    size : int, optional
        if set, will create a DataFrame of exactly this size (and ignore min/max len
        params).
    min_size : int, optional
        if not passing an exact size, set the minimum number of rows in the
        DataFrame.
    max_size : int, optional
        if not passing an exact size, set the maximum number of rows in the
        DataFrame.
    chunked : bool, optional
        ensure that DataFrames with more than row have ``n_chunks`` > 1. if
        omitted, chunking will be randomised at the level of individual Series.
    include_cols : [column], optional
        a list of `column` objects to include in the generated DataFrame. note that
        explicitly provided columns are appended onto the list of existing columns
        (if any present).
    null_probability : {float, dict[str,float]}, optional
        percentage chance (expressed between 0.0 => 1.0) that a generated value is
        None. this is applied independently of any None values generated by the
        underlying strategy, and can be applied either on a per-column basis (if
        given as a ``{col:pct}`` dict), or globally. if null_probability is defined
        on a column, it takes precedence over the global value.
    allow_infinities : bool, optional
        optionally disallow generation of +/-inf values for floating-point dtypes.
    allowed_dtypes : {list,set}, optional
        when automatically generating data, allow only these dtypes.
    excluded_dtypes : {list,set}, optional
        when automatically generating data, exclude these dtypes.

    Notes
    -----
    In actual usage this is deployed as a unit test decorator, providing a strategy
    that generates DataFrames or LazyFrames with the given characteristics for
    the unit test. While developing a strategy/test, it can also be useful to
    call `.example()` directly on a given strategy to see concrete instances of
    the generated data.

    Examples
    --------
    Use `column` or `columns` to specify the schema of the types of DataFrame to
    generate. Note: in actual use the strategy is applied as a test decorator, not
    used standalone.

    >>> from polars.testing.parametric import column, columns, dataframes
    >>> from hypothesis import given
    >>>
    >>> # generate arbitrary DataFrames
    >>> @given(df=dataframes())
    ... def test_repr(df: pl.DataFrame) -> None:
    ...     assert isinstance(repr(df), str)
    >>>
    >>> # generate LazyFrames with at least 1 column, random dtypes, and specific size:
    >>> df = dataframes(min_cols=1, lazy=True, max_size=5)
    >>> df.example()  # doctest: +SKIP
    >>>
    >>> # generate DataFrames with known colnames, random dtypes (per test, not per-frame):
    >>> df_strategy = dataframes(columns(["x", "y", "z"]))
    >>> df.example()  # doctest: +SKIP
    >>>
    >>> # generate frames with explicitly named/typed columns and a fixed size:
    >>> df_strategy = dataframes(
    ...     [
    ...         column("x", dtype=pl.Int32),
    ...         column("y", dtype=pl.Float64),
    ...     ],
    ...     size=2,
    ... )
    >>> df_strategy.example()  # doctest: +SKIP
    shape: (2, 2)
    ┌───────────┬────────────┐
    │ x         ┆ y          │
    │ ---       ┆ ---        │
    │ i32       ┆ f64        │
    ╞═══════════╪════════════╡
    │ -15836    ┆ 1.1755e-38 │
    ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 575050513 ┆ NaN        │
    └───────────┴────────────┘
    """  # noqa: 501
    if isinstance(cols, int):
        cols = columns(cols)
    if isinstance(min_size, int):
        if min_cols in (0, None):
            min_cols = 1

    selectable_dtypes = [
        dtype
        for dtype in (allowed_dtypes or strategy_dtypes)
        if dtype not in (excluded_dtypes or ())
    ]

    @composite
    def draw_frames(draw: DrawFn) -> pli.DataFrame | pli.LazyFrame:
        # if not given, create 'n' cols with random dtypes
        if cols is None:
            n = between(draw, int, min_=(min_cols or 0), max_=(max_cols or MAX_COLS))
            dtypes_ = [draw(sampled_from(selectable_dtypes)) for _ in range(n)]
            coldefs = columns(cols=n, dtype=dtypes_)
        elif isinstance(cols, column):
            coldefs = [cols]
        else:
            coldefs = list(cols)  # type: ignore[arg-type]

        # append any explicitly provided cols
        coldefs.extend(include_cols or ())

        # assign dataframe/series size
        series_size = (
            between(draw, int, min_=(min_size or 0), max_=(max_size or MAX_DATA_SIZE))
            if size is None
            else size
        )
        # init dataframe from generated series data; series data is
        # given as a python-native sequence (TODO: or as an arrow array).
        for idx, c in enumerate(coldefs):
            if c.name is None:
                c.name = f"col{idx}"
            if c.null_probability is None:
                if isinstance(null_probability, dict):
                    c.null_probability = null_probability.get(c.name, 0.0)
                else:
                    c.null_probability = null_probability

        frame_columns = [
            c.name if (c.dtype is None) else (c.name, c.dtype) for c in coldefs
        ]
        df = pli.DataFrame(
            data={
                c.name: draw(
                    series(
                        name=c.name,
                        dtype=c.dtype,
                        size=series_size,
                        null_probability=(c.null_probability or 0.0),
                        allow_infinities=allow_infinities,
                        strategy=c.strategy,
                        unique=c.unique,
                        chunked=(chunked is None and draw(booleans())),
                    )
                )
                for c in coldefs
            },
            columns=frame_columns,  # type: ignore[arg-type]
        )
        # optionally generate frames with n_chunks > 1
        if series_size > 1 and chunked is True:
            split_at = series_size // 2
            df = df[:split_at].vstack(df[split_at:])

        # optionally make lazy
        return df.lazy() if lazy else df

    return draw_frames()
