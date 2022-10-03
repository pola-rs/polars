from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence, overload

from polars.internals import DataFrame, Series

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    from polars.internals.type_aliases import Orientation


def from_dict(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]]],
    columns: Sequence[str] | None = None,
) -> DataFrame:
    """
    Construct a DataFrame from a dictionary of sequences.

    This operation clones data, unless you pass in a ``Dict[str, pl.Series]``.

    Parameters
    ----------
    data : dict of sequences
        Two-dimensional data represented as a dictionary. dict must contain
        Sequences.
    columns : Sequence of str, default None
        Column labels to use for resulting DataFrame. If specified, overrides any
        labels already present in the data. Must match data dimensions.

    Returns
    -------
    :class:`DataFrame`

    Examples
    --------
    >>> data = {"a": [1, 2], "b": [3, 4]}
    >>> df = pl.from_dict(data)
    >>> df
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
    return DataFrame._from_dict(data=data, columns=columns)


def from_dicts(
    dicts: Sequence[dict[str, Any]], infer_schema_length: int | None = 50
) -> DataFrame:
    """
    Construct a DataFrame from a sequence of dictionaries. This operation clones data.

    Parameters
    ----------
    dicts
        Sequence with dictionaries mapping column name to value
    infer_schema_length
        How many dictionaries/rows to scan to determine the data types
        if set to `None` all rows are scanned. This will be slow.

    Returns
    -------
    :class:`DataFrame`

    Examples
    --------
    >>> data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    >>> df = pl.from_dicts(data)
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 5   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 6   │
    └─────┴─────┘

    """
    return DataFrame._from_dicts(dicts, infer_schema_length)


def from_records(
    data: Sequence[Sequence[Any]],
    columns: Sequence[str] | None = None,
    orient: Orientation | None = None,
    infer_schema_length: int | None = 50,
) -> DataFrame:
    """
    Construct a DataFrame from a sequence of sequences. This operation clones data.

    Note that this is slower than creating from columnar memory.

    Parameters
    ----------
    data : Sequence of sequences
        Two-dimensional data represented as a sequence of sequences.
    columns : Sequence of str, default None
        Column labels to use for resulting DataFrame. Must match data dimensions.
        If not specified, columns will be named `column_0`, `column_1`, etc.
    orient : {None, 'col', 'row'}
        Whether to interpret two-dimensional data as columns or as rows. If None,
        the orientation is inferred by matching the columns and data dimensions. If
        this does not yield conclusive results, column orientation is used.
    infer_schema_length
        How many dictionaries/rows to scan to determine the data types
        if set to `None` all rows are scanned. This will be slow.

    Returns
    -------
    :class:`DataFrame`

    Examples
    --------
    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> df = pl.from_records(data, columns=["a", "b"])
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 5   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 6   │
    └─────┴─────┘

    """
    return DataFrame._from_records(
        data, columns=columns, orient=orient, infer_schema_length=infer_schema_length
    )


def from_numpy(
    data: np.ndarray[Any, Any],
    columns: Sequence[str] | None = None,
    orient: Orientation | None = None,
) -> DataFrame:
    """
    Construct a DataFrame from a numpy ndarray. This operation clones data.

    Note that this is slower than creating from columnar memory.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Two-dimensional data represented as a numpy ndarray.
    columns : Sequence of str, default None
        Column labels to use for resulting DataFrame. Must match data dimensions.
        If not specified, columns will be named `column_0`, `column_1`, etc.
    orient : {None, 'col', 'row'}
        Whether to interpret two-dimensional data as columns or as rows. If None,
        the orientation is inferred by matching the columns and data dimensions. If
        this does not yield conclusive results, column orientation is used.

    Returns
    -------
    :class:`DataFrame`

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> df = pl.from_numpy(data, columns=["a", "b"], orient="col")
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 5   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 6   │
    └─────┴─────┘

    """
    if not _NUMPY_AVAILABLE:
        raise ImportError("'numpy' is required when using from_numpy().")
    return DataFrame._from_numpy(data, columns=columns, orient=orient)


def from_arrow(
    a: pa.Table | pa.Array | pa.ChunkedArray, rechunk: bool = True
) -> DataFrame | Series:
    """
    Create a DataFrame or Series from an Arrow Table or Array.

    This operation will be zero copy for the most part. Types that are not
    supported by Polars may be cast to the closest supported type.

    Parameters
    ----------
    a : :class:`pyarrow.Table` or :class:`pyarrow.Array`
        Data represented as Arrow Table or Array.
    rechunk : bool, default True
        Make sure that all data is in contiguous memory.

    Returns
    -------
    :class:`DataFrame` or :class:`Series`

    Examples
    --------
    Constructing a DataFrame from an Arrow Table:

    >>> import pyarrow as pa
    >>> data = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> df = pl.from_arrow(data)
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 5   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 6   │
    └─────┴─────┘

    Constructing a Series from an Arrow Array:

    >>> import pyarrow as pa
    >>> data = pa.array([1, 2, 3])
    >>> series = pl.from_arrow(data)
    >>> series
    shape: (3,)
    Series: '' [i64]
    [
        1
        2
        3
    ]

    """
    if not _PYARROW_AVAILABLE:
        raise ImportError("'pyarrow' is required when using from_arrow().")
    if isinstance(a, pa.Table):
        return DataFrame._from_arrow(a, rechunk=rechunk)
    elif isinstance(a, (pa.Array, pa.ChunkedArray)):
        return Series._from_arrow("", a, rechunk)
    else:
        raise ValueError(f"Expected Arrow Table or Array, got {type(a)}.")


@overload
def from_pandas(
    df: pd.DataFrame,
    rechunk: bool = True,
    nan_to_none: bool = True,
) -> DataFrame:
    ...


@overload
def from_pandas(
    df: pd.Series | pd.DatetimeIndex,
    rechunk: bool = True,
    nan_to_none: bool = True,
) -> Series:
    ...


def from_pandas(
    df: pd.DataFrame | pd.Series | pd.DatetimeIndex,
    rechunk: bool = True,
    nan_to_none: bool = True,
) -> DataFrame | Series:
    """
    Construct a Polars DataFrame or Series from a pandas DataFrame or Series.

    This operation clones data.

    This requires that :mod:`pandas` and :mod:`pyarrow` are installed.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`, :class:`pandas.Series`, :class:`pandas.DatetimeIndex`
        Data represented as a pandas DataFrame, Series, or DatetimeIndex.
    rechunk : bool, default True
        Make sure that all data is in contiguous memory.
    nan_to_none : bool, default True
        If data contains `NaN` values PyArrow will convert the ``NaN`` to ``None``

    Returns
    -------
    :class:`DataFrame`

    Examples
    --------
    Constructing a :class:`DataFrame` from a :class:`pandas.DataFrame`:

    >>> import pandas as pd
    >>> pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    >>> df = pl.from_pandas(pd_df)
    >>> df
        shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 4   ┆ 5   ┆ 6   │
    └─────┴─────┴─────┘

    Constructing a Series from a :class:`pd.Series`:

    >>> import pandas as pd
    >>> pd_series = pd.Series([1, 2, 3], name="pd")
    >>> df = pl.from_pandas(pd_series)
    >>> df
    shape: (3,)
    Series: 'pd' [i64]
    [
        1
        2
        3
    ]

    """
    if not _PYARROW_AVAILABLE:
        raise ImportError("'pyarrow' is required when using from_pandas().")
    if not _PANDAS_AVAILABLE:
        raise ImportError("'pandas' is required when using from_pandas().")

    if isinstance(df, (pd.Series, pd.DatetimeIndex)):
        return Series._from_pandas("", df, nan_to_none=nan_to_none)
    elif isinstance(df, pd.DataFrame):
        return DataFrame._from_pandas(df, rechunk=rechunk, nan_to_none=nan_to_none)
    else:
        raise ValueError(f"Expected pandas DataFrame or Series, got {type(df)}.")
