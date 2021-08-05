from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np
import pyarrow as pa

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "from_dict",
    "from_records",
    "from_arrow",
    "from_pandas",
    "from_rows",  # deprecated
    "from_arrow_table",  # deprecated
]


def from_dict(
    data: Dict[str, Sequence[Any]],
    columns: Optional[Sequence[str]] = None,
    nullable: bool = True,
) -> "pl.DataFrame":
    """
    Construct a DataFrame from a dictionary of sequences.

    Parameters
    ----------
    data : dict of sequences
        Two-dimensional data represented as a dictionary. dict must contain
        Sequences.
    columns : Sequence of str, default None
        Column labels to use for resulting DataFrame. If specified, overrides any
        labels already present in the data. Must match data dimensions.
    nullable : bool, default True
        If your data does not contain null values, set to False to speed up
        DataFrame creation.

    Returns
    -------
    DataFrame

    Examples
    --------

    >>> data = {'a': [1, 2], 'b': [3, 4]}
    >>> df = pl.from_dict(data)
    >>> df
    shape: (2, 2)
    ╭─────┬─────╮
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 4   │
    ╰─────┴─────╯
    """
    return pl.DataFrame._from_dict(data=data, columns=columns, nullable=nullable)


def from_records(
    data: Union[np.ndarray, Sequence[Sequence[Any]]],
    columns: Optional[Sequence[str]] = None,
    orient: Optional[str] = None,
    nullable: bool = True,
) -> "pl.DataFrame":
    """
    Construct a DataFrame from a numpy ndarray or sequence of sequences.

    Parameters
    ----------
    data : numpy ndarray or Sequence of sequences
        Two-dimensional data represented as numpy ndarray or sequence of sequences.
    columns : Sequence of str, default None
        Column labels to use for resulting DataFrame. Must match data dimensions.
        If not specified, columns will be named `column_0`, `column_1`, etc.
    orient : {'col', 'row'}, default None
        Whether to interpret two-dimensional data as columns or as rows. If None,
        the orientation is infered by matching the columns and data dimensions. If
        this does not yield conclusive results, column orientation is used.
    nullable : bool, default True
        If your data does not contain null values, set to False to speed up
        DataFrame creation.

    Returns
    -------
    DataFrame

    Examples
    --------

    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> df = pl.from_records(data, columns=['a', 'b'])
    >>> df
    shape: (3, 2)
    ╭─────┬─────╮
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 5   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 6   │
    ╰─────┴─────╯
    """
    return pl.DataFrame._from_records(
        data, columns=columns, orient=orient, nullable=nullable
    )


def from_arrow(
    a: Union[pa.Table, pa.Array], rechunk: bool = True
) -> Union["pl.DataFrame", "pl.Series"]:
    """
    Create a DataFrame or Series from an Arrow table or array.

    This operation will be zero copy for the most part. Types that are not
    supported by Polars may be cast to the closest supported type.

    Parameters
    ----------
    a : Arrow table or array
        Data represented as Arrow table or array.
    rechunk : bool, default True
        Make sure that all data is contiguous.

    Returns
    -------
    DataFrame or Series

    Examples
    --------
    Constructing a DataFrame from an Arrow table:

    >>> data = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df = pl.from_arrow(data)
    >>> df
    shape: (3, 2)
    ╭─────┬─────╮
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 5   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 3   ┆ 6   │
    ╰─────┴─────╯

    Constructing a Series from an Arrow array:

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
    if isinstance(a, pa.Table):
        return pl.DataFrame._from_arrow(a, rechunk=rechunk)
    elif isinstance(a, pa.Array):
        return pl.Series._from_arrow("", a)
    else:
        raise ValueError(f"Expected Arrow table or array, got {type(a)}.")


def from_pandas(
    df: Union["pd.DataFrame", "pd.Series", "pd.DatetimeIndex"],
    rechunk: bool = True,
) -> Union["pl.Series", "pl.DataFrame"]:
    """
    Construct a Polars DataFrame or Series from a pandas DataFrame or Series.

    Requires the pandas package to be installed.

    Parameters
    ----------
    data : pandas DataFrame, Series, or DatetimeIndex
        Data represented as a pandas DataFrame, Series, or DatetimeIndex.
    columns : Sequence of str, default None
        Column labels to use for resulting DataFrame. If specified, overrides any
        labels already present in the data. Must match data dimensions.
    rechunk : bool, default True
        Make sure that all data is contiguous.

    Returns
    -------
    DataFrame

    Examples
    --------
    Constructing a DataFrame from a pandas DataFrame:

    >>> pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    >>> df = pl.from_pandas(pd_df)
    >>> df
    shape: (2, 3)
    ╭─────┬─────┬─────╮
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ 4   ┆ 5   ┆ 6   │
    ╰─────┴─────┴─────╯

    Constructing a Series from a pandas Series:

    >>> pd_series = pd.Series([1,2,3], name='pd')
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
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("from_pandas requires pandas to be installed.") from e

    if isinstance(df, (pd.Series, pd.DatetimeIndex)):
        return pl.Series._from_pandas("", df)
    elif isinstance(df, pd.DataFrame):
        return pl.DataFrame._from_pandas(df, rechunk=rechunk)
    else:
        raise ValueError(f"Expected pandas DataFrame or Series, got {type(df)}.")


def from_rows(
    rows: Sequence[Sequence[Any]],
    column_names: Optional[Sequence[str]] = None,
    column_name_mapping: Optional[Dict[int, str]] = None,
) -> "pl.DataFrame":
    """
    .. deprecated:: 0.8.13
          `from_rows` will be removed in Polars 0.9.0, it is replaced by
          `from_records` because the latter offers more versatility. To keep the same
          functionality, call `from_records` with `orient='row'`

    Create a DataFrame from rows. This should only be used as a last resort, as this is
    more expensive than creating from columnar data.

    Parameters
    ----------
    rows
        rows.
    column_names
        column names to use for the DataFrame.
    column_name_mapping
        map column index to a new name:

        Example:
        --------

        >>> column_mapping: {0: "first_column", 3: "fourth column"}
    """
    return pl.DataFrame.from_rows(rows, column_names, column_name_mapping)


def from_arrow_table(table: pa.Table, rechunk: bool = True) -> "pl.DataFrame":
    """
    .. deprecated:: 7.3
        use `from_arrow`

    Create a DataFrame from an arrow Table.

    Parameters
    ----------
    a
        Arrow Table.
    rechunk
        Make sure that all data is contiguous.
    """
    import warnings

    warnings.warn(
        "from_arrow_table is deprecated, use DataFrame.from_arrow instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pl.DataFrame.from_arrow(table, rechunk)
