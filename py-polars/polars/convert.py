from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "from_dict",
    "from_rows",
    "from_arrow",
    "from_pandas",
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
    ```python
    >>> data = {'a': [1, 2], 'b': [3, 4]}
    >>> df = pl.DataFrame.from_dict(data)
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
    ```
    """
    return pl.DataFrame._from_dict(data=data, columns=columns, nullable=nullable)


def from_rows(
    rows: Sequence[Sequence[Any]],
    column_names: Optional[Sequence[str]] = None,
    column_name_mapping: Optional[Dict[int, str]] = None,
) -> "pl.DataFrame":
    """
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
        ```python
            column_mapping: {0: "first_column, 3: "fourth column"}
        ```
    """
    return pl.DataFrame.from_rows(rows, column_names, column_name_mapping)


def from_arrow(
    a: Union[pa.Table, pa.Array], rechunk: bool = True
) -> Union["pl.DataFrame", "pl.Series"]:
    """
    Create a DataFrame from an arrow Table.

    Parameters
    ----------
    a
        Arrow Table.
    rechunk
        Make sure that all data is contiguous.
    """
    if isinstance(a, pa.Table):
        return pl.DataFrame.from_arrow(a, rechunk)
    elif isinstance(a, pa.Array):
        return pl.Series.from_arrow("", a)
    else:
        raise ValueError(f"expected arrow table / array, got {a}")


def _from_pandas_helper(a: Union["pd.Series", "pd.DatetimeIndex"]) -> pa.Array:
    dtype = a.dtype
    if dtype == "datetime64[ns]":
        # We first cast to ms because that's the unit of Date64,
        # Then we cast to via int64 to date64. Casting directly to Date64 lead to
        # loss of time information https://github.com/ritchie46/polars/issues/476
        arr = pa.array(np.array(a.values, dtype="datetime64[ms]"))
        arr = pa.compute.cast(arr, pa.int64())
        return pa.compute.cast(arr, pa.date64())
    elif dtype == "object" and isinstance(a.iloc[0], str):
        return pa.array(a, pa.large_utf8())
    else:
        return pa.array(a)


def from_pandas(
    df: Union["pd.DataFrame", "pd.Series", "pd.DatetimeIndex"],
    rechunk: bool = True,
) -> Union["pl.Series", "pl.DataFrame"]:
    """
    Convert from a pandas DataFrame to a polars DataFrame.

    Parameters
    ----------
    df
        DataFrame to convert.
    rechunk
        Make sure that all data is contiguous.

    Returns
    -------
    A Polars DataFrame
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("from_pandas requires pandas to be installed.") from e

    if isinstance(df, (pd.Series, pd.DatetimeIndex)):
        return from_arrow(_from_pandas_helper(df))

    # Note: we first tried to infer the schema via pyarrow and then modify the schema if
    # needed. However arrow 3.0 determines the type of a string like this:
    #       pa.array(array).type
    # Needlessly allocating and failing when the string is too large for the string dtype.

    data = {}

    for name in df.columns:
        s = df[name]
        data[name] = _from_pandas_helper(s)

    table = pa.table(data)
    return from_arrow(table, rechunk)


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
