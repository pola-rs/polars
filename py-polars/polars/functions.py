from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute

import polars as pl

try:
    import pandas as pd
except ImportError:
    pass

__all__ = [
    "get_dummies",
    "concat",
    "repeat",
    "arg_where",
    "from_rows",
    "from_arrow",
    "from_pandas",
    "from_arrow_table",  # deprecated
]


def get_dummies(df: "pl.DataFrame") -> "pl.DataFrame":
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    """
    return df.to_dummies()


def concat(dfs: Sequence["pl.DataFrame"], rechunk: bool = True) -> "pl.DataFrame":
    """
    Aggregate all the Dataframes in a List of DataFrames to a single DataFrame.

    Parameters
    ----------
    dfs
        DataFrames to concatenate.
    rechunk
        rechunk the final DataFrame.
    """
    assert len(dfs) > 0
    df = dfs[0].clone()
    for i in range(1, len(dfs)):
        try:
            df = df.vstack(dfs[i], in_place=False)  # type: ignore[assignment]
        # could have a double borrow (one mutable one ref)
        except RuntimeError:
            df.vstack(dfs[i].clone(), in_place=True)

    if rechunk:
        return df.rechunk()
    return df


def repeat(
    val: Union[int, float, str], n: int, name: Optional[str] = None
) -> "pl.Series":
    """
    Repeat a single value n times and collect into a Series.

    Parameters
    ----------
    val
        Value to repeat.
    n
        Number of repeats.
    name
        Optional name of the Series.
    """
    if name is None:
        name = ""
    if isinstance(val, str):
        s = pl.Series._repeat(name, val, n)
        s.rename(name)
        return s
    else:
        return pl.Series.from_arrow(name, pa.repeat(val, n))


def arg_where(mask: "pl.Series") -> "pl.Series":
    """
    Get index values where Boolean mask evaluate True.

    Parameters
    ----------
    mask
        Boolean Series.

    Returns
    -------
    UInt32 Series
    """
    return mask.arg_true()


def from_rows(
    rows: Sequence[Sequence[Any]],
    column_names: Optional[Sequence[str]] = None,
    column_name_mapping: Optional[Dict[int, str]] = None,
) -> "pl.DataFrame":
    """
    Create a DataFrame from rows. This should only be used as a last resort, as this is more expensive than
    creating from columnar data.
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


def _from_pandas_helper(a: "pd.Series") -> pa.Array:  # noqa: F821
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
    rechunk: bool = True,  # noqa: F821
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
    if isinstance(df, pd.Series) or isinstance(df, pd.DatetimeIndex):
        return from_arrow(_from_pandas_helper(df))

    # Note: we first tried to infer the schema via pyarrow and then modify the schema if needed.
    # However arrow 3.0 determines the type of a string like this:
    #       pa.array(array).type
    # needlessly allocating and failing when the string is too large for the string dtype.
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
    return pl.DataFrame.from_arrow(table, rechunk)
