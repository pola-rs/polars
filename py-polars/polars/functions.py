from typing import Optional, Sequence, Union

import pyarrow as pa

import polars as pl

try:
    from polars.polars import concat_df as _concat_df

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

__all__ = [
    "get_dummies",
    "concat",
    "repeat",
    "arg_where",
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
    if not len(dfs) > 0:
        raise ValueError("cannot concat empty list")

    df = pl.wrap_df(_concat_df(dfs))

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
        return pl.Series._from_arrow(name, pa.repeat(val, n))


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
