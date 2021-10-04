from typing import Optional, Sequence, Union

try:
    import pyarrow as pa

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

import polars as pl

try:
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_series as _concat_series

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


def concat(
    items: Union[Sequence["pl.DataFrame"], Sequence["pl.Series"]], rechunk: bool = True
) -> Union["pl.DataFrame", "pl.Series"]:
    """
    Aggregate all the Dataframes/Series in a List of DataFrames/Series to a single DataFrame/Series.

    Parameters
    ----------
    items
        DataFrames/Series to concatenate.
    rechunk
        rechunk the final DataFrame/Series.
    """
    if not len(items) > 0:
        raise ValueError("cannot concat empty list")

    out: Union["pl.Series", "pl.DataFrame"]
    if isinstance(items[0], pl.DataFrame):
        out = pl.wrap_df(_concat_df(items))
    else:
        out = pl.wrap_s(_concat_series(items))

    if rechunk:
        return out.rechunk()  # type: ignore
    return out


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
        if not _PYARROW_AVAILABLE:
            raise ImportError(
                "'pyarrow' is required for repeating a int or a float value."
            )
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
