from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence, overload

from polars.datatypes import N_INFER_DEFAULT, SchemaDefinition, SchemaDict
from polars.dependencies import _PYARROW_AVAILABLE
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.internals import DataFrame, Series
from polars.internals.construction import _unpack_schema, include_unknowns
from polars.utils import deprecated_alias

if TYPE_CHECKING:
    from polars.internals.type_aliases import Orientation


@deprecated_alias(columns="schema")
def from_dict(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series],
    schema: SchemaDefinition | None = None,
    *,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame:
    """
    Construct a DataFrame from a dictionary of sequences.

    This operation clones data, unless you pass a ``{str: pl.Series,}`` dict.

    Parameters
    ----------
    data : dict of sequences
        Two-dimensional data represented as a dictionary. dict must contain
        Sequences.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the columns param will be overridden.

    Returns
    -------
    :class:`DataFrame`

    Examples
    --------
    >>> df = pl.from_dict({"a": [1, 2], "b": [3, 4]})
    >>> df
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    """
    return DataFrame._from_dict(
        data=data, schema=schema, schema_overrides=schema_overrides
    )


def from_dicts(
    dicts: Sequence[dict[str, Any]],
    infer_schema_length: int | None = N_INFER_DEFAULT,
    *,
    schema: SchemaDefinition | None = None,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame:
    """
    Construct a DataFrame from a sequence of dictionaries. This operation clones data.

    Parameters
    ----------
    dicts
        Sequence with dictionaries mapping column name to value
    infer_schema_length
        How many dictionaries/rows to scan to determine the data types
        if set to `None` then ALL dicts are scanned; this will be slow.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If a list of column names is supplied that does NOT match the names in the
        underlying data, the names given here will overwrite the actual fields in
        the order that they appear - however, in this case it is typically clearer
        to rename after loading the frame.

        If you want to drop some of the fields found in the input dictionaries, a
        _partial_ schema can be declared, in which case omitted fields will not be
        loaded. Similarly you can extend the loaded frame with empty columns by adding
        them to the schema.
    schema_overrides : dict, default None
        Support override of inferred types for one or more columns.

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
    │ 2   ┆ 5   │
    │ 3   ┆ 6   │
    └─────┴─────┘

    Declaring a partial ``schema`` will drop the omitted columns.

    >>> df = pl.from_dicts(data, schema={"a": pl.Int32})
    >>> df
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i32 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 3   │
    └─────┘

    Can also use the ``schema`` param to extend the loaded columns with one
    or more additional (empty) columns that are not present in the input dicts:

    >>> pl.from_dicts(
    ...     data,
    ...     schema=["a", "b", "c", "d"],
    ...     schema_overrides={"c": pl.Float64, "d": pl.Utf8},
    ... )
    shape: (3, 4)
    ┌─────┬─────┬──────┬──────┐
    │ a   ┆ b   ┆ c    ┆ d    │
    │ --- ┆ --- ┆ ---  ┆ ---  │
    │ i64 ┆ i64 ┆ f64  ┆ str  │
    ╞═════╪═════╪══════╪══════╡
    │ 1   ┆ 4   ┆ null ┆ null │
    │ 2   ┆ 5   ┆ null ┆ null │
    │ 3   ┆ 6   ┆ null ┆ null │
    └─────┴─────┴──────┴──────┘

    """
    column_names, schema = _unpack_schema(
        schema, schema_overrides=schema_overrides, include_overrides_in_columns=True
    )
    schema = include_unknowns(schema, column_names or list(schema))
    return DataFrame._from_dicts(
        dicts,
        infer_schema_length,
        schema=(column_names and schema),
        schema_overrides=schema_overrides,
    )


@deprecated_alias(columns="schema")
def from_records(
    data: Sequence[Sequence[Any]],
    schema: Sequence[str] | None = None,
    *,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
) -> DataFrame:
    """
    Construct a DataFrame from a sequence of sequences. This operation clones data.

    Note that this is slower than creating from columnar memory.

    Parameters
    ----------
    data : Sequence of sequences
        Two-dimensional data represented as a sequence of sequences.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the columns param will be overridden.
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
    >>> df = pl.from_records(data, schema=["a", "b"])
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    │ 2   ┆ 5   │
    │ 3   ┆ 6   │
    └─────┴─────┘

    """
    return DataFrame._from_records(
        data,
        schema=schema,
        schema_overrides=schema_overrides,
        orient=orient,
        infer_schema_length=infer_schema_length,
    )


@deprecated_alias(columns="schema")
def from_numpy(
    data: np.ndarray[Any, Any],
    schema: SchemaDefinition | None = None,
    *,
    schema_overrides: SchemaDict | None = None,
    orient: Orientation | None = None,
) -> DataFrame:
    """
    Construct a DataFrame from a numpy ndarray. This operation clones data.

    Note that this is slower than creating from columnar memory.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Two-dimensional data represented as a numpy ndarray.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the columns param will be overridden.
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
    >>> df = pl.from_numpy(data, schema=["a", "b"], orient="col")
    >>> df
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    │ 2   ┆ 5   │
    │ 3   ┆ 6   │
    └─────┴─────┘

    """
    return DataFrame._from_numpy(
        data, schema=schema, orient=orient, schema_overrides=schema_overrides
    )


def from_arrow(
    a: pa.Table | pa.Array | pa.ChunkedArray,
    rechunk: bool = True,
    schema: Sequence[str] | None = None,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame | Series:
    """
    Create a DataFrame or Series from an Arrow Table or Array.

    This operation will be zero copy for the most part. Types that are not
    supported by Polars may be cast to the closest supported type.

    Parameters
    ----------
    a : :class:`pyarrow.Table` or :class:`pyarrow.Array`
        Data representing an Arrow Table or Array.
    rechunk : bool, default True
        Make sure that all data is in contiguous memory.
    schema : Sequence of str, dict, default None
        Column labels to use for resulting DataFrame. Must match data dimensions.
        If not specified, existing Array table columns are used, with missing names
        named as `column_0`, `column_1`, etc.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The DataFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the schema param will be overridden.

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
    │ 2   ┆ 5   │
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
    if isinstance(a, pa.Table):
        return DataFrame._from_arrow(
            a, rechunk=rechunk, schema=schema, schema_overrides=schema_overrides
        )
    elif isinstance(a, (pa.Array, pa.ChunkedArray)):
        return Series._from_arrow("", a, rechunk)
    else:
        raise ValueError(f"Expected Arrow Table or Array, got {type(a)}.")


@overload
def from_pandas(
    df: pd.DataFrame,
    rechunk: bool = True,
    nan_to_null: bool = True,
    schema_overrides: SchemaDict | None = None,
    *,
    include_index: bool = False,
) -> DataFrame:
    ...


@overload
def from_pandas(
    df: pd.Series | pd.DatetimeIndex,
    rechunk: bool = True,
    nan_to_null: bool = True,
    schema_overrides: SchemaDict | None = None,
    *,
    include_index: bool = False,
) -> Series:
    ...


@deprecated_alias(nan_to_none="nan_to_null")
def from_pandas(
    df: pd.DataFrame | pd.Series | pd.DatetimeIndex,
    rechunk: bool = True,
    nan_to_null: bool = True,
    schema_overrides: SchemaDict | None = None,
    *,
    include_index: bool = False,
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
    nan_to_null : bool, default True
        If data contains `NaN` values PyArrow will convert the ``NaN`` to ``None``
    schema_overrides : dict, default None
        Support override of inferred types for one or more columns.
    include_index : bool, default False
        Load any non-default pandas indexes as columns.

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
    if isinstance(df, (pd.Series, pd.DatetimeIndex)):
        return Series._from_pandas("", df, nan_to_null=nan_to_null)
    elif isinstance(df, pd.DataFrame):
        return DataFrame._from_pandas(
            df,
            rechunk=rechunk,
            nan_to_null=nan_to_null,
            schema_overrides=schema_overrides,
            include_index=include_index,
        )
    else:
        raise ValueError(f"Expected pandas DataFrame or Series, got {type(df)}.")


def from_dataframe(df: Any, allow_copy: bool = True) -> DataFrame:
    """
    Build a Polars DataFrame from any dataframe supporting the interchange protocol.

    Parameters
    ----------
    df
        Object supporting the dataframe interchange protocol, i.e. must have implemented
        the ``__dataframe__`` method.
    allow_copy
        Allow memory to be copied to perform the conversion. If set to False, causes
        conversions that are not zero-copy to fail.

    Notes
    -----
    Details on the dataframe interchange protocol:
    https://data-apis.org/dataframe-protocol/latest/index.html

    Zero-copy conversions currently cannot be guaranteed and will throw a
    ``RuntimeError``.

    Using a dedicated function like :func:`from_pandas` or :func:`from_arrow` is a more
    efficient method of conversion.

    """
    if isinstance(df, DataFrame):
        return df
    if not hasattr(df, "__dataframe__"):
        raise TypeError(
            f"`df` of type {type(df)} does not support the dataframe interchange"
            " protocol."
        )
    if not _PYARROW_AVAILABLE or int(pa.__version__.split(".")[0]) < 11:
        raise ImportError(
            "pyarrow>=11.0.0 is required for converting a dataframe interchange object"
            " to a Polars dataframe."
        )

    import pyarrow.interchange  # noqa: F401

    pa_table = pa.interchange.from_dataframe(df, allow_copy=allow_copy)
    return from_arrow(pa_table, rechunk=allow_copy)  # type: ignore[return-value]
