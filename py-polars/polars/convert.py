from __future__ import annotations

import io
import re
from itertools import chain, zip_longest
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, overload

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import N_INFER_DEFAULT, Categorical, List, Object, Struct, Utf8
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import NoDataError
from polars.io import read_csv
from polars.utils.various import _cast_repr_strings_with_schema

if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.dependencies import numpy as np
    from polars.type_aliases import Orientation, SchemaDefinition, SchemaDict


def from_dict(
    data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series],
    schema: SchemaDefinition | None = None,
    *,
    schema_overrides: SchemaDict | None = None,
) -> DataFrame:
    """
    Construct a DataFrame from a dictionary of sequences.

    This operation clones data, unless you pass a `{str: pl.Series,}` dict.

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
    DataFrame

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
    return pl.DataFrame._from_dict(
        data, schema=schema, schema_overrides=schema_overrides
    )


def from_dicts(
    data: Sequence[dict[str, Any]],
    schema: SchemaDefinition | None = None,
    *,
    schema_overrides: SchemaDict | None = None,
    infer_schema_length: int | None = N_INFER_DEFAULT,
) -> DataFrame:
    """
    Construct a DataFrame from a sequence of dictionaries. This operation clones data.

    Parameters
    ----------
    data
        Sequence with dictionaries mapping column name to value
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
        *partial* schema can be declared, in which case omitted fields will not be
        loaded. Similarly, you can extend the loaded frame with empty columns by
        adding them to the schema.
    schema_overrides : dict, default None
        Support override of inferred types for one or more columns.
    infer_schema_length
        How many dictionaries/rows to scan to determine the data types
        if set to `None` then ALL dicts are scanned; this will be slow.

    Returns
    -------
    DataFrame

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

    Declaring a partial `schema` will drop the omitted columns.

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

    Can also use the `schema` param to extend the loaded columns with one
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
    if not data and not (schema or schema_overrides):
        raise NoDataError("no data, cannot infer schema")

    return pl.DataFrame(
        data,
        schema=schema,
        schema_overrides=schema_overrides,
        infer_schema_length=infer_schema_length,
    )


def from_records(
    data: Sequence[Any],
    schema: SchemaDefinition | None = None,
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
    DataFrame

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
    return pl.DataFrame._from_records(
        data,
        schema=schema,
        schema_overrides=schema_overrides,
        orient=orient,
        infer_schema_length=infer_schema_length,
    )


def _from_dataframe_repr(m: re.Match[str]) -> DataFrame:
    """Reconstruct a DataFrame from a regex-matched table repr."""
    from polars.datatypes.convert import dtype_short_repr_to_dtype

    # extract elements from table structure
    lines = m.group().split("\n")[1:-1]
    rows = [
        [re.sub(r"^[\W+]*│", "", elem).strip() for elem in row]
        for row in [re.split("[┆|]", row.rstrip("│ ")) for row in lines]
        if len(row) > 1 or not re.search("├[╌┼]+┤", row[0])
    ]

    # determine beginning/end of the header block
    table_body_start = 2
    for idx, (elem, *_) in enumerate(rows):
        if re.match(r"^\W*╞", elem):
            table_body_start = idx
            break

    # handle headers with wrapped column names and determine headers/dtypes
    header_block = ["".join(h).split("---") for h in zip(*rows[:table_body_start])]
    dtypes: list[str | None]
    if all(len(h) == 1 for h in header_block):
        headers = [h[0] for h in header_block]
        dtypes = [None] * len(headers)
    else:
        headers, dtypes = (list(h) for h in zip_longest(*header_block))

    body = rows[table_body_start + 1 :]
    no_dtypes = all(d is None for d in dtypes)

    # transpose rows into columns, detect/omit truncated columns
    coldata = list(zip(*(row for row in body if not all((e == "…") for e in row))))
    for el in ("…", "..."):
        if el in headers:
            idx = headers.index(el)
            for table_elem in (headers, dtypes):
                table_elem.pop(idx)  # type: ignore[attr-defined]
            if coldata:
                coldata.pop(idx)

    # init cols as utf8 Series, handle "null" -> None, create schema from repr dtype
    data = [
        pl.Series([(None if v == "null" else v) for v in cd], dtype=Utf8)
        for cd in coldata
    ]
    schema = dict(zip(headers, (dtype_short_repr_to_dtype(d) for d in dtypes)))
    if schema and data and (n_extend_cols := (len(schema) - len(data))) > 0:
        empty_data = [None] * len(data[0])
        data.extend((pl.Series(empty_data, dtype=Utf8)) for _ in range(n_extend_cols))
    for dtype in set(schema.values()):
        if dtype in (List, Struct, Object):
            raise NotImplementedError(
                f"`from_repr` does not support data type {dtype.base_type().__name__!r}"
            )

    # construct DataFrame from string series and cast from repr to native dtype
    df = pl.DataFrame(data=data, orient="col", schema=list(schema))
    if no_dtypes:
        if df.is_empty():
            # if no dtypes *and* empty, default to string
            return df.with_columns(F.all().cast(Utf8))
        else:
            # otherwise, take a trip through our CSV inference logic
            if all(tp == Utf8 for tp in df.schema.values()):
                buf = io.BytesIO()
                df.write_csv(file=buf)
                df = read_csv(buf, new_columns=df.columns, try_parse_dates=True)
            return df
    elif schema and not data:
        return df.cast(schema)  # type: ignore[arg-type]
    else:
        return _cast_repr_strings_with_schema(df, schema)


def _from_series_repr(m: re.Match[str]) -> Series:
    """Reconstruct a Series from a regex-matched series repr."""
    from polars.datatypes.convert import dtype_short_repr_to_dtype

    shape = m.groups()[0]
    name = m.groups()[1][1:-1]
    length = int(shape[1:-2] if shape else -1)
    dtype = dtype_short_repr_to_dtype(m.groups()[2])

    if length == 0:
        string_values = []
    else:
        string_values = [
            v.strip()
            for v in re.findall(r"[\s>#]*(?:\t|\s{4,})([^\n]*)\n", m.groups()[-1])
        ]
        if string_values == ["[", "]"]:
            string_values = []
        elif string_values and string_values[0].lstrip("#> ") == "[":
            string_values = string_values[1:]

    values = string_values[:length] if length > 0 else string_values
    values = [(None if v == "null" else v) for v in values if v not in ("…", "...")]

    if not values:
        return pl.Series(name=name, values=values, dtype=dtype)
    else:
        srs = pl.Series(name=name, values=values, dtype=Utf8)
        if dtype is None:
            return srs
        elif dtype in (Categorical, Utf8):
            return srs.str.replace('^"(.*)"$', r"$1").cast(dtype)

        return _cast_repr_strings_with_schema(
            srs.to_frame(), schema={srs.name: dtype}
        ).to_series()


def from_repr(tbl: str) -> DataFrame | Series:
    """
    Utility function that reconstructs a DataFrame or Series from the object's repr.

    Parameters
    ----------
    tbl
        A string containing a polars DataFrame or Series repr; does not need
        to be trimmed of whitespace (or leading prompts) as the repr will be
        found/extracted automatically.

    Notes
    -----
    This function handles the default UTF8_FULL and UTF8_FULL_CONDENSED DataFrame
    tables (with or without rounded corners). Truncated columns/rows are omitted,
    wrapped headers are accounted for, and dtypes automatically identified.

    Currently compound/nested dtypes such as List and Struct are not supported;
    neither are Object dtypes.

    See Also
    --------
    polars.DataFrame.to_init_repr
    polars.Series.to_init_repr

    Examples
    --------
    From DataFrame table repr:

    >>> df = pl.from_repr(
    ...     '''
    ...     Out[3]:
    ...     shape: (1, 5)
    ...     ┌───────────┬────────────┬───┬───────┬────────────────────────────────┐
    ...     │ source_ac ┆ source_cha ┆ … ┆ ident ┆ timestamp                      │
    ...     │ tor_id    ┆ nnel_id    ┆   ┆ ---   ┆ ---                            │
    ...     │ ---       ┆ ---        ┆   ┆ str   ┆ datetime[μs, Asia/Tokyo]       │
    ...     │ i32       ┆ i64        ┆   ┆       ┆                                │
    ...     ╞═══════════╪════════════╪═══╪═══════╪════════════════════════════════╡
    ...     │ 123456780 ┆ 9876543210 ┆ … ┆ a:b:c ┆ 2023-03-25 10:56:59.663053 JST │
    ...     │ …         ┆ …          ┆ … ┆ …     ┆ …                              │
    ...     │ 803065983 ┆ 2055938745 ┆ … ┆ x:y:z ┆ 2023-03-25 12:38:18.050545 JST │
    ...     └───────────┴────────────┴───┴───────┴────────────────────────────────┘
    ... '''
    ... )
    >>> df
    shape: (2, 4)
    ┌─────────────────┬───────────────────┬───────┬────────────────────────────────┐
    │ source_actor_id ┆ source_channel_id ┆ ident ┆ timestamp                      │
    │ ---             ┆ ---               ┆ ---   ┆ ---                            │
    │ i32             ┆ i64               ┆ str   ┆ datetime[μs, Asia/Tokyo]       │
    ╞═════════════════╪═══════════════════╪═══════╪════════════════════════════════╡
    │ 123456780       ┆ 9876543210        ┆ a:b:c ┆ 2023-03-25 10:56:59.663053 JST │
    │ 803065983       ┆ 2055938745        ┆ x:y:z ┆ 2023-03-25 12:38:18.050545 JST │
    └─────────────────┴───────────────────┴───────┴────────────────────────────────┘

    From Series repr:

    >>> s = pl.from_repr(
    ...     '''
    ...     shape: (3,)
    ...     Series: 's' [bool]
    ...     [
    ...        true
    ...        false
    ...        true
    ...     ]
    ...     '''
    ... )
    >>> s.to_list()
    [True, False, True]

    """
    # find DataFrame table...
    m = re.search(r"([┌╭].*?[┘╯])", tbl, re.DOTALL)
    if m is not None:
        return _from_dataframe_repr(m)

    # ...or Series in the given string
    m = re.search(
        pattern=r"(?:shape: (\(\d+,\))\n.*?)?Series:\s+([^\n]+)\s+\[([^\n]+)](.*)",
        string=tbl,
        flags=re.DOTALL,
    )
    if m is not None:
        return _from_series_repr(m)

    raise ValueError("input string does not contain DataFrame or Series")


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
    DataFrame

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
    return pl.DataFrame._from_numpy(
        data, schema=schema, orient=orient, schema_overrides=schema_overrides
    )


# Note: we cannot @overload the typing (Series vs DataFrame) here, as pyarrow
# does not implement any support for type hints; attempts to hint here will
# simply result in mypy inferring "Any", which isn't useful...


def from_arrow(
    data: (
        pa.Table
        | pa.Array
        | pa.ChunkedArray
        | pa.RecordBatch
        | Iterable[pa.RecordBatch | pa.Table]
    ),
    schema: SchemaDefinition | None = None,
    *,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
) -> DataFrame | Series:
    """
    Create a DataFrame or Series from an Arrow Table or Array.

    This operation will be zero copy for the most part. Types that are not
    supported by Polars may be cast to the closest supported type.

    Parameters
    ----------
    data : :class:`pyarrow.Table`, :class:`pyarrow.Array`, one or more :class:`pyarrow.RecordBatch`
        Data representing an Arrow Table, Array, or sequence of RecordBatches or Tables.
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
    rechunk : bool, default True
        Make sure that all data is in contiguous memory.

    Returns
    -------
    DataFrame or Series

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
    >>> series = pl.from_arrow(data, schema={"s": pl.Int32})
    >>> series
    shape: (3,)
    Series: 's' [i32]
    [
        1
        2
        3
    ]

    """  # noqa: W505
    if isinstance(data, pa.Table):
        return pl.DataFrame._from_arrow(
            data=data, rechunk=rechunk, schema=schema, schema_overrides=schema_overrides
        )
    elif isinstance(data, (pa.Array, pa.ChunkedArray)):
        name = getattr(data, "_name", "") or ""
        s = pl.DataFrame(
            data=pl.Series._from_arrow(name, data, rechunk=rechunk),
            schema=schema,
            schema_overrides=schema_overrides,
        ).to_series()
        return s if (name or schema or schema_overrides) else s.alias("")
    elif not data:
        return pl.DataFrame(
            schema=schema,
            schema_overrides=schema_overrides,
        )

    if isinstance(data, pa.RecordBatch):
        data = [data]
    if isinstance(data, Iterable):
        return pl.DataFrame._from_arrow(
            data=pa.Table.from_batches(
                chain.from_iterable(
                    (b.to_batches() if isinstance(b, pa.Table) else [b]) for b in data
                )
            ),
            rechunk=rechunk,
            schema=schema,
            schema_overrides=schema_overrides,
        )

    raise TypeError(
        f"expected PyArrow Table, Array, or one or more RecordBatches; got {type(data).__name__!r}"
    )


@overload
def from_pandas(
    data: pd.DataFrame,
    *,
    schema_overrides: SchemaDict | None = ...,
    rechunk: bool = ...,
    nan_to_null: bool = ...,
    include_index: bool = ...,
) -> DataFrame:
    ...


@overload
def from_pandas(
    data: pd.Series[Any] | pd.Index[Any],
    *,
    schema_overrides: SchemaDict | None = ...,
    rechunk: bool = ...,
    nan_to_null: bool = ...,
    include_index: bool = ...,
) -> Series:
    ...


def from_pandas(
    data: pd.DataFrame | pd.Series[Any] | pd.Index[Any],
    *,
    schema_overrides: SchemaDict | None = None,
    rechunk: bool = True,
    nan_to_null: bool = True,
    include_index: bool = False,
) -> DataFrame | Series:
    """
    Construct a Polars DataFrame or Series from a pandas DataFrame or Series.

    This operation clones data.

    This requires that :mod:`pandas` and :mod:`pyarrow` are installed.

    Parameters
    ----------
    data : :class:`pandas.DataFrame` or :class:`pandas.Series` or :class:`pandas.Index`
        Data represented as a pandas DataFrame, Series, or Index.
    schema_overrides : dict, default None
        Support override of inferred types for one or more columns.
    rechunk : bool, default True
        Make sure that all data is in contiguous memory.
    nan_to_null : bool, default True
        If data contains `NaN` values PyArrow will convert the `NaN` to `None`
    include_index : bool, default False
        Load any non-default pandas indexes as columns.

    Returns
    -------
    DataFrame

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
    if isinstance(data, (pd.Series, pd.DatetimeIndex)):
        return pl.Series._from_pandas("", data, nan_to_null=nan_to_null)
    elif isinstance(data, pd.DataFrame):
        return pl.DataFrame._from_pandas(
            data,
            rechunk=rechunk,
            nan_to_null=nan_to_null,
            schema_overrides=schema_overrides,
            include_index=include_index,
        )
    else:
        raise TypeError(
            f"expected pandas DataFrame or Series, got {type(data).__name__!r}"
        )
