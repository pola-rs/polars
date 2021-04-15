try:
    from .polars import (  # noqa: F401
        PyDataFrame,
        PySeries,
        toggle_string_cache as pytoggle_string_cache,
        version,
    )
except ImportError:
    import warnings

    warnings.warn("binary files missing")
    __pdoc__ = {"wrap_df": False}

from typing import (
    Dict,
    Sequence,
    List,
    Tuple,
    Optional,
    Union,
    TextIO,
    BinaryIO,
    Callable,
    Any,
)
from .series import Series, wrap_s
from . import datatypes
from .datatypes import DataType, pytype_to_polars_type
from ._html import NotebookFormatter
import pyarrow as pa
import pyarrow.compute
import pyarrow.parquet
import numpy as np
import os
from pathlib import Path


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lazy import LazyFrame, Expr


def wrap_df(df: "PyDataFrame") -> "DataFrame":
    return DataFrame._from_pydf(df)


def _prepare_other_arg(other: Any) -> Series:
    # if not a series create singleton series such that it will broadcast
    if not isinstance(other, Series):
        if isinstance(other, str):
            pass
        elif isinstance(other, Sequence):
            raise ValueError("operation not supported")

        other = Series("", [other])
    return other


class DataFrame:
    def __init__(
        self,
        data: "Union[Dict[str, Sequence], List[Series], np.ndarray]",
        nullable: bool = True,
    ):
        """
        A DataFrame is a two dimensional data structure that represents data as a table with rows and columns.
        """

        columns = []
        if isinstance(data, dict):
            for k, v in data.items():
                columns.append(Series(k, v, nullable=nullable).inner())
        elif isinstance(data, np.ndarray):
            shape = data.shape
            if len(shape) == 2:
                for i, c in enumerate(range(shape[1])):
                    columns.append(Series(str(i), data[:, c], nullable=False).inner())
            else:
                raise ValueError("a numpy array should have 2 dimensions")
        elif isinstance(data, list):
            for s in data:
                if not isinstance(s, Series):
                    raise ValueError("a list should contain Series")
                columns.append(s.inner())
        else:
            try:
                import pandas as pd

                if isinstance(data, pd.DataFrame):
                    for c in data.columns:
                        if nullable:
                            s = Series(c, data[c].to_list(), nullable=True).inner()
                        else:
                            s = Series(c, data[c].values, nullable=False).inner()
                        columns.append(s)
                else:
                    raise ValueError("a dictionary was expected.")
            except ImportError:
                raise ValueError("a dictionary was expected.")

        self._df = PyDataFrame(columns)

    @staticmethod
    def _from_pydf(df: "PyDataFrame") -> "DataFrame":
        self = DataFrame.__new__(DataFrame)
        self._df = df
        return self

    @staticmethod
    def read_csv(
        file: Union[str, TextIO],
        infer_schema_length: int = 100,
        batch_size: int = 64,
        has_headers: bool = True,
        ignore_errors: bool = False,
        stop_after_n_rows: Optional[int] = None,
        skip_rows: int = 0,
        projection: "Optional[List[int]]" = None,
        sep: str = ",",
        columns: "Optional[List[str]]" = None,
        rechunk: bool = True,
        encoding: str = "utf8",
        n_threads: Optional[int] = None,
        dtype: "Optional[Dict[str, DataType]]" = None,
    ) -> "DataFrame":
        """
        Read a CSV file into a Dataframe.

        Parameters
        ---
        file
            Path to a file or a file like object. Any valid filepath can be used. Example: `file.csv`.
        sep
            Character to use as delimiter in the file.
        stop_after_n_rows
            Only read specified number of rows of the dataset. After `n` stops reading.
        has_headers
            Indicate if first row of dataset is header or not. If set to False first row will be set to `column_x`, `x` being an enumeration over every column in the dataset.
        encoding
            Allowed encodings: `utf8`, `utf8-lossy`. Lossy means that invalid utf8 values are replaced with `�` character.

        Example
        ---
        ```python
        dataframe = pl.read_csv('file.csv', sep=';', stop_after_n_rows=25)
        ```

        Returns
        ---
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)

        if isinstance(file, str):
            path = file
        else:
            path = None

        if dtype is not None:
            new_dtype = []
            for k, v in dtype.items():
                new_dtype.append((k, pytype_to_polars_type(v)))
            dtype = new_dtype

        self._df = PyDataFrame.read_csv(
            file,
            infer_schema_length,
            batch_size,
            has_headers,
            ignore_errors,
            stop_after_n_rows,
            skip_rows,
            projection,
            sep,
            rechunk,
            columns,
            encoding,
            n_threads,
            path,
            dtype,
        )
        return self

    @staticmethod
    def read_parquet(
        file: Union[str, BinaryIO], stop_after_n_rows: "Optional[int]" = None
    ) -> "DataFrame":
        """
        Read into a DataFrame from a parquet file.

        Parameters
        ---
        file
            Path to a file or a file like object. Any valid filepath can be used.
        stop_after_n_rows
            Only read specified number of rows of the dataset. After `n` stops reading.

        Returns
        ---
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.read_parquet(file, stop_after_n_rows)
        return self

    @staticmethod
    def read_ipc(file: Union[str, BinaryIO]) -> "DataFrame":
        """
        Read into a DataFrame from Arrow IPC stream format. This is also called the feather format.

        Parameters
        ----------
        file
            Path to a file or a file like object.

        Returns
        -------
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.read_ipc(file)
        return self

    @staticmethod
    def from_arrow(table: pa.Table, rechunk: bool = True) -> "DataFrame":
        """
        Create DataFrame from arrow Table.
        Most will be zero copy. Types that are not supported by Polars may be cast to a closest
        supported type.

        Parameters
        ----------
        table
            Arrow Table
        rechunk
            Make sure that all data is contiguous.
        """
        data = {}
        for i, column in enumerate(table):
            if column.type == pa.timestamp("s"):
                column = pa.compute.cast(
                    pa.compute.multiply(pa.compute.cast(column, pa.int64()), 1000),
                    pa.date64(),
                )
            elif column.type == pa.timestamp("ms"):
                column = pa.compute.cast(
                    pa.compute.cast(column, pa.int64()), pa.date64()
                )
            elif column.type == pa.timestamp("us"):
                column = pa.compute.cast(
                    pa.compute.divide(pa.compute.cast(column, pa.int64()), 1000),
                    pa.date64(),
                )
            elif column.type == pa.timestamp("ns"):
                column = pa.compute.cast(
                    pa.compute.divide(pa.compute.cast(column, pa.int64()), 1000000),
                    pa.date64(),
                )
            # note: Decimal256 could not be cast to float
            elif isinstance(column.type, pa.Decimal128Type):
                column = pa.compute.cast(column, pa.float64())

            if column._name is None:
                name = f"column_{i}"
            else:
                name = column._name

            if column.num_chunks > 1:
                column = column.combine_chunks()

            data[name] = column

        table = pa.table(data)
        batches = table.to_batches()
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_arrow_record_batches(batches)
        if rechunk:
            return self.rechunk()
        return self

    def to_arrow(self) -> pa.Table:
        """
        Collect the underlying arrow arrays in an Arrow Table.
        This operation is zero copy.
        """
        record_batches = self._df.to_arrow()
        return pa.Table.from_batches(record_batches)

    def to_pandas(
        self, *args, date_as_object=False, **kwargs
    ) -> "pd.DataFrame":  # noqa: F821
        """
        Cast to a Pandas DataFrame. This requires that Pandas is installed.
        This operation clones data.

        Parameters
        ----------
        args
            arguments will be sent to pyarrow.Table.to_pandas
        date_as_object
            Cast dates to objects. If False, convert to datetime64[ns] dtype
        kwargs
            arguments will be sent to pyarrow.Table.to_pandas

        Example
        ---
        ```python
        >>> import pandas
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ['a', 'b', 'c']
            })

        >>> pandas_df = dataframe.to_pandas()
        >>> type(pandas_df)
        pandas.core.frame.DataFrame
        ```
        """
        return self.to_arrow().to_pandas(*args, date_as_object=date_as_object, **kwargs)

    def to_csv(
        self,
        file: Union[TextIO, str, Path],
        batch_size: int = 100000,
        has_headers: bool = True,
        delimiter: str = ",",
    ):
        """
        Write Dataframe to comma-separated values file (csv)

        Parameters
        ---
        file
            File path to which the file should be written.
        batch_size
            Size of the write buffer. Increase to have faster io.
        has_headers
            Whether or not to include header in the CSV output.
        delimiter
            Separate CSV fields with this symbol.

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3, 4, 5],
            "bar": [6, 7, 8, 9, 10],
            "ham": ['a', 'b', 'c', 'd','e']
            })
        >>> dataframe.to_csv('new_file.csv', sep=',')
        ```
        """
        if isinstance(file, Path):
            file = str(file)

        self._df.to_csv(file, batch_size, has_headers, ord(delimiter))

    def to_ipc(self, file: Union[BinaryIO, str, Path]):
        """
        Write to Arrow IPC binary stream, or a feather file.

        Parameters
        ----------
        file
            File path to which the file should be written.
        """
        if isinstance(file, Path):
            file = str(file)

        self._df.to_ipc(file)

    def to_parquet(
        self,
        file: Union[str, Path],
        compression: str = "snappy",
        use_pyarrow: bool = True,
        **kwargs,
    ):
        """
        Write the DataFrame disk in parquet format.

        Parameters
        ----------
        file
            File path to which the file should be written.
        compression
            Compression method (only supported if `use_pyarrow`)
        use_pyarrow
            Use C++ parquet implementation vs rust parquet implementation.
            At the moment C++ supports more features

        **kwargs are passed to pyarrow.parquet.write_table
        """
        if isinstance(file, Path):
            file = str(file)

        if use_pyarrow:
            tbl = self.to_arrow()
            pa.parquet.write_table(
                table=tbl, where=file, compression=compression, **kwargs
            )
        else:
            self._df.to_parquet(file)

    def to_numpy(self) -> np.ndarray:
        """
        Convert DataFrame to a 2d numpy array.
        This operation clones data.

        Example
        ---
        ```python
        >>> import pandas
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ['a', 'b', 'c']
            })

        >>> numpy_array = dataframe.to_numpy()
        >>> type(numpy_array)
        numpy.ndarray
        ```
        """
        return np.vstack([self[:, i].to_numpy() for i in range(self.width)]).T

    def __mul__(self, other):
        other = _prepare_other_arg(other)
        return wrap_df(self._df.mul(other._s))

    def __truediv__(self, other):
        other = _prepare_other_arg(other)
        return wrap_df(self._df.div(other._s))

    def __add__(self, other):
        other = _prepare_other_arg(other)
        return wrap_df(self._df.add(other._s))

    def __sub__(self, other):
        other = _prepare_other_arg(other)
        return wrap_df(self._df.sub(other._s))

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, item) -> "PySeries":
        """
        Access columns as attribute
        """
        try:
            return wrap_s(self._df.column(item))
        except RuntimeError:
            raise AttributeError(f"{item} not found")

    def __iter__(self):
        return self.get_columns().__iter__()

    def find_idx_by_name(self, name: str) -> int:
        """
        Find the index of a column by name

        Parameters
        ----------
        name
            Name of the column to find
        """
        return self._df.find_idx_by_name(name)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            item = Series("", item)
        # select rows and columns at once
        # every 2d selection, i.e. tuple is row column order, just like numpy
        if isinstance(item, tuple):
            row_selection, col_selection = item

            # df[:, unknown]
            if isinstance(row_selection, slice):

                # multiple slices
                # df[:, :]
                if isinstance(col_selection, slice):
                    # slice can be
                    # by index
                    #   [1:8]
                    # or by column name
                    #   ["foo":"bar"]
                    # first we make sure that the slice is by index
                    start = col_selection.start
                    stop = col_selection.stop
                    if isinstance(col_selection.start, str):
                        start = self.find_idx_by_name(col_selection.start)
                    if isinstance(col_selection.stop, str):
                        stop = self.find_idx_by_name(col_selection.stop) + 1

                    col_selection = slice(start, stop, col_selection.step)

                    df = self.__getitem__(self.columns[col_selection])
                    return df[row_selection]

                # single slice
                # df[:, unknown]
                series = self.__getitem__(col_selection)
                # s[:]
                wrap_s(series[row_selection])

            # df[2, :] (select row as df)
            if isinstance(row_selection, int):
                if isinstance(col_selection, slice):
                    df = self[:, col_selection]
                    return df.slice(row_selection, 1)

            # column selection can be "a" and ["a", "b"]
            if isinstance(col_selection, str):
                col_selection = [col_selection]
            df = self.__getitem__(col_selection)
            return df.__getitem__(row_selection)

        # select single column
        # df["foo"]
        if isinstance(item, str):
            return wrap_s(self._df.column(item))

        # df[idx]
        if isinstance(item, int):
            return wrap_s(self._df.select_at_idx(item))

        # df[:]
        if isinstance(item, slice):
            if getattr(item, "end", False):
                raise ValueError("a slice with steps larger than 1 is not supported")
            if item.start is None:
                start = 0
            else:
                start = item.start
            if item.stop is None:
                stop = self.height
            else:
                stop = item.stop
            length = stop - start
            return self.slice(start, length)

        # select multiple columns
        # df["foo", "bar"]
        if isinstance(item, Sequence) and isinstance(item[0], str):
            return wrap_df(self._df.select(item))

        # select rows by mask or index
        # df[[1, 2, 3]]
        # df[true, false, true]
        if isinstance(item, np.ndarray):
            if item.dtype == int:
                return wrap_df(self._df.take(item))
            if isinstance(item[0], str):
                return wrap_df(self._df.select(item))
        if isinstance(item, (Series, Sequence)):
            if isinstance(item, Sequence):
                # only bool or integers allowed
                if type(item[0]) == bool:
                    item = Series("", item)
                else:
                    return wrap_df(self._df.take(item))
            dtype = item.dtype
            if dtype == datatypes.Boolean:
                return wrap_df(self._df.filter(item.inner()))
            if dtype == datatypes.UInt32:
                return wrap_df(self._df.take_with_series(item.inner()))

    def __setitem__(self, key, value):
        # df["foo"] = series
        if isinstance(key, str):
            try:
                self.drop_in_place(key)
            except Exception:
                pass
            self.hstack([Series(key, value)], in_place=True)
        # df[idx] = series
        elif isinstance(key, int):
            assert isinstance(value, Series)
            self.replace_at_idx(key, value)
        # df[a, b]
        elif isinstance(key, tuple):
            row_selection, col_selection = key
            # get series column selection
            s = self.__getitem__(col_selection)

            # dispatch to __setitem__ of Series to do modification
            s[row_selection] = value

            # now find the location to place series
            # df[idx]
            if isinstance(col_selection, int):
                self.replace_at_idx(col_selection, s)
            # df["foo"]
            elif isinstance(col_selection, str):
                self.replace(col_selection, s)
        else:
            return NotImplemented

    def __len__(self):
        return self.height

    def _repr_html_(self) -> str:
        max_cols = int(os.environ.get("POLARS_FMT_MAX_COLS", default=75))
        max_rows = int(os.environ.get("POLARS_FMT_MAX_rows", 25))
        return "\n".join(NotebookFormatter(self, max_cols, max_rows).render())

    def insert_at_idx(self, index: int, series: Series):
        self._df.insert_at_idx(index, series._s)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get shape of the DataFrame.

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
        >>> dataframe.shape
        shape: (5, 1)
        ```
        """
        return self._df.shape()

    @property
    def height(self) -> int:
        """
        Get height of the DataFrame.

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
        >>> dataframe.height
        5
        ```
        """
        return self._df.height()

    @property
    def width(self) -> int:
        """
        Get width of the DataFrame

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
        >>> dataframe.width
        1
        ```
        """
        return self._df.width()

    @property
    def columns(self) -> "List[str]":
        """
        Get or set column names

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ['a', 'b', 'c']
            })

        >>> dataframe.columns
        ['foo', 'bar', 'ham']

        # Set column names
        >>> dataframe.columns = ['apple', 'banana', 'orange']
        shape: (3, 3)
        ╭───────┬────────┬────────╮
        │ apple ┆ banana ┆ orange │
        │ ---   ┆ ---    ┆ ---    │
        │ i64   ┆ i64    ┆ str    │
        ╞═══════╪════════╪════════╡
        │ 1     ┆ 6      ┆ "a"    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2     ┆ 7      ┆ "b"    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 3     ┆ 8      ┆ "c"    │
        ╰───────┴────────┴────────╯
        ```
        """
        return self._df.columns()

    @columns.setter
    def columns(self, columns: "List[str]"):
        self._df.set_column_names(columns)

    @property
    def dtypes(self) -> "List[type]":
        """
        Get dtypes of columns in DataFrame. Dtypes can also be found in column headers when printing the DataFrame.

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ['a', 'b', 'c']
            })

        >>> dataframe.dtypes
        [polars.datatypes.Int64, polars.datatypes.Float64, polars.datatypes.Utf8]
        >>> dataframe
        shape: (3, 3)
        ╭─────┬─────┬─────╮
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ "a" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ "b" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ "c" │
        ╰─────┴─────┴─────╯
        ```
        """
        return [datatypes.dtypes[idx] for idx in self._df.dtypes()]

    def replace_at_idx(self, index: int, series: Series):
        """
        Replace a column at an index  location.

        Parameters
        ----------
        index
            Column index
        series
            Series that will replace the column
        """
        self._df.replace_at_idx(index, series._s)

    def sort(
        self, by_column: str, in_place: bool = False, reverse: bool = False
    ) -> Optional["DataFrame"]:
        """
        Sort the DataFrame by column

        Parameters
        ----------
        by_column
            By which column to sort. Only accepts string.
        in_place
            Perform operation in-place.
        reverse
            Reverse/descending sort.

        Example
        ---
        ```python
        >>> pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ['a', 'b', 'c']
            })

        >>> dataframe.sort('foo', reverse=True)
        shape: (3, 3)
        ╭─────┬─────┬─────╮
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ "c" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ "b" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 1   ┆ 6   ┆ "a" │
        ╰─────┴─────┴─────╯
        ```
        """
        if in_place:
            self._df.sort_in_place(by_column, reverse)
        else:
            return wrap_df(self._df.sort(by_column, reverse))

    def frame_equal(self, other: "DataFrame", null_equal: bool = False) -> bool:
        """
        Check if DataFrame is equal to other.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.

        Example
        ---
        ```python
        >>> df1 = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ['a', 'b', 'c']
            })

        >>> df2 = pl.DataFrame({
            "foo": [3, 2, 1],
            "bar": [8.0, 7.0, 6.0],
            "ham": ['c', 'b', 'a']
            })

        >>> df1.frame_equal(df1)
        True

        >>> df1.frame_equal(df2)
        False
        ```
        """
        return self._df.frame_equal(other._df, null_equal)

    def replace(self, column: str, new_col: Series):
        """
        Replace a column by a new Series.

        Parameters
        ----------
        column
            Column to replace.
        new_col
            New column to insert.
        """
        self._df.replace(column, new_col.inner())

    def slice(self, offset: int, length: int) -> "DataFrame":
        """
        Slice this DataFrame over the rows direction.

        Parameters
        ----------
        offset
            Offset index.
        length
            Length of the slice.
        """
        return wrap_df(self._df.slice(offset, length))

    def head(self, length: int = 5) -> "DataFrame":
        """
        Get first N rows as DataFrame

        Parameters
        ----------
        length
            Length of the head

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3, 4, 5],
            "bar": [6, 7, 8, 9, 10],
            "ham": ['a', 'b', 'c', 'd','e']
            })

        >>> dataframe.head(3)
        shape: (3, 3)
        ╭─────┬─────┬─────╮
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ "a" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ "b" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   ┆ "c" │
        ╰─────┴─────┴─────╯
        ```
        """
        return wrap_df(self._df.head(length))

    def tail(self, length: int = 5) -> "DataFrame":
        """
        Get last N rows as DataFrame

        Parameters
        ----------
        length
            Length of the tail

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3, 4, 5],
            "bar": [6, 7, 8, 9, 10],
            "ham": ['a', 'b', 'c', 'd','e']
            })

        >>> dataframe.tail(3)
        shape: (3, 3)
        ╭─────┬─────┬─────╮
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ "c" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 4   ┆ 9   ┆ "d" │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ 5   ┆ 10  ┆ "e" │
        ╰─────┴─────┴─────╯
        ```
        """
        return wrap_df(self._df.tail(length))

    def drop_nulls(self, subset: "Optional[List[str]]" = None) -> "DataFrame":
        """
        Return a new DataFrame where the null values are dropped
        """
        if subset is not None and isinstance(subset, str):
            subset = [subset]
        return wrap_df(self._df.drop_nulls(subset))

    def pipe(self, func: Callable, *args, **kwargs):
        """
        Apply a function on Self

        Parameters
        ----------
        func
            Callable
        args
            Arguments
        kwargs
            Keyword arguments
        """
        return func(self, *args, **kwargs)

    def groupby(self, by: "Union[str, List[str]]") -> "GroupBy":
        """
        Start a groupby operation

        Parameters
        ----------
        by
            Column(s) to group by.

        # Example

        Below we group by column `"a"`, and we sum column `"b"`.

        ```python
        >>> df = pl.DataFrame(
            {
                "a": ["a", "b", "a", "b", "b", "c"],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )

        assert (
            df.groupby("a")["b"]
            .sum()
            .sort(by_column="a")
            .frame_equal(DataFrame({"a": ["a", "b", "c"], "": [4, 11, 6]}))
        )
        ```

        We can also loop over the grouped `DataFrame`

        ```python
        for sub_df in df.groupby("a"):
            print(sub_df)
        ```
        Outputs:
        ```text
        shape: (3, 3)
        ╭─────┬─────┬─────╮
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ "b" ┆ 2   ┆ 5   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ "b" ┆ 4   ┆ 3   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ "b" ┆ 5   ┆ 2   │
        ╰─────┴─────┴─────╯
        shape: (1, 3)
        ╭─────┬─────┬─────╮
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ "c" ┆ 6   ┆ 1   │
        ╰─────┴─────┴─────╯
        shape: (2, 3)
        ╭─────┬─────┬─────╮
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ "a" ┆ 1   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ "a" ┆ 3   ┆ 4   │
        ╰─────┴─────┴─────╯
        ```

        """
        if isinstance(by, str):
            by = [by]
        return GroupBy(self._df, by, downsample=False)

    def downsample(self, by: str, rule: str, n: int) -> "GroupBy":
        """
        Start a downsampling groupby operation.

        Parameters
        ----------
        by
            Column that will be used as key in the groupby operation.
            This should be a date64/date32 column
        rule
            Units of the downscaling operation.

            Any of:
                - "month"
                - "week"
                - "day"
                - "hour"
                - "minute"
                - "second"

        n
            Number of units (e.g. 5 "day", 15 "minute"
        """
        return GroupBy(self._df, by, downsample=True, rule=rule, downsample_n=n)

    def join(
        self,
        df: "DataFrame",
        left_on: "Optional[Union[str, List[str]]]" = None,
        right_on: "Optional[Union[str, List[str]]]" = None,
        on: "Optional[Union[str, List[str]]]" = None,
        how="inner",
    ) -> "DataFrame":
        """
        SQL like joins

        Parameters
        ----------
        df
            DataFrame to join with
        left_on
            Name(s) of the left join column(s)
        right_on
            Name(s) of the right join column(s)
        on
            Name(s) of the join columns in both DataFrames
        how
            Join strategy
                - "inner"
                - "left"
                - "outer"

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ['a', 'b', 'c']
            })

        >>> other_dataframe = pl.DataFrame({
            "apple": ['x', 'y', 'z'],
            "ham": ['a', 'b', 'd']
            })

        >>> dataframe.join(other_dataframe, on='ham')
        shape: (2, 4)
        ╭─────┬─────┬─────┬───────╮
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6   ┆ "a" ┆ "x"   │
        ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2   ┆ 7   ┆ "b" ┆ "y"   │
        ╰─────┴─────┴─────┴───────╯

        >>> dataframe.join(other_dataframe, on='ham', how='outer')
        shape: (4, 4)
        ╭──────┬──────┬─────┬───────╮
        │ foo  ┆ bar  ┆ ham ┆ apple │
        │ ---  ┆ ---  ┆ --- ┆ ---   │
        │ i64  ┆ f64  ┆ str ┆ str   │
        ╞══════╪══════╪═════╪═══════╡
        │ 1    ┆ 6    ┆ "a" ┆ "x"   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 2    ┆ 7    ┆ "b" ┆ "y"   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ null ┆ null ┆ "d" ┆ "z"   │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ 3    ┆ 8    ┆ "c" ┆ null  │
        ╰──────┴──────┴─────┴───────╯
        ```

        Returns
        -------
            Joined DataFrame
        """
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if isinstance(on, str):
            left_on = [on]
            right_on = [on]
        elif isinstance(on, list):
            left_on = on
            right_on = on
        if left_on is None or right_on is None:
            raise ValueError("you should pass the column to join on as an argument")

        out = self._df.join(df._df, left_on, right_on, how)

        return wrap_df(out)

    def apply(
        self, f: "Callable[[Tuple[Any]], Any]", output_type: "Optional[DataType]" = None
    ) -> "Series":
        """
        Apply a custom function over the rows of the DataFrame. The rows are passed as tuple.

        Beware, this is slow.

        Parameters
        ----------
        f
            Custom function/ lambda function
        output_type
            Output type of the operation. If none given, Polars tries to infer the type.
        """
        return wrap_s(self._df.apply(f, output_type))

    def with_column(self, column: "Series") -> "DataFrame":
        """
        Return a new DataFrame with the column added or replaced

        Parameters
        ----------
        column
            Series, where the name of the Series refers to the column in the DataFrame.
        """
        return wrap_df(self._df.with_column(column._s))

    def hstack(
        self, columns: "Union[List[Series], DataFrame]", in_place=False
    ) -> Optional["DataFrame"]:
        """
        Return a new DataFrame grown horizontally by stacking multiple Series to it.

        Parameters
        ----------
        columns
            Series to stack
        in_place
            Modify in place
        """
        if not isinstance(columns, list):
            columns = columns.get_columns()
        if in_place:
            self._df.hstack_mut([s.inner() for s in columns])
        else:
            return wrap_df(self._df.hstack([s.inner() for s in columns]))

    def vstack(self, df: "DataFrame", in_place: bool = False) -> Optional["DataFrame"]:
        """
        Grow this DataFrame vertically by stacking a DataFrame to it.

        Parameters
        ----------
        df
            DataFrame to stack
        in_place
            Modify in place
        """
        if in_place:
            self._df.vstack_mut(df._df)
        else:
            return wrap_df(self._df.vstack(df._df))

    def drop(self, name: "Union[str, List[str]]") -> "DataFrame":
        """
        Remove column from DataFrame and return as new.

        Parameters
        ----------
        name
            Column(s) to drop

        Example
        ---
        ```python
        >>> dataframe = pl.DataFrame({
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ['a', 'b', 'c']
            })

        >>> dataframe.drop('ham')
        shape: (3, 2)
        ╭─────┬─────╮
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 6   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 2   ┆ 7   │
        ├╌╌╌╌╌┼╌╌╌╌╌┤
        │ 3   ┆ 8   │
        ╰─────┴─────╯
        ```
        """
        if isinstance(name, list):
            df = self.clone()

            for name in name:
                df._df.drop_in_place(name)
            return df

        return wrap_df(self._df.drop(name))

    def drop_in_place(self, name: str) -> Series:
        """
        Drop in place

        Parameters
        ----------
        name
            Column to drop
        """
        return wrap_s(self._df.drop_in_place(name))

    def select_at_idx(self, idx: int) -> Series:
        """
        Select column at index location.

        Parameters
        ----------
        idx
            Location of selection
        """
        return wrap_s(self._df.select_at_idx(idx))

    def clone(self) -> "DataFrame":
        """
        Very cheap deep clone
        """
        return wrap_df(self._df.clone())

    def get_columns(self) -> "List[Series]":
        """
        Get the DataFrame as a List of Series
        """
        return list(map(lambda s: wrap_s(s), self._df.get_columns()))

    def fill_none(self, strategy: str) -> "DataFrame":
        """
        Fill None values by a filling strategy.

        Parameters
        ----------
        strategy
            - "backward"
            - "forward"
            - "mean"
            - "min'
            - "max"

        Returns
        -------
            DataFrame with None replaced with the filling strategy.
        """
        return wrap_df(self._df.fill_none(strategy))

    def explode(self, columns: "Union[str, List[str]]") -> "DataFrame":
        """
        Explode `DataFrame` to long format by exploding a column with Lists.

        Parameters
        ----------
        columns
            Column of LargeList type

        Returns
        -------
        DataFrame
        """
        if isinstance(columns, str):
            columns = [columns]
        return wrap_df(self._df.explode(columns))

    def melt(
        self, id_vars: "Union[List[str], str]", value_vars: "Union[List[str], str]"
    ) -> "DataFrame":
        """
        Unpivot DataFrame to long format.

        Parameters
        ----------
        id_vars
            Columns to use as identifier variables

        value_vars
            Values to use as identifier variables

        Returns
        -------

        """
        if isinstance(value_vars, str):
            value_vars = [value_vars]
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        return wrap_df(self._df.melt(id_vars, value_vars))

    def shift(self, periods: int) -> "DataFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_df(self._df.shift(periods))

    def shift_and_fill(
        self, periods: int, fill_value: Union[int, str, float]
    ) -> "DataFrame":
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with the result of the `fill_value` expression.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            fill None values with this value.
        """
        return (
            self.lazy()
            .shift_and_fill(periods, fill_value)
            .collect(
                predicate_pushdown=False,
                projection_pushdown=False,
                simplify_expression=False,
            )
        )

    def is_duplicated(self) -> Series:
        """
        Get a mask of all duplicated rows in this DataFrame
        """
        return wrap_s(self._df.is_duplicated())

    def is_unique(self) -> Series:
        """
        Get a mask of all unique rows in this DataFrame
        """
        return wrap_s(self._df.is_unique())

    def lazy(self) -> "LazyFrame":
        """
        Start a lazy query from this point. This returns a `LazyFrame` object.

        Operations on a `LazyFrame` are not executed until this is requested by either calling:

        * `.fetch()` (run on a small number of rows)
        * `.collect()` (run on all data)
        * `.describe_plan()` (print unoptimized query plan)
        * `.describe_optimized_plan()` (print optimized query plan)
        * `.show_graph()` (show (un)optimized query plan) as graphiz graph.

        Lazy operations are advised because they allow for query optimization and more parallelization.
        """
        from polars.lazy import wrap_ldf

        return wrap_ldf(self._df.lazy())

    def n_chunks(self) -> int:
        """
        Get number of chunks used by the ChunkedArrays of this DataFrame
        """
        return self._df.n_chunks()

    def max(self, axis: int = 0) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their maximum value
        """
        if axis == 0:
            return wrap_df(self._df.max())
        if axis == 1:
            return wrap_s(self._df.hmax()).to_frame()
        raise ValueError("axis should be 0 or 1")

    def min(self, axis: int = 0) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their minimum value
        """
        if axis == 0:
            return wrap_df(self._df.min())
        if axis == 1:
            return wrap_s(self._df.hmin()).to_frame()
        raise ValueError("axis should be 0 or 1")

    def sum(self, axis: int = 0) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their sum value
        """
        if axis == 0:
            return wrap_df(self._df.sum())
        if axis == 1:
            return wrap_s(self._df.hsum()).to_frame()
        raise ValueError("axis should be 0 or 1")

    def mean(self, axis: int = 0) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their mean value
        """
        if axis == 0:
            return wrap_df(self._df.mean())
        if axis == 1:
            return wrap_s(self._df.hmean()).to_frame()
        raise ValueError("axis should be 0 or 1")

    def std(self) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their standard deviation value
        """
        return wrap_df(self._df.std())

    def var(self) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their variance value
        """
        return wrap_df(self._df.var())

    def median(self) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their median value
        """
        return wrap_df(self._df.median())

    def quantile(self, quantile: float) -> "DataFrame":
        """
        Aggregate the columns of this DataFrame to their quantile value
        """
        return wrap_df(self._df.quantile(quantile))

    def to_dummies(self) -> "DataFrame":
        """
        Get one hot encoded dummy variables.
        """
        return wrap_df(self._df.to_dummies())

    def drop_duplicates(
        self, maintain_order=True, subset: "Optional[List[str]]" = None
    ) -> "DataFrame":
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        if subset is not None and not isinstance(subset, list):
            subset = [subset]
        return wrap_df(self._df.drop_duplicates(maintain_order, subset))

    def rechunk(self) -> "DataFrame":
        """
        Rechunk the data in this DataFrame to a contiguous allocation.

        This will make sure all subsequent operations have optimal and predictable performance
        """
        return wrap_df(self._df.rechunk())

    def null_count(self) -> "DataFrame":
        """
        Create a new DataFrame that shows the null counts per column.
        """
        return wrap_df(self._df.null_count())

    def sample(
        self,
        n: "Optional[int]" = None,
        frac: "Optional[float]" = None,
        with_replacement: bool = False,
    ) -> "DataFrame":
        """
        Sample from this DataFrame by setting either `n` or `frac`

        Parameters
        ----------
        n
            Number of samples < self.len()
        frac
            Fraction between 0.0 and 1.0
        with_replacement
            Sample with replacement
        """
        if n is not None:
            return wrap_df(self._df.sample_n(n, with_replacement))
        return wrap_df(self._df.sample_frac(frac, with_replacement))

    def fold(self, operation: "Callable[['Series', 'Series'], 'Series']") -> "Series":
        """
        Apply a horizontal reduction on a DataFrame. This can be used to effectively
        determine aggregations on a row level, and can be applied to any DataType that
        can be supercasted (casted to a similar parent type).

        An example of the supercast rules when applying an arithmetic operation on two DataTypes are for instance:

        Int8 + Utf8 = Utf8
        Float32 + Int64 = Float32
        Float32 + Float64 = Float64

        # Examples

        ## A horizontal sum operation
        ```python
        >>> df = pl.DataFrame(
            {"a": [2, 1, 3],
            "b": [1, 2, 3],
            "c": [1.0, 2.0, 3.0]
        })

        >>> df.fold(lambda s1, s2: s1 + s2)
        ```
        ```text
        Series: 'a' [f64]
        [
            4
            5
            9
        ]
        ```

        ## A horizontal minimum operation

        ```python
        >>> df = pl.DataFrame(
            {"a": [2, 1, 3],
            "b": [1, 2, 3],
            "c": [1.0, 2.0, 3.0]
        })

        >>> df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2))
        ```
        ```text
        Series: 'a' [f64]
        [
            1
            1
            3
        ]
        ```

        ## A horizontal string concattenation
        ```python
        >>> df = pl.DataFrame(
            {"a": ["foo", "bar", 2],
            "b": [1, 2, 3],
            "c": [1.0, 2.0, 3.0]
        })

        >>> df.fold(lambda s1, s2: s1 + s2)
        ```
        ```text
        Series: '' [f64]
        [
            "foo11"
            "bar22
            "233"
        ]
        ```

        Parameters
        ----------
        operation
            function that takes two `Series` and returns a `Series`
        """
        if self.width == 1:
            return self
        df = self
        acc = operation(df[0], df[1])

        for i in range(2, df.width):
            acc = operation(acc, df[i])
        return acc

    def row(self, index: int) -> Tuple[Any]:
        """
        Get a row as tuple

        Parameters
        ----------
        index
            Row index
        """
        return self._df.row_tuple(index)


class GroupBy:
    def __init__(
        self,
        df: "PyDataFrame",
        by: "List[str]",
        downsample: bool = False,
        rule=None,
        downsample_n: int = 0,
    ):
        self._df = df
        self.by = by
        self.downsample = downsample
        self.rule = rule
        self.downsample_n = downsample_n

    def __getitem__(self, item):
        return self.select(item)

    def __iter__(self):
        groups_df = self.groups()
        groups = groups_df["groups"]
        df = wrap_df(self._df)
        for i in range(groups_df.height):
            yield df[groups[i]]

    def get_group(self, group_value: "Union[Any, Tuple[Any]]") -> DataFrame:
        groups_df = self.groups()
        groups = groups_df["groups"]

        if not isinstance(group_value, list):
            group_value = [group_value]

        by = self.by
        if not isinstance(by, list):
            by = [by]

        mask = None
        for column, group_val in zip(by, group_value):
            local_mask = groups_df[column] == group_val
            if mask is None:
                mask = local_mask
            else:
                mask = mask & local_mask

        # should be only one match
        try:
            groups_idx = groups[mask][0]
        except IndexError:
            raise ValueError(f"no group: {group_value} found")

        df = wrap_df(self._df)
        return df[groups_idx]

    def groups(self) -> DataFrame:
        return wrap_df(self._df.groupby(self.by, None, "groups"))

    def apply(self, f: "Callable[[DataFrame], DataFrame]"):
        """
        Apply a function over the groups as a sub-DataFrame.

        Parameters
        ----------
        f
            Custom function

        Returns
        -------
        DataFrame
        """
        return wrap_df(self._df.groupby_apply(self.by, f))

    def agg(
        self,
        column_to_agg: "Union[List[Tuple[str, List[str]]], Dict[str, List[str]]], List[Expr]",
    ) -> DataFrame:
        """
        Use multiple aggregations on columns. This can be combined with complete lazy API.

        Parameters
        ----------
        column_to_agg
            map column to aggregation functions

            Examples:
                ## column name to aggregation with tuples:
                [("foo", ["sum", "n_unique", "min"]),
                 ("bar": ["max"])]

                ## column name to aggregation with dict:
                {"foo": ["sum", "n_unique", "min"],
                "bar": "max" }

                ## use lazy API syntax
                [col("foo").sum(), col("bar").min()]

        Returns
        -------
        Result of groupby split apply operations.


        # Example

        ```python

        # use lazy API
        (df.groupby(["foo", "bar])
            .agg([pl.sum("ham"), col("spam").tail(4).sum()])

        # use a dict
        (df.groupby(["foo", "bar])
            .agg({"spam": ["sum", "min"})
        ```
        """
        if isinstance(column_to_agg, dict):
            column_to_agg = [
                (column, [agg] if isinstance(agg, str) else agg)
                for (column, agg) in column_to_agg.items()
            ]
        elif isinstance(column_to_agg, list):
            from .lazy import Expr

            if isinstance(column_to_agg[0], tuple):
                column_to_agg = [
                    (column, [agg] if isinstance(agg, str) else agg)
                    for (column, agg) in column_to_agg
                ]

            elif isinstance(column_to_agg[0], Expr):
                return (
                    wrap_df(self._df)
                    .lazy()
                    .groupby(self.by)
                    .agg(column_to_agg)
                    .collect(no_optimization=True)
                )

                pass
            else:
                raise ValueError(
                    f"argument: {column_to_agg} not understood, have you passed a list of expressions?"
                )
        else:
            raise ValueError(
                f"argument: {column_to_agg} not understood, have you passed a list of expressions?"
            )

        if self.downsample:
            return wrap_df(
                self._df.downsample_agg(
                    self.by, self.rule, self.downsample_n, column_to_agg
                )
            )

        return wrap_df(self._df.groupby_agg(self.by, column_to_agg))

    def select(self, columns: "Union[str, List[str]]") -> "GBSelection":
        """
        Select the columns that will be aggregated.

        Parameters
        ----------
        columns
            One or multiple columns
        """
        if self.downsample:
            raise ValueError("select not supported in downsample operation")
        if isinstance(columns, str):
            columns = [columns]
        return GBSelection(self._df, self.by, columns)

    def select_all(self):
        """
        Select all columns for aggregation.
        """
        return GBSelection(
            self._df, self.by, None, self.downsample, self.rule, self.downsample_n
        )

    def pivot(self, pivot_column: str, values_column: str) -> "PivotOps":
        """
        Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.

        Parameters
        ----------
        pivot_column
            Column to pivot.
        values_column
            Column that will be aggregated
        """
        if self.downsample:
            raise ValueError("pivot not supported in downsample operation")
        return PivotOps(self._df, self.by, pivot_column, values_column)

    def first(self) -> DataFrame:
        """
        Aggregate the first values in the group.
        """
        return self.select_all().first()

    def last(self) -> DataFrame:
        """
        Aggregate the last values in the group.
        """
        return self.select_all().last()

    def sum(self) -> DataFrame:
        """
        Reduce the groups to the sum.
        """
        return self.select_all().sum()

    def min(self) -> DataFrame:
        """
        Reduce the groups to the minimal value.
        """
        return self.select_all().min()

    def max(self) -> DataFrame:
        """
        Reduce the groups to the maximal value.
        """
        return self.select_all().max()

    def count(self) -> DataFrame:
        """
        Count the number of values in each group.
        """
        return self.select_all().count()

    def mean(self) -> DataFrame:
        """
        Reduce the groups to the mean values.
        """
        return self.select_all().mean()

    def n_unique(self) -> DataFrame:
        """
        Count the unique values per group.
        """
        return self.select_all().n_unique()

    def quantile(self, quantile: float) -> DataFrame:
        """
        Count the unique values per group.
        """
        return self.select_all().quantile(quantile)

    def median(self) -> DataFrame:
        """
        Return the median per group.
        """
        return self.select_all().median()

    def agg_list(self) -> DataFrame:
        """
        Aggregate the groups into Series.
        """
        return self.select_all().agg_list()


class PivotOps:
    def __init__(
        self, df: DataFrame, by: "List[str]", pivot_column: str, values_column: str
    ):
        self._df = df
        self.by = by
        self.pivot_column = pivot_column
        self.values_column = values_column

    def first(self):
        """
        Get the first value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "first")
        )

    def sum(self):
        """
        Get the sum per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "sum")
        )

    def min(self):
        """
        Get the minimal value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "min")
        )

    def max(self):
        """
        Get the maximal value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "max")
        )

    def mean(self):
        """
        Get the mean value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "mean")
        )

    def count(self):
        """
        Count the values per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "count")
        )

    def median(self):
        """
        Get the median value per group.
        """
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )


class GBSelection:
    def __init__(
        self,
        df: DataFrame,
        by: "List[str]",
        selection: "Optional[List[str]]",
        downsample: bool = False,
        rule=None,
        downsample_n: int = 0,
    ):
        self._df = df
        self.by = by
        self.selection = selection
        self.downsample = downsample
        self.rule = rule
        self.n = downsample_n

    def first(self) -> DataFrame:
        """
        Aggregate the first values in the group.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "first"))

        return wrap_df(self._df.groupby(self.by, self.selection, "first"))

    def last(self) -> DataFrame:
        """
        Aggregate the last values in the group.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "last"))
        return wrap_df(self._df.groupby(self.by, self.selection, "last"))

    def sum(self) -> DataFrame:
        """
        Reduce the groups to the sum.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "sum"))
        return wrap_df(self._df.groupby(self.by, self.selection, "sum"))

    def min(self) -> DataFrame:
        """
        Reduce the groups to the minimal value.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "min"))
        return wrap_df(self._df.groupby(self.by, self.selection, "min"))

    def max(self) -> DataFrame:
        """
        Reduce the groups to the maximal value.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "max"))
        return wrap_df(self._df.groupby(self.by, self.selection, "max"))

    def count(self) -> DataFrame:
        """
        Count the number of values in each group.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "count"))
        return wrap_df(self._df.groupby(self.by, self.selection, "count"))

    def mean(self) -> DataFrame:
        """
        Reduce the groups to the mean values.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "mean"))
        return wrap_df(self._df.groupby(self.by, self.selection, "mean"))

    def n_unique(self) -> DataFrame:
        """
        Count the unique values per group.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "n_unique"))
        return wrap_df(self._df.groupby(self.by, self.selection, "n_unique"))

    def quantile(self, quantile: float) -> DataFrame:
        """
        Compute the quantile per group
        """
        if self.downsample:
            raise ValueError("quantile operation not supported during downsample")
        return wrap_df(self._df.groupby_quantile(self.by, self.selection, quantile))

    def median(self) -> DataFrame:
        """
        Return the median per group.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "median"))
        return wrap_df(self._df.groupby(self.by, self.selection, "median"))

    def agg_list(self) -> DataFrame:
        """
        Aggregate the groups into Series.
        """
        if self.downsample:
            return wrap_df(self._df.downsample(self.by, self.rule, self.n, "agg_list"))
        return wrap_df(self._df.groupby(self.by, self.selection, "agg_list"))

    def apply(
        self,
        func: "Union[Callable[['Any'], 'Any'], Callable[['Any'], 'Any']]",
        dtype_out: "Optional['DataType']" = None,
    ) -> "DataFrame":
        """
        Apply a function over the groups
        """
        df = self.agg_list()
        if isinstance(self.selection, str):
            selection = [self.selection]
        else:
            selection = self.selection
        for name in selection:
            s = df.drop_in_place(name + "_agg_list").apply(func, dtype_out)
            s.rename(name)
            df[name] = s

        return df


def _series_to_frame(self: "Series") -> "DataFrame":
    return wrap_df(PyDataFrame([self._s]))


Series.to_frame = _series_to_frame


class StringCache:
    """
    Context manager that allows to data sources to share the same categorical features.
    This will temporarily cache the string categories until the context manager is finished.

    """

    def __init__(self):
        pass

    def __enter__(self):
        pytoggle_string_cache(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pytoggle_string_cache(False)


def toggle_string_cache(toggle: bool):
    """
    Turn on/off the global string cache. This ensures that casts to Categorical types have the categories when string
    values are equal
    """
    pytoggle_string_cache(toggle)
