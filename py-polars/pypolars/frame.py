from __future__ import annotations

try:
    from .pypolars import PyDataFrame, PySeries, PyLazyFrame
except:
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
)
from .series import Series, wrap_s
from .datatypes import *
import numpy as np


def wrap_df(df: PyDataFrame) -> DataFrame:
    return DataFrame._from_pydf(df)


class DataFrame:
    def __init__(self, data: Dict[str, Sequence], nullable: bool = False):

        columns = []
        if isinstance(data, dict):
            for k, v in data.items():
                columns.append(Series(k, v, nullable=nullable).inner())
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
        batch_size: int = 1000,
        has_headers: bool = True,
        ignore_errors: bool = False,
        stop_after_n_rows: Optional[int] = None,
        skip_rows: int = 0,
        projection: Optional[List[int]] = None,
        sep: str = ",",
        columns: Optional[List[str]] = None,
        rechunk: bool = True,
        encoding: str = "utf8",
    ) -> DataFrame:
        """
        Read into a DataFrame from a csv file.

        Parameters
        ----------
        file
            Path to a file or a file like object.
        infer_schema_length
            Maximum number of lines to read to infer schema.
        batch_size
            Number of lines to read into the buffer at once. Modify this to change performance.
        has_headers
            If the CSV file has headers or not.
        ignore_errors
            Try to keep reading lines if some lines yield errors.
        stop_after_n_rows
            After n rows are read from the CSV stop reading. This probably not stops exactly at `n_rows` it is dependent
            on the batch size.
        skip_rows
            Start reading after `skip_rows`.
        projection
            Indexes of columns to select
        sep
            Delimiter/ value seperator
        columns
            Columns to project/ select
        rechunk
            Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
        encoding
            - "utf8"
            _ "utf8-lossy"

        Returns
        -------
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
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
        )
        return self

    @staticmethod
    def read_parquet(
        file: Union[str, BinaryIO],
    ) -> DataFrame:
        """
        Read into a DataFrame from a parquet file.

        Parameters
        ----------
        file
            Path to a file or a file like object.

        Returns
        -------
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.read_parquet(file)
        return self

    @staticmethod
    def read_ipc(file: Union[str, BinaryIO]) -> DataFrame:
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

    def to_pandas(self) -> "pd.DataFrame":
        """
        cast to a Pandas DataFrame.
        """
        import pandas as pd

        data = {}
        for col in self.columns:
            series = self[col]
            if series.dtype == List:
                data[col] = series.to_list()
            elif series.dtype == Utf8:
                data[col] = series.to_list()
            else:
                data[col] = series.to_numpy()
        return pd.DataFrame(data)

    def to_csv(
        self,
        file: Union[TextIO, str],
        batch_size: int = 100000,
        has_headers: bool = True,
        delimiter: str = ",",
    ):
        """
        Write DataFrame to CSV

        Parameters
        ----------
        file
            write location
        batch_size
            Size of the write buffer. Increase to have faster io.
        has_headers
            Whether or not to include header in the CSV output.
        delimiter
            Space elements with this symbol.
        """
        self._df.to_csv(file, batch_size, has_headers, ord(delimiter))

    def to_ipc(self, file: Union[BinaryIO, str], batch_size):
        """
        Write to Arrow IPC binary stream, or a feather file.

        Parameters
        ----------
        file
            write location
        batch_size
            Size of the write buffer. Increase to have faster io.
        """
        self._df.to_ipc(file, batch_size)

    def to_numpy(self) -> np.ndarray:
        """
        Convert DataFrame to a 2d numpy array.
        This operation clones data.
        """
        return np.vstack([self[:, i].to_numpy() for i in range(self.width)]).T

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, item) -> PySeries:
        """
        Access columns as attribute
        """
        try:
            return wrap_s(self._df.column(item))
        except RuntimeError:
            raise AttributeError(f"{item} not found")

    def __getitem__(self, item):
        # select rows and columns at once
        # every 2d selection, i.e. tuple is row column order, just like numpy
        if isinstance(item, tuple):
            row_selection, col_selection = item

            # df[:, unknown]
            if isinstance(row_selection, slice):

                # multiple slices
                # df[:, :]
                if isinstance(col_selection, slice):
                    # TODO: select by indexes as column names can be duplicates
                    df = self.__getitem__(self.columns[col_selection])
                    return df[row_selection]

                # single slice
                # df[:, unknown]
                series = self.__getitem__(col_selection)
                # s[:]
                wrap_s(series[row_selection])

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
        if isinstance(item, (Series, Sequence)):
            if isinstance(item, Sequence):
                # only bool or integers allowed
                if type(item[0]) == bool:
                    item = Series("", item)
                else:
                    return wrap_df(self._df.take(item))
            dtype = item.dtype
            if dtype == Bool:
                return wrap_df(self._df.filter(item.inner()))
            if dtype == UInt32:
                return wrap_df(self._df.take_with_series(item.inner()))

    def __setitem__(self, key, value):
        # df["foo"] = series
        if isinstance(key, str):
            try:
                self.drop_in_place(key)
            except:
                pass
            self.hstack([Series(key, value)])
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
                self.replace_at_idx(0, s)
            # df["foo"]
            elif isinstance(col_selection, str):
                self.replace(col_selection, s)
        else:
            return NotImplemented

    def __len__(self):
        return self.height

    def insert_at_idx(self, index: int, series: Series):
        self._df.insert_at_idx(index, series._s)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get shape of the DataFrame
        """
        return self._df.shape()

    @property
    def height(self) -> int:
        """
        Get height of the DataFrame
        """
        return self._df.height()

    @property
    def width(self) -> int:
        """
        Get width of the DataFrame
        """
        return self._df.width()

    @property
    def columns(self) -> List[str]:
        """
        get column names
        """
        return self._df.columns()

    @columns.setter
    def columns(self, columns: List[str]):
        self._df.set_column_names(columns)

    @property
    def dtypes(self) -> List[type]:
        """
        get dtypes
        """
        return [dtypes[idx] for idx in self._df.dtypes()]

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
    ) -> Optional[DataFrame]:
        """
        Sort the DataFrame by column

        Parameters
        ----------
        by_column
            by which column to sort
        in_place
            sort in place or return a sorted DataFrame
        reverse
            reverse sort
        """
        if in_place:
            self._df.sort_in_place(by_column, reverse)
        else:
            return wrap_df(self._df.sort(by_column, reverse))

    def frame_equal(self, other: DataFrame, null_equal: bool = False) -> bool:
        """
        Check if DataFrame is equal to other.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.
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

    def slice(self, offset: int, length: int) -> DataFrame:
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

    def head(self, length: int = 5) -> DataFrame:
        """
        Get first N rows as DataFrame

        Parameters
        ----------
        length
            Length of the head
        """
        return wrap_df(self._df.head(length))

    def tail(self, length: int = 5) -> DataFrame:
        """
        Get last N rows as DataFrame

        Parameters
        ----------
        length
            Length of the tail
        """
        return wrap_df(self._df.tail(length))

    def drop_nulls(self) -> "DataFrame":
        """
        Return a new DataFrame where the null values are dropped
        """
        return wrap_df(self._df.drop_nulls())

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

    def groupby(self, by: Union[str, List[str]]) -> GroupBy:
        """
        Start a groupby operation

        Parameters
        ----------
        by
            Column(s) to group by.
        """
        if isinstance(by, str):
            by = [by]
        return GroupBy(self._df, by)

    def join(
        self,
        df: DataFrame,
        left_on: str,
        right_on: str,
        how="inner",
    ) -> DataFrame:
        """
        SQL like joins

        Parameters
        ----------
        df
            DataFrame to join with
        left_on
            Name of the left join column
        right_on
            Name of the right join column
        how
            Join strategy
                - "inner"
                - "left"
                - "outer"

        Returns
        -------
            Joined DataFrame
        """
        try:
            if how == "inner":
                inner = self._df.inner_join(df._df, left_on, right_on)
            elif how == "left":
                inner = self._df.left_join(df._df, left_on, right_on)
            elif how == "outer":
                inner = self._df.outer_join(df._df, left_on, right_on)
            else:
                return NotImplemented
        except Exception as e:
            self._df.with_parallel(False)
            raise e
        return wrap_df(inner)

    def hstack(self, columns: List[Series]):
        """
        Grow this DataFrame horizontally by stacking Series to it.

        Parameters
        ----------
        columns
            Series to stack
        """
        self._df.hstack([s.inner() for s in columns])

    def vstack(self, df: DataFrame):
        """
        Grow this DataFrame vertically by stacking a DataFrame to it.

        Parameters
        ----------
        df
            DataFrame to stack
        """
        self._df.vstack(df._df)

    def drop(self, name: str) -> DataFrame:
        """
        Remove column from DataFrame and return as new.

        Parameters
        ----------
        name
            Column to drop
        """
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

    def clone(self) -> DataFrame:
        """
        Very cheap deep clone
        """
        return wrap_df(self._df.clone())

    def get_columns(self) -> List[Series]:
        """
        Get the DataFrame as a List of Series
        """
        return list(map(lambda s: wrap_s(s), self._df.get_columns()))

    def fill_none(self, strategy: str) -> DataFrame:
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

    def explode(self, column: str) -> DataFrame:
        """
        Explode `DataFrame` to long format by exploding a column with Lists.

        Parameters
        ----------
        column
            Column of LargeList type

        Returns
        -------
        DataFrame
        """
        return wrap_df(self._df.explode(column))

    def melt(
        self, id_vars: Union[List[str], str], value_vars: Union[List[str], str]
    ) -> DataFrame:
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

    def shift(self, periods: int) -> DataFrame:
        """
        Shift the values by a given period and fill the parts that will be empty due to this operation
        with `Nones`.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        """
        return wrap_df(self._df.shift(periods))

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
        pass

    def n_chunks(self) -> int:
        """
        Get number of chunks used by the ChunkedArrays of this DataFrame
        """
        return self._df.n_chunks()

    def max(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their maximum value
        """
        return wrap_df(self._df.max())

    def min(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their minimum value
        """
        return wrap_df(self._df.min())

    def sum(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their sum value
        """
        return wrap_df(self._df.sum())

    def mean(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their mean value
        """
        return self._df.mean()

    def median(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their median value
        """
        return wrap_df(self._df.median())

    def quantile(self, quantile: float) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their quantile value
        """
        return wrap_df(self._df.quantile(quantile))

    def to_dummies(self) -> DataFrame:
        """
        Get one hot encoded dummy variables.
        """
        return wrap_df(self._df.to_dummies())

    def drop_duplicates(self) -> DataFrame:
        """
        Drop duplicate rows from this DataFrame.
        Note that this fails if there is a column of type `List` in the DataFrame.
        """
        return wrap_df(self._df.drop_duplicates())

    def _rechunk(self) -> DataFrame:
        return wrap_df(self._df.rechunk())


class GroupBy:
    def __init__(self, df: DataFrame, by: List[str]):
        self._df = df
        self.by = by

    def agg(
        self, column_to_agg: Union[List[Tuple[str, List[str]]], Dict[str, List[str]]]
    ) -> DataFrame:
        """
        Use multiple aggregations on columns

        Parameters
        ----------
        column_to_agg
            map column to aggregation functions

            Examples:
                [("foo", ["sum", "n_unique", "min"]),
                 ("bar": ["max"])]

                {"foo": ["sum", "n_unique", "min"],
                "bar": "max" }

        Returns
        -------
        Result of groupby split apply operations.
        """
        if isinstance(column_to_agg, dict):
            column_to_agg = [
                (column, [agg] if isinstance(agg, str) else agg)
                for (column, agg) in column_to_agg.items()
            ]
        else:
            column_to_agg = [
                (column, [agg] if isinstance(agg, str) else agg)
                for (column, agg) in column_to_agg
            ]

        return wrap_df(self._df.groupby_agg(self.by, column_to_agg))

    def select(self, columns: Union[str, List[str]]) -> GBSelection:
        """
        Select the columns that will be aggregated.

        Parameters
        ----------
        columns
            One or multiple columns
        """
        if isinstance(columns, str):
            columns = [columns]
        return GBSelection(self._df, self.by, columns)

    def select_all(self):
        """
        Select all columns for aggregation.
        """
        return GBSelection(self._df, self.by, None)

    def pivot(self, pivot_column: str, values_column: str) -> PivotOps:
        """
        Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.

        Parameters
        ----------
        pivot_column
            Column to pivot.
        values_column
            Column that will be aggregated
        """
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
        self, df: DataFrame, by: List[str], pivot_column: str, values_column: str
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
    def __init__(self, df: DataFrame, by: List[str], selection: Optional[List[str]]):
        self._df = df
        self.by = by
        self.selection = selection

    def first(self) -> DataFrame:
        """
        Aggregate the first values in the group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "first"))

    def last(self) -> DataFrame:
        """
        Aggregate the last values in the group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "last"))

    def sum(self) -> DataFrame:
        """
        Reduce the groups to the sum.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "sum"))

    def min(self) -> DataFrame:
        """
        Reduce the groups to the minimal value.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "min"))

    def max(self) -> DataFrame:
        """
        Reduce the groups to the maximal value.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "max"))

    def count(self) -> DataFrame:
        """
        Count the number of values in each group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "count"))

    def mean(self) -> DataFrame:
        """
        Reduce the groups to the mean values.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "mean"))

    def n_unique(self) -> DataFrame:
        """
        Count the unique values per group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "n_unique"))

    def quantile(self, quantile: float) -> DataFrame:
        """
        Count the unique values per group.
        """
        return wrap_df(self._df.groupby_quantile(self.by, self.selection, quantile))

    def median(self) -> DataFrame:
        """
        Return the median per group.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "median"))

    def agg_list(self) -> DataFrame:
        """
        Aggregate the groups into Series.
        """
        return wrap_df(self._df.groupby(self.by, self.selection, "agg_list"))
