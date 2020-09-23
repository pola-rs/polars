from __future__ import annotations
from .pypolars import PyDataFrame, PySeries
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


def wrap_df(df: PyDataFrame) -> DataFrame:
    return DataFrame.from_pydf(df)


class DataFrame:
    def __init__(self, data: Dict[str, Sequence], nullable: bool = False):
        columns = []
        for k, v in data.items():
            columns.append(Series(k, v, nullable=nullable).inner())

        self._df = PyDataFrame(columns)

    @staticmethod
    def from_pydf(df: PyDataFrame) -> DataFrame:
        self = DataFrame.__new__(DataFrame)
        self._df = df
        return self

    @staticmethod
    def from_csv(
        file: Union[str, TextIO],
        infer_schema_length: int = 100,
        batch_size: int = 100000,
        has_headers: bool = True,
        ignore_errors: bool = False,
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

        Returns
        -------
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_csv(
            file, infer_schema_length, batch_size, has_headers, ignore_errors
        )
        return self

    @staticmethod
    def from_parquet(
        file: Union[str, BinaryIO], batch_size: int = 250000,
    ) -> DataFrame:
        """
        Read into a DataFrame from a parquet file.

        Parameters
        ----------
        file
            Path to a file or a file like object.
        batch_size
            Number of lines to read into the buffer at once. Modify this to change performance.

        Returns
        -------
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_parquet(file, batch_size)
        return self

    @staticmethod
    def from_ipc(file: Union[str, BinaryIO]) -> DataFrame:
        """
        Read into a DataFrame from Arrow IPC stream format.

        Parameters
        ----------
        file
            Path to a file or a file like object.

        Returns
        -------
        DataFrame
        """
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_ipc(file)
        return self

    def to_csv(
        self,
        path: str,
        batch_size: int = 100000,
        has_headers: bool = True,
        delimiter: str = ",",
    ):
        """
        Write DataFrame to CSV

        Parameters
        ----------
        path
            write location
        batch_size
            Size of the write buffer. Increase to have faster io.
        has_headers
            Whether or not to include header in the CSV output.
        delimiter
            Space elements with this symbol.
        """
        self._df.to_csv(path, batch_size, has_headers, ord(delimiter))

    def to_ipc(self, path: str, batch_size):
        """
        Write to Arrow IPC binary stream.

        Parameters
        ----------
        path
            write location
        batch_size
            Size of the write buffer. Increase to have faster io.
        """
        self._df.to_ipc(path, batch_size)

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
        return NotImplemented

    def __setitem__(self, key, value):
        # df["foo"] = series
        if isinstance(key, str):
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

    def frame_equal(self, other: DataFrame) -> bool:
        return self._df.frame_equal(other._df)

    def replace(self, column: str, new_col: Series):
        self._df.replace(column, new_col.inner())

    def slice(self, offset: int, length: int) -> DataFrame:
        return wrap_df(self._df.slice(offset, length))

    def head(self, length: int = 5) -> DataFrame:
        return wrap_df(self._df.head(length))

    def tail(self, length: int = 5) -> DataFrame:
        return wrap_df(self._df.tail(length))

    def pipe(self, func: Callable, *args, **kwargs):
        return func(self, *args, **kwargs)

    def groupby(self, by: Union[str, List[str]]) -> GroupBy:
        if isinstance(by, str):
            by = [by]
        return GroupBy(self._df, by)

    def join(
        self,
        df: DataFrame,
        left_on: str,
        right_on: str,
        how="inner",
        parallel: bool = False,
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
        parallel
            Use parallel join strategy.

        Returns
        -------
            Joined DataFrame
        """
        self._df.with_parallel(parallel)
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
        self._df.hstack([s.inner() for s in columns])

    def vstack(self, df: DataFrame):
        self._df.vstack(df._df)

    def drop(self, name: str) -> DataFrame:
        return wrap_df(self._df.drop(name))

    def drop_in_place(self, name: str) -> Series:
        return wrap_s(self._df.drop_in_place(name))

    def select_at_idx(self, idx: int) -> Series:
        return wrap_s(self._df.select_at_idx(idx))

    def clone(self) -> DataFrame:
        """
        Cheap deep clone
        """
        return wrap_df(self._df.clone())

    def get_columns(self) -> List[Series]:
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


class GroupBy:
    def __init__(self, df: DataFrame, by: List[str]):
        self._df = df
        self.by = by

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
        return GBSelection(self._df, self.by, self._df.columns())

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


class PivotOps:
    def __init__(
        self, df: DataFrame, by: List[str], pivot_column: str, values_column: str
    ):
        self._df = df
        self.by = by
        self.pivot_column = pivot_column
        self.values_column = values_column

    def first(self):
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "first")
        )

    def sum(self):
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "sum")
        )

    def min(self):
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "min")
        )

    def max(self):
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "max")
        )

    def mean(self):
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "mean")
        )

    def median(self):
        return wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )


class GBSelection:
    def __init__(self, df: DataFrame, by: List[str], selection: List[str]):
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
