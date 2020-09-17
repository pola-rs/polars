from __future__ import annotations
from .pypolars import PyDataFrame, PySeries
from typing import Dict, Sequence, List, Tuple, Optional, Union
from .series import Series, wrap_s
import numpy as np


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
        path: str,
        infer_schema_length: int = 100,
        batch_size: int = 100000,
        has_headers: bool = True,
        ignore_errors: bool = False,
    ) -> DataFrame:
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_csv(
            path, infer_schema_length, batch_size, has_headers, ignore_errors
        )
        return self

    @staticmethod
    def from_parquet(path: str, batch_size: int = 250000,) -> DataFrame:
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_parquet(path, batch_size)
        return self

    @staticmethod
    def from_ipc(path: str) -> DataFrame:
        self = DataFrame.__new__(DataFrame)
        self._df = PyDataFrame.from_ipc(path)
        return self

    def to_csv(
        self,
        path: str,
        batch_size: int = 100000,
        has_headers: bool = True,
        delimiter: str = ",",
    ):
        self._df.to_csv(path, batch_size, has_headers, ord(delimiter))

    def to_ipc(self, path: str, batch_size):
        self._df.to_ipc(path, batch_size)

    def __str__(self) -> str:
        return self._df.as_str()

    def __repr__(self) -> str:
        return self.__str__()

    def __getattr__(self, item) -> PySeries:
        """
        Access columns as attribute
        """
        return wrap_s(self._df.column(item))

    def __getitem__(self, item):
        # select rows and columns at once
        if isinstance(item, tuple):
            row_selection, col_selection = item

            if isinstance(row_selection, slice):
                series = self.__getitem__(col_selection)
                wrap_s(series[row_selection])

            # column selection can be "a" and ["a", "b"]
            if isinstance(col_selection, str):
                col_selection = [col_selection]
            df = self.__getitem__(col_selection)
            return df.__getitem__(row_selection)

        # select single column
        if isinstance(item, str):
            return wrap_s(self._df.column(item))

        if isinstance(item, int):
            return wrap_s(self._df.select_at_idx(item))

        # select multiple columns
        if isinstance(item, Sequence) and isinstance(item[0], str):
            return wrap_df(self._df.select(item))

        # select rows by mask or index
        if isinstance(item, (Series, Sequence)):
            if isinstance(item, Sequence):
                # only bool or integers allowed
                if type(item[0]) == bool:
                    item = Series("", item)
                else:
                    return wrap_df(self._df.take(item))
            dtype = item.dtype
            if dtype == "bool":
                return wrap_df(self._df.filter(item.inner()))
            if dtype == "u32":
                return wrap_df(self._df.take_with_series(item.inner()))
        return NotImplemented

    def __len__(self):
        return self.height

    @property
    def shape(self) -> Tuple[int, int]:
        return self._df.shape()

    @property
    def height(self) -> int:
        return self._df.height()

    @property
    def width(self) -> int:
        return self._df.width()

    @property
    def columns(self) -> List[str]:
        return self._df.columns()

    def sort(
        self, by_column: str, in_place: bool = False, reverse: bool = False
    ) -> Optional[DataFrame]:
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

    def groupby(self, by: Union[str, List[str]]) -> GroupBy:
        if isinstance(by, str):
            by = [by]
        return GroupBy(self._df, by)

    def join(
        self, df: DataFrame, left_on: str, right_on: str, how="inner"
    ) -> DataFrame:
        if how == "inner":
            inner = self._df.inner_join(df._df, left_on, right_on)
        elif how == "left":
            inner = self._df.left_join(df._df, left_on, right_on)
        elif how == "outer":
            inner = self._df.outer_join(df._df, left_on, right_on)
        else:
            raise NotImplemented
        return wrap_df(inner)

    def hstack(self, columns: List[Series]):
        self._df.hstack([s.inner() for s in columns])

    def drop(self, name: str) -> DataFrame:
        return wrap_df(self._df.drop(name))

    def drop_in_place(self, name: str) -> Series:
        return wrap_s(self._df.drop_in_place(name))

    def select_at_idx(self, idx: int) -> Series:
        return wrap_s(self._df.select_at_idx(idx))


class GroupBy:
    def __init__(self, df: DataFrame, by: List[str]):
        self._df = df
        self.by = by

    def select(self, columns: Union[str, List[str]]) -> GBSelection:
        if isinstance(columns, str):
            columns = [columns]
        return GBSelection(self._df, self.by, columns)

    def pivot(self, pivot_column: str, values_column: str) -> PivotOps:
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
        wrap_df(self._df.pivot(self.by, self.pivot_column, self.values_column, "first"))

    def sum(self):
        wrap_df(self._df.pivot(self.by, self.pivot_column, self.values_column, "sum"))

    def min(self):
        wrap_df(self._df.pivot(self.by, self.pivot_column, self.values_column, "min"))

    def max(self):
        wrap_df(self._df.pivot(self.by, self.pivot_column, self.values_column, "max"))

    def mean(self):
        wrap_df(self._df.pivot(self.by, self.pivot_column, self.values_column, "mean"))

    def median(self):
        wrap_df(
            self._df.pivot(self.by, self.pivot_column, self.values_column, "median")
        )


class GBSelection:
    def __init__(self, df: DataFrame, by: List[str], selection: List[str]):
        self._df = df
        self.by = by
        self.selection = selection

    def first(self):
        return wrap_df(self._df.groupby(self.by, self.selection, "first"))

    def sum(self):
        return wrap_df(self._df.groupby(self.by, self.selection, "sum"))

    def min(self):
        return wrap_df(self._df.groupby(self.by, self.selection, "min"))

    def max(self):
        return wrap_df(self._df.groupby(self.by, self.selection, "max"))

    def count(self):
        return wrap_df(self._df.groupby(self.by, self.selection, "count"))

    def mean(self):
        return wrap_df(self._df.groupby(self.by, self.selection, "mean"))
