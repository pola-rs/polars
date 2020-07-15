from __future__ import annotations
from .polars import PyDataFrame, PySeries
from typing import Dict, Sequence, List, Tuple
from .series import Series, wrap_s
import numpy as np


def wrap_df(df: PyDataFrame) -> DataFrame:
    return DataFrame.from_pydf(df)


class DataFrame:
    def __init__(self, data: Dict[str, Sequence]):
        columns = []
        for k, v in data.items():
            columns.append(Series(k, v).inner())

        self._df = PyDataFrame(columns)

    @staticmethod
    def from_pydf(df: PyDataFrame) -> DataFrame:
        self = DataFrame.__new__(DataFrame)
        self._df = df
        return self

    def __str__(self) -> str:
        return self._df.as_str()

    def __getitem__(self, item):
        # select rows and columns at once
        if isinstance(item, tuple):
            row_selection, col_selection = item

            # column selection can be "a" and ["a", "b"]
            if isinstance(col_selection, str):
                col_selection = [col_selection]
            df = self.__getitem__(col_selection)
            return df.__getitem__(row_selection)

        # select single column
        if isinstance(item, str):
            return wrap_s(self._df.column(item))

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

    def sort(self, by_column: str):
        self._df.sort(by_column)

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

    def groupby(self, by: str, select: str, agg: str) -> DataFrame:
        return wrap_df(self._df.groupby(by, select, agg))
