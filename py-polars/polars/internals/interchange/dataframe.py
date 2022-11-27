from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from polars.internals.interchange.column import PolarsColumn
from polars.internals.interchange.dataframe_protocol import DataFrame as DataFrameXchg

if TYPE_CHECKING:
    from typing import Any

    import polars as pl


class PolarsDataFrameXchg(DataFrameXchg):
    """
    DataFrame class with only the methods required by the interchange protocol defined.

    Instances of this (private) class are returned from ``pl.DataFrame.__dataframe__``
    as objects with the methods and attributes defined on this class.
    """

    def __init__(
        self, df: pl.DataFrame, nan_as_null: bool = False, allow_copy: bool = True
    ):
        """
        Constructor - an instance of this (private) class is returned from
        `pd.DataFrame.__dataframe__`.
        """
        self._df = df
        self._nan_as_null = nan_as_null  # Has no effect for now
        self._allow_copy = allow_copy

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> PolarsDataFrameXchg:
        return PolarsDataFrameXchg(self._df, nan_as_null, allow_copy)

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    def num_columns(self) -> int:
        return self._df.shape[1]

    def num_rows(self) -> int:
        return self._df.shape[0]

    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.

        It is possible for a Polars DataFrame to consist of Columns with a varying
        number of chunks. This method returns the number of chunks of the first
        column.

        See Also
        --------
        polars.internals.dataframe.frame.DataFrame.n_chunks

        """
        return self._df.n_chunks()

    def column_names(self) -> list[str]:
        return self._df.columns

    def get_column(self, i: int) -> PolarsColumn:
        return PolarsColumn(self._df.to_series(i), allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> PolarsColumn:
        return PolarsColumn(self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> list[PolarsColumn]:
        return [
            PolarsColumn(column, allow_copy=self._allow_copy)
            for column in self._df.get_columns()
        ]

    def select_columns(self, indices: Sequence[int]) -> PolarsDataFrameXchg:
        if not isinstance(indices, Sequence):
            raise ValueError("`indices` is not a sequence")
        if not isinstance(indices, list):
            indices = list(indices)

        return PolarsDataFrameXchg(
            self._df[:, indices], self._nan_as_null, self._allow_copy
        )

    def select_columns_by_name(self, names: Sequence[str]) -> PolarsDataFrameXchg:
        if not isinstance(names, Sequence):
            raise ValueError("`names` is not a sequence")

        return PolarsDataFrameXchg(
            self._df.select(names), self._nan_as_null, self._allow_copy
        )

    def get_chunks(self, n_chunks: int | None = None) -> Iterable[PolarsDataFrameXchg]:
        total_n_chunks = self.num_chunks()
        chunks = self._df.get_chunks()  # TODO: Implement

        if (n_chunks is None) or (n_chunks == total_n_chunks):
            for chunk in chunks:
                yield PolarsDataFrameXchg(chunk, self._allow_copy)

        elif (n_chunks <= 0) or (n_chunks % total_n_chunks != 0):
            raise ValueError(
                "`n_chunks` must be a multiple of the number of chunks of this"
                f" dataframe ({total_n_chunks})"
            )

        else:
            subchunks_per_chunk = n_chunks // total_n_chunks
            for chunk in chunks:
                size = len(chunk)
                step = size // subchunks_per_chunk
                if size % subchunks_per_chunk != 0:
                    step += 1
                for start in range(0, step * subchunks_per_chunk, step):
                    yield PolarsDataFrameXchg(
                        chunk[start : start + step, :], self._allow_copy
                    )
