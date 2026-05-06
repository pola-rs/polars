from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from polars.dataframe import DataFrame
    from polars.lazyframe import LazyFrame


@runtime_checkable
class QueryResult(Protocol):
    """The result of a Polars query.

    .. note::
     This object should not be instantiated directly by the user.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    @property
    def head(self) -> DataFrame | None:
        """The first n rows of the result."""
        ...

    @property
    def n_rows_total(self) -> int | None:
        """Total rows that are outputted by the result."""
        ...

    def lazy(self) -> LazyFrame:
        """Convert the `QueryResult` into a `LazyFrame`."""
        ...


class SingleNodeQueryResult:
    """The result of a Polars query.

    .. note::
     This object should not be instantiated directly by the user.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    def __init__(self, df: DataFrame) -> None:
        self._df = df

    @property
    def head(self) -> DataFrame | None:
        """The first n rows of the result."""
        return self._df.head()

    @property
    def n_rows_total(self) -> int | None:
        """Total rows that are outputted by the result."""
        return self._df.height

    def lazy(self) -> LazyFrame:
        """Convert the `QueryResult` into a `LazyFrame`."""
        return self._df.lazy()

    def __repr__(self) -> str:
        import polars as pl

        with pl.Config(tbl_hide_dataframe_shape=True):
            return f"""
        QueryResult; head:
            {self.head}
        """

    def _repr_html_(self) -> str:
        """Format output data in HTML for display in Jupyter Notebooks."""
        import polars as pl

        with pl.Config(tbl_hide_dataframe_shape=True):
            head_html = self._df.head()._repr_html_()

        head_section = f"""
        <div style="margin-bottom: 16px;">
            <h3 style="margin: 0 0 8px 0; color: #333; font-family: sans-serif; font-size: 14px; font-weight: 600;">QueryResult; head:</h3>
            <div>{head_html}</div>
        </div>
        """

        return f"""
        <div style="padding: 12px; border: 1px solid #ddd; border-radius: 4px; background: white;">
            {head_section}
        </div>
        """
