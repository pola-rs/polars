"""Module for formatting output data in HTML."""
from __future__ import annotations

import html
import os
from textwrap import dedent
from types import TracebackType
from typing import Iterable


class Tag:
    """Class for representing an HTML tag."""

    def __init__(
        self,
        elements: list[str],
        tag: str,
        attributes: dict[str, str] | None = None,
    ):
        self.tag = tag
        self.elements = elements
        self.attributes = attributes

    def __enter__(self) -> None:
        if self.attributes is not None:
            s = f"<{self.tag} "
            for k, v in self.attributes.items():
                s += f'{k}="{v}" '
            s = f"{s.rstrip()}>"
            self.elements.append(s)
        else:
            self.elements.append(f"<{self.tag}>")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elements.append(f"</{self.tag}>")


class HTMLFormatter:
    def __init__(
        self,
        df: DataFrame,  # type: ignore[name-defined] # noqa: F821
        max_cols: int = 75,
        max_rows: int = 40,
        from_series: bool = False,
    ):
        self.df = df
        self.elements: list[str] = []
        self.max_cols = max_cols
        self.max_rows = max_rows
        self.series = from_series
        self.row_idx: Iterable[int]
        self.col_idx: Iterable[int]

        if max_rows < df.height:
            self.row_idx = [
                *list(range(0, max_rows // 2)),
                -1,
                *list(range(df.height - max_rows // 2, df.height)),
            ]
        else:
            self.row_idx = range(0, df.height)
        if max_cols < df.width:
            self.col_idx = [
                *list(range(0, max_cols // 2)),
                -1,
                *list(range(df.width - max_cols // 2, df.width)),
            ]
        else:
            self.col_idx = range(0, df.width)

    def write_header(self) -> None:
        """Write the header of an HTML table."""
        shape = self.df.shape
        if self.series:
            shape = shape[:1]

        self.elements.append(f"<small>shape: {shape}</small>")
        with Tag(self.elements, "thead"):
            with Tag(self.elements, "tr"):
                columns = self.df.columns
                for c in self.col_idx:
                    with Tag(self.elements, "th"):
                        if c == -1:
                            self.elements.append("...")
                        else:
                            self.elements.append(columns[c])
            with Tag(self.elements, "tr"):
                dtypes = self.df._df.dtype_strings()
                for c in self.col_idx:
                    with Tag(self.elements, "td"):
                        if c == -1:
                            self.elements.append("...")
                        else:
                            self.elements.append(dtypes[c])

    def write_body(self) -> None:
        """Write the body of an HTML table."""
        str_lengths = int(os.environ.get("POLARS_FMT_STR_LEN", "15"))
        with Tag(self.elements, "tbody"):
            for r in self.row_idx:
                with Tag(self.elements, "tr"):
                    for c in self.col_idx:
                        with Tag(self.elements, "td"):
                            if r == -1 or c == -1:
                                self.elements.append("...")
                            else:
                                series = self.df[:, c]

                                self.elements.append(
                                    html.escape(series._s.get_fmt(r, str_lengths))
                                )

    def write(self, inner: str) -> None:
        self.elements.append(inner)

    def render(self) -> list[str]:
        with Tag(
            self.elements, "table", {"border": "1", "class": "dataframe pl-dataframe"}
        ):
            self.write_header()
            self.write_body()
        return self.elements


class NotebookFormatter(HTMLFormatter):
    """
    Class for formatting output data in HTML for display in Jupyter Notebooks.

    This class is intended for functionality specific to DataFrame._repr_html_().

    """

    def write_style(self) -> None:
        style = """\
            <style>
            .pl-dataframe > thead > tr > th {
              text-align: right;
            }
            </style>
        """
        self.write(dedent(style))

    def render(self) -> list[str]:
        """Return the lines needed to render a HTML table."""
        with Tag(self.elements, "div"):
            self.write_style()
            super().render()
        return self.elements
