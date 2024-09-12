"""Module for formatting output data in HTML."""

from __future__ import annotations

import os
from textwrap import dedent
from typing import TYPE_CHECKING, Iterable, Literal

from polars.dependencies import html

if TYPE_CHECKING:
    from types import TracebackType

    from polars import DataFrame


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
    """
    Class for HTML formatting.

    Table cell alignment will be set by environment variables.

    Parameters
    ----------
    df: DataFrame
    max_cols: int
    max_rows: int
    from_series: bool
    overall_alignment: Literal["LEFT", "CENTER", "RIGHT"]
    numeric_alignment: Literal["LEFT", "CENTER", "RIGHT"]
    """

    def __init__(
        self,
        df: DataFrame,
        *,
        max_cols: int = 75,
        max_rows: int = 40,
        from_series: bool = False,
        overall_alignment: Literal["LEFT", "CENTER", "RIGHT"] = "RIGHT",
        numeric_alignment: Literal["LEFT", "CENTER", "RIGHT"] = "RIGHT",
    ):
        self.df = df
        self.elements: list[str] = []
        self.max_cols = max_cols
        self.max_rows = max_rows
        self.from_series = from_series
        self.row_idx: Iterable[int]
        self.col_idx: Iterable[int]
        self.overall_align_lower: str
        self.numeric_align_lower: str
        self.attribute_nested_dict: dict[int, dict[str, str] | None]

        if max_rows < df.height:
            half, rest = divmod(max_rows, 2)
            self.row_idx = [
                *list(range(half + rest)),
                -1,
                *list(range(df.height - half, df.height)),
            ]
        else:
            self.row_idx = range(df.height)
        if max_cols < df.width:
            self.col_idx = [
                *list(range(max_cols // 2)),
                -1,
                *list(range(df.width - max_cols // 2, df.width)),
            ]
        else:
            self.col_idx = range(df.width)

        self.overall_align_lower = overall_alignment.lower()
        self.numeric_align_lower = numeric_alignment.lower()
        self.attribute_nested_dict = {
            c: self.get_attributes(col_idx=c) for c in self.col_idx
        }

    def write_header(self) -> None:
        """Write the header of an HTML table."""
        with Tag(self.elements, "thead"):
            if not bool(int(os.environ.get("POLARS_FMT_TABLE_HIDE_COLUMN_NAMES", "0"))):
                with Tag(self.elements, "tr"):
                    columns = self.df.columns
                    for c in self.col_idx:
                        _dict = self.attribute_nested_dict[c]
                        with Tag(self.elements, "th", _dict):
                            if c == -1:
                                self.elements.append("&hellip;")
                            else:
                                self.elements.append(html.escape(columns[c]))
            if not bool(
                int(os.environ.get("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES", "0"))
            ):
                with Tag(self.elements, "tr"):
                    dtypes = self.df._df.dtype_strings()
                    for c in self.col_idx:
                        _dict = self.attribute_nested_dict[c]
                        with Tag(self.elements, "td", _dict):
                            if c == -1:
                                self.elements.append("&hellip;")
                            else:
                                self.elements.append(dtypes[c])

    def write_body(self) -> None:
        """Write the body of an HTML table."""
        str_len_limit = int(os.environ.get("POLARS_FMT_STR_LEN", default=30))
        with Tag(self.elements, "tbody"):
            for r in self.row_idx:
                with Tag(self.elements, "tr"):
                    for c in self.col_idx:
                        _dict = self.attribute_nested_dict[c]
                        with Tag(self.elements, "td", _dict):
                            if r == -1 or c == -1:
                                self.elements.append("&hellip;")
                            else:
                                series = self.df[:, c]
                                self.elements.append(
                                    html.escape(series._s.get_fmt(r, str_len_limit))
                                )

    def get_attributes(self, col_idx: int) -> dict[str, str] | None:
        """
        Get HTML td/th attributes of a column.

        Parameters
        ----------
        col_idx: int
            index number of the target column.
        """
        if self.numeric_align_lower != self.overall_align_lower:
            series = self.df[:, col_idx]
            if series.dtype.is_numeric():
                return {"align": self.numeric_align_lower}
        return None

    def write(self, inner: str) -> None:
        """Append a raw string to the inner HTML."""
        self.elements.append(inner)

    def render(self) -> list[str]:
        """Return the lines needed to render a HTML table."""
        if not bool(
            int(
                os.environ.get("POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION", "0")
            )
        ):
            # format frame/series shape with '_' thousand-separators
            s = self.df.shape
            shape = f"({s[0]:_},)" if self.from_series else f"({s[0]:_}, {s[1]:_})"

            self.elements.append(f"<small>shape: {shape}</small>")

        with Tag(
            # be careful changing the CSS class ref here...
            # ref: https://github.com/pola-rs/polars/issues/7443
            self.elements,
            "table",
            {"border": "1", "class": "dataframe"},
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
        """Write <style> tag."""
        style = """
            <style>
            .dataframe > thead > tr,
            .dataframe > tbody > tr {{
              text-align: {};
              white-space: pre-wrap;
            }}
            </style>
        """
        style_string = dedent(style.format(self.overall_align_lower)).strip()
        self.write(style_string)

    def render(self) -> list[str]:
        """Return the lines needed to render a HTML table."""
        with Tag(self.elements, "div"):
            self.write_style()
            super().render()
        return self.elements
