"""
Module for formatting output data in HTML.
"""

from textwrap import dedent
from typing import (
    Dict,
    List,
    Optional,
)
import polars as pl


class Tag:
    def __init__(
        self, elements: List[str], tag: str, attributes: Optional[Dict[str, str]] = None
    ):
        self.tag = tag
        self.elements = elements
        self.attributes = attributes

    def __enter__(self):
        if self.attributes is not None:
            s = f"<{self.tag} "
            for k, v in self.attributes.items():
                s += f'{k}="{v} "'
            s += ">"
            self.elements.append(s)
        else:
            self.elements.append(f"<{self.tag}>")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elements.append(f"</{self.tag}>")


def tag(inner: str, tag: str) -> str:
    return f"<{tag}>{inner}</{tag}>"


def th(inner: str) -> str:
    return tag(inner, "th")


def tr(inner: str) -> str:
    return tag(inner, "tr")


def thead(inner: str) -> str:
    return tag(inner, "thead")


def tbody(inner: str) -> str:
    return tag(inner, "tbody")


def table(inner: str) -> str:
    return tag(inner, "table")


def div(inner: str) -> str:
    return tag(inner, "div")


class HTMLFormatter:
    def __init__(self, df: "pl.DataFrame", max_cols=75, max_rows=40):
        self.df = df
        self.elements = []
        self.max_cols = max_cols
        self.max_rows = max_rows
        if max_rows < df.height:
            self.row_idx = (
                list(range(0, max_rows // 2))
                + [-1]
                + list(range(df.height - max_rows // 2, df.height))
            )
        else:
            self.row_idx = range(0, df.height)
        if max_cols < df.width:
            self.col_idx = (
                list(range(0, max_cols // 2))
                + [-1]
                + list(range(df.width - max_cols // 2, df.width))
            )
        else:
            self.col_idx = range(0, df.width)

    def write_header(self):
        """
        Writes the header of an HTML table
        """
        with Tag(self.elements, "thead"):
            with Tag(self.elements, "tr"):
                for col in self.df.columns:
                    with Tag(self.elements, "th"):
                        self.elements.append(col)
            with Tag(self.elements, "tr"):
                for dtype in self.df.dtypes:
                    dtype = pl.datatypes.DTYPE_TO_FFINAME[dtype]
                    with Tag(self.elements, "td"):
                        self.elements.append(dtype)

    def write_body(self):
        """
        Writes the body of an HTML table
        """
        with Tag(self.elements, "tbody"):
            for r in self.row_idx:
                with Tag(self.elements, "tr"):
                    for c in self.col_idx:
                        with Tag(self.elements, "td"):
                            if r == -1:
                                self.elements.append("...")
                            elif c == -1:
                                self.elements.append("...")
                            else:
                                self.elements.append(f"{self.df[c]._s.get_fmt(r)}")

    def write(self, inner: str):
        self.elements.append(inner)

    def render(self):
        with Tag(self.elements, "table", {"border": "1", "class": "dataframe"}):
            self.write_header()
            self.write_body()


class NotebookFormatter(HTMLFormatter):
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    def write_style(self) -> None:
        # SNIPPET Forked from pandas.

        # We use the "scoped" attribute here so that the desired
        # style properties for the data frame are not then applied
        # throughout the entire notebook.
        template_first = """\
            <style scoped>"""
        template_last = """\
            </style>"""
        template_select = """\
                .dataframe %s {
                    %s: %s;
                }"""
        element_props = [
            ("tbody tr th:only-of-type", "vertical-align", "middle"),
            ("tbody tr th", "vertical-align", "top"),
        ]
        element_props.append(("thead th", "text-align", "right"))
        template_mid = "\n\n".join(map(lambda t: template_select % t, element_props))
        template = dedent("\n".join((template_first, template_mid, template_last)))
        self.write(template)

    def render(self) -> List[str]:
        """
        Return the lines needed to render a HTML table.
        """
        with Tag(self.elements, "div"):
            self.write_style()
            super().render()
        return self.elements


if __name__ == "__main__":
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    fmt = NotebookFormatter(df)

    print("\n".join(fmt.render()))
