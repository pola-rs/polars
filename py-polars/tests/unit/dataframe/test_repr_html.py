import pytest

import polars as pl


def test_repr_html() -> None:
    # check it does not panic/error, and appears to contain
    # a reasonable table with suitably escaped html entities.
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "<bar>": ["a", "b", "c"],
            "<baz": ["a", "b", "c"],
            "spam>": ["a", "b", "c"],
        }
    )
    html = df._repr_html_()
    for match in (
        "<table",
        'class="dataframe"',
        "<th>foo</th>",
        "<th>&lt;bar&gt;</th>",
        "<th>&lt;baz</th>",
        "<th>spam&gt;</th>",
        "<td>1</td>",
        "<td>2</td>",
        "<td>3</td>",
    ):
        assert match in html, f"Expected to find {match!r} in html repr"


def test_html_tables() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    # default: header contains names/dtypes
    header = "<thead><tr><th>a</th><th>b</th><th>c</th></tr><tr><td>i64</td><td>i64</td><td>i64</td></tr></thead>"
    assert header in df._repr_html_()

    # validate that relevant config options are respected
    with pl.Config(tbl_hide_column_names=True):
        header = "<thead><tr><td>i64</td><td>i64</td><td>i64</td></tr></thead>"
        assert header in df._repr_html_()

    with pl.Config(tbl_hide_column_data_types=True):
        header = "<thead><tr><th>a</th><th>b</th><th>c</th></tr></thead>"
        assert header in df._repr_html_()

    with pl.Config(
        tbl_hide_column_data_types=True,
        tbl_hide_column_names=True,
    ):
        header = "<thead></thead>"
        assert header in df._repr_html_()


def test_df_repr_html_max_rows_default() -> None:
    df = pl.DataFrame({"a": range(50)})

    html = df._repr_html_()

    expected_rows = 10
    assert html.count("<td>") - 2 == expected_rows


def test_df_repr_html_max_rows_odd() -> None:
    df = pl.DataFrame({"a": range(50)})

    with pl.Config(tbl_rows=9):
        html = df._repr_html_()

    expected_rows = 9
    assert html.count("<td>") - 2 == expected_rows


def test_series_repr_html_max_rows_default() -> None:
    s = pl.Series("a", range(50))

    html = s._repr_html_()

    expected_rows = 10
    assert html.count("<td>") - 2 == expected_rows


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("single space", "single space"),
        ("multiple   spaces", "multiple&nbsp;&nbsp;&nbsp;spaces"),
        (
            "  trailing & leading spaces  ",
            "&nbsp;&nbsp;trailing &amp; leading spaces&nbsp;&nbsp;",
        ),
    ],
)
def test_html_representation_multiple_spaces(text: str, expected: str) -> None:
    with pl.Config(fmt_str_lengths=100):
        html_repr = pl.DataFrame({"s": [text]})._repr_html_()
        assert f"<td>&quot;{expected}&quot;</td>" in html_repr
