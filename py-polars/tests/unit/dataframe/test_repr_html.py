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


def test_html_representation_multiple_spaces() -> None:
    df = pl.DataFrame(
        {"string_col": ["multiple   spaces", "  trailing and leading   "]}
    )
    html_repr = df._repr_html_()

    assert (
        html_repr
        == """<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (2, 1)</small><table border="1" class="dataframe"><thead><tr><th>string_col</th></tr><tr><td>str</td></tr></thead><tbody><tr><td>&quot;multiple&nbsp;&nbsp;&nbsp;spaces&quot;</td></tr><tr><td>&quot;&nbsp;&nbsp;trailing&nbsp;and&nbsp;leading&nbsp;&nbsp;&nbsp;&quot;</td></tr></tbody></table></div>"""
    )
