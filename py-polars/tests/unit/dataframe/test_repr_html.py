import re

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

    # Regex explanation:
    # Matches cell content inside <td>...</td> tags, but only within the <tbody> section
    # 1. <tbody>: Ensures matching starts within the <tbody> section.
    # 2. .*?: Lazily matches any content until the first <td> tag.
    # 3. <td>(.*?)</td>: Captures the content inside each <td> tag (non-greedy).
    # 4. .*?: Lazily matches any content between <td>...</td> and </tbody>.
    # 5. </tbody>: Ensures matching ends at the closing </tbody> tag.
    # The re.S flag allows the regex to work across multiple lines.
    cell_pattern = re.compile(r"<tbody>.*?<td>(.*?)</td>.*?</tbody>", re.S)

    cells = cell_pattern.findall(html_repr)

    for cell_content in cells:
        # Check that there are no regular spaces in the content
        assert " " not in cell_content, f"Unexpected space in cell: {cell_content}"
        # Check that the content contains &nbsp; as required
        assert (
            "&nbsp;" in cell_content
        ), f"Expected &nbsp; in cell but found: {cell_content}"
