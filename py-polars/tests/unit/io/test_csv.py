from __future__ import annotations

import gzip
import io
import sys
import textwrap
import zlib
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import ComputeError, NoDataError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.utils.various import normalize_filepath

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import TimeUnit


@pytest.fixture()
def foods_file_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.csv"


def test_quoted_date() -> None:
    csv = textwrap.dedent(
        """\
        a,b
        "2022-01-01",1
        "2022-01-02",2
        """
    )
    result = pl.read_csv(csv.encode(), try_parse_dates=True)
    expected = pl.DataFrame({"a": [date(2022, 1, 1), date(2022, 1, 2)], "b": [1, 2]})
    assert_frame_equal(result, expected)


# Issue: https://github.com/pola-rs/polars/issues/10826
def test_date_pattern_with_datetime_override_10826() -> None:
    result = pl.read_csv(
        source=io.StringIO("col\n2023-01-01\n2023-02-01\n2023-03-01"),
        dtypes={"col": pl.Datetime},
    )
    expected = pl.Series(
        "col", [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1)]
    ).to_frame()
    assert_frame_equal(result, expected)

    result = pl.read_csv(
        source=io.StringIO("col\n2023-01-01T01:02:03\n2023-02-01\n2023-03-01"),
        dtypes={"col": pl.Datetime},
    )
    expected = pl.Series(
        "col",
        [datetime(2023, 1, 1, 1, 2, 3), datetime(2023, 2, 1), datetime(2023, 3, 1)],
    ).to_frame()
    assert_frame_equal(result, expected)


def test_to_from_buffer(df_no_lists: pl.DataFrame) -> None:
    df = df_no_lists
    buf = io.BytesIO()
    df.write_csv(buf)
    buf.seek(0)

    read_df = pl.read_csv(buf, try_parse_dates=True)
    read_df = read_df.with_columns(
        [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    )
    assert_frame_equal(df, read_df, categorical_as_str=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(df.select("time", "cat"), read_df, categorical_as_str=True)


@pytest.mark.write_disk()
def test_to_from_file(df_no_lists: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = df_no_lists.drop("strings_nulls")

    file_path = tmp_path / "small.csv"
    df.write_csv(file_path)
    read_df = pl.read_csv(file_path, try_parse_dates=True)

    read_df = read_df.with_columns(
        [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    )
    assert_frame_equal(df, read_df, categorical_as_str=True)


def test_normalize_filepath(io_files_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        normalize_filepath(io_files_path)

    assert normalize_filepath(str(io_files_path), check_not_directory=False) == str(
        io_files_path
    )


def test_csv_null_values() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c
        na,b,c
        a,na,c
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values="na")
    assert df.rows() == [(None, "b", "c"), ("a", None, "c")]

    # note: after reading, the buffer position in StringIO will have been
    # advanced; reading again will raise NoDataError, so we provide a hint
    # in the error string about this, suggesting "seek(0)" as a possible fix...
    with pytest.raises(
        NoDataError, match=r"empty CSV data .* position = 20; try seek\(0\)"
    ):
        pl.read_csv(f)

    # ... unless we explicitly tell read_csv not to raise an
    # exception, in which case we expect an empty dataframe
    assert_frame_equal(pl.read_csv(f, raise_if_empty=False), pl.DataFrame())

    out = io.BytesIO()
    df.write_csv(out, null_value="na")
    assert csv == out.getvalue().decode("ascii")

    csv = textwrap.dedent(
        """\
        a,b,c
        na,b,c
        a,n/a,c
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values=["na", "n/a"])
    assert df.rows() == [(None, "b", "c"), ("a", None, "c")]

    csv = textwrap.dedent(
        r"""
        a,b,c
        na,b,c
        a,\N,c
        ,b,
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values={"a": "na", "b": r"\N"})
    assert df.rows() == [(None, "b", "c"), ("a", None, "c"), (None, "b", None)]


def test_csv_missing_utf8_is_empty_string() -> None:
    # validate 'missing_utf8_is_empty_string' for missing fields that are...
    # >> ...leading
    # >> ...trailing (both EOL & EOF)
    # >> ...in lines that have missing fields
    # >> ...in cols containing no other strings
    # >> ...interacting with other user-supplied null values

    csv = textwrap.dedent(
        r"""
        a,b,c
        na,b,c
        a,\N,c
        ,b,
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(
        f,
        null_values={"a": "na", "b": r"\N"},
        missing_utf8_is_empty_string=True,
    )
    # ┌──────┬──────┬─────┐
    # │ a    ┆ b    ┆ c   │
    # ╞══════╪══════╪═════╡
    # │ null ┆ b    ┆ c   │
    # │ a    ┆ null ┆ c   │
    # │      ┆ b    ┆     │
    # └──────┴──────┴─────┘
    assert df.rows() == [(None, "b", "c"), ("a", None, "c"), ("", "b", "")]

    csv = textwrap.dedent(
        r"""
        a,b,c,d,e,f,g
        na,,,,\N,,
        a,\N,c,,,,g
        ,,,
        ,,,na,,,
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values=["na", r"\N"])
    # ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
    # │ a    ┆ b    ┆ c    ┆ d    ┆ e    ┆ f    ┆ g    │
    # ╞══════╪══════╪══════╪══════╪══════╪══════╪══════╡
    # │ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
    # │ a    ┆ null ┆ c    ┆ null ┆ null ┆ null ┆ g    │
    # │ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
    # │ null ┆ null ┆ null ┆ null ┆ null ┆ null ┆ null │
    # └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
    assert df.rows() == [
        (None, None, None, None, None, None, None),
        ("a", None, "c", None, None, None, "g"),
        (None, None, None, None, None, None, None),
        (None, None, None, None, None, None, None),
    ]

    f.seek(0)
    df = pl.read_csv(
        f,
        null_values=["na", r"\N"],
        missing_utf8_is_empty_string=True,
    )
    # ┌──────┬──────┬─────┬──────┬──────┬──────┬─────┐
    # │ a    ┆ b    ┆ c   ┆ d    ┆ e    ┆ f    ┆ g   │
    # ╞══════╪══════╪═════╪══════╪══════╪══════╪═════╡
    # │ null ┆      ┆     ┆      ┆ null ┆      ┆     │
    # │ a    ┆ null ┆ c   ┆      ┆      ┆      ┆ g   │
    # │      ┆      ┆     ┆      ┆      ┆      ┆     │
    # │      ┆      ┆     ┆ null ┆      ┆      ┆     │
    # └──────┴──────┴─────┴──────┴──────┴──────┴─────┘
    assert df.rows() == [
        (None, "", "", "", None, "", ""),
        ("a", None, "c", "", "", "", "g"),
        ("", "", "", "", "", "", ""),
        ("", "", "", None, "", "", ""),
    ]


def test_csv_float_parsing() -> None:
    lines_with_floats = [
        "123.86,+123.86,-123.86\n",
        ".987,+.987,-.987\n",
        "5.,+5.,-5.\n",
        "inf,+inf,-inf\n",
        "NaN,+NaN,-NaN\n",
    ]

    for line_with_floats in lines_with_floats:
        f = io.StringIO(line_with_floats)
        df = pl.read_csv(f, has_header=False, new_columns=["a", "b", "c"])
        assert df.dtypes == [pl.Float64, pl.Float64, pl.Float64]

    lines_with_scientific_numbers = [
        "1e27,1E65,1e-28,1E-9\n",
        "+1e27,+1E65,+1e-28,+1E-9\n",
        "1e+27,1E+65,1e-28,1E-9\n",
        "+1e+27,+1E+65,+1e-28,+1E-9\n",
        "-1e+27,-1E+65,-1e-28,-1E-9\n",
        #        "e27,E65,e-28,E-9\n",
        #        "+e27,+E65,+e-28,+E-9\n",
        #        "-e27,-E65,-e-28,-E-9\n",
    ]

    for line_with_scientific_numbers in lines_with_scientific_numbers:
        f = io.StringIO(line_with_scientific_numbers)
        df = pl.read_csv(f, has_header=False, new_columns=["a", "b", "c", "d"])
        assert df.dtypes == [pl.Float64, pl.Float64, pl.Float64, pl.Float64]


def test_datetime_parsing() -> None:
    csv = textwrap.dedent(
        """\
        timestamp,open,high
        2021-01-01 00:00:00,0.00305500,0.00306000
        2021-01-01 00:15:00,0.00298800,0.00300400
        2021-01-01 00:30:00,0.00298300,0.00300100
        2021-01-01 00:45:00,0.00299400,0.00304000
        """
    )

    f = io.StringIO(csv)
    df = pl.read_csv(f, try_parse_dates=True)
    assert df.dtypes == [pl.Datetime, pl.Float64, pl.Float64]


def test_datetime_parsing_default_formats() -> None:
    csv = textwrap.dedent(
        """\
        ts_dmy,ts_dmy_f,ts_dmy_p
        01/01/21 00:00:00,31-01-2021T00:00:00.123,31-01-2021 11:00
        01/01/21 00:15:00,31-01-2021T00:15:00.123,31-01-2021 01:00
        01/01/21 00:30:00,31-01-2021T00:30:00.123,31-01-2021 01:15
        01/01/21 00:45:00,31-01-2021T00:45:00.123,31-01-2021 01:30
        """
    )

    f = io.StringIO(csv)
    df = pl.read_csv(f, try_parse_dates=True)
    assert df.dtypes == [pl.Datetime, pl.Datetime, pl.Datetime]


def test_partial_dtype_overwrite() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c
        1,2,3
        1,2,3
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, dtypes=[pl.Utf8])
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Int64]


def test_dtype_overwrite_with_column_name_selection() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c,d
        1,2,3,4
        1,2,3,4
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, columns=["c", "b", "d"], dtypes=[pl.Int32, pl.Utf8])
    assert df.dtypes == [pl.Utf8, pl.Int32, pl.Int64]


def test_dtype_overwrite_with_column_idx_selection() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c,d
        1,2,3,4
        1,2,3,4
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, columns=[2, 1, 3], dtypes=[pl.Int32, pl.Utf8])
    # Columns without an explicit dtype set will get pl.Utf8 if dtypes is a list
    # if the column selection is done with column indices instead of column names.
    assert df.dtypes == [pl.Utf8, pl.Int32, pl.Utf8]
    # Projections are sorted.
    assert df.columns == ["b", "c", "d"]


def test_partial_column_rename() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c
        1,2,3
        1,2,3
        """
    )
    f = io.StringIO(csv)
    for use in [True, False]:
        f.seek(0)
        df = pl.read_csv(f, new_columns=["foo"], use_pyarrow=use)
        assert df.columns == ["foo", "b", "c"]


@pytest.mark.parametrize(
    ("col_input", "col_out"),
    [([0, 1], ["a", "b"]), ([0, 2], ["a", "c"]), (["b"], ["b"])],
)
def test_read_csv_columns_argument(
    col_input: list[int] | list[str], col_out: list[str]
) -> None:
    csv = textwrap.dedent(
        """\
        a,b,c
        1,2,3
        1,2,3
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, columns=col_input)
    assert df.shape[0] == 2
    assert df.columns == col_out


def test_read_csv_buffer_ownership() -> None:
    bts = b"\xf0\x9f\x98\x80,5.55,333\n\xf0\x9f\x98\x86,-5.0,666"
    buf = io.BytesIO(bts)
    df = pl.read_csv(
        buf,
        has_header=False,
        new_columns=["emoji", "flt", "int"],
    )
    # confirm that read_csv succeeded, and didn't close the input buffer (#2696)
    assert df.shape == (2, 3)
    assert df.rows() == [("😀", 5.55, 333), ("😆", -5.0, 666)]
    assert not buf.closed
    assert buf.read() == bts


@pytest.mark.write_disk()
def test_read_csv_encoding(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    bts = (
        b"Value1,Value2,Value3,Value4,Region\n"
        b"-30,7.5,2578,1,\xa5x\xa5_\n-32,7.97,3006,1,\xa5x\xa4\xa4\n"
        b"-31,8,3242,2,\xb7s\xa6\xcb\n-33,7.97,3300,3,\xb0\xaa\xb6\xaf\n"
        b"-20,7.91,3384,4,\xac\xfc\xb0\xea\n"
    )

    file_path = tmp_path / "encoding.csv"
    file_path.write_bytes(bts)

    file_str = str(file_path)
    bytesio = io.BytesIO(bts)

    for use_pyarrow in (False, True):
        bytesio.seek(0)
        for file in [file_path, file_str, bts, bytesio]:
            assert_series_equal(
                pl.read_csv(
                    file,  # type: ignore[arg-type]
                    encoding="big5",
                    use_pyarrow=use_pyarrow,
                ).get_column("Region"),
                pl.Series("Region", ["台北", "台中", "新竹", "高雄", "美國"]),
            )


def test_column_rename_and_dtype_overwrite() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c
        1,2,3
        1,2,3
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(
        f,
        new_columns=["A", "B", "C"],
        dtypes={"A": pl.Utf8, "B": pl.Int64, "C": pl.Float32},
    )
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Float32]

    f = io.StringIO(csv)
    df = pl.read_csv(
        f,
        columns=["a", "c"],
        new_columns=["A", "C"],
        dtypes={"A": pl.Utf8, "C": pl.Float32},
    )
    assert df.dtypes == [pl.Utf8, pl.Float32]

    csv = textwrap.dedent(
        """\
        1,2,3
        1,2,3
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(
        f,
        new_columns=["A", "B", "C"],
        dtypes={"A": pl.Utf8, "C": pl.Float32},
        has_header=False,
    )
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Float32]


def test_compressed_csv(io_files_path: Path) -> None:
    # gzip compression
    csv = textwrap.dedent(
        """\
        a,b,c
        1,a,1.0
        2,b,2.0
        3,c,3.0
        """
    )
    fout = io.BytesIO()
    with gzip.GzipFile(fileobj=fout, mode="w") as f:
        f.write(csv.encode())

    csv_bytes = fout.getvalue()
    out = pl.read_csv(csv_bytes)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1.0, 2.0, 3.0]}
    )
    assert_frame_equal(out, expected)

    # now from disk
    csv_file = io_files_path / "gzipped.csv.gz"
    out = pl.read_csv(str(csv_file), truncate_ragged_lines=True)
    assert_frame_equal(out, expected)

    # now with column projection
    out = pl.read_csv(csv_bytes, columns=["a", "b"])
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    assert_frame_equal(out, expected)

    # zlib compression
    csv_bytes = zlib.compress(csv.encode())
    out = pl.read_csv(csv_bytes)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1.0, 2.0, 3.0]}
    )
    assert_frame_equal(out, expected)

    # no compression
    f2 = io.BytesIO(b"a,b\n1,2\n")
    out2 = pl.read_csv(f2)
    expected = pl.DataFrame({"a": [1], "b": [2]})
    assert_frame_equal(out2, expected)


def test_partial_decompression(foods_file_path: Path) -> None:
    f_out = io.BytesIO()
    with gzip.GzipFile(fileobj=f_out, mode="w") as f:
        f.write(foods_file_path.read_bytes())

    csv_bytes = f_out.getvalue()
    for n_rows in [1, 5, 26]:
        out = pl.read_csv(csv_bytes, n_rows=n_rows)
        assert out.shape == (n_rows, 4)


def test_empty_bytes() -> None:
    b = b""
    with pytest.raises(NoDataError):
        pl.read_csv(b)

    df = pl.read_csv(b, raise_if_empty=False)
    assert_frame_equal(df, pl.DataFrame())


def test_csv_quote_char() -> None:
    expected = pl.DataFrame(
        [
            pl.Series("linenum", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            pl.Series(
                "last_name",
                [
                    "Jagger",
                    'O"Brian',
                    "Richards",
                    'L"Etoile',
                    "Watts",
                    "Smith",
                    '"Wyman"',
                    "Woods",
                    'J"o"ne"s',
                ],
            ),
            pl.Series(
                "first_name",
                [
                    "Mick",
                    '"Mary"',
                    "Keith",
                    "Bennet",
                    "Charlie",
                    'D"Shawn',
                    "Bill",
                    "Ron",
                    "Brian",
                ],
            ),
        ]
    )
    rolling_stones = textwrap.dedent(
        """\
        linenum,last_name,first_name
        1,Jagger,Mick
        2,O"Brian,"Mary"
        3,Richards,Keith
        4,L"Etoile,Bennet
        5,Watts,Charlie
        6,Smith,D"Shawn
        7,"Wyman",Bill
        8,Woods,Ron
        9,J"o"ne"s,Brian
        """
    )
    for use_pyarrow in (False, True):
        out = pl.read_csv(
            rolling_stones.encode(), quote_char=None, use_pyarrow=use_pyarrow
        )
        assert out.shape == (9, 3)
        assert_frame_equal(out, expected)

    # non-standard quote char
    df = pl.DataFrame({"x": ["", "0*0", "xyz"]})
    csv_data = df.write_csv(quote_char="*")

    assert csv_data == "x\n**\n*0**0*\nxyz\n"
    assert_frame_equal(df, pl.read_csv(io.StringIO(csv_data), quote_char="*"))


def test_csv_empty_quotes_char_1622() -> None:
    pl.read_csv(b"a,b,c,d\nA1,B1,C1,1\nA2,B2,C2,2\n", quote_char="")


def test_ignore_try_parse_dates() -> None:
    csv = textwrap.dedent(
        """\
        a,b,c
        1,i,16200126
        2,j,16250130
        3,k,17220012
        4,l,17290009
        """
    ).encode()

    headers = ["a", "b", "c"]
    dtypes: dict[str, type[pl.DataType]] = {
        k: pl.Utf8 for k in headers
    }  # Forces Utf8 type for every column
    df = pl.read_csv(csv, columns=headers, dtypes=dtypes)
    assert df.dtypes == [pl.Utf8, pl.Utf8, pl.Utf8]


def test_csv_date_handling() -> None:
    csv = textwrap.dedent(
        """\
        date
        1745-04-02
        1742-03-21
        1743-06-16
        1730-07-22
        ""
        1739-03-16
        """
    )
    expected = pl.DataFrame(
        {
            "date": [
                date(1745, 4, 2),
                date(1742, 3, 21),
                date(1743, 6, 16),
                date(1730, 7, 22),
                None,
                date(1739, 3, 16),
            ]
        }
    )
    out = pl.read_csv(csv.encode(), try_parse_dates=True)
    assert_frame_equal(out, expected)
    dtypes = {"date": pl.Date}
    out = pl.read_csv(csv.encode(), dtypes=dtypes)
    assert_frame_equal(out, expected)


def test_csv_globbing(io_files_path: Path) -> None:
    path = io_files_path / "foods*.csv"
    df = pl.read_csv(path)
    assert df.shape == (135, 4)

    with pytest.raises(ValueError):
        _ = pl.read_csv(path, columns=[0, 1])

    df = pl.read_csv(path, columns=["category", "sugars_g"])
    assert df.shape == (135, 2)
    assert df.row(-1) == ("seafood", 1)
    assert df.row(0) == ("vegetables", 2)

    with pytest.raises(ValueError):
        _ = pl.read_csv(path, dtypes=[pl.Utf8, pl.Int64, pl.Int64, pl.Int64])

    dtypes = {
        "category": pl.Utf8,
        "calories": pl.Int32,
        "fats_g": pl.Float32,
        "sugars_g": pl.Int32,
    }

    df = pl.read_csv(path, dtypes=dtypes)
    assert df.dtypes == list(dtypes.values())


def test_csv_schema_offset(foods_file_path: Path) -> None:
    csv = textwrap.dedent(
        """\
        metadata
        line
        col1,col2,col3
        alpha,beta,gamma
        1,2.0,"A"
        3,4.0,"B"
        5,6.0,"C"
        """
    ).encode()

    df = pl.read_csv(csv, skip_rows=3)
    assert df.columns == ["alpha", "beta", "gamma"]
    assert df.shape == (3, 3)
    assert df.dtypes == [pl.Int64, pl.Float64, pl.Utf8]

    df = pl.read_csv(csv, skip_rows=2, skip_rows_after_header=1)
    assert df.columns == ["col1", "col2", "col3"]
    assert df.shape == (3, 3)
    assert df.dtypes == [pl.Int64, pl.Float64, pl.Utf8]

    df = pl.scan_csv(foods_file_path, skip_rows=4).collect()
    assert df.columns == ["fruit", "60", "0", "11"]
    assert df.shape == (23, 4)
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Float64, pl.Int64]

    df = pl.scan_csv(foods_file_path, skip_rows_after_header=24).collect()
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (3, 4)
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Int64, pl.Int64]

    df = pl.scan_csv(
        foods_file_path, skip_rows_after_header=24, infer_schema_length=1
    ).collect()
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (3, 4)
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Int64, pl.Int64]


def test_empty_string_missing_round_trip() -> None:
    df = pl.DataFrame({"varA": ["A", "", None], "varB": ["B", "", None]})
    for null in (None, "NA", "NULL", r"\N"):
        f = io.BytesIO()
        df.write_csv(f, null_value=null)
        f.seek(0)
        df_read = pl.read_csv(f, null_values=null)
        assert_frame_equal(df, df_read)


def test_write_csv_separator() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    f = io.BytesIO()
    df.write_csv(f, separator="\t")
    f.seek(0)
    assert f.read() == b"a\tb\n1\t1\n2\t2\n3\t3\n"
    assert_frame_equal(df, pl.read_csv(f, separator="\t"))


def test_write_csv_line_terminator() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    f = io.BytesIO()
    df.write_csv(f, line_terminator="\r\n")
    f.seek(0)
    assert f.read() == b"a,b\r\n1,1\r\n2,2\r\n3,3\r\n"
    assert_frame_equal(df, pl.read_csv(f, eol_char="\n"))


def test_escaped_null_values() -> None:
    csv = textwrap.dedent(
        """\
        "a","b","c"
        "a","n/a","NA"
        "None","2","3.0"
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(
        f,
        null_values={"a": "None", "b": "n/a", "c": "NA"},
        dtypes={"a": pl.Utf8, "b": pl.Int64, "c": pl.Float64},
    )
    assert df[1, "a"] is None
    assert df[0, "b"] is None
    assert df[0, "c"] is None


def test_quoting_round_trip() -> None:
    f = io.BytesIO()
    df = pl.DataFrame(
        {
            "a": [
                "tab,separated,field",
                "newline\nseparated\nfield",
                'quote"separated"field',
            ]
        }
    )
    df.write_csv(f)
    read_df = pl.read_csv(f)
    assert_frame_equal(read_df, df)


def test_fallback_chrono_parser() -> None:
    data = textwrap.dedent(
        """\
    date_1,date_2
    2021-01-01,2021-1-1
    2021-02-02,2021-2-2
    2021-10-10,2021-10-10
    """
    )
    df = pl.read_csv(data.encode(), try_parse_dates=True)
    assert df.null_count().row(0) == (0, 0)


def test_tz_aware_try_parse_dates() -> None:
    data = (
        "a,b,c,d\n"
        "2020-01-01T02:00:00+01:00,2021-04-28T00:00:00+02:00,2021-03-28T00:00:00+01:00,2\n"
        "2020-01-01T03:00:00+01:00,2021-04-29T00:00:00+02:00,2021-03-29T00:00:00+02:00,3\n"
    )
    result = pl.read_csv(io.StringIO(data), try_parse_dates=True)
    expected = pl.DataFrame(
        {
            "a": [
                datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 1, 2, tzinfo=timezone.utc),
            ],
            "b": [
                datetime(2021, 4, 27, 22, tzinfo=timezone.utc),
                datetime(2021, 4, 28, 22, tzinfo=timezone.utc),
            ],
            "c": [
                datetime(2021, 3, 27, 23, tzinfo=timezone.utc),
                datetime(2021, 3, 28, 22, tzinfo=timezone.utc),
            ],
            "d": [2, 3],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("try_parse_dates", [True, False])
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_csv_overwrite_datetime_dtype(
    try_parse_dates: bool, time_unit: TimeUnit
) -> None:
    data = """\
    a
    2020-1-1T00:00:00.123456789
    2020-1-2T00:00:00.987654321
    2020-1-3T00:00:00.132547698
    """
    result = pl.read_csv(
        io.StringIO(data),
        try_parse_dates=try_parse_dates,
        dtypes={"a": pl.Datetime(time_unit)},
    )
    expected = pl.DataFrame(
        {
            "a": pl.Series(
                [
                    "2020-01-01T00:00:00.123456789",
                    "2020-01-02T00:00:00.987654321",
                    "2020-01-03T00:00:00.132547698",
                ]
            ).str.to_datetime(time_unit=time_unit)
        }
    )
    assert_frame_equal(result, expected)


def test_csv_string_escaping() -> None:
    df = pl.DataFrame({"a": ["Free trip to A,B", '''Special rate "1.79"''']})
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)
    df_read = pl.read_csv(f)
    assert_frame_equal(df_read, df)


@pytest.mark.write_disk()
def test_glob_csv(df_no_lists: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = df_no_lists.drop("strings_nulls")
    file_path = tmp_path / "small.csv"
    df.write_csv(file_path)

    path_glob = tmp_path / "small*.csv"
    assert pl.scan_csv(path_glob).collect().shape == (3, 11)
    assert pl.read_csv(path_glob).shape == (3, 11)


def test_csv_whitespace_separator_at_start_do_not_skip() -> None:
    csv = "\t\t\t\t0\t1"
    assert pl.read_csv(csv.encode(), separator="\t", has_header=False).to_dict(
        False
    ) == {
        "column_1": [None],
        "column_2": [None],
        "column_3": [None],
        "column_4": [None],
        "column_5": [0],
        "column_6": [1],
    }


def test_csv_whitespace_separator_at_end_do_not_skip() -> None:
    csv = "0\t1\t\t\t\t"
    assert pl.read_csv(csv.encode(), separator="\t", has_header=False).to_dict(
        False
    ) == {
        "column_1": [0],
        "column_2": [1],
        "column_3": [None],
        "column_4": [None],
        "column_5": [None],
        "column_6": [None],
    }


def test_csv_multiple_null_values() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": ["2022-01-01", "__NA__", "", "NA"],
        }
    )
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)

    df2 = pl.read_csv(f, null_values=["__NA__", "NA"])
    expected = pl.DataFrame(
        {
            "a": [1, 2, None, 4],
            "b": ["2022-01-01", None, "", None],
        }
    )
    assert_frame_equal(df2, expected)


def test_different_eol_char() -> None:
    csv = "a,1,10;b,2,20;c,3,30"
    expected = pl.DataFrame(
        {"column_1": ["a", "b", "c"], "column_2": [1, 2, 3], "column_3": [10, 20, 30]}
    )
    assert_frame_equal(
        pl.read_csv(csv.encode(), eol_char=";", has_header=False), expected
    )


def test_csv_write_escape_headers() -> None:
    df0 = pl.DataFrame({"col,1": ["data,1"], 'col"2': ['data"2'], "col:3": ["data:3"]})
    out = io.BytesIO()
    df0.write_csv(out)
    assert out.getvalue() == b'"col,1","col""2",col:3\n"data,1","data""2",data:3\n'

    df1 = pl.DataFrame({"c,o,l,u,m,n": [123]})
    out = io.BytesIO()
    df1.write_csv(out)

    df2 = pl.read_csv(out)
    assert_frame_equal(df1, df2)
    assert df2.schema == {"c,o,l,u,m,n": pl.Int64}


def test_csv_write_escape_newlines() -> None:
    df = pl.DataFrame({"escape": ["n\nn"]})
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)
    read_df = pl.read_csv(f)
    assert_frame_equal(df, read_df)


def test_skip_new_line_embedded_lines() -> None:
    csv = r"""a,b,c,d,e\n
        1,2,3,"\n Test",\n
        4,5,6,"Test A",\n
        7,8,,"Test B \n",\n"""

    for empty_string, missing_value in ((True, ""), (False, None)):
        df = pl.read_csv(
            csv.encode(),
            skip_rows_after_header=1,
            infer_schema_length=0,
            missing_utf8_is_empty_string=empty_string,
        )
        assert df.to_dict(False) == {
            "a": ["4", "7"],
            "b": ["5", "8"],
            "c": ["6", missing_value],
            "d": ["Test A", "Test B \\n"],
            "e\\n": ["\\n", "\\n"],
        }


def test_csv_dtype_overwrite_bool() -> None:
    csv = "a, b\n" + ",false\n" + ",false\n" + ",false"
    df = pl.read_csv(
        csv.encode(),
        dtypes={"a": pl.Boolean, "b": pl.Boolean},
    )
    assert df.dtypes == [pl.Boolean, pl.Boolean]


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (None, "dt\n2022-01-02T00:00:00.000000\n"),
        ("%F %T%.3f", "dt\n2022-01-02 00:00:00.000\n"),
        ("%Y", "dt\n2022\n"),
        ("%m", "dt\n01\n"),
        ("%m$%d", "dt\n01$02\n"),
        ("%R", "dt\n00:00\n"),
    ],
)
def test_datetime_format(fmt: str, expected: str) -> None:
    df = pl.DataFrame({"dt": [datetime(2022, 1, 2)]})
    csv = df.write_csv(datetime_format=fmt)
    assert csv == expected


def test_invalid_datetime_format() -> None:
    tz_naive = pl.Series(["2020-01-01T00:00:00"]).str.strptime(pl.Datetime)
    tz_aware = tz_naive.dt.replace_time_zone("UTC")
    with pytest.raises(
        ComputeError, match="cannot format NaiveDateTime with format '%q'"
    ):
        tz_naive.to_frame().write_csv(datetime_format="%q")
    with pytest.raises(ComputeError, match="cannot format DateTime with format '%q'"):
        tz_aware.to_frame().write_csv(datetime_format="%q")


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (None, "dt\n2022-01-02T00:00:00.000000+0000\n"),
        ("%F %T%.3f%z", "dt\n2022-01-02 00:00:00.000+0000\n"),
        ("%Y%z", "dt\n2022+0000\n"),
        ("%m%z", "dt\n01+0000\n"),
        ("%m$%d%z", "dt\n01$02+0000\n"),
        ("%R%z", "dt\n00:00+0000\n"),
    ],
)
@pytest.mark.parametrize("tzinfo", [timezone.utc, timezone(timedelta(hours=0))])
def test_datetime_format_tz_aware(fmt: str, expected: str, tzinfo: timezone) -> None:
    df = pl.DataFrame({"dt": [datetime(2022, 1, 2, tzinfo=tzinfo)]})
    csv = df.write_csv(datetime_format=fmt)
    assert csv == expected


@pytest.mark.parametrize(
    ("tu1", "tu2", "expected"),
    [
        (
            "ns",
            "ns",
            "x,y\n2022-09-04T10:30:45.123000000,2022-09-04T10:30:45.123000000\n",
        ),
        (
            "ns",
            "us",
            "x,y\n2022-09-04T10:30:45.123000000,2022-09-04T10:30:45.123000\n",
        ),
        (
            "ns",
            "ms",
            "x,y\n2022-09-04T10:30:45.123000000,2022-09-04T10:30:45.123\n",
        ),
        ("us", "us", "x,y\n2022-09-04T10:30:45.123000,2022-09-04T10:30:45.123000\n"),
        ("us", "ms", "x,y\n2022-09-04T10:30:45.123000,2022-09-04T10:30:45.123\n"),
        ("ms", "us", "x,y\n2022-09-04T10:30:45.123,2022-09-04T10:30:45.123000\n"),
        ("ms", "ms", "x,y\n2022-09-04T10:30:45.123,2022-09-04T10:30:45.123\n"),
    ],
)
def test_datetime_format_inferred_precision(
    tu1: TimeUnit, tu2: TimeUnit, expected: str
) -> None:
    df = pl.DataFrame(
        data={
            "x": [datetime(2022, 9, 4, 10, 30, 45, 123000)],
            "y": [datetime(2022, 9, 4, 10, 30, 45, 123000)],
        },
        schema=[
            ("x", pl.Datetime(tu1)),
            ("y", pl.Datetime(tu2)),
        ],
    )
    assert expected == df.write_csv()


def test_inferred_datetime_format_mixed() -> None:
    ts = pl.datetime_range(datetime(2000, 1, 1), datetime(2000, 1, 2), eager=True)
    df = pl.DataFrame({"naive": ts, "aware": ts.dt.replace_time_zone("UTC")})
    result = df.write_csv()
    expected = (
        "naive,aware\n"
        "2000-01-01T00:00:00.000000,2000-01-01T00:00:00.000000+0000\n"
        "2000-01-02T00:00:00.000000,2000-01-02T00:00:00.000000+0000\n"
    )
    assert result == expected


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (None, "dt\n2022-01-02\n"),
        ("%Y", "dt\n2022\n"),
        ("%m", "dt\n01\n"),
        ("%m$%d", "dt\n01$02\n"),
    ],
)
def test_date_format(fmt: str, expected: str) -> None:
    df = pl.DataFrame({"dt": [date(2022, 1, 2)]})
    csv = df.write_csv(date_format=fmt)
    assert csv == expected


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (None, "dt\n16:15:30.000000000\n"),
        ("%R", "dt\n16:15\n"),
    ],
)
def test_time_format(fmt: str, expected: str) -> None:
    df = pl.DataFrame({"dt": [time(16, 15, 30)]})
    csv = df.write_csv(time_format=fmt)
    assert csv == expected


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_float_precision(dtype: pl.Float32 | pl.Float64) -> None:
    df = pl.Series("col", [1.0, 2.2, 3.33], dtype=dtype).to_frame()

    assert df.write_csv(float_precision=None) == "col\n1.0\n2.2\n3.33\n"
    assert df.write_csv(float_precision=0) == "col\n1\n2\n3\n"
    assert df.write_csv(float_precision=1) == "col\n1.0\n2.2\n3.3\n"
    assert df.write_csv(float_precision=2) == "col\n1.00\n2.20\n3.33\n"
    assert df.write_csv(float_precision=3) == "col\n1.000\n2.200\n3.330\n"


def test_skip_rows_different_field_len() -> None:
    csv = io.StringIO(
        textwrap.dedent(
            """a,b
        1,A
        2,
        3,B
        4,
        """
        )
    )
    for empty_string, missing_value in ((True, ""), (False, None)):
        csv.seek(0)
        assert pl.read_csv(
            csv, skip_rows_after_header=2, missing_utf8_is_empty_string=empty_string
        ).to_dict(False) == {
            "a": [3, 4],
            "b": ["B", missing_value],
        }


def test_duplicated_columns() -> None:
    csv = textwrap.dedent(
        """a,a
    1,2
    """
    )
    assert pl.read_csv(csv.encode()).columns == ["a", "a_duplicated_0"]
    new = ["c", "d"]
    assert pl.read_csv(csv.encode(), new_columns=new).columns == new


def test_error_message() -> None:
    data = io.StringIO("target,wind,energy,miso\n" "1,2,3,4\n" "1,2,1e5,1\n")
    with pytest.raises(
        ComputeError,
        match=r"Could not parse `1e5` as dtype `i64` at column 'energy' \(column number 3\)",
    ):
        pl.read_csv(data, infer_schema_length=1)


def test_csv_categorical_lifetime() -> None:
    # escaped strings do some heap allocates in the builder
    # this tests of the lifetimes remains valid
    csv = textwrap.dedent(
        r"""
    a,b
    "needs_escape",b
    "" ""needs" escape" foo"",b
    "" ""needs" escape" foo"",
    """
    )

    df = pl.read_csv(csv.encode(), dtypes={"a": pl.Categorical, "b": pl.Categorical})
    assert df.dtypes == [pl.Categorical, pl.Categorical]
    assert df.to_dict(False) == {
        "a": ["needs_escape", ' "needs escape foo', ' "needs escape foo'],
        "b": ["b", "b", None],
    }

    assert (df["a"] == df["b"]).to_list() == [False, False, None]


def test_csv_categorical_categorical_merge() -> None:
    N = 50
    f = io.BytesIO()
    pl.DataFrame({"x": ["A"] * N + ["B"] * N}).write_csv(f)
    f.seek(0)
    assert pl.read_csv(f, dtypes={"x": pl.Categorical}, sample_size=10).unique(
        maintain_order=True
    )["x"].to_list() == ["A", "B"]


def test_batched_csv_reader(foods_file_path: Path) -> None:
    reader = pl.read_csv_batched(foods_file_path, batch_size=4)
    batches = reader.next_batches(5)

    assert batches is not None
    assert len(batches) == 5
    assert batches[0].to_dict(False) == {
        "category": ["vegetables", "seafood", "meat", "fruit", "seafood", "meat"],
        "calories": [45, 150, 100, 60, 140, 120],
        "fats_g": [0.5, 5.0, 5.0, 0.0, 5.0, 10.0],
        "sugars_g": [2, 0, 0, 11, 1, 1],
    }
    assert batches[-1].to_dict(False) == {
        "category": ["fruit", "meat", "vegetables", "fruit"],
        "calories": [130, 100, 30, 50],
        "fats_g": [0.0, 7.0, 0.0, 0.0],
        "sugars_g": [25, 0, 5, 11],
    }
    assert_frame_equal(pl.concat(batches), pl.read_csv(foods_file_path))
    # the final batch of the low-memory variant is different
    reader = pl.read_csv_batched(foods_file_path, batch_size=4, low_memory=True)
    batches = reader.next_batches(5)
    assert len(batches) == 5  # type: ignore[arg-type]
    batches += reader.next_batches(5)  # type: ignore[operator]
    assert_frame_equal(pl.concat(batches), pl.read_csv(foods_file_path))

    reader = pl.read_csv_batched(foods_file_path, batch_size=4, low_memory=True)
    batches = reader.next_batches(10)
    assert_frame_equal(pl.concat(batches), pl.read_csv(foods_file_path))  # type: ignore[arg-type]


def test_batched_csv_reader_empty(io_files_path: Path) -> None:
    empty_csv = io_files_path / "empty.csv"
    with pytest.raises(NoDataError, match="empty CSV"):
        pl.read_csv_batched(source=empty_csv)

    reader = pl.read_csv_batched(source=empty_csv, raise_if_empty=False)
    assert reader.next_batches(1) is None


def test_batched_csv_reader_all_batches(foods_file_path: Path) -> None:
    for new_columns in [None, ["Category", "Calories", "Fats_g", "Sugars_g"]]:
        out = pl.read_csv(foods_file_path, new_columns=new_columns)
        reader = pl.read_csv_batched(
            foods_file_path, new_columns=new_columns, batch_size=4
        )
        batches = reader.next_batches(5)
        batched_dfs = []

        while batches:
            batched_dfs.extend(batches)
            batches = reader.next_batches(5)

        batched_concat_df = pl.concat(batched_dfs, rechunk=True)
        assert_frame_equal(out, batched_concat_df)


def test_batched_csv_reader_no_batches(foods_file_path: Path) -> None:
    reader = pl.read_csv_batched(foods_file_path, batch_size=4)
    batches = reader.next_batches(0)

    assert batches is None


def test_csv_single_categorical_null() -> None:
    f = io.BytesIO()
    pl.DataFrame(
        {
            "x": ["A"],
            "y": [None],
            "z": ["A"],
        }
    ).write_csv(f)
    f.seek(0)

    df = pl.read_csv(
        f,
        dtypes={"y": pl.Categorical},
    )

    assert df.dtypes == [pl.Utf8, pl.Categorical, pl.Utf8]
    assert df.to_dict(False) == {"x": ["A"], "y": [None], "z": ["A"]}


def test_csv_quoted_missing() -> None:
    csv = (
        '"col1"|"col2"|"col3"|"col4"\n'
        '"0"|"Free text with a line\nbreak"|"123"|"456"\n'
        '"1"|"Free text without a linebreak"|""|"789"\n'
        '"0"|"Free text with \ntwo \nlinebreaks"|"101112"|"131415"'
    )
    result = pl.read_csv(csv.encode(), separator="|", dtypes={"col3": pl.Int32})
    expected = pl.DataFrame(
        {
            "col1": [0, 1, 0],
            "col2": [
                "Free text with a line\nbreak",
                "Free text without a linebreak",
                "Free text with \ntwo \nlinebreaks",
            ],
            "col3": [123, None, 101112],
            "col4": [456, 789, 131415],
        },
        schema_overrides={"col3": pl.Int32},
    )
    assert_frame_equal(result, expected)


def test_csv_write_tz_aware() -> None:
    df = pl.DataFrame({"times": datetime(2021, 1, 1)}).with_columns(
        pl.col("times")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/Zurich")
    )
    assert df.write_csv() == "times\n2021-01-01T01:00:00.000000+0100\n"


def test_csv_statistics_offset() -> None:
    # this would fail if the statistics sample did not also sample
    # from the end of the file
    # the lines at the end have larger rows as the numbers increase
    N = 5_000
    csv = "\n".join(str(x) for x in range(N))
    assert pl.read_csv(io.StringIO(csv), n_rows=N).height == 4999


@pytest.mark.write_disk()
def test_csv_scan_categorical(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    N = 5_000
    df = pl.DataFrame({"x": ["A"] * N})

    file_path = tmp_path / "test_csv_scan_categorical.csv"
    df.write_csv(file_path)
    result = pl.scan_csv(file_path, dtypes={"x": pl.Categorical}).collect()

    assert result["x"].dtype == pl.Categorical


@pytest.mark.write_disk()
def test_csv_scan_new_columns_less_than_original_columns(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"x": ["A"], "y": ["A"], "z": "A"})

    file_path = tmp_path / "test_csv_scan_new_columns.csv"
    df.write_csv(file_path)
    result = pl.scan_csv(file_path, new_columns=["x_new", "y_new"]).collect()

    assert result.columns == ["x_new", "y_new", "z"]


def test_read_csv_chunked() -> None:
    """Check that row count is properly functioning."""
    N = 10_000
    csv = "1\n" * N
    df = pl.read_csv(io.StringIO(csv), row_count_name="count")

    # The next value should always be higher if monotonically increasing.
    assert df.filter(pl.col("count") < pl.col("count").shift(1)).is_empty()


def test_read_empty_csv(io_files_path: Path) -> None:
    with pytest.raises(NoDataError) as err:
        pl.read_csv(io_files_path / "empty.csv")
    assert "empty CSV" in str(err.value)

    df = pl.read_csv(io_files_path / "empty.csv", raise_if_empty=False)
    assert_frame_equal(df, pl.DataFrame())

    with pytest.raises(pa.ArrowInvalid) as err:
        pl.read_csv(io_files_path / "empty.csv", use_pyarrow=True)
    assert "Empty CSV" in str(err.value)

    df = pl.read_csv(
        io_files_path / "empty.csv", raise_if_empty=False, use_pyarrow=True
    )
    assert_frame_equal(df, pl.DataFrame())


@pytest.mark.slow()
def test_read_web_file() -> None:
    url = "https://raw.githubusercontent.com/pola-rs/polars/main/examples/datasets/foods1.csv"
    df = pl.read_csv(url)
    assert df.shape == (27, 4)


@pytest.mark.slow()
def test_csv_multiline_splits() -> None:
    # create a very unlikely csv file with many multilines in a
    # single field (e.g. 5000). polars must reject multi-threading here
    # as it cannot find proper file chunks without sequentially parsing.

    np.random.seed(0)
    f = io.BytesIO()

    def some_multiline_str(n: int) -> str:
        strs = []
        strs.append('"')
        # sample between 0 and 5 so that it is likely
        # the multiline field also go 3 separators.
        for length in np.random.randint(0, 5, n):
            strs.append(f"{'xx,' * length}")

        strs.append('"')
        return "\n".join(strs)

    for _ in range(4):
        f.write(f"field1,field2,{some_multiline_str(5000)}\n".encode())

    f.seek(0)
    assert pl.read_csv(f, has_header=False).shape == (4, 3)


def test_read_csv_n_rows_outside_heuristic() -> None:
    # create a fringe case csv file that breaks the heuristic determining how much of
    # the file to read, and ensure n_rows is still adhered to

    f = io.StringIO()

    f.write(",,,?????????\n" * 1000)
    f.write("?????????????????????????????????????????????????,,,\n")
    f.write(",,,?????????\n" * 1048)

    f.seek(0)
    assert pl.read_csv(f, n_rows=2048, has_header=False).shape == (2048, 4)


def test_write_csv_stdout_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    # The capsys fixture allows pytest to access stdout/stderr. See
    # https://docs.pytest.org/en/7.1.x/how-to/capture-stdout-stderr.html
    df = pl.DataFrame(
        {
            "numbers": [1, 2, 3],
            "strings": ["test", "csv", "stdout"],
            "dates": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        }
    )

    # pytest hijacks sys.stdout and changes its type, which causes mypy failure
    df.write_csv(sys.stdout)  # type: ignore[call-overload]
    captured = capsys.readouterr()
    assert captured.out == (
        "numbers,strings,dates\n"
        "1,test,2023-01-01\n"
        "2,csv,2023-01-02\n"
        "3,stdout,2023-01-03\n"
    )

    df.write_csv(sys.stderr)  # type: ignore[call-overload]
    captured = capsys.readouterr()
    assert captured.err == (
        "numbers,strings,dates\n"
        "1,test,2023-01-01\n"
        "2,csv,2023-01-02\n"
        "3,stdout,2023-01-03\n"
    )


def test_csv_9929() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3]})
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)
    with pytest.raises(pl.NoDataError):
        pl.read_csv(f, skip_rows=10**6)


def test_csv_quote_styles() -> None:
    class TemporalFormats(TypedDict):
        datetime_format: str
        time_format: str

    temporal_formats: TemporalFormats = {
        "datetime_format": "%Y-%m-%dT%H:%M:%S",
        "time_format": "%H:%M:%S",
    }

    dtm = datetime(2077, 7, 5, 3, 1, 0)
    dt = dtm.date()
    tm = dtm.time()

    df = pl.DataFrame(
        {
            "float": [1.0, 2.0, None],
            "string": ["a", "a,bc", '"hello'],
            "int": [1, 2, 3],
            "bool": [True, False, None],
            "date": [dt, None, dt],
            "datetime": [None, dtm, dtm],
            "time": [tm, tm, None],
        }
    )

    assert df.write_csv(quote_style="always", **temporal_formats) == (
        '"float","string","int","bool","date","datetime","time"\n'
        '"1.0","a","1","true","2077-07-05","","03:01:00"\n'
        '"2.0","a,bc","2","false","","2077-07-05T03:01:00","03:01:00"\n'
        '"","""hello","3","","2077-07-05","2077-07-05T03:01:00",""\n'
    )
    assert df.write_csv(quote_style="necessary", **temporal_formats) == (
        "float,string,int,bool,date,datetime,time\n"
        "1.0,a,1,true,2077-07-05,,03:01:00\n"
        '2.0,"a,bc",2,false,,2077-07-05T03:01:00,03:01:00\n'
        ',"""hello",3,,2077-07-05,2077-07-05T03:01:00,\n'
    )
    assert df.write_csv(quote_style="never", **temporal_formats) == (
        "float,string,int,bool,date,datetime,time\n"
        "1.0,a,1,true,2077-07-05,,03:01:00\n"
        "2.0,a,bc,2,false,,2077-07-05T03:01:00,03:01:00\n"
        ',"hello,3,,2077-07-05,2077-07-05T03:01:00,\n'
    )
    assert df.write_csv(
        quote_style="non_numeric", quote_char="8", **temporal_formats
    ) == (
        "8float8,8string8,8int8,8bool8,8date8,8datetime8,8time8\n"
        "1.0,8a8,1,8true8,82077-07-058,,803:01:008\n"
        "2.0,8a,bc8,2,8false8,,82077-07-05T03:01:008,803:01:008\n"
        ',8"hello8,3,,82077-07-058,82077-07-05T03:01:008,\n'
    )


def test_ignore_errors_casting_dtypes() -> None:
    csv = """inventory
    10

    400
    90
    """

    assert pl.read_csv(
        source=io.StringIO(csv),
        dtypes={"inventory": pl.Int8},
        ignore_errors=True,
    ).to_dict(False) == {"inventory": [10, None, None, 90]}

    with pytest.raises(pl.ComputeError):
        pl.read_csv(
            source=io.StringIO(csv),
            dtypes={"inventory": pl.Int8},
            ignore_errors=False,
        )


def test_ignore_errors_date_parser() -> None:
    data_invalid_date = "int,float,date\n3,3.4,X"
    with pytest.raises(pl.ComputeError):
        pl.read_csv(
            source=io.StringIO(data_invalid_date),
            dtypes={"date": pl.Date},
            ignore_errors=False,
        )


def test_csv_ragged_lines() -> None:
    expected = {"column_1": ["A", "B", "C"]}
    assert (
        pl.read_csv(
            io.StringIO("A\nB,ragged\nC"), has_header=False, truncate_ragged_lines=True
        ).to_dict(False)
        == expected
    )
    assert (
        pl.read_csv(
            io.StringIO("A\nB\nC,ragged"), has_header=False, truncate_ragged_lines=True
        ).to_dict(False)
        == expected
    )

    for s in ["A\nB,ragged\nC", "A\nB\nC,ragged"]:
        with pytest.raises(pl.ComputeError, match=r"found more fields than defined"):
            pl.read_csv(io.StringIO(s), has_header=False, truncate_ragged_lines=False)
        with pytest.raises(pl.ComputeError, match=r"found more fields than defined"):
            pl.read_csv(io.StringIO(s), has_header=False, truncate_ragged_lines=False)


def test_provide_schema() -> None:
    # can be used to overload schema with ragged csv files
    assert pl.read_csv(
        io.StringIO("A\nB,ragged\nC"),
        has_header=False,
        schema={"A": pl.Utf8, "B": pl.Utf8, "C": pl.Utf8},
    ).to_dict(False) == {
        "A": ["A", "B", "C"],
        "B": [None, "ragged", None],
        "C": [None, None, None],
    }
