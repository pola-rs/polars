from __future__ import annotations

import gzip
import io
import os
import textwrap
import zlib
from datetime import date, datetime, time
from pathlib import Path

import pytest

import polars as pl
from polars import DataType
from polars.internals.type_aliases import TimeUnit
from polars.testing import assert_frame_equal_local_categoricals


def test_quoted_date() -> None:
    csv = textwrap.dedent(
        """a,b
    "2022-01-01",1
    "2022-01-02",2
    """
    )

    expected = pl.DataFrame({"a": [date(2022, 1, 1), date(2022, 1, 2)], "b": [1, 2]})

    assert pl.read_csv(csv.encode(), parse_dates=True).frame_equal(expected)


def test_to_from_buffer(df_no_lists: pl.DataFrame) -> None:
    df = df_no_lists
    buf = io.BytesIO()
    df.write_csv(buf)
    buf.seek(0)

    read_df = pl.read_csv(buf, parse_dates=True)

    read_df = read_df.with_columns(
        [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    )
    assert_frame_equal_local_categoricals(df, read_df)


def test_to_from_file(io_test_dir: str, df_no_lists: pl.DataFrame) -> None:
    df = df_no_lists
    df = df.drop("strings_nulls")

    f = os.path.join(io_test_dir, "small.csv")
    df.write_csv(f)

    read_df = pl.read_csv(f, parse_dates=True)

    read_df = read_df.with_columns(
        [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    )
    assert_frame_equal_local_categoricals(df, read_df)


def test_read_web_file() -> None:
    url = "https://raw.githubusercontent.com/pola-rs/polars/master/examples/datasets/foods1.csv"  # noqa: E501
    df = pl.read_csv(url)
    assert df.shape == (27, 4)


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
        """
    )
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values={"a": "na", "b": r"\N"})
    assert df.rows() == [(None, "b", "c"), ("a", None, "c")]


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
    df = pl.read_csv(f, parse_dates=True)
    assert df.dtypes == [pl.Datetime, pl.Float64, pl.Float64]


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
    "col_input, col_out", [([0, 1], ["a", "b"]), ([0, 2], ["a", "c"]), (["b"], ["b"])]
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


def test_read_csv_encoding() -> None:
    bts = (
        b"Value1,Value2,Value3,Value4,Region\n"
        b"-30,7.5,2578,1,\xa5x\xa5_\n-32,7.97,3006,1,\xa5x\xa4\xa4\n"
        b"-31,8,3242,2,\xb7s\xa6\xcb\n-33,7.97,3300,3,\xb0\xaa\xb6\xaf\n"
        b"-20,7.91,3384,4,\xac\xfc\xb0\xea\n"
    )

    file_path = os.path.join(os.path.dirname(__file__), "encoding.csv")
    file_str = str(file_path)

    with open(file_path, "wb") as f:
        f.write(bts)

    bytesio = io.BytesIO(bts)

    for use_pyarrow in (False, True):
        for file in (file_path, file_str, bts, bytesio):
            print(type(file))
            assert pl.read_csv(
                file,  # type: ignore[arg-type]
                encoding="big5",
                use_pyarrow=use_pyarrow,
            ).get_column("Region") == pl.Series(
                "Region", ["台北", "台中", "新竹", "高雄", "美國"]
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


def test_compressed_csv() -> None:
    # gzip compression
    csv = textwrap.dedent(
        """\
        a,b,c
        1,a,1.0
        2,b,2.0,
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
    assert out.frame_equal(expected)

    # now from disk
    csv_file = Path(__file__).parent.parent / "files" / "gzipped.csv"
    out = pl.read_csv(str(csv_file))
    assert out.frame_equal(expected)

    # now with column projection
    out = pl.read_csv(csv_bytes, columns=["a", "b"])
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    assert out.frame_equal(expected)

    # zlib compression
    csv_bytes = zlib.compress(csv.encode())
    out = pl.read_csv(csv_bytes)
    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1.0, 2.0, 3.0]}
    )
    assert out.frame_equal(expected)

    # no compression
    f2 = io.BytesIO(b"a,b\n1,2\n")
    out2 = pl.read_csv(f2)
    expected = pl.DataFrame({"a": [1], "b": [2]})
    assert out2.frame_equal(expected)


def test_partial_decompression(foods_csv: str) -> None:
    fout = io.BytesIO()
    with open(foods_csv, "rb") as fread:
        with gzip.GzipFile(fileobj=fout, mode="w") as f:
            f.write(fread.read())

    csv_bytes = fout.getvalue()
    for n_rows in [1, 5, 26]:
        out = pl.read_csv(csv_bytes, n_rows=n_rows)
        assert out.shape == (n_rows, 4)


def test_empty_bytes() -> None:
    b = b""
    with pytest.raises(ValueError):
        pl.read_csv(b)


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
        out.frame_equal(expected)


def test_csv_empty_quotes_char() -> None:
    # panicked in: https://github.com/pola-rs/polars/issues/1622
    pl.read_csv(b"a,b,c,d\nA1,B1,C1,1\nA2,B2,C2,2\n", quote_char="")


def test_ignore_parse_dates() -> None:
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
    dtypes: dict[str, type[DataType]] = {
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
    out = pl.read_csv(csv.encode(), parse_dates=True)
    assert out.frame_equal(expected, null_equal=True)
    dtypes = {"date": pl.Date}
    out = pl.read_csv(csv.encode(), dtypes=dtypes)
    assert out.frame_equal(expected, null_equal=True)


def test_csv_globbing(examples_dir: str) -> None:
    path = os.path.abspath(
        os.path.join(
            examples_dir,
            "*.csv",
        )
    )
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


def test_csv_schema_offset(foods_csv: str) -> None:
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

    df = pl.scan_csv(foods_csv, skip_rows=4).collect()
    assert df.columns == ["fruit", "60", "0", "11"]
    assert df.shape == (23, 4)
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Float64, pl.Int64]

    df = pl.scan_csv(foods_csv, skip_rows_after_header=24).collect()
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
        assert df.frame_equal(df_read)


def test_write_csv_delimiter() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    f = io.BytesIO()
    df.write_csv(f, sep="\t")
    f.seek(0)
    assert f.read() == b"a\tb\n1\t1\n2\t2\n3\t3\n"


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
    f.seek(0)
    read_df = pl.read_csv(f)

    assert read_df.frame_equal(df)


def test_fallback_chrono_parser() -> None:
    data = textwrap.dedent(
        """\
    date_1,date_2
    2021-01-01,2021-1-1
    2021-02-02,2021-2-2
    2021-10-10,2021-10-10
    """
    )
    df = pl.read_csv(data.encode(), parse_dates=True)
    assert df.null_count().row(0) == (0, 0)


def test_csv_string_escaping() -> None:
    df = pl.DataFrame({"a": ["Free trip to A,B", '''Special rate "1.79"''']})
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)
    df_read = pl.read_csv(f)
    assert df_read.frame_equal(df)


def test_glob_csv(io_test_dir: str) -> None:
    path = os.path.join(io_test_dir, "small*.csv")
    assert pl.scan_csv(path).collect().shape == (3, 11)
    assert pl.read_csv(path).shape == (3, 11)


def test_csv_whitepsace_delimiter_at_start_do_not_skip() -> None:
    csv = "\t\t\t\t0\t1"
    assert pl.read_csv(csv.encode(), sep="\t", has_header=False).to_dict(False) == {
        "column_1": [None],
        "column_2": [None],
        "column_3": [None],
        "column_4": [None],
        "column_5": [0],
        "column_6": [1],
    }


def test_csv_whitepsace_delimiter_at_end_do_not_skip() -> None:
    csv = "0\t1\t\t\t\t"
    assert pl.read_csv(csv.encode(), sep="\t", has_header=False).to_dict(False) == {
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

    assert df2.frame_equal(expected)


def test_different_eol_char() -> None:
    csv = "a,1,10;b,2,20;c,3,30"
    expected = pl.DataFrame(
        {"column_1": ["a", "b", "c"], "column_2": [1, 2, 3], "column_3": [10, 20, 30]}
    )
    assert pl.read_csv(csv.encode(), eol_char=";", has_header=False).frame_equal(
        expected
    )


def test_csv_write_escape_newlines() -> None:
    df = pl.DataFrame({"escape": ["n\nn"]})
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)
    read_df = pl.read_csv(f)
    assert df.frame_equal(read_df)


def test_skip_new_line_embedded_lines() -> None:
    csv = r"""a,b,c,d,e\n
        1,2,3,"\n Test",\n
        4,5,6,"Test A",\n
        7,8,9,"Test B \n",\n"""

    df = pl.read_csv(csv.encode(), skip_rows_after_header=1, infer_schema_length=0)
    assert df.to_dict(False) == {
        "a": ["4", "7"],
        "b": ["5", "8"],
        "c": ["6", "9"],
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
    "fmt,expected",
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


@pytest.mark.parametrize(
    "tu1,tu2,expected",
    [
        (
            "ns",
            "ns",
            "x,y\n2022-09-04T10:30:45.123000000,2022-09-04T10:30:45.123000000\n",
        ),
        (
            "ns",
            "us",
            "x,y\n2022-09-04T10:30:45.123000000,2022-09-04T10:30:45.123000000\n",
        ),
        (
            "ns",
            "ms",
            "x,y\n2022-09-04T10:30:45.123000000,2022-09-04T10:30:45.123000000\n",
        ),
        ("us", "us", "x,y\n2022-09-04T10:30:45.123000,2022-09-04T10:30:45.123000\n"),
        ("us", "ms", "x,y\n2022-09-04T10:30:45.123000,2022-09-04T10:30:45.123000\n"),
        ("ms", "us", "x,y\n2022-09-04T10:30:45.123000,2022-09-04T10:30:45.123000\n"),
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
        columns=[
            ("x", pl.Datetime(tu1)),
            ("y", pl.Datetime(tu2)),
        ],
    )
    assert expected == df.write_csv()


@pytest.mark.parametrize(
    "fmt,expected",
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
    "fmt,expected",
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
    assert pl.read_csv(csv, skip_rows_after_header=2).to_dict(False) == {
        "a": [3, 4],
        "b": ["B", None],
    }
