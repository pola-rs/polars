# flake8: noqa: W191,E101
import gzip
import io
import os
import zlib
from datetime import date
from pathlib import Path
from typing import Dict, List, Type, Union

import pytest

import polars as pl
from polars import DataType


def test_to_from_buffer(df: pl.DataFrame) -> None:
    buf = io.BytesIO()
    df.write_csv(buf)
    buf.seek(0)

    read_df = pl.read_csv(buf, parse_dates=True)

    read_df = read_df.with_columns(
        [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    )
    assert df.frame_equal(read_df)


def test_to_from_file(io_test_dir: str, df: pl.DataFrame) -> None:
    df = df.drop("strings_nulls")

    f = os.path.join(io_test_dir, "small.csv")
    df.write_csv(f)

    read_df = pl.read_csv(f, parse_dates=True)

    read_df = read_df.with_columns(
        [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    )
    assert df.frame_equal(read_df)


def test_read_web_file() -> None:
    url = "https://raw.githubusercontent.com/pola-rs/polars/master/examples/datasets/foods1.csv"
    df = pl.read_csv(url)
    assert df.shape == (27, 4)


def test_csv_null_values() -> None:
    csv = """
a,b,c
na,b,c
a,na,c"""
    f = io.StringIO(csv)

    df = pl.read_csv(f, null_values="na")
    assert df[0, "a"] is None
    assert df[1, "b"] is None

    csv = """
a,b,c
na,b,c
a,n/a,c"""
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values=["na", "n/a"])
    assert df[0, "a"] is None
    assert df[1, "b"] is None

    csv = """
a,b,c
na,b,c
a,n/a,c"""
    f = io.StringIO(csv)
    df = pl.read_csv(f, null_values={"a": "na", "b": "n/a"})
    assert df[0, "a"] is None
    assert df[1, "b"] is None


def test_datetime_parsing() -> None:
    csv = """
timestamp,open,high
2021-01-01 00:00:00,0.00305500,0.00306000
2021-01-01 00:15:00,0.00298800,0.00300400
2021-01-01 00:30:00,0.00298300,0.00300100
2021-01-01 00:45:00,0.00299400,0.00304000
"""

    f = io.StringIO(csv)
    df = pl.read_csv(f, parse_dates=True)
    assert df.dtypes == [pl.Datetime, pl.Float64, pl.Float64]


def test_partial_dtype_overwrite() -> None:
    csv = """
a,b,c
1,2,3
1,2,3
"""
    f = io.StringIO(csv)
    df = pl.read_csv(f, dtypes=[pl.Utf8])
    assert df.dtypes == [pl.Utf8, pl.Int64, pl.Int64]


def test_partial_column_rename() -> None:
    csv = """
a,b,c
1,2,3
1,2,3
"""
    f = io.StringIO(csv)
    for use in [True, False]:
        f.seek(0)
        df = pl.read_csv(f, new_columns=["foo"], use_pyarrow=use)
        assert df.columns == ["foo", "b", "c"]


@pytest.mark.parametrize(
    "col_input, col_out", [([0, 1], ["a", "b"]), ([0, 2], ["a", "c"]), (["b"], ["b"])]
)
def test_read_csv_columns_argument(
    col_input: Union[List[int], List[str]], col_out: List[str]
) -> None:
    csv = """a,b,c
    1,2,3
    1,2,3
    """
    f = io.StringIO(csv)
    df = pl.read_csv(f, columns=col_input)
    assert df.shape[0] == 2
    assert df.columns == col_out


def test_read_csv_buffer_ownership() -> None:
    buf = io.BytesIO(b"\xf0\x9f\x98\x80,5.55,333\n\xf0\x9f\x98\x86,-5.0,666")
    df = pl.read_csv(
        buf,
        has_header=False,
        new_columns=["emoji", "flt", "int"],
    )
    # confirm that read_csv succeeded, and didn't close the input buffer (#2696)
    assert df.shape == (2, 3)
    assert not buf.closed


def test_column_rename_and_dtype_overwrite() -> None:
    csv = """
a,b,c
1,2,3
1,2,3
"""
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

    csv = """
1,2,3
1,2,3
"""
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
    csv = """
a,b,c
1,a,1.0
2,b,2.0,
3,c,3.0
"""
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


def test_empty_bytes() -> None:
    b = b""
    with pytest.raises(ValueError):
        pl.read_csv(b)


def test_csq_quote_char() -> None:
    rolling_stones = """
    linenum,last_name,first_name
    1,Jagger,Mick
    2,O"Brian,Mary
    3,Richards,Keith
    4,L"Etoile,Bennet
    5,Watts,Charlie
    6,Smith,D"Shawn
    7,Wyman,Bill
    8,Woods,Ron
    9,Jones,Brian
    """

    assert pl.read_csv(rolling_stones.encode(), quote_char=None).shape == (9, 3)


def test_csv_empty_quotes_char() -> None:
    # panicked in: https://github.com/pola-rs/polars/issues/1622
    pl.read_csv(b"a,b,c,d\nA1,B1,C1,1\nA2,B2,C2,2\n", quote_char="")


def test_ignore_parse_dates() -> None:
    csv = """a,b,c
1,i,16200126
2,j,16250130
3,k,17220012
4,l,17290009""".encode()

    headers = ["a", "b", "c"]
    dtypes: Dict[str, Type[DataType]] = {
        k: pl.Utf8 for k in headers
    }  # Forces Utf8 type for every column
    df = pl.read_csv(csv, columns=headers, dtypes=dtypes)
    assert df.dtypes == [pl.Utf8, pl.Utf8, pl.Utf8]


def test_csv_date_handling() -> None:
    csv = """date
1745-04-02
1742-03-21
1743-06-16
1730-07-22
""
1739-03-16
"""
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
    csv = """metadata
line
foo,bar
1,2
3,4
5,6
""".encode()
    df = pl.read_csv(csv, skip_rows=2)
    assert df.columns == ["foo", "bar"]
    assert df.shape == (3, 2)
    df = pl.read_csv(csv, skip_rows=2, skip_rows_after_header=2)
    assert df.columns == ["foo", "bar"]
    assert df.shape == (1, 2)

    df = pl.scan_csv(foods_csv, skip_rows=4).collect()
    assert df.columns == ["fruit", "60", "0", "11"]
    assert df.shape == (23, 4)

    df = pl.scan_csv(foods_csv, skip_rows_after_header=10).collect()
    assert df.columns == ["category", "calories", "fats_g", "sugars_g"]
    assert df.shape == (17, 4)


def test_empty_string_missing_round_trip() -> None:
    df = pl.DataFrame({"varA": ["A", "", None], "varB": ["B", "", None]})
    f = io.BytesIO()
    df.write_csv(f)
    f.seek(0)
    df_read = pl.read_csv(f)
    assert df.frame_equal(df_read)


def test_write_csv_delimiter() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    f = io.BytesIO()
    df.write_csv(f, sep="\t")
    f.seek(0)
    assert f.read() == b"a\tb\n1\t1\n2\t2\n3\t3\n"


def test_escaped_null_values() -> None:
    csv = """
"a","b","c"
"a","n/a","NA"
"None","2","3.0"
    """
    f = io.StringIO(csv)
    df = pl.read_csv(
        f,
        null_values={"a": "None", "b": "n/a", "c": "NA"},
        dtypes={"a": pl.Utf8, "b": pl.Int64, "c": pl.Float64},
    )
    assert df[1, "a"] is None
    assert df[0, "b"] is None
    assert df[0, "c"] is None


def quoting_round_trip() -> None:
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
