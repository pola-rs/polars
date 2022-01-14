# flake8: noqa: W191,E101
import copy
import gzip
import io
import os
import pickle
import zlib
from datetime import date
from functools import partial
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars import DataType


def test_to_from_buffer(df: pl.DataFrame) -> None:
    df = df.drop("strings_nulls")

    for to_fn, from_fn, text_based in zip(
        [df.to_parquet, df.to_csv, df.to_ipc, df.to_json],
        [
            pl.read_parquet,
            partial(pl.read_csv, parse_dates=True),
            pl.read_ipc,
            pl.read_json,
        ],
        [False, True, False, True],
    ):
        f = io.BytesIO()
        to_fn(f)  # type: ignore
        f.seek(0)

        df_1 = from_fn(f)  # type: ignore
        # some type information is lost due to text conversion
        if text_based:
            df_1 = df_1.with_columns(
                [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
            )
        assert df.frame_equal(df_1)


def test_select_columns_and_projection_from_buffer() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})
    for to_fn, from_fn in zip(
        [df.to_parquet, df.to_ipc], [pl.read_parquet, pl.read_ipc]
    ):
        f = io.BytesIO()
        to_fn(f)  # type: ignore
        f.seek(0)

        df_1 = from_fn(f, columns=["b", "c"], use_pyarrow=False)  # type: ignore
        assert df_1.frame_equal(expected)

    for to_fn, from_fn in zip(
        [df.to_parquet, df.to_ipc], [pl.read_parquet, pl.read_ipc]
    ):
        f = io.BytesIO()
        to_fn(f)  # type: ignore
        f.seek(0)

        df_2 = from_fn(f, columns=[1, 2], use_pyarrow=False)  # type: ignore
        assert df_2.frame_equal(expected)


def test_compressed_to_ipc() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    compressions = [None, "uncompressed", "lz4", "zstd"]

    for compression in compressions:
        f = io.BytesIO()
        df.to_ipc(f, compression)  # type: ignore
        f.seek(0)

        df_read = pl.read_ipc(f, use_pyarrow=False)
        assert df_read.frame_equal(df)


def test_read_web_file() -> None:
    url = "https://raw.githubusercontent.com/pola-rs/polars/master/examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv"
    df = pl.read_csv(url)
    assert df.shape == (27, 4)


def test_parquet_chunks() -> None:
    """
    This failed in https://github.com/pola-rs/polars/issues/545
    """
    cases = [
        1048576,
        1048577,
    ]

    for case in cases:
        f = io.BytesIO()
        # repeat until it has case instances
        df = pd.DataFrame(
            np.tile([1.0, pd.to_datetime("2010-10-10")], [case, 1]),
            columns=["floats", "dates"],
        )

        # write as parquet
        df.to_parquet(f)
        f.seek(0)

        # read it with polars
        polars_df = pl.read_parquet(f)
        assert pl.DataFrame(df).frame_equal(polars_df)


def test_parquet_datetime() -> None:
    """
    This failed because parquet writers cast datetime to Date
    """
    f = io.BytesIO()
    data = {
        "datetime": [  # unix timestamp in ms
            1618354800000,
            1618354740000,
            1618354680000,
            1618354620000,
            1618354560000,
        ],
        "laf_max": [73.1999969482, 71.0999984741, 74.5, 69.5999984741, 69.6999969482],
        "laf_eq": [59.5999984741, 61.0, 62.2999992371, 56.9000015259, 60.0],
    }
    df = pl.DataFrame(data)
    df = df.with_column(df["datetime"].cast(pl.Datetime))

    df.to_parquet(f, use_pyarrow=True)
    f.seek(0)
    read = pl.read_parquet(f)
    assert read.frame_equal(df)


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
    csv_file = Path(__file__).parent / "files" / "gzipped.csv"
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


def test_pickle() -> None:
    a = pl.Series("a", [1, 2])
    b = pickle.dumps(a)
    out = pickle.loads(b)
    assert a.series_equal(out)
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    b = pickle.dumps(df)
    out = pickle.loads(b)
    assert df.frame_equal(out, null_equal=True)


def test_copy() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    assert copy.copy(df).frame_equal(df, True)
    assert copy.deepcopy(df).frame_equal(df, True)

    a = pl.Series("a", [1, 2])
    assert copy.copy(a).series_equal(a, True)
    assert copy.deepcopy(a).series_equal(a, True)


def test_to_json() -> None:
    # tests if it runs if no arg given
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.to_json() == '{"columns":[{"name":"a","datatype":"Int64","values":[1,2,3]}]}'
    )
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})

    out = df.to_json(row_oriented=True)
    assert out == r"""[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]"""
    out = df.to_json(json_lines=True)
    assert (
        out
        == r"""{"a":1,"b":"a"}
{"a":2,"b":"b"}
{"a":3,"b":null}
"""
    )


def test_ipc_schema() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    f = io.BytesIO()
    df.to_ipc(f)
    f.seek(0)

    assert pl.read_ipc_schema(f) == {"a": pl.Int64, "b": pl.Utf8, "c": pl.Boolean}


def test_categorical_round_trip() -> None:
    df = pl.DataFrame({"ints": [1, 2, 3], "cat": ["a", "b", "c"]})
    df = df.with_column(pl.col("cat").cast(pl.Categorical))

    tbl = df.to_arrow()
    assert "dictionary" in str(tbl["cat"].type)

    df2: pl.DataFrame = pl.from_arrow(tbl)  # type: ignore
    assert df2.dtypes == [pl.Int64, pl.Categorical]


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


def test_date_list_fmt() -> None:
    df = pl.DataFrame(
        {
            "mydate": ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-05"],
            "index": [1, 2, 5, 5],
        }
    )

    df = df.with_column(pl.col("mydate").str.strptime(pl.Date, "%Y-%m-%d"))
    assert (
        str(df.groupby("index", maintain_order=True).agg(pl.col("mydate"))["mydate"])
        == """shape: (3,)
Series: 'mydate' [list]
[
	[2020-01-01]
	[2020-01-02]
	[2020-01-05, 2020-01-05]
]"""
    )


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


def test_scan_csv() -> None:
    df = pl.scan_csv(Path(__file__).parent / "files" / "small.csv")
    assert df.collect().shape == (4, 3)


def test_scan_parquet() -> None:
    df = pl.scan_parquet(Path(__file__).parent / "files" / "small.parquet")
    assert df.collect().shape == (4, 3)


def test_read_sql() -> None:
    import sqlite3
    import tempfile

    try:
        import connectorx  # noqa

        with tempfile.TemporaryDirectory() as tmpdir_name:
            test_db = os.path.join(tmpdir_name, "test.db")
            conn = sqlite3.connect(test_db)
            conn.executescript(
                """
                CREATE TABLE test_data (
                    id    INTEGER PRIMARY KEY,
                    name  TEXT NOT NULL,
                    value FLOAT,
                    date  DATE
                );
                INSERT INTO test_data(name,value,date) VALUES ('misc',100.0,'2020-01-01'), ('other',-99.5,'2021-12-31');
                """
            )
            conn.close()

            df = pl.read_sql(
                connection_uri=f"sqlite:///{test_db}", sql="SELECT * FROM test_data"
            )
            # ┌─────┬───────┬───────┬────────────┐
            # │ id  ┆ name  ┆ value ┆ date       │
            # │ --- ┆ ---   ┆ ---   ┆ ---        │
            # │ i64 ┆ str   ┆ f64   ┆ date       │
            # ╞═════╪═══════╪═══════╪════════════╡
            # │ 1   ┆ misc  ┆ 100.0 ┆ 2020-01-01 │
            # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
            # │ 2   ┆ other ┆ -99.5 ┆ 2021-12-31 │
            # └─────┴───────┴───────┴────────────┘

            expected = {
                "id": pl.Int64,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Date,
            }
            assert df.schema == expected
            assert df.shape == (2, 4)
            assert df["date"].to_list() == [date(2020, 1, 1), date(2021, 12, 31)]
            # assert df.rows() == ...

    except ImportError:
        pass  # if connectorx not installed on test machine


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


def test_csv_globbing() -> None:
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "examples",
            "aggregate_multiple_files_in_chunks",
            "datasets",
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


def test_csv_schema_offset() -> None:
    csv = r"""
c,d,3
a,b,c
--
-
a,b,c
1,2,3
1,2,3
    """.encode()
    df = pl.read_csv(csv, offset_schema_inference=4, skip_rows=4)
    assert df.columns == ["a", "b", "c"]
