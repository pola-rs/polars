from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, ShapeError
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_file_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.csv"


def test_scan_csv(io_files_path: Path) -> None:
    df = pl.scan_csv(io_files_path / "small.csv")
    assert df.collect().shape == (4, 3)


def test_scan_csv_no_cse_deadlock(io_files_path: Path) -> None:
    dfs = [pl.scan_csv(io_files_path / "small.csv")] * (pl.thread_pool_size() + 1)
    pl.concat(dfs, parallel=True).collect(comm_subplan_elim=False)


def test_scan_empty_csv(io_files_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        pl.scan_csv(io_files_path / "empty.csv").collect()
    assert "empty CSV" in str(excinfo.value)

    lf = pl.scan_csv(io_files_path / "empty.csv", raise_if_empty=False)
    assert_frame_equal(lf, pl.LazyFrame())


@pytest.mark.write_disk()
def test_invalid_utf8(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    np.random.seed(1)
    bts = bytes(np.random.randint(0, 255, 200))

    file_path = tmp_path / "nonutf8.csv"
    file_path.write_bytes(bts)

    a = pl.read_csv(file_path, has_header=False, encoding="utf8-lossy")
    b = pl.scan_csv(file_path, has_header=False, encoding="utf8-lossy").collect()

    assert_frame_equal(a, b)


def test_row_index(foods_file_path: Path) -> None:
    df = pl.read_csv(foods_file_path, row_index_name="row_index")
    assert df["row_index"].to_list() == list(range(27))

    df = (
        pl.scan_csv(foods_file_path, row_index_name="row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_csv(foods_file_path, row_index_name="row_index")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


@pytest.mark.parametrize("file_name", ["foods1.csv", "foods*.csv"])
def test_scan_csv_schema_overwrite_and_dtypes_overwrite(
    io_files_path: Path, file_name: str
) -> None:
    file_path = io_files_path / file_name
    df = pl.scan_csv(
        file_path,
        schema_overrides={"calories_foo": pl.String, "fats_g_foo": pl.Float32},
        with_column_names=lambda names: [f"{a}_foo" for a in names],
    ).collect()
    assert df.dtypes == [pl.String, pl.String, pl.Float32, pl.Int64]
    assert df.columns == [
        "category_foo",
        "calories_foo",
        "fats_g_foo",
        "sugars_g_foo",
    ]


@pytest.mark.parametrize("file_name", ["foods1.csv", "foods*.csv"])
@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16])
def test_scan_csv_schema_overwrite_and_small_dtypes_overwrite(
    io_files_path: Path, file_name: str, dtype: pl.DataType
) -> None:
    file_path = io_files_path / file_name
    df = pl.scan_csv(
        file_path,
        schema_overrides={"calories_foo": pl.String, "sugars_g_foo": dtype},
        with_column_names=lambda names: [f"{a}_foo" for a in names],
    ).collect()
    assert df.dtypes == [pl.String, pl.String, pl.Float64, dtype]
    assert df.columns == [
        "category_foo",
        "calories_foo",
        "fats_g_foo",
        "sugars_g_foo",
    ]


@pytest.mark.parametrize("file_name", ["foods1.csv", "foods*.csv"])
def test_scan_csv_schema_new_columns_dtypes(
    io_files_path: Path, file_name: str
) -> None:
    file_path = io_files_path / file_name

    for dtype in [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16]:
        # assign 'new_columns', providing partial dtype overrides
        df1 = pl.scan_csv(
            file_path,
            schema_overrides={"calories": pl.String, "sugars": dtype},
            new_columns=["category", "calories", "fats", "sugars"],
        ).collect()
        assert df1.dtypes == [pl.String, pl.String, pl.Float64, dtype]
        assert df1.columns == ["category", "calories", "fats", "sugars"]

        # assign 'new_columns' with 'dtypes' list
        df2 = pl.scan_csv(
            file_path,
            schema_overrides=[pl.String, pl.String, pl.Float64, dtype],
            new_columns=["category", "calories", "fats", "sugars"],
        ).collect()
        assert df1.rows() == df2.rows()

    # rename existing columns, then lazy-select disjoint cols
    lf = pl.scan_csv(
        file_path,
        new_columns=["colw", "colx", "coly", "colz"],
    )
    schema = lf.collect_schema()
    assert schema.dtypes() == [pl.String, pl.Int64, pl.Float64, pl.Int64]
    assert schema.names() == ["colw", "colx", "coly", "colz"]
    assert (
        lf.select("colz", "colx").collect().rows()
        == df1.select("sugars", pl.col("calories").cast(pl.Int64)).rows()
    )

    # partially rename columns / overwrite dtypes
    df4 = pl.scan_csv(
        file_path,
        schema_overrides=[pl.String, pl.String],
        new_columns=["category", "calories"],
    ).collect()
    assert df4.dtypes == [pl.String, pl.String, pl.Float64, pl.Int64]
    assert df4.columns == ["category", "calories", "fats_g", "sugars_g"]

    # cannot have len(new_columns) > len(actual columns)
    with pytest.raises(ShapeError):
        pl.scan_csv(
            file_path,
            schema_overrides=[pl.String, pl.String],
            new_columns=["category", "calories", "c3", "c4", "c5"],
        ).collect()

    # cannot set both 'new_columns' and 'with_column_names'
    with pytest.raises(ValueError, match="mutually.exclusive"):
        pl.scan_csv(
            file_path,
            schema_overrides=[pl.String, pl.String],
            new_columns=["category", "calories", "fats", "sugars"],
            with_column_names=lambda cols: [col.capitalize() for col in cols],
        ).collect()


def test_lazy_n_rows(foods_file_path: Path) -> None:
    df = (
        pl.scan_csv(foods_file_path, n_rows=4, row_index_name="idx")
        .filter(pl.col("idx") > 2)
        .collect()
    )
    assert df.to_dict(as_series=False) == {
        "idx": [3],
        "category": ["fruit"],
        "calories": [60],
        "fats_g": [0.0],
        "sugars_g": [11],
    }


def test_lazy_row_index_no_push_down(foods_file_path: Path) -> None:
    plan = (
        pl.scan_csv(foods_file_path)
        .with_row_index()
        .filter(pl.col("index") == 1)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .explain(predicate_pushdown=True)
    )
    # related to row count is not pushed.
    assert 'FILTER [(col("index")) == (1)] FROM' in plan
    # unrelated to row count is pushed.
    assert 'SELECTION: [(col("category")) == (String(vegetables))]' in plan


@pytest.mark.write_disk()
def test_glob_skip_rows(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    for i in range(2):
        file_path = tmp_path / f"test_{i}.csv"
        file_path.write_text(
            f"""
metadata goes here
file number {i}
foo,bar,baz
1,2,3
4,5,6
7,8,9
"""
        )
    file_path = tmp_path / "*.csv"
    assert pl.read_csv(file_path, skip_rows=2).to_dict(as_series=False) == {
        "foo": [1, 4, 7, 1, 4, 7],
        "bar": [2, 5, 8, 2, 5, 8],
        "baz": [3, 6, 9, 3, 6, 9],
    }


def test_glob_n_rows(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.csv"
    df = pl.scan_csv(file_path, n_rows=40).collect()

    # 27 rows from foods1.csv and 13 from foods2.csv
    assert df.shape == (40, 4)

    # take first and last rows
    assert df[[0, 39]].to_dict(as_series=False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }


def test_scan_csv_schema_overwrite_not_projected_8483(foods_file_path: Path) -> None:
    df = (
        pl.scan_csv(
            foods_file_path,
            schema_overrides={"calories": pl.String, "sugars_g": pl.Int8},
        )
        .select(pl.len())
        .collect()
    )
    expected = pl.DataFrame({"len": 27}, schema={"len": pl.UInt32})
    assert_frame_equal(df, expected)


def test_csv_list_arg(io_files_path: Path) -> None:
    first = io_files_path / "foods1.csv"
    second = io_files_path / "foods2.csv"

    df = pl.scan_csv(source=[first, second]).collect()
    assert df.shape == (54, 4)
    assert df.row(-1) == ("seafood", 194, 12.0, 1)
    assert df.row(0) == ("vegetables", 45, 0.5, 2)


# https://github.com/pola-rs/polars/issues/9887
def test_scan_csv_slice_offset_zero(io_files_path: Path) -> None:
    lf = pl.scan_csv(io_files_path / "small.csv")
    result = lf.slice(0)
    assert result.collect().height == 4


@pytest.mark.write_disk()
def test_scan_empty_csv_with_row_index(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.parquet"
    df = pl.DataFrame({"a": []})
    df.write_csv(file_path)

    read = pl.scan_csv(file_path).with_row_index("idx")
    assert read.collect().schema == OrderedDict([("idx", pl.UInt32), ("a", pl.String)])


@pytest.mark.write_disk()
def test_csv_null_values_with_projection_15515() -> None:
    data = """IndCode,SireCode,BirthDate,Flag
ID00316,.,19940315,
"""

    with tempfile.NamedTemporaryFile() as f:
        f.write(data.encode())
        f.seek(0)

        q = (
            pl.scan_csv(f.name, null_values={"SireCode": "."})
            .with_columns(pl.col("SireCode").alias("SireKey"))
            .select("SireKey", "BirthDate")
        )

        assert q.collect().to_dict(as_series=False) == {
            "SireKey": [None],
            "BirthDate": [19940315],
        }


@pytest.mark.write_disk()
def test_csv_respect_user_schema_ragged_lines_15254() -> None:
    with tempfile.NamedTemporaryFile() as f:
        f.write(
            b"""
A,B,C
1,2,3
4,5,6,7,8
9,10,11
""".strip()
        )
        f.seek(0)

        df = pl.scan_csv(
            f.name, schema=dict.fromkeys("ABCDE", pl.String), truncate_ragged_lines=True
        ).collect()
        assert df.to_dict(as_series=False) == {
            "A": ["1", "4", "9"],
            "B": ["2", "5", "10"],
            "C": ["3", "6", "11"],
            "D": [None, "7", None],
            "E": [None, "8", None],
        }


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "dfs",
    [
        [pl.DataFrame({"a": [1, 2, 3]}), pl.DataFrame({"b": [4, 5, 6]})],
        [
            pl.DataFrame({"a": [1, 2, 3]}),
            pl.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9]}),
        ],
    ],
)
def test_file_list_schema_mismatch(
    tmp_path: Path, dfs: list[pl.DataFrame], streaming: bool
) -> None:
    tmp_path.mkdir(exist_ok=True)

    paths = [f"{tmp_path}/{i}.csv" for i in range(len(dfs))]

    for df, path in zip(dfs, paths):
        df.write_csv(path)

    lf = pl.scan_csv(paths)
    with pytest.raises(ComputeError):
        lf.collect(streaming=streaming)

    if len({df.width for df in dfs}) == 1:
        expect = pl.concat(df.select(x=pl.first().cast(pl.Int8)) for df in dfs)
        out = pl.scan_csv(paths, schema={"x": pl.Int8}).collect(streaming=streaming)

        assert_frame_equal(out, expect)


@pytest.mark.parametrize("streaming", [True, False])
def test_file_list_schema_supertype(tmp_path: Path, streaming: bool) -> None:
    tmp_path.mkdir(exist_ok=True)

    data_lst = [
        """\
a
1
2
""",
        """\
a
b
c
""",
    ]

    paths = [f"{tmp_path}/{i}.csv" for i in range(len(data_lst))]

    for data, path in zip(data_lst, paths):
        with Path(path).open("w") as f:
            f.write(data)

    expect = pl.Series("a", ["1", "2", "b", "c"]).to_frame()
    out = pl.scan_csv(paths).collect(streaming=streaming)

    assert_frame_equal(out, expect)


@pytest.mark.parametrize("streaming", [True, False])
def test_file_list_comment_skip_rows_16327(tmp_path: Path, streaming: bool) -> None:
    tmp_path.mkdir(exist_ok=True)

    data_lst = [
        """\
# comment
a
b
c
""",
        """\
a
b
c
""",
    ]

    paths = [f"{tmp_path}/{i}.csv" for i in range(len(data_lst))]

    for data, path in zip(data_lst, paths):
        with Path(path).open("w") as f:
            f.write(data)

    expect = pl.Series("a", ["b", "c", "b", "c"]).to_frame()
    out = pl.scan_csv(paths, comment_prefix="#").collect(streaming=streaming)

    assert_frame_equal(out, expect)


@pytest.mark.xfail(reason="Bug: https://github.com/pola-rs/polars/issues/17634")
def test_scan_csv_with_column_names_nonexistent_file() -> None:
    path_str = "my-nonexistent-data.csv"
    path = Path(path_str)
    assert not path.exists()

    # Just calling the scan function should not raise any errors
    result = pl.scan_csv(path, with_column_names=lambda x: [c.upper() for c in x])
    assert isinstance(result, pl.LazyFrame)

    # Upon collection, it should fail
    with pytest.raises(FileNotFoundError):
        result.collect()
