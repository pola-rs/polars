from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.write_disk
def test_streaming_parquet_glob_5900(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.parquet"
    df.write_parquet(file_path)

    path_glob = tmp_path / "small*.parquet"
    result = (
        pl.scan_parquet(path_glob).select(pl.all().first()).collect(engine="streaming")
    )
    assert result.shape == (1, df.width)


def test_scan_slice_streaming(io_files_path: Path) -> None:
    foods_file_path = io_files_path / "foods1.csv"
    df = pl.scan_csv(foods_file_path).head(5).collect(engine="streaming")
    assert df.shape == (5, 4)

    # globbing
    foods_file_path = io_files_path / "foods*.csv"
    df = pl.scan_csv(foods_file_path).head(5).collect(engine="streaming")
    assert df.shape == (5, 4)


@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt8, pl.Int16, pl.UInt16])
def test_scan_csv_overwrite_small_dtypes(
    io_files_path: Path, dtype: pl.DataType
) -> None:
    file_path = io_files_path / "foods1.csv"
    df = pl.scan_csv(file_path, schema_overrides={"sugars_g": dtype}).collect(
        engine="streaming"
    )
    assert df.dtypes == [pl.String, pl.Int64, pl.Float64, dtype]


@pytest.mark.write_disk
def test_sink_parquet(io_files_path: Path, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file = io_files_path / "small.parquet"

    file_path = tmp_path / "sink.parquet"

    df_scanned = pl.scan_parquet(file)
    df_scanned.sink_parquet(file_path)

    result = pl.read_parquet(file_path)
    df_read = pl.read_parquet(file)
    assert_frame_equal(result, df_read)


@pytest.mark.write_disk
def test_sink_parquet_10115(tmp_path: Path) -> None:
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"

    # this fails if the schema will be incorrectly due to the projection
    # pushdown
    (pl.DataFrame([{"x": 1, "y": "foo"}]).write_parquet(in_path))

    joiner = pl.LazyFrame([{"y": "foo", "z": "_"}])

    (
        pl.scan_parquet(in_path)
        .join(joiner, how="left", on="y")
        .select("x", "y", "z")
        .sink_parquet(out_path)  #
    )

    assert pl.read_parquet(out_path).to_dict(as_series=False) == {
        "x": [1],
        "y": ["foo"],
        "z": ["_"],
    }


@pytest.mark.write_disk
def test_sink_ipc(io_files_path: Path, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file = io_files_path / "small.parquet"

    file_path = tmp_path / "sink.ipc"

    df_scanned = pl.scan_parquet(file)
    df_scanned.sink_ipc(file_path)

    result = pl.read_ipc(file_path)
    df_read = pl.read_parquet(file)
    assert_frame_equal(result, df_read)


@pytest.mark.write_disk
def test_sink_csv(io_files_path: Path, tmp_path: Path) -> None:
    source_file = io_files_path / "small.parquet"
    target_file = tmp_path / "sink.csv"

    pl.scan_parquet(source_file).sink_csv(target_file)

    source_data = pl.read_parquet(source_file)
    target_data = pl.read_csv(target_file)
    assert_frame_equal(target_data, source_data)


@pytest.mark.write_disk
def test_sink_csv_14494(tmp_path: Path) -> None:
    pl.LazyFrame({"c": [1, 2, 3]}, schema={"c": pl.Int64}).filter(
        pl.col("c") > 10
    ).sink_csv(tmp_path / "sink.csv")
    assert pl.read_csv(tmp_path / "sink.csv").columns == ["c"]


@pytest.mark.parametrize(("value"), ["abc", ""])
def test_sink_csv_exception_for_separator(value: str) -> None:
    df = pl.LazyFrame({"dummy": ["abc"]})
    with pytest.raises(ValueError, match="should be a single byte character, but is"):
        df.sink_csv("path", separator=value)


@pytest.mark.parametrize(("value"), ["abc", ""])
def test_sink_csv_exception_for_quote(value: str) -> None:
    df = pl.LazyFrame({"dummy": ["abc"]})
    with pytest.raises(ValueError, match="should be a single byte character, but is"):
        df.sink_csv("path", quote_char=value)


def test_sink_csv_batch_size_zero() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="invalid zero value"):
        lf.sink_csv("test.csv", batch_size=0)


@pytest.mark.write_disk
def test_sink_csv_nested_data(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.csv"

    lf = pl.LazyFrame({"list": [[1, 2, 3, 4, 5]]})
    with pytest.raises(
        pl.exceptions.ComputeError, match="CSV format does not support nested data"
    ):
        lf.sink_csv(path)


def test_scan_csv_only_header_10792(io_files_path: Path) -> None:
    foods_file_path = io_files_path / "only_header.csv"
    df = pl.scan_csv(foods_file_path).collect(engine="streaming")
    assert df.to_dict(as_series=False) == {"Name": [], "Address": []}


def test_scan_empty_csv_10818(io_files_path: Path) -> None:
    empty_file_path = io_files_path / "empty.csv"
    df = pl.scan_csv(empty_file_path, raise_if_empty=False).collect(engine="streaming")
    assert df.is_empty()


@pytest.mark.write_disk
def test_streaming_cross_join_schema(tmp_path: Path) -> None:
    file_path = tmp_path / "temp.parquet"
    a = pl.DataFrame({"a": [1, 2]}).lazy()
    b = pl.DataFrame({"b": ["b"]}).lazy()
    a.join(b, how="cross").sink_parquet(file_path)
    read = pl.read_parquet(file_path, parallel="none")
    assert read.to_dict(as_series=False) == {"a": [1, 2], "b": ["b", "b"]}


@pytest.mark.write_disk
def test_sink_ndjson_should_write_same_data(
    io_files_path: Path, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    source_path = io_files_path / "foods1.csv"
    target_path = tmp_path / "foods_test.ndjson"

    expected = pl.read_csv(source_path)

    lf = pl.scan_csv(source_path)
    lf.sink_ndjson(target_path)
    df = pl.read_ndjson(target_path)

    assert_frame_equal(df, expected)


@pytest.mark.write_disk
@pytest.mark.parametrize("streaming", [False, True])
def test_parquet_eq_statistics(
    plmonkeypatch: PlMonkeyPatch, capfd: Any, tmp_path: Path, streaming: bool
) -> None:
    tmp_path.mkdir(exist_ok=True)

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    df = pl.DataFrame({"idx": pl.arange(100, 200, eager=True)}).with_columns(
        (pl.col("idx") // 25).alias("part")
    )
    df = pl.concat(df.partition_by("part", as_dict=False), rechunk=False)
    assert df.n_chunks("all") == [4, 4]

    file_path = tmp_path / "stats.parquet"
    df.write_parquet(file_path, statistics=True, use_pyarrow=False)

    for pred in [
        pl.col("idx") == 50,
        pl.col("idx") == 150,
        pl.col("idx") == 210,
    ]:
        result = (
            pl.scan_parquet(file_path)
            .filter(pred)
            .collect(engine="streaming" if streaming else "in-memory")
        )
        assert_frame_equal(result, df.filter(pred))

    captured = capfd.readouterr().err
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 1 / 1 row groups" in captured
    )
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 0 / 1 row groups" in captured
    )


@pytest.mark.write_disk
def test_streaming_empty_parquet_16523(tmp_path: Path) -> None:
    file_path = tmp_path / "foo.parquet"
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int32})
    df.write_parquet(file_path)
    q = pl.scan_parquet(file_path)
    q2 = pl.LazyFrame({"a": [1]}, schema={"a": pl.Int32})
    assert q.join(q2, on="a").collect(engine="streaming").shape == (0, 1)


@pytest.mark.parametrize(
    "method",
    ["parquet", "csv", "ipc", "ndjson"],
)
@pytest.mark.write_disk
def test_sink_phases(tmp_path: Path, method: str) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [
                "some",
                "text",
                "over-here-is-very-long",
                "and",
                "some",
                "more",
                "text",
            ],
        }
    )

    # Ordered Unions lead to many phase transitions.
    ref_df = pl.concat([df] * 100)
    lf = pl.concat([df.lazy()] * 100)

    (getattr(lf, f"sink_{method}"))(tmp_path / f"t.{method}", engine="streaming")
    df = (getattr(pl, f"scan_{method}"))(tmp_path / f"t.{method}").collect()

    assert_frame_equal(df, ref_df)

    (getattr(lf, f"sink_{method}"))(
        tmp_path / f"t.{method}", maintain_order=False, engine="streaming"
    )
    height = (
        (getattr(pl, f"scan_{method}"))(tmp_path / f"t.{method}")
        .select(pl.len())
        .collect()[0, 0]
    )
    assert height == ref_df.height


def test_empty_sink_parquet_join_14863(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.parquet"
    lf = pl.LazyFrame(schema=["a", "b", "c"]).cast(pl.String)
    lf.sink_parquet(file_path)
    assert_frame_equal(
        pl.LazyFrame({"a": ["uno"]}).join(pl.scan_parquet(file_path), on="a").collect(),
        lf.collect(),
    )


@pytest.mark.write_disk
def test_scan_non_existent_file_21527() -> None:
    with pytest.raises(
        FileNotFoundError,
        match=r"a-file-that-does-not-exist",
    ):
        pl.scan_parquet("a-file-that-does-not-exist").sink_ipc(
            "x.ipc", engine="streaming"
        )
