import io
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars._typing import EngineType
from polars.testing import assert_frame_equal

SINKS = [
    (pl.scan_ipc, pl.LazyFrame.sink_ipc),
    (pl.scan_parquet, pl.LazyFrame.sink_parquet),
    (pl.scan_csv, pl.LazyFrame.sink_csv),
    (pl.scan_ndjson, pl.LazyFrame.sink_ndjson),
]


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.write_disk
def test_mkdir(tmp_path: Path, scan: Any, sink: Any, engine: EngineType) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    with pytest.raises(FileNotFoundError):
        sink(df.lazy(), tmp_path / "a" / "b" / "c" / "file", engine=engine)

    f = tmp_path / "a" / "b" / "c" / "file2"
    sink(df.lazy(), f, mkdir=True)

    assert_frame_equal(scan(f).collect(), df)


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.write_disk
def test_lazy_sinks(tmp_path: Path, scan: Any, sink: Any, engine: EngineType) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    lf1 = sink(df.lazy(), tmp_path / "a", lazy=True)
    lf2 = sink(df.lazy(), tmp_path / "b", lazy=True)

    assert not Path(tmp_path / "a").exists()
    assert not Path(tmp_path / "b").exists()

    pl.collect_all([lf1, lf2], engine=engine)

    assert_frame_equal(scan(tmp_path / "a").collect(), df)
    assert_frame_equal(scan(tmp_path / "b").collect(), df)


@pytest.mark.parametrize(
    "sink",
    [
        pl.LazyFrame.sink_ipc,
        pl.LazyFrame.sink_parquet,
        pl.LazyFrame.sink_csv,
        pl.LazyFrame.sink_ndjson,
    ],
)
@pytest.mark.write_disk
def test_double_lazy_error(sink: Any) -> None:
    df = pl.DataFrame({})

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="cannot create a sink on top of another sink",
    ):
        sink(sink(df.lazy(), "a", lazy=True), "b")


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_to_memory(sink: Any, scan: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    f = io.BytesIO()
    sink(df.lazy(), f)

    f.seek(0)
    assert_frame_equal(
        scan(f).collect(),
        df,
    )


@pytest.mark.parametrize(("scan", "sink"), SINKS)
@pytest.mark.write_disk
def test_sink_to_file(tmp_path: Path, sink: Any, scan: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [5, 10, 1996],
        }
    )

    with (tmp_path / "f").open("w+") as f:
        sink(df.lazy(), f, sync_on_close="all")
        f.seek(0)
        assert_frame_equal(
            scan(f).collect(),
            df,
        )


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_empty(sink: Any, scan: Any) -> None:
    df = pl.LazyFrame(data={"col1": ["a"]})

    df_empty = pl.LazyFrame(
        data={"col1": []},
        schema={"col1": str},
    )

    expected = df_empty.join(df, how="cross").collect()
    expected_schema = expected.schema

    kwargs = {}
    if scan == pl.scan_ndjson:
        kwargs["schema"] = expected_schema

    # right empty
    f = io.BytesIO()
    sink(df.join(df_empty, how="cross"), f)
    f.seek(0)
    assert_frame_equal(scan(f, **kwargs), expected.lazy())

    # left empty
    f.seek(0)
    sink(df_empty.join(df, how="cross"), f)
    f.truncate()
    f.seek(0)
    assert_frame_equal(scan(f, **kwargs), expected.lazy())

    # both empty
    f.seek(0)
    sink(df_empty.join(df_empty, how="cross"), f)
    f.truncate()
    f.seek(0)
    assert_frame_equal(scan(f, **kwargs), expected.lazy())


@pytest.mark.parametrize(("scan", "sink"), SINKS)
def test_sink_null_upcast(scan: Any, sink: Any) -> None:
    scan_kwargs: dict[str, Any] = {}
    sink_kwargs: dict[str, Any] = {}
    if scan == pl.scan_csv:
        scan_kwargs["null_values"] = "<NULL>"
        scan_kwargs["schema"] = pl.Schema({"a": pl.Int64})
        sink_kwargs["null_value"] = "<NULL>"

    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.select(a=pl.lit(None))

    f = io.BytesIO()
    g = io.BytesIO()

    sink(df1.lazy(), f, **sink_kwargs)
    sink(df2.lazy(), g, **sink_kwargs)

    f.seek(0)
    g.seek(0)
    assert_frame_equal(
        scan([f, g], **scan_kwargs).collect(),
        pl.concat([df1, df2]),
    )
