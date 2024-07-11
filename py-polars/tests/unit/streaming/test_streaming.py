from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import PolarsInefficientMapWarning
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import JoinStrategy

pytestmark = pytest.mark.xdist_group("streaming")


def test_streaming_categoricals_5921() -> None:
    with pl.StringCache():
        out_lazy = (
            pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
            .lazy()
            .with_columns(pl.col("X").cast(pl.Categorical))
            .group_by("X")
            .agg(pl.col("Y").min())
            .sort("Y", descending=True)
            .collect(streaming=True)
        )

        out_eager = (
            pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
            .with_columns(pl.col("X").cast(pl.Categorical))
            .group_by("X")
            .agg(pl.col("Y").min())
            .sort("Y", descending=True)
        )

    for out in [out_eager, out_lazy]:
        assert out.dtypes == [pl.Categorical, pl.Int64]
        assert out.to_dict(as_series=False) == {"X": ["a", "b"], "Y": [2, 1]}


def test_streaming_block_on_literals_6054() -> None:
    df = pl.DataFrame({"col_1": [0] * 5 + [1] * 5})
    s = pl.Series("col_2", list(range(10)))

    assert df.lazy().with_columns(s).group_by("col_1").agg(pl.all().first()).collect(
        streaming=True
    ).sort("col_1").to_dict(as_series=False) == {"col_1": [0, 1], "col_2": [0, 5]}


def test_streaming_streamable_functions(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    assert (
        pl.DataFrame({"a": [1, 2, 3]})
        .lazy()
        .map_batches(
            function=lambda df: df.with_columns(pl.col("a").alias("b")),
            schema={"a": pl.Int64, "b": pl.Int64},
            streamable=True,
        )
    ).collect(streaming=True).to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [1, 2, 3],
    }

    (_, err) = capfd.readouterr()
    assert "df -> function -> ordered_sink" in err


@pytest.mark.slow()
def test_cross_join_stack() -> None:
    a = pl.Series(np.arange(100_000)).to_frame().lazy()
    t0 = time.time()
    # this should be instant if directly pushed into sink
    # if not the cross join will first fill the stack with all matches of a single chunk
    assert a.join(a, how="cross").head().collect(streaming=True).shape == (5, 2)
    t1 = time.time()
    assert (t1 - t0) < 0.5


def test_streaming_literal_expansion() -> None:
    df = pl.DataFrame(
        {
            "y": ["a", "b"],
            "z": [1, 2],
        }
    )

    q = df.lazy().select(
        x=pl.lit("constant"),
        y=pl.col("y"),
        z=pl.col("z"),
    )

    assert q.collect(streaming=True).to_dict(as_series=False) == {
        "x": ["constant", "constant"],
        "y": ["a", "b"],
        "z": [1, 2],
    }
    assert q.group_by(["x", "y"]).agg(pl.mean("z")).sort("y").collect(
        streaming=True
    ).to_dict(as_series=False) == {
        "x": ["constant", "constant"],
        "y": ["a", "b"],
        "z": [1.0, 2.0],
    }
    assert q.group_by(["x"]).agg(pl.mean("z")).collect().to_dict(as_series=False) == {
        "x": ["constant"],
        "z": [1.5],
    }


def test_streaming_apply(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    q = pl.DataFrame({"a": [1, 2]}).lazy()

    with pytest.warns(PolarsInefficientMapWarning, match="with this one instead"):
        (
            q.select(
                pl.col("a").map_elements(lambda x: x * 2, return_dtype=pl.Int64)
            ).collect(streaming=True)
        )
        (_, err) = capfd.readouterr()
        assert "df -> projection -> ordered_sink" in err


def test_streaming_ternary() -> None:
    q = pl.LazyFrame({"a": [1, 2, 3]})

    assert (
        q.with_columns(
            pl.when(pl.col("a") >= 2).then(pl.col("a")).otherwise(None).alias("b"),
        )
        .explain(streaming=True)
        .startswith("STREAMING")
    )


def test_streaming_sortedness_propagation_9494() -> None:
    assert (
        pl.DataFrame(
            {
                "when": [date(2023, 5, 10), date(2023, 5, 20), date(2023, 6, 10)],
                "what": [1, 2, 3],
            }
        )
        .lazy()
        .sort("when")
        .group_by_dynamic("when", every="1mo")
        .agg(pl.col("what").sum())
        .collect(streaming=True)
    ).to_dict(as_series=False) == {
        "when": [date(2023, 5, 1), date(2023, 6, 1)],
        "what": [3, 3],
    }


@pytest.mark.write_disk()
@pytest.mark.slow()
def test_streaming_generic_left_and_inner_join_from_disk(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    p0 = tmp_path / "df0.parquet"
    p1 = tmp_path / "df1.parquet"
    # by loading from disk, we get different chunks
    n = 200_000
    k = 100

    d0: dict[str, np.ndarray[Any, Any]] = {
        f"x{i}": np.random.random(n) for i in range(k)
    }
    d0.update({"id": np.arange(n)})

    df0 = pl.DataFrame(d0)
    df1 = df0.clone().select(pl.all().shuffle(111))

    df0.write_parquet(p0)
    df1.write_parquet(p1)

    lf0 = pl.scan_parquet(p0)
    lf1 = pl.scan_parquet(p1).select(pl.all().name.suffix("_r"))

    join_strategies: list[JoinStrategy] = ["left", "inner"]
    for how in join_strategies:
        q = lf0.join(lf1, left_on="id", right_on="id_r", how=how)
        assert_frame_equal(
            q.collect(streaming=True),
            q.collect(streaming=False),
            check_row_order=how == "left",
        )


def test_streaming_9776() -> None:
    df = pl.DataFrame({"col_1": ["a"] * 1000, "ID": [None] + ["a"] * 999})
    ordered = (
        df.group_by("col_1", "ID", maintain_order=True)
        .len()
        .filter(pl.col("col_1") == "a")
    )
    unordered = (
        df.group_by("col_1", "ID", maintain_order=False)
        .len()
        .filter(pl.col("col_1") == "a")
    )
    expected = [("a", None, 1), ("a", "a", 999)]
    assert ordered.rows() == expected
    assert unordered.sort(["col_1", "ID"]).rows() == expected


@pytest.mark.write_disk()
def test_stream_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "in.parquet"
    schema = {
        "KLN_NR": pl.String,
    }

    df = pl.DataFrame(
        {
            "KLN_NR": [],
        },
        schema=schema,
    )
    df.write_parquet(p)
    assert pl.scan_parquet(p).collect(streaming=True).schema == schema


def test_streaming_empty_df() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", ["a", "b", "c", "b", "a", "a"], dtype=pl.Categorical()),
            pl.Series("b", ["b", "c", "c", "b", "a", "c"], dtype=pl.Categorical()),
        ]
    )

    result = (
        df.lazy()
        .join(df.lazy(), on="a", how="inner")
        .filter(False)
        .collect(streaming=True)
    )

    assert result.to_dict(as_series=False) == {"a": [], "b": [], "b_right": []}


def test_streaming_duplicate_cols_5537() -> None:
    assert pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).lazy().with_columns(
        (pl.col("a") * 2).alias("foo"), (pl.col("a") * 3)
    ).collect(streaming=True).to_dict(as_series=False) == {
        "a": [3, 6, 9],
        "b": [1, 2, 3],
        "foo": [2, 4, 6],
    }


def test_null_sum_streaming_10455() -> None:
    df = pl.DataFrame(
        {
            "x": [1] * 10,
            "y": [None] * 10,
        },
        schema={"x": pl.Int64, "y": pl.Float32},
    )
    result = df.lazy().group_by("x").sum().collect(streaming=True)
    expected = {"x": [1], "y": [0.0]}
    assert result.to_dict(as_series=False) == expected


def test_boolean_agg_schema() -> None:
    df = pl.DataFrame(
        {
            "x": [1, 1, 1],
            "y": [False, True, False],
        }
    ).lazy()

    agg_df = df.group_by("x").agg(pl.col("y").max().alias("max_y"))

    for streaming in [True, False]:
        assert (
            agg_df.collect(streaming=streaming).schema
            == agg_df.collect_schema()
            == {"x": pl.Int64, "max_y": pl.Boolean}
        )


@pytest.mark.write_disk()
def test_streaming_csv_headers_but_no_data_13770(tmp_path: Path) -> None:
    with Path.open(tmp_path / "header_no_data.csv", "w") as f:
        f.write("name, age\n")

    schema = {"name": pl.String, "age": pl.Int32}
    df = (
        pl.scan_csv(tmp_path / "header_no_data.csv", schema=schema)
        .head()
        .collect(streaming=True)
    )
    assert len(df) == 0
    assert df.schema == schema


@pytest.mark.write_disk()
def test_custom_temp_dir(tmp_path: Path, monkeypatch: Any) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    s = pl.arange(0, 100_000, eager=True).rename("idx")
    df = s.shuffle().to_frame()
    df.lazy().sort("idx").collect(streaming=True)

    assert os.listdir(tmp_path), f"Temp directory '{tmp_path}' is empty"


@pytest.mark.write_disk()
def test_streaming_with_hconcat(tmp_path: Path) -> None:
    df1 = pl.DataFrame(
        {
            "id": [0, 0, 1, 1, 2, 2],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    df1.write_parquet(tmp_path / "df1.parquet")

    df2 = pl.DataFrame(
        {
            "y": [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )
    df2.write_parquet(tmp_path / "df2.parquet")

    lf1 = pl.scan_parquet(tmp_path / "df1.parquet")
    lf2 = pl.scan_parquet(tmp_path / "df2.parquet")
    query = (
        pl.concat([lf1, lf2], how="horizontal")
        .group_by("id")
        .agg(pl.all().mean())
        .sort(pl.col("id"))
    )

    plan_lines = [line.strip() for line in query.explain(streaming=True).splitlines()]

    # Each input of the concatenation should be a streaming section,
    # as these might be streamable even though the horizontal concatenation itself
    # doesn't yet support streaming.
    for i, line in enumerate(plan_lines):
        if line.startswith("PLAN"):
            assert plan_lines[i + 1].startswith(
                "STREAMING"
            ), f"{line} does not contain a streaming section"

    result = query.collect(streaming=True)

    expected = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "x": [0.5, 2.5, 4.5],
            "y": [6.5, 8.5, 10.5],
        }
    )

    assert_frame_equal(result, expected)
