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
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


def test_streaming_categoricals_5921() -> None:
    out_lazy = (
        pl.DataFrame({"X": ["a", "a", "a", "b", "b"], "Y": [2, 2, 2, 1, 1]})
        .lazy()
        .with_columns(pl.col("X").cast(pl.Categorical))
        .group_by("X")
        .agg(pl.col("Y").min())
        .sort("Y", descending=True)
        .collect(engine="streaming")
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
        engine="streaming"
    ).sort("col_1").to_dict(as_series=False) == {"col_1": [0, 1], "col_2": [0, 5]}


@pytest.mark.may_fail_auto_streaming
@pytest.mark.may_fail_cloud  # reason: non-pure map_batches
def test_streaming_streamable_functions(
    plmonkeypatch: PlMonkeyPatch, capfd: Any
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "1")
    calls = 0

    def func(df: pl.DataFrame) -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return df.with_columns(pl.col("a").alias("b"))

    assert (
        pl.DataFrame({"a": list(range(100))})
        .lazy()
        .map_batches(
            function=func,
            schema={"a": pl.Int64, "b": pl.Int64},
            streamable=True,
        )
    ).collect(engine="streaming").to_dict(as_series=False) == {
        "a": list(range(100)),
        "b": list(range(100)),
    }

    assert calls > 1


@pytest.mark.slow
@pytest.mark.may_fail_auto_streaming
@pytest.mark.may_fail_cloud  # reason: timing
def test_cross_join_stack() -> None:
    morsel_size = os.environ.get("POLARS_IDEAL_MORSEL_SIZE")
    if morsel_size is not None and int(morsel_size) < 1000:
        pytest.skip("test is too slow for small morsel sizes")

    a = pl.Series(np.arange(100_000)).to_frame().lazy()
    t0 = time.time()
    assert a.join(a, how="cross").head().collect(engine="streaming").shape == (5, 2)
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

    assert q.collect(engine="streaming").to_dict(as_series=False) == {
        "x": ["constant", "constant"],
        "y": ["a", "b"],
        "z": [1, 2],
    }
    assert q.group_by(["x", "y"]).agg(pl.mean("z")).sort("y").collect(
        engine="streaming"
    ).to_dict(as_series=False) == {
        "x": ["constant", "constant"],
        "y": ["a", "b"],
        "z": [1.0, 2.0],
    }
    assert q.group_by(["x"]).agg(pl.mean("z")).collect().to_dict(as_series=False) == {
        "x": ["constant"],
        "z": [1.5],
    }


@pytest.mark.may_fail_auto_streaming
def test_streaming_apply(plmonkeypatch: PlMonkeyPatch, capfd: Any) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    q = pl.DataFrame({"a": [1, 2]}).lazy()
    with pytest.warns(
        PolarsInefficientMapWarning,
        match="with this one instead",
    ):
        (
            q.select(
                pl.col("a").map_elements(lambda x: x * 2, return_dtype=pl.Int64)
            ).collect(engine="streaming")
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
        .collect(engine="streaming")
    ).to_dict(as_series=False) == {
        "when": [date(2023, 5, 1), date(2023, 6, 1)],
        "what": [3, 3],
    }


@pytest.mark.write_disk
@pytest.mark.slow
def test_streaming_generic_left_and_inner_join_from_disk(tmp_path: Path) -> None:
    morsel_size = os.environ.get("POLARS_IDEAL_MORSEL_SIZE")
    if morsel_size is not None and int(morsel_size) < 1000:
        pytest.skip("test is too slow for small morsel sizes")

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
        assert_frame_equal(
            lf0.join(lf1, left_on="id", right_on="id_r", how=how).collect(
                engine="streaming"
            ),
            lf0.join(lf1, left_on="id", right_on="id_r", how=how).collect(
                engine="in-memory"
            ),
            check_row_order=False,
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


@pytest.mark.write_disk
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
    assert pl.scan_parquet(p).collect(engine="streaming").schema == schema


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
        .collect(engine="streaming")
    )

    assert result.to_dict(as_series=False) == {"a": [], "b": [], "b_right": []}


def test_streaming_duplicate_cols_5537() -> None:
    assert pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).lazy().with_columns(
        (pl.col("a") * 2).alias("foo"), (pl.col("a") * 3)
    ).collect(engine="streaming").to_dict(as_series=False) == {
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
    result = df.lazy().group_by("x").sum().collect(engine="streaming")
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
            agg_df.collect(engine="streaming" if streaming else "in-memory").schema
            == agg_df.collect_schema()
            == {"x": pl.Int64, "max_y": pl.Boolean}
        )


@pytest.mark.write_disk
def test_streaming_csv_headers_but_no_data_13770(tmp_path: Path) -> None:
    with Path.open(tmp_path / "header_no_data.csv", "w") as f:
        f.write("name, age\n")

    schema = {"name": pl.String, "age": pl.Int32}
    df = (
        pl.scan_csv(tmp_path / "header_no_data.csv", schema=schema)
        .head()
        .collect(engine="streaming")
    )
    assert df.height == 0
    assert df.schema == schema


@pytest.mark.write_disk
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

    result = query.collect(engine="streaming")

    expected = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "x": [0.5, 2.5, 4.5],
            "y": [6.5, 8.5, 10.5],
        }
    )

    assert_frame_equal(result, expected)


@pytest.mark.write_disk
def test_elementwise_identification_in_ternary_15767(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    (
        pl.LazyFrame({"a": pl.Series([1])})
        .with_columns(b=pl.col("a").is_in(pl.Series([1, 2, 3])))
        .sink_parquet(tmp_path / "1")
    )

    (
        pl.LazyFrame({"a": pl.Series([1])})
        .with_columns(
            b=pl.when(pl.col("a").is_in(pl.Series([1, 2, 3]))).then(pl.col("a"))
        )
        .sink_parquet(tmp_path / "1")
    )


def test_streaming_temporal_17669() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2, 3]}, schema={"a": pl.Datetime("us")})
        .with_columns(
            b=pl.col("a").dt.date(),
            c=pl.col("a").dt.time(),
        )
        .collect(engine="streaming")
    )
    assert df.schema == {
        "a": pl.Datetime("us"),
        "b": pl.Date,
        "c": pl.Time,
    }


def test_i128_sum_reduction() -> None:
    assert (
        pl.Series("a", [1, 2, 3], pl.Int128)
        .to_frame()
        .lazy()
        .sum()
        .collect(engine="streaming")
        .item()
        == 6
    )
