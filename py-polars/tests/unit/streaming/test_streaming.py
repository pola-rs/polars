from __future__ import annotations

import os
import tempfile
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
    from polars.type_aliases import JoinStrategy

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


def test_tree_validation_streaming() -> None:
    # this query leads to a tree collection with an invalid branch
    # this test triggers the tree validation function.
    df_1 = pl.DataFrame(
        {
            "a": [22, 1, 1],
            "b": [500, 37, 20],
        },
    ).lazy()

    df_2 = pl.DataFrame(
        {"a": [23, 4, 20, 28, 3]},
    ).lazy()

    dfs = [df_2]
    cat = pl.concat(dfs, how="vertical")

    df_3 = df_1.select(
        [
            "a",
            # this expression is not allowed streaming, so it invalidates a branch
            pl.col("b")
            .filter(pl.col("a").min() > pl.col("a").rank())
            .alias("b_not_streaming"),
        ]
    ).join(
        cat,
        on=[
            "a",
        ],
    )

    out = df_1.join(df_3, on="a", how="left")
    assert out.collect(streaming=True).shape == (3, 3)


def test_streaming_apply(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    q = pl.DataFrame({"a": [1, 2]}).lazy()

    with pytest.warns(
        PolarsInefficientMapWarning, match="In this case, you can replace"
    ):
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
        .startswith("--- STREAMING")
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
        assert_frame_equal(q.collect(streaming=True), q.collect(streaming=False))


def test_streaming_9776() -> None:
    df = pl.DataFrame({"col_1": ["a"] * 1000, "ID": [None] + ["a"] * 999})
    ordered = (
        df.group_by("col_1", "ID", maintain_order=True)
        .count()
        .filter(pl.col("col_1") == "a")
    )
    unordered = (
        df.group_by("col_1", "ID", maintain_order=False)
        .count()
        .filter(pl.col("col_1") == "a")
    )
    expected = [("a", None, 1), ("a", "a", 999)]
    assert ordered.rows() == expected
    assert unordered.sort(["col_1", "ID"]).rows() == expected


@pytest.mark.write_disk()
def test_stream_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "in.parquet"
    schema = {
        "KLN_NR": pl.Utf8,
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
        [(pl.col("a") * 2).alias("foo"), (pl.col("a") * 3)]
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
            == agg_df.schema
            == {"x": pl.Int64, "max_y": pl.Boolean}
        )


def test_streaming_11219() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "c", None]})
    lf_other = pl.LazyFrame({"c": ["foo", "ham"]})
    lf_other2 = pl.LazyFrame({"c": ["foo", "ham"]})

    assert lf.with_context([lf_other, lf_other2]).select(
        pl.col("b") + pl.col("c").first()
    ).collect(streaming=True).to_dict(as_series=False) == {"b": ["afoo", "cfoo", None]}


@pytest.mark.write_disk()
def test_custom_temp_dir(monkeypatch: Any) -> None:
    test_temp_dir = "test_temp_dir"
    temp_dir = Path(tempfile.gettempdir()) / test_temp_dir

    monkeypatch.setenv("POLARS_VERBOSE", "1")
    monkeypatch.setenv("POLARS_FORCE_OOC", "1")
    monkeypatch.setenv("POLARS_TEMP_DIR", str(temp_dir))

    s = pl.arange(0, 100_000, eager=True).rename("idx")
    df = s.shuffle().to_frame()
    df.lazy().sort("idx").collect(streaming=True)

    assert os.listdir(temp_dir), f"Temp directory '{temp_dir}' is empty"
