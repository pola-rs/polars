from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.xdist_group("streaming")

ENGINES = ["in-memory", "streaming"]


def _frame(n: int = 20_000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.int_range(n, eager=True),
            "k": pl.int_range(n, eager=True) % 5_003,
            "g": pl.int_range(n, eager=True) % 97,
            "h": pl.int_range(n, eager=True) % 13,
            "x": (pl.int_range(n, eager=True) * 7919) % 1_000,
        }
    )


def _write_parts(df: pl.DataFrame, path: Path, parts: int = 8) -> pl.LazyFrame:
    size = df.height // parts + 1
    for i in range(parts):
        df.slice(i * size, size).write_parquet(path / f"{i:02}.parquet")
    return pl.scan_parquet(path / "*.parquet")


@pytest.mark.parametrize("engine", ENGINES)
def test_stacked_windows_over_group_by_evaluate_in_same_order(engine: str) -> None:
    q = (
        _frame()
        .lazy()
        .group_by("k")
        .agg(pl.col("g").first(), pl.col("h").first(), pl.col("x").sum())
        .with_columns(w1=pl.col("x").cum_sum().over("g"))
        .with_columns(w2=pl.col("w1").cum_sum().over("h"))
        .with_columns(w3=pl.col("x").cum_sum().over("g") + pl.col("w2") * 0)
    )
    out = q.collect(engine=engine)  # type: ignore[call-overload]
    assert_series_equal(out["w1"], out["w3"], check_names=False)


@pytest.mark.parametrize("engine", ENGINES)
def test_window_in_cached_branch_stays_aligned(engine: str) -> None:
    base = (
        _frame()
        .lazy()
        .group_by("k")
        .agg(pl.col("g").first(), pl.col("x").sum())
        .cache()
    )
    left = base.select("k", c=pl.col("x").cum_sum().over("g"))
    right = base.select(k2="k", g2="g", x2="x")
    out = pl.concat([left, right], how="horizontal").collect(engine=engine)  # type: ignore[call-overload]

    assert_series_equal(out["k"], out["k2"], check_names=False)
    expected = out.select(pl.col("x2").cum_sum().over("g2"))
    assert_series_equal(out["c"], expected.to_series(), check_names=False)


def test_window_with_order_dependent_key(tmp_path: Path) -> None:
    lf = _write_parts(_frame(), tmp_path)
    key = (pl.col("h") == 0).cum_sum()

    q = lf.with_columns(s=pl.col("x").sum().over(key))
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))

    q = q.group_by("s").agg(pl.col("id").sum()).sort("s")
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("reverse", [False, True])
def test_window_seeded_random_rank(reverse: bool) -> None:
    df = _frame().with_columns(pl.col("x") // 100)
    if reverse:
        df = df.reverse()

    q = df.lazy().with_columns(r=pl.col("x").rank("random", seed=1).over("g"))
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_seeded_random_rank_unordered_consumer(tmp_path: Path) -> None:
    lf = _write_parts(_frame().with_columns(pl.col("x") // 100), tmp_path)

    q = (
        lf.with_columns(r=pl.col("x").rank("random", seed=1).over("g"))
        .group_by("g", "r")
        .agg(pl.col("id").sum())
        .sort("g", "r")
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def _window_headers(q: pl.LazyFrame) -> list[str]:
    return [
        line.strip()
        for line in q.explain(engine="streaming").splitlines()
        if "WINDOW[" in line
    ]


def test_window_extracted_per_spec() -> None:
    lf = _frame().lazy()
    q = lf.with_columns(
        a=pl.col("x") - pl.col("x").mean().over("g"),
        b=pl.col("x").sum().over("g"),
        c=pl.col("x").cum_sum().over("g", order_by="h"),
        d=pl.col("x").rank().over(pl.col("h") + 1),
    )
    headers = _window_headers(q)
    assert len(headers) == 3
    assert "WINDOW[" not in q.explain(engine="in-memory")

    out = q.collect(engine="streaming")
    assert out.columns == ["id", "k", "g", "h", "x", "a", "b", "c", "d"]
    assert_frame_equal(out, q.collect(engine="in-memory"))


def test_window_nested_only_outer_extracted() -> None:
    lf = _frame().lazy()
    q = lf.select("id", w=pl.col("x").sum().over("h").sum().over("g"))
    headers = _window_headers(q)
    assert len(headers) == 1
    assert 'PARTITION BY ["g"]' in headers[0]
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize(
    ("expr", "extracted"),
    [
        (pl.col("x").sum().over("g"), True),
        (pl.col("x").sum().over("g") + pl.col("x"), True),
        (pl.col("x").sum().over("g").sum(), False),
        (pl.col("x").sum().over("g", mapping_strategy="join"), False),
        (pl.col("x").sum().over(pl.col("g").sum().over("h")), False),
        (pl.col("x").cum_sum().over(order_by="h"), False),
    ],
)
def test_window_extraction_scope(expr: pl.Expr, extracted: bool) -> None:
    q = _frame().lazy().select("id", w=expr)
    assert bool(_window_headers(q)) == extracted
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize(
    ("build", "maintain_order", "ordered_eval"),
    [
        (lambda lf: lf.with_columns(w=pl.col("x").mean().over("g")), True, False),
        (lambda lf: lf.with_columns(w=pl.col("x").cum_sum().over("g")), True, True),
        (
            lambda lf: (
                lf.with_columns(w=pl.col("x").mean().over("g"))
                .group_by("g")
                .agg(pl.col("w").sum())
            ),
            False,
            False,
        ),
        (
            lambda lf: (
                lf.with_columns(w=pl.col("x").cum_sum().over("g"))
                .group_by("g")
                .agg(pl.col("w").sum())
            ),
            False,
            True,
        ),
        (
            lambda lf: lf.with_columns(w=pl.col("x").cum_sum().over("g")).sort("h"),
            False,
            True,
        ),
        (
            lambda lf: lf.with_columns(w=pl.col("x").cum_sum().over("g")).sort(
                "h", maintain_order=True
            ),
            True,
            True,
        ),
    ],
)
def test_window_order_flags(
    build: Any, maintain_order: bool, ordered_eval: bool
) -> None:
    q = build(_frame().lazy())
    headers = _window_headers(q)
    assert len(headers) == 1
    flags = f"WINDOW[maintain_order: {str(maintain_order).lower()}, ordered_eval: {str(ordered_eval).lower()}]"
    assert headers[0].startswith(flags)
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=maintain_order,
    )


def _keyed_frame(n: int = 20_000) -> pl.DataFrame:
    i = pl.col("id")
    return pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        g=pl.when(i % 101 == 0).then(None).otherwise(i % 97),
        h=(i % 13).cast(pl.String),
        t=pl.when(i % 89 == 0).then(None).otherwise((i * 7919) % 50),
        x=((i * 104729) % 1_000).cast(pl.Float64),
    )


def _assert_unordered_window(q: pl.LazyFrame, n_nodes: int = 1) -> None:
    q = q.sort("x", "id")
    headers = _window_headers(q)
    assert len(headers) == n_nodes
    assert all("maintain_order: false" in h for h in headers)
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert dot.count("window[") == n_nodes
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g", order_by="t"),
        pl.col("x").implode().over("g", order_by="t"),
        pl.lit(1).over("g", order_by="t"),
        pl.col("x").cum_sum().over("g"),
        pl.col("x").cum_sum().over("g", order_by="t"),
        pl.col("x").cum_sum().over("g", order_by="t", descending=True),
        pl.col("x").cum_sum().over("g", order_by="t", nulls_last=True),
        pl.col("x").cum_sum().over(["g", "h"], order_by=["t", "x"]),
        pl.col("x").shift(1).over("h", order_by="t"),
        pl.col("x").rank().over("g"),
        pl.col("x").rank("ordinal").over("g", order_by="t"),
        pl.col("t").rank("dense").over("h"),
        (pl.col("x") + 1).over("g"),
        (pl.col("x") - pl.col("x").mean()).over("g", order_by="t"),
        pl.col("x").sort().over("g"),
        pl.col("x").reverse().over("g"),
        pl.int_range(pl.len()).over("g", order_by="t"),
        pl.col("t").cum_count().over("g"),
    ],
)
def test_window_unordered_results(expr: pl.Expr) -> None:
    _assert_unordered_window(_keyed_frame().lazy().with_columns(w=expr))


def test_window_unordered_several_specs() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .with_columns(
            a=pl.col("x").cum_sum().over("g"),
            b=pl.col("x").shift().over("g"),
            c=pl.col("x").rank().over("h"),
        )
    )
    _assert_unordered_window(q, n_nodes=2)


def test_window_unordered_empty() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .filter(pl.col("id") < 0)
        .with_columns(w=pl.col("x").cum_sum().over("g"))
    )
    _assert_unordered_window(q)
    assert q.sort("x", "id").collect(engine="streaming").schema["w"] == pl.Float64


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").filter(pl.col("x") > 500).over("g", order_by="t"),
        pl.col("x").head(1).over("g", order_by="t"),
    ],
)
def test_window_unordered_shape_error(expr: pl.Expr) -> None:
    q = _keyed_frame().lazy().with_columns(w=expr).sort("x", "id")
    assert len(_window_headers(q)) == 1
    for engine in ENGINES:
        with pytest.raises(pl.exceptions.ShapeError):
            q.collect(engine=engine)  # type: ignore[call-overload]


def _physical_windows(q: pl.LazyFrame) -> list[str]:
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    return [line for line in dot.splitlines() if "window[" in line]


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").rank().over("g"),
        pl.col("x").cum_sum().over("g", order_by="t"),
        pl.col("x").cum_sum().over("g"),
        pl.col("x").shift(2).over(["g", "h"], order_by=["t", "x"]),
        (pl.col("x") + 1).over("g"),
        pl.col("x").sum().over("g", order_by="t"),
    ],
)
def test_window_row_index_mode(expr: pl.Expr) -> None:
    q = _keyed_frame().lazy().with_columns(w=expr)
    windows = _physical_windows(q)
    assert len(windows) == 1
    assert "maintain_order: true" in windows[0]
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_row_index_mode_over_group_by() -> None:
    gb = (
        _frame()
        .lazy()
        .group_by("k")
        .agg(pl.col("g").first(), pl.col("x").sum())
        .cache()
    )
    q = pl.concat(
        [
            gb.select(k2="k", g2="g", x2="x"),
            gb.with_columns(w=pl.col("x").cum_sum().over("g")),
        ],
        how="horizontal",
    )
    windows = _physical_windows(q)
    assert len(windows) == 1
    assert "maintain_order: true" in windows[0]

    out = q.collect(engine="streaming")
    assert_series_equal(out["k"], out["k2"], check_names=False)
    expected = out.select(pl.col("x2").cum_sum().over("g2")).to_series()
    assert_series_equal(out["w"], expected, check_names=False)


def test_window_before_stable_sort() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over("g"))
        .sort("t", maintain_order=True)
    )
    assert "maintain_order: true" in _physical_windows(q)[0]
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_after_sort() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .sort("t", "id")
        .with_columns(w=pl.col("x").cum_sum().over("g"))
    )
    assert "SORT BY" in q.explain(engine="streaming")
    assert "maintain_order: true" in _physical_windows(q)[0]
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_row_index_mode_several_specs() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .with_columns(
            a=pl.col("x").cum_sum().over("g"),
            b=pl.col("x").rank().over("g"),
            c=pl.col("x").cum_sum().over("h", order_by="t"),
        )
    )
    windows = _physical_windows(q)
    assert len(windows) == 2
    assert all("maintain_order: true" in w for w in windows)
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_sql_window_peers_any_order() -> None:
    lf = _keyed_frame().lazy().with_columns(pl.col("t").fill_null(0))
    ctx = pl.SQLContext(t=lf)
    q = ctx.execute(
        """
        SELECT g, MAX(c) AS c, SUM(x) AS x FROM (
            SELECT g, x, SUM(x) OVER (PARTITION BY g ORDER BY t) AS c FROM t
        ) GROUP BY g
        """
    )
    headers = _window_headers(q)
    assert len(headers) == 1
    assert headers[0].startswith("WINDOW[maintain_order: false, ordered_eval: false]")

    out = q.collect(engine="streaming")
    assert_series_equal(out["c"], out["x"], check_names=False)

    q = ctx.execute(
        """
        SELECT id, SUM(x) OVER (PARTITION BY g ORDER BY t, id) AS c FROM t ORDER BY x, id
        """
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_dataframe_window_ties_in_input_order() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .with_columns(c=pl.col("x").cum_sum().over("g", order_by="t"))
        .group_by("g")
        .agg(pl.col("c").sort())
    )
    headers = _window_headers(q)
    assert headers[0].startswith("WINDOW[maintain_order: false, ordered_eval: true]")
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=False,
    )
