from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

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
    build: Callable[[pl.LazyFrame], pl.LazyFrame],
    maintain_order: bool,
    ordered_eval: bool,
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


def _physical_windows(q: pl.LazyFrame) -> list[str]:
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    return [line for line in dot.splitlines() if "window[" in line]


def _assert_unordered_physical_windows(q: pl.LazyFrame, n_nodes: int = 1) -> None:
    windows = _physical_windows(q)
    assert len(windows) == n_nodes
    assert all("maintain_order: false" in w for w in windows)


def _assert_unordered_window(q: pl.LazyFrame, n_nodes: int = 1) -> None:
    q = q.sort("x", "id")
    headers = _window_headers(q)
    assert len(headers) == n_nodes
    assert all("maintain_order: false" in h for h in headers)
    _assert_unordered_physical_windows(q, n_nodes)
    assert "zip" not in q.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True
    )
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


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").cum_sum().over("g", order_by="t"),
        pl.col("x").rank().over("g"),
        pl.col("x").shift(1).over("h", order_by="t"),
        (pl.col("x") - pl.col("x").mean()).over("g", order_by="t"),
        pl.col("x").sort().over("g"),
    ],
)
@pytest.mark.parametrize("n", [0, 20_000])
def test_window_unordered_reads_all_columns(expr: pl.Expr, n: int) -> None:
    columns = list(dict.fromkeys(expr.meta.root_names()))
    q = _keyed_frame(n).lazy().select(columns).with_columns(w=expr).sort(pl.all())
    windows = _physical_windows(q)
    assert len(windows) == 1
    assert "maintain_order: false" in windows[0]
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


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


def test_window_scalar_order_key() -> None:
    lf = _keyed_frame().lazy()
    q = lf.with_columns(w=pl.col("x").cum_sum().over("g", order_by=pl.col("t").sum()))
    expected = lf.with_columns(w=pl.col("x").cum_sum().over("g")).collect()
    for engine in ENGINES:
        assert_frame_equal(q.collect(engine=engine), expected)  # type: ignore[call-overload]


def test_window_repeated_partition_keys() -> None:
    q = (
        _keyed_frame()
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over(["g", "g"], order_by="t"))
    )
    assert len(_physical_windows(q)) == 1
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


class _Key:
    def __init__(self, value: int) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Key) and self.value == other.value


def test_window_object_partition_key() -> None:
    df = _keyed_frame(1_000).with_columns(
        o=pl.Series([_Key(i % 7) for i in range(1_000)], dtype=pl.Object)
    )
    q = df.lazy().with_columns(w=pl.col("x").shift().over("o")).select("id", "w")
    assert _physical_windows(q) == []
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


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


@pytest.mark.parametrize(
    ("predicate", "extracted"),
    [
        (
            pl.int_range(pl.len()).over("g", order_by=["t", "id"], descending=True)
            == 0,
            True,
        ),
        ((pl.col("x") > pl.col("x").mean().over("g")) & (pl.col("id") % 3 != 0), True),
        (pl.col("x").cum_sum().over("h", order_by="t") < 5_000, True),
        (pl.col("x").sum().over("g").sum() > 0, False),
    ],
)
def test_window_in_filter(predicate: pl.Expr, extracted: bool) -> None:
    q = _keyed_frame().lazy().filter(predicate)
    graph = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert ("window[" in graph) == extracted
    if extracted:
        assert "in-memory-map" not in graph


@pytest.mark.parametrize(
    ("order_by", "descending", "nulls_last", "header"),
    [
        (["t", "h"], False, False, 'ORDER BY "t", "h"'),
        (["t", "h"], True, True, 'ORDER BY "t" DESC NULLS LAST, "h" DESC NULLS LAST'),
        ([pl.col("t") * 2, "h"], False, True, '"h" NULLS LAST'),
    ],
)
def test_window_order_by_keys_not_encoded(
    order_by: list[str | pl.Expr], descending: bool, nulls_last: bool, header: str
) -> None:
    w = pl.int_range(pl.len()).over(
        "g", order_by=order_by, descending=descending, nulls_last=nulls_last
    )
    q = _keyed_frame().lazy().with_columns(w=w, c=pl.col("x").cum_sum().over("g"))
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert not any(
        "row_encode" in line for line in dot.splitlines() if "window[" not in line
    )
    assert any(header in h for h in _window_headers(q))
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize(
    "order_by",
    [
        [pl.col("t"), pl.lit(1)],
        [pl.col("t").shuffle(), pl.col("t").shuffle()],
    ],
)
def test_window_order_by_stays_encoded(order_by: list[pl.Expr]) -> None:
    w = pl.int_range(pl.len()).over("g", order_by=order_by)
    q = _keyed_frame().lazy().with_columns(w=w)
    order_by_names = _window_headers(q)[0].split(" ORDER BY ")[1]
    assert order_by_names.startswith('"_POLARS_TMP_PLAN_')
    assert "," not in order_by_names
    q.collect(engine="streaming")


def test_sql_window_order_by_mixed_directions() -> None:
    ctx = pl.SQLContext(t=_keyed_frame().lazy())
    q = ctx.execute(
        """
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY g ORDER BY t DESC NULLS LAST, h ASC NULLS FIRST, id DESC
        ) AS rn FROM t
        """
    )
    assert 'ORDER BY "t" DESC NULLS LAST, "h", "id" DESC' in _window_headers(q)[0]
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


SPILL_ENV = {
    "POLARS_OOC_MEMORY_BUDGET_MB": "0",
    "POLARS_OOC_SPILL_MIN_BYTES": "1",
    "POLARS_OOC_DRIFT_THRESHOLD": "0",
}


@pytest.fixture(
    params=[
        "at_once",
        "stored",
        pytest.param("spilled", marks=[pytest.mark.write_disk, pytest.mark.slow]),
    ]
)
def window_mode(
    request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> str:
    """Makes small inputs take the paths of large ones."""
    mode = str(request.param)
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "32")
    plmonkeypatch.setenv("POLARS_WINDOW_MIN_ROWS_PER_PARTITION", "1")
    if mode != "at_once":
        num_partitions = 4 * pl.thread_pool_size() + 3
        plmonkeypatch.setenv("POLARS_WINDOW_NUM_PARTITIONS", str(num_partitions))
        plmonkeypatch.setenv("POLARS_WINDOW_BATCH_BYTES", "4096")
        plmonkeypatch.setenv("POLARS_WINDOW_RANGE_BYTES", "4096")
    if mode == "spilled":
        for name, value in SPILL_ENV.items():
            plmonkeypatch.setenv(name, value)
        plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))
    return mode


MODE_EXPRS = [
    pl.col("x").sum().over("g", order_by="t"),
    pl.col("x").cum_sum().over("g"),
    pl.col("x").cum_sum().over("g", order_by="t"),
    pl.col("x").cum_sum().over(["g", "h"], order_by=["t", "x"]),
    pl.col("x").rank("ordinal").over("g", order_by="t"),
    pl.col("h").sort().over("g"),
    pl.col("x").implode().over("g", order_by="t"),
    pl.int_range(pl.len()).over("g", order_by="t"),
]


@pytest.mark.parametrize("expr", MODE_EXPRS)
def test_window_mode_ordered(window_mode: str, expr: pl.Expr) -> None:
    q = _keyed_frame(2000).lazy().with_columns(w=expr)
    assert _physical_windows(q)
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("expr", MODE_EXPRS)
def test_window_mode_unordered(window_mode: str, expr: pl.Expr) -> None:
    q = _keyed_frame(2000).lazy().with_columns(w=expr).sort("x", "id")
    _assert_unordered_physical_windows(q)
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("expr", MODE_EXPRS)
def test_window_mode_partitions(window_mode: str, expr: pl.Expr) -> None:
    columns = list(dict.fromkeys(expr.meta.root_names()))
    q = _keyed_frame(2000).lazy().select(columns).with_columns(w=expr)
    q = q.sort(pl.all().exclude("w"), "w")
    _assert_unordered_physical_windows(q)
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_several_specs(window_mode: str) -> None:
    q = (
        _keyed_frame(2000)
        .lazy()
        .with_columns(
            a=pl.col("x").cum_sum().over("g", order_by="t"),
            b=pl.col("x").shift().over("g"),
            c=pl.col("x").rank().over("h"),
        )
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_empty(window_mode: str) -> None:
    q = (
        _keyed_frame(2000)
        .lazy()
        .filter(pl.col("id") < 0)
        .with_columns(w=pl.col("x").cum_sum().over("g", order_by="t"))
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_one_partition_key(window_mode: str) -> None:
    q = (
        _keyed_frame(2000)
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over(pl.lit(1), order_by="t"))
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_one_partition_interleaved_keys(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_WINDOW_NUM_PARTITIONS", "1")
    df = pl.DataFrame({"g": ["a", "b", "a", "b"], "v": [10, 20, 30, 40]})
    out = df.lazy().with_columns(w=pl.col("v").cum_sum().over("g"))
    assert out.collect(engine="streaming")["w"].to_list() == [10, 20, 40, 60]


@pytest.mark.parametrize("key", [pl.col("g"), pl.col("id") * 0])
def test_window_resplit_large_partitions(
    window_mode: str, key: pl.Expr, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_WINDOW_NUM_PARTITIONS", "2")
    plmonkeypatch.setenv("POLARS_WINDOW_MAX_PARTITION_BYTES", "2000")
    q = (
        _keyed_frame(2000)
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over(key, order_by="t"))
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_repeated_input_seqs(window_mode: str) -> None:
    # A cross join gives several morsels the same sequence number.
    left = pl.LazyFrame({"a": pl.int_range(40, eager=True)})
    right = pl.LazyFrame({"b": pl.int_range(50, eager=True)})
    q = left.join(right, how="cross", maintain_order="left_right").with_columns(
        w=pl.col("b").cum_sum().over(pl.col("a") % 7)
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_error_in_partition(window_mode: str) -> None:
    n = 2000
    s = pl.Series("s", [str(j) if j < n - 3 else "x" for j in range(n)])
    q = (
        _keyed_frame(n)
        .with_columns(s)
        .lazy()
        .with_columns(w=pl.col("s").cast(pl.Int64).cum_sum().over("g"))
    )
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="streaming")


def test_window_mode_resume_over_join(
    window_mode: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    # The join pauses its inputs after sampling a few rows.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "16")
    lf = (
        _keyed_frame(2000)
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over("g", order_by="t"))
    )
    other = pl.LazyFrame({"h": [str(i) for i in range(13)], "extra": range(13)})
    q = lf.join(other, on="h", how="inner", maintain_order="left")
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_resume_over_horizontal_concat(window_mode: str) -> None:
    lf = (
        _keyed_frame(2000)
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over("g", order_by="t"))
    )
    other = pl.LazyFrame({"extra": pl.int_range(2000, eager=True)})
    q = pl.concat([lf, other], how="horizontal")
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_resume_over_multiplexer(window_mode: str) -> None:
    lf = (
        _keyed_frame(2000)
        .lazy()
        .with_columns(w=pl.col("x").cum_sum().over("g", order_by="t"))
        .with_row_index("rn")
    )
    q = lf.join(lf.select("rn", w2="w"), on="rn", maintain_order="left")
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_window_mode_empty_morsels(
    window_mode: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_WINDOW_RANGE_BYTES", "1")
    q = (
        _keyed_frame(2000)
        .lazy()
        .map_batches(lambda df: df.filter(pl.col("id") >= 1000), streamable=True)
        .with_columns(w=pl.col("x").cum_sum().over("g", order_by="t"))
    )
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))
