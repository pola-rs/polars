from __future__ import annotations

import json
import time
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

import polars as pl
from polars._plr import _expr_nodes, _ir_nodes  # type: ignore[attr-defined]
from polars._utils.wrap import wrap_df
from tests.unit.io.conftest import format_file_uri

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd


class Timer:
    """Simple-minded timing of nodes."""

    def __init__(self, start: int | None) -> None:
        self.start = start
        self.timings: list[tuple[int, int, str]] = []

    def record(self, fn: Callable[[], pd.DataFrame], name: str) -> pd.DataFrame:
        start = time.monotonic_ns()
        result = fn()
        end = time.monotonic_ns()
        if self.start is not None:
            self.timings.append((start - self.start, end - self.start, name))
        return result


def test_run_on_pandas() -> None:
    # Simple join example, missing multiple columns, slices, etc.
    def join(
        inputs: list[Callable[[], pd.DataFrame]],
        obj: Any,
        _node_traverser: Any,
        timer: Timer,
    ) -> Callable[[], pd.DataFrame]:
        assert len(obj.left_on) == 1
        assert len(obj.right_on) == 1
        left_on = obj.left_on[0].output_name
        right_on = obj.right_on[0].output_name

        assert len(inputs) == 2

        def run(inputs: list[Callable[[], pd.DataFrame]]) -> pd.DataFrame:
            # materialize inputs
            dataframes = [call() for call in inputs]
            return timer.record(
                lambda: dataframes[0].merge(
                    dataframes[1], left_on=left_on, right_on=right_on
                ),
                "pandas-join",
            )

        return partial(run, inputs)

    # Simple scan example, missing predicates, columns pruning, slices, etc.
    def df_scan(
        _inputs: None, obj: Any, _: Any, timer: Timer
    ) -> Callable[[], pd.DataFrame]:
        assert obj.selection is None
        return lambda: timer.record(lambda: wrap_df(obj.df).to_pandas(), "pandas-scan")

    @lru_cache(1)
    def get_node_converters() -> dict[
        type, Callable[[Any, Any, Any, Timer], Callable[[], pd.DataFrame]]
    ]:
        return {
            _ir_nodes.Join: join,
            _ir_nodes.DataFrameScan: df_scan,
        }

    def get_input(node_traverser: Any, *, timer: Timer) -> Callable[[], pd.DataFrame]:
        current_node = node_traverser.get_node()

        inputs_callable = []
        for inp in node_traverser.get_inputs():
            node_traverser.set_node(inp)
            inputs_callable.append(get_input(node_traverser, timer=timer))

        node_traverser.set_node(current_node)
        ir_node = node_traverser.view_current_node()
        return get_node_converters()[ir_node.__class__](
            inputs_callable, ir_node, node_traverser, timer
        )

    def run_on_pandas(node_traverser: Any, query_start: int | None) -> None:
        timer = Timer(
            time.monotonic_ns() - query_start if query_start is not None else None
        )
        current_node = node_traverser.get_node()

        callback = get_input(node_traverser, timer=timer)

        def run_callback(
            columns: list[str] | None,
            _: Any,
            n_rows: int | None,
            should_time: bool,
        ) -> pl.DataFrame | tuple[pl.DataFrame, list[tuple[int, int, str]]]:
            assert n_rows is None
            assert columns is None

            # produce a wrong result to ensure the callback has run.
            result = pl.from_pandas(callback() * 2)
            if should_time:
                return result, timer.timings
            else:
                return result

        node_traverser.set_node(current_node)
        node_traverser.set_udf(run_callback)

    # Polars query that will run on pandas
    q1 = pl.LazyFrame({"foo": [1, 2, 3]})
    q2 = pl.LazyFrame({"foo": [1], "bar": [2]})
    q = q1.join(q2, on="foo")
    assert q.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=run_on_pandas  # type: ignore[call-overload]
    ).to_dict(as_series=False) == {
        "foo": [2],
        "bar": [4],
    }

    result, timings = q.profile(post_opt_callback=run_on_pandas)
    assert result.to_dict(as_series=False) == {
        "foo": [2],
        "bar": [4],
    }
    assert timings["node"].to_list() == [
        "optimization",
        "pandas-scan",
        "pandas-scan",
        "pandas-join",
    ]


def test_path_uri_to_python_conversion_22766(tmp_path: Path) -> None:
    path = format_file_uri(f"{tmp_path / 'data.parquet'}")

    df = pl.DataFrame({"x": 1})
    df.write_parquet(path)

    q = pl.scan_parquet(path)

    out: list[str] = q._ldf.visit().view_current_node().paths
    assert len(out) == 1

    assert out[0].startswith("file://")
    assert out == [path]


def test_node_traverse_sink(tmp_path: Path) -> None:
    def callback(node_traverser: Any, query_start: int | None) -> None:
        assert list(json.loads(node_traverser.view_current_node().payload)["File"]) == [
            "target",
            "file_format",
            "unified_sink_args",
        ]

    q = pl.LazyFrame({"x": [0, 1, 2]}).sink_parquet(tmp_path / "a", lazy=True)
    q.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=callback  # type: ignore[call-overload]
    )


def _collect_rolling_function_data(
    query: pl.LazyFrame,
) -> list[tuple[Any, ...]]:
    """Traverse a query's IR and return function_data tuples for rolling expressions."""
    results: list[tuple[Any, ...]] = []

    def callback(node_traverser: Any, query_start: int | None) -> None:
        for expr_ir in node_traverser.get_exprs():
            expr_node = node_traverser.view_expression(expr_ir.node)
            if isinstance(expr_node, _expr_nodes.Function):
                name, *options = expr_node.function_data
                if isinstance(name, _expr_nodes.RollingFunction):
                    results.append((name, *options))

    query.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=callback  # type: ignore[call-overload]
    )
    return results


def test_rolling_expr_visitor() -> None:
    """Test that fixed-size rolling expressions are exposed via the visitor."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").rolling_sum(window_size=3).alias("rolling_sum"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, window_size, min_periods, weights, center, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Sum
    assert window_size == 3
    assert min_periods == 3
    assert weights is None
    assert center is False
    assert fn_params == ()


def test_rolling_expr_visitor_var() -> None:
    """Test that rolling_var serializes ddof in fn_params."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").rolling_var(window_size=3, ddof=2).alias("rolling_var"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, window_size, min_periods, weights, center, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Var
    assert window_size == 3
    assert min_periods == 3
    assert weights is None
    assert center is False
    assert fn_params == (2,)


def test_rolling_expr_visitor_min_centered() -> None:
    """Test rolling_min with center=True."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").rolling_min(window_size=3, center=True).alias("rmin"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, window_size, _, _, center, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Min
    assert window_size == 3
    assert center is True
    assert fn_params == ()


def test_rolling_expr_visitor_quantile() -> None:
    """Test that rolling_quantile serializes probability and method."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x")
        .rolling_quantile(0.75, window_size=3, interpolation="linear")
        .alias("rq"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, window_size, _, _, _, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Quantile
    assert window_size == 3
    assert fn_params == (0.75, "linear")


def test_rolling_expr_visitor_std() -> None:
    """Test that rolling_std serializes ddof in fn_params."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").rolling_std(window_size=3, ddof=0).alias("rstd"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, _, _, _, _, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Std
    assert fn_params == (0,)


def test_rolling_expr_visitor_skew() -> None:
    """Test that rolling_skew serializes the bias parameter."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").rolling_skew(window_size=3, bias=False).alias("rskew"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, _, _, _, _, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Skew
    assert fn_params == (False,)


def test_rolling_expr_visitor_kurtosis() -> None:
    """Test that rolling_kurtosis serializes fisher and bias parameters."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x")
        .rolling_kurtosis(window_size=3, fisher=False, bias=True)
        .alias("rkurt"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, _, _, _, _, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Kurtosis
    assert fn_params == (False, True)


def test_rolling_expr_visitor_rank() -> None:
    """Test that rolling_rank serializes method and seed parameters."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").rolling_rank(window_size=3, method="dense", seed=42).alias("rrank"),
    )
    rolling_exprs = _collect_rolling_function_data(q)
    assert len(rolling_exprs) == 1
    name, _, _, _, _, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunction.Rank
    assert fn_params == ("dense", 42)
