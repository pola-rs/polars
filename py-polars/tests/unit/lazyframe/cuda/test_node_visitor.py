from __future__ import annotations

import json
import time
from datetime import datetime
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

import pytest

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

    with pytest.deprecated_call():
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


def _collect_ewm_functions(query: pl.LazyFrame) -> list[tuple[Any, list[Any]]]:
    """Traverse a query's IR; return (Function, resolved input nodes) for ewm exprs."""
    results: list[tuple[Any, list[Any]]] = []

    def callback(node_traverser: Any, query_start: int | None) -> None:
        for expr_ir in node_traverser.get_exprs():
            expr_node = node_traverser.view_expression(expr_ir.node)
            if isinstance(expr_node, _expr_nodes.Function):
                function_data = expr_node.function_data
                if function_data and isinstance(
                    function_data[0], _expr_nodes.EwmFunction
                ):
                    inputs = [
                        node_traverser.view_expression(i) for i in expr_node.input
                    ]
                    results.append((expr_node, inputs))

    query.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=callback  # type: ignore[call-overload]
    )
    return results


def test_ewm_mean_expr_visitor() -> None:
    """Test that ewm_mean is exposed with its EWMOptions fields."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").ewm_mean(alpha=0.5).alias("ewm_mean"),
    )
    ewm_exprs = _collect_ewm_functions(q)
    assert len(ewm_exprs) == 1
    fn, _inputs = ewm_exprs[0]
    name, alpha, adjust, bias, min_periods, ignore_nulls = fn.function_data
    assert name == _expr_nodes.EwmFunction.Mean
    assert alpha == 0.5
    # `ewm_mean` has no `bias` kwarg; it is always serialized as False.
    assert adjust is True
    assert bias is False
    assert min_periods == 1
    assert ignore_nulls is False


def test_ewm_std_expr_visitor() -> None:
    """Test that ewm_std serializes all five EWMOptions fields, incl. non-defaults."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x")
        .ewm_std(alpha=0.3, adjust=False, bias=True, min_samples=2, ignore_nulls=True)
        .alias("ewm_std"),
    )
    ewm_exprs = _collect_ewm_functions(q)
    assert len(ewm_exprs) == 1
    fn, _inputs = ewm_exprs[0]
    name, alpha, adjust, bias, min_periods, ignore_nulls = fn.function_data
    assert name == _expr_nodes.EwmFunction.Std
    assert alpha == pytest.approx(0.3)
    assert adjust is False
    assert bias is True
    assert min_periods == 2
    assert ignore_nulls is True


def test_ewm_var_expr_visitor() -> None:
    """Test that ewm_var is exposed and distinguished from ewm_std by its enum."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").ewm_var(alpha=0.3, bias=False).alias("ewm_var"),
    )
    ewm_exprs = _collect_ewm_functions(q)
    assert len(ewm_exprs) == 1
    fn, _inputs = ewm_exprs[0]
    name, alpha, _adjust, bias, _min_periods, _ignore_nulls = fn.function_data
    assert name == _expr_nodes.EwmFunction.Var
    assert alpha == pytest.approx(0.3)
    assert bias is False


def test_ewm_mean_span_expr_visitor() -> None:
    """Test that only the derived alpha (not span) is visible."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").ewm_mean(span=3).alias("ewm_mean"),
    )
    ewm_exprs = _collect_ewm_functions(q)
    assert len(ewm_exprs) == 1
    fn, _inputs = ewm_exprs[0]
    name, alpha, *_ = fn.function_data
    assert name == _expr_nodes.EwmFunction.Mean
    # span=3 -> alpha = 2 / (span + 1)
    assert alpha == pytest.approx(2.0 / (3.0 + 1.0))


def test_ewm_mean_by_expr_visitor() -> None:
    """Test that ewm_mean_by exposes its half_life Duration and the `by` input."""
    q = pl.LazyFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "t": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
        }
    ).with_columns(
        pl.col("x").ewm_mean_by(by="t", half_life="1w2d3h4m5s6ms").alias("ewm_mean_by"),
    )
    ewm_exprs = _collect_ewm_functions(q)
    assert len(ewm_exprs) == 1
    fn, inputs = ewm_exprs[0]
    # Two inputs: the values column, then the `by` column second.
    assert len(inputs) == 2
    by_node = inputs[1]
    assert isinstance(by_node, _expr_nodes.Column)
    assert by_node.name == "t"
    name, half_life = fn.function_data
    assert name == _expr_nodes.EwmFunction.MeanBy
    # Wrap<Duration> 6-tuple: (months, weeks, days, nanoseconds, parsed_int, negative)
    expected_ns = (3 * 3600 + 4 * 60 + 5) * 1_000_000_000 + 6 * 1_000_000
    assert half_life == (0, 1, 2, expected_ns, False, False)


def test_ewm_mean_by_parsed_int_expr_visitor() -> None:
    """Test the parsed_int path of the half_life Duration tuple (the `i` unit)."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0], "t": [1, 2, 3]}).with_columns(
        pl.col("x").ewm_mean_by(by="t", half_life="2i").alias("ewm_mean_by"),
    )
    ewm_exprs = _collect_ewm_functions(q)
    assert len(ewm_exprs) == 1
    fn, _inputs = ewm_exprs[0]
    name, half_life = fn.function_data
    assert name == _expr_nodes.EwmFunction.MeanBy
    assert half_life == (0, 0, 0, 2, True, False)
