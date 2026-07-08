from __future__ import annotations

import json
import time
from datetime import datetime
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars._plr import _expr_nodes, _ir_nodes  # type: ignore[attr-defined]
from polars._utils.wrap import wrap_df, wrap_expr
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


def _collect_ewm_function_data(
    query: pl.LazyFrame,
) -> list[tuple[Any, ...]]:
    """Traverse a query's IR and return function_data tuples for ewm expressions."""
    results: list[tuple[Any, ...]] = []

    def callback(node_traverser: Any, query_start: int | None) -> None:
        for expr_ir in node_traverser.get_exprs():
            expr_node = node_traverser.view_expression(expr_ir.node)
            if isinstance(expr_node, _expr_nodes.Function):
                name, *options = expr_node.function_data
                if isinstance(name, _expr_nodes.EwmFunction):
                    results.append((name, *options))

    query.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=callback  # type: ignore[call-overload]
    )
    return results


def test_ewm_mean_expr_visitor() -> None:
    """Test that ewm_mean is exposed with its EWMOptions fields."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").ewm_mean(alpha=0.5).alias("ewm_mean"),
    )
    ewm_exprs = _collect_ewm_function_data(q)
    assert len(ewm_exprs) == 1
    name, alpha, adjust, bias, min_periods, ignore_nulls = ewm_exprs[0]
    assert name == _expr_nodes.EwmFunction.Mean
    assert hash(name) == hash(_expr_nodes.EwmFunction.Mean)
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
    ewm_exprs = _collect_ewm_function_data(q)
    assert len(ewm_exprs) == 1
    name, alpha, adjust, bias, min_periods, ignore_nulls = ewm_exprs[0]
    assert name == _expr_nodes.EwmFunction.Std
    assert alpha == 0.3
    assert adjust is False
    assert bias is True
    assert min_periods == 2
    assert ignore_nulls is True


def test_ewm_var_expr_visitor() -> None:
    """Test that ewm_var is exposed and distinguished from ewm_std by its enum."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").ewm_var(alpha=0.3, bias=False).alias("ewm_var"),
    )
    ewm_exprs = _collect_ewm_function_data(q)
    assert len(ewm_exprs) == 1
    name, alpha, _, bias, _, _ = ewm_exprs[0]
    assert name == _expr_nodes.EwmFunction.Var
    assert alpha == 0.3
    assert bias is False


def test_ewm_mean_span_expr_visitor() -> None:
    """Test that only the derived alpha (not span) is visible."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).with_columns(
        pl.col("x").ewm_mean(span=3).alias("ewm_mean"),
    )
    ewm_exprs = _collect_ewm_function_data(q)
    assert len(ewm_exprs) == 1
    name, alpha, _, _, _, _ = ewm_exprs[0]
    assert name == _expr_nodes.EwmFunction.Mean
    # span=3 -> alpha = 2 / (span + 1)
    assert alpha == 2.0 / (3.0 + 1.0)


def test_ewm_mean_by_expr_visitor() -> None:
    """Test that ewm_mean_by exposes its half_life Duration."""
    q = pl.LazyFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "t": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
        }
    ).with_columns(
        pl.col("x").ewm_mean_by(by="t", half_life="1w2d3h4m5s6ms").alias("ewm_mean_by"),
    )
    ewm_exprs = _collect_ewm_function_data(q)
    assert len(ewm_exprs) == 1
    name, half_life = ewm_exprs[0]
    assert name == _expr_nodes.EwmFunction.MeanBy
    # Wrap<Duration> 6-tuple: (months, weeks, days, nanoseconds, parsed_int, negative)
    expected_ns = (3 * 3600 + 4 * 60 + 5) * 1_000_000_000 + 6 * 1_000_000
    assert half_life == (0, 1, 2, expected_ns, False, False)


def test_ewm_mean_by_parsed_int_expr_visitor() -> None:
    """Test the parsed_int path of the half_life Duration tuple (the `i` unit)."""
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0], "t": [1, 2, 3]}).with_columns(
        pl.col("x").ewm_mean_by(by="t", half_life="2i").alias("ewm_mean_by"),
    )
    ewm_exprs = _collect_ewm_function_data(q)
    assert len(ewm_exprs) == 1
    name, half_life = ewm_exprs[0]
    assert name == _expr_nodes.EwmFunction.MeanBy
    assert half_life == (0, 0, 0, 2, True, False)


def _collect_rolling_by_function_data(
    query: pl.LazyFrame,
) -> list[tuple[Any, ...]]:
    """Traverse a query's IR; return function_data tuples for rolling ``*_by`` exprs."""
    results: list[tuple[Any, ...]] = []

    def callback(node_traverser: Any, query_start: int | None) -> None:
        for expr_ir in node_traverser.get_exprs():
            expr_node = node_traverser.view_expression(expr_ir.node)
            if isinstance(expr_node, _expr_nodes.Function):
                name, *options = expr_node.function_data
                if isinstance(name, _expr_nodes.RollingFunctionBy):
                    results.append((name, *options))

    query.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=callback  # type: ignore[call-overload]
    )
    return results


def test_rolling_mean_by_expr_visitor() -> None:
    """Test that rolling_mean_by exposes its Duration window and closed_window."""
    q = pl.LazyFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "t": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        }
    ).with_columns(
        pl.col("x").rolling_mean_by("t", window_size="2h").alias("rmean_by"),
    )
    rolling_exprs = _collect_rolling_by_function_data(q)
    assert len(rolling_exprs) == 1
    name, window_size, min_periods, closed, fn_params = rolling_exprs[0]
    assert name == _expr_nodes.RollingFunctionBy.MeanBy
    assert hash(name) == hash(_expr_nodes.RollingFunctionBy.MeanBy)
    # Wrap<Duration> 6-tuple: (months, weeks, days, nanoseconds, parsed_int, negative)
    assert window_size == (0, 0, 0, 2 * 3600 * 1_000_000_000, False, False)
    assert min_periods == 1
    assert closed == "right"
    assert fn_params == ()


@pytest.mark.parametrize(
    ("expr", "expected_name", "expected_fn_params"),
    [
        (pl.col("x").rolling_min_by("t", "2h"), "MinBy", ()),
        (pl.col("x").rolling_max_by("t", "2h"), "MaxBy", ()),
        (pl.col("x").rolling_sum_by("t", "2h"), "SumBy", ()),
        (pl.col("x").rolling_std_by("t", "2h"), "StdBy", (1,)),
        (pl.col("x").rolling_var_by("t", "2h", ddof=2), "VarBy", (2,)),
        (
            pl.col("x").rolling_quantile_by("t", "2h", quantile=0.25),
            "QuantileBy",
            (0.25, "nearest"),
        ),
        (pl.col("x").rolling_rank_by("t", "2h"), "RankBy", ("average", None)),
    ],
)
def test_rolling_by_variant_fn_params(
    expr: pl.Expr, expected_name: str, expected_fn_params: tuple[Any, ...]
) -> None:
    """Each rolling ``*_by`` variant exposes its enum discriminant and fn_params."""
    q = pl.LazyFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "t": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        }
    ).with_columns(expr.alias("out"))
    rolling_exprs = _collect_rolling_by_function_data(q)
    assert len(rolling_exprs) == 1
    name, _, _, _, fn_params = rolling_exprs[0]
    assert name == getattr(_expr_nodes.RollingFunctionBy, expected_name)
    assert fn_params == expected_fn_params


def _collect_named_function_data(
    query: pl.LazyFrame, kind: type
) -> list[tuple[Any, ...]]:
    """Return function_data tuples whose name is an instance of ``kind``."""
    results: list[tuple[Any, ...]] = []

    def callback(node_traverser: Any, query_start: int | None) -> None:
        for expr_ir in node_traverser.get_exprs():
            expr_node = node_traverser.view_expression(expr_ir.node)
            if isinstance(expr_node, _expr_nodes.Function):
                name, *options = expr_node.function_data
                if isinstance(name, kind):
                    results.append((name, *options))

    query.collect(  # pyrefly: ignore[no-matching-overload]
        post_opt_callback=callback  # type: ignore[call-overload]
    )
    return results


def test_array_expr_visitor() -> None:
    """An array function node is exposed as an ArrayFunction typed view."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.sum().alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert len(data) == 1
    (name,) = data[0]
    assert name == _expr_nodes.ArrayFunction.Sum
    assert hash(name) == hash(_expr_nodes.ArrayFunction.Sum)


@pytest.mark.parametrize(
    ("expr", "schema", "expected"),
    [
        (
            pl.col("x").arr.len(),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.Length,),
        ),
        (
            pl.col("x").arr.min(),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.Min,),
        ),
        (
            pl.col("x").arr.max(),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.Max,),
        ),
        (
            pl.col("x").arr.to_list(),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.ToList,),
        ),
        (
            pl.col("x").arr.var(ddof=2),
            {"x": pl.Array(pl.Float64, 3)},
            (_expr_nodes.ArrayFunction.Var, 2),
        ),
        (
            pl.col("x").arr.mean(),
            {"x": pl.Array(pl.Float64, 3)},
            (_expr_nodes.ArrayFunction.Mean,),
        ),
        (
            pl.col("x").arr.median(),
            {"x": pl.Array(pl.Float64, 3)},
            (_expr_nodes.ArrayFunction.Median,),
        ),
        (
            pl.col("x").arr.arg_min(),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.ArgMin,),
        ),
        (
            pl.col("x").arr.arg_max(),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.ArgMax,),
        ),
        (
            pl.col("x").arr.join("-", ignore_nulls=False),
            {"x": pl.Array(pl.String, 3)},
            (_expr_nodes.ArrayFunction.Join, False),
        ),
        (
            pl.col("x").arr.count_matches(1),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.CountMatches,),
        ),
        (
            pl.col("x").arr.shift(2),
            {"x": pl.Array(pl.Int64, 3)},
            (_expr_nodes.ArrayFunction.Shift,),
        ),
        (
            pl.concat_arr("x", "y"),
            {"x": pl.Array(pl.Int64, 3), "y": pl.Array(pl.Int64, 2)},
            (_expr_nodes.ArrayFunction.Concat,),
        ),
    ],
)
def test_array_variant_function_data(
    expr: pl.Expr, schema: dict[str, pl.DataType], expected: tuple[Any, ...]
) -> None:
    """Each Array variant exposes its discriminant and complete option payload."""
    q = pl.LazyFrame(schema=schema).with_columns(expr.alias("out"))
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert data == [expected]


@pytest.mark.parametrize("null_on_oob", [True, False])
def test_array_get_exposes_option(null_on_oob: bool) -> None:
    """arr.get carries its ``null_on_oob`` flag as a trailing option."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.get(0, null_on_oob=null_on_oob).alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert len(data) == 1
    name, actual_null_on_oob = data[0]
    assert name == _expr_nodes.ArrayFunction.Get
    assert actual_null_on_oob is null_on_oob


def test_array_std_exposes_ddof() -> None:
    """arr.std carries its ddof as a trailing option."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Float64, 3)}).with_columns(
        pl.col("x").arr.std(ddof=2).alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert data == [(_expr_nodes.ArrayFunction.Std, 2)]


def test_array_sort_exposes_sort_options() -> None:
    """arr.sort mirrors list.sort's public option encoding."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.sort(descending=True, nulls_last=True).alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert data == [(_expr_nodes.ArrayFunction.Sort, True, True)]


def test_array_contains_exposes_nulls_equal() -> None:
    """arr.contains carries its ``nulls_equal`` flag as a trailing option."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.contains(1, nulls_equal=False).alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert data == [(_expr_nodes.ArrayFunction.Contains, False)]


def test_array_explode_exposes_options() -> None:
    """arr.explode exposes both ExplodeOptions fields in declaration order."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.explode(empty_as_null=False, keep_nulls=False).alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert data == [(_expr_nodes.ArrayFunction.Explode, False, False)]


def test_array_slice_exposes_offset_and_length() -> None:
    """Fixed-width arr.slice exposes its constant offset and length."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.slice(-2, 1, as_array=True).alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert data == [(_expr_nodes.ArrayFunction.Slice, -2, 1)]


def test_array_to_struct_default_is_exposed() -> None:
    """arr.to_struct() exposes the absent name generator as ``None``."""
    q = pl.LazyFrame(schema={"x": pl.Array(pl.Int64, 3)}).with_columns(
        pl.col("x").arr.to_struct().alias("out"),
    )
    data = _collect_named_function_data(q, _expr_nodes.ArrayFunction)
    assert len(data) == 1
    name, fields = data[0]
    assert name == _expr_nodes.ArrayFunction.ToStruct
    assert fields == [f"field_{i}" for i in range(3)]


@pytest.mark.parametrize(
    ("kwargs", "is_elementwise", "changes_length", "expected_is_elementwise"),
    [
        (b"\x80visitor-kwargs", True, False, True),
        (b"", False, False, False),
        (b"", True, True, False),
    ],
)
def test_ffi_plugin_expr_visitor(
    kwargs: bytes,
    is_elementwise: bool,
    changes_length: bool,
    expected_is_elementwise: bool,
) -> None:
    """An FFI plugin exposes its dispatch metadata without loading the library."""
    from polars._plr import register_plugin_function  # type: ignore[attr-defined]

    expr = wrap_expr(
        register_plugin_function(
            plugin_path="plugins/libvisitor.so",
            function_name="score",
            args=[pl.col("x")._pyexpr],
            kwargs=kwargs,
            is_elementwise=is_elementwise,
            input_wildcard_expansion=False,
            returns_scalar=False,
            cast_to_supertype=False,
            pass_name_to_apply=False,
            changes_length=changes_length,
        )
    )
    visitor = pl.LazyFrame(schema={"x": pl.Float64})._ldf.visit()
    [node], _ = visitor.add_expressions([expr._pyexpr])
    view = visitor.view_expression(node)
    assert isinstance(view, _expr_nodes.Function)
    assert view.function_data == (
        "ffi_plugin",
        "plugins/libvisitor.so",
        "score",
        kwargs,
        expected_is_elementwise,
    )
