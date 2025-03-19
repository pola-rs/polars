from __future__ import annotations

import time
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Callable

import polars as pl
from polars._utils.wrap import wrap_df
from polars.polars import _ir_nodes

if TYPE_CHECKING:
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
    assert q.collect(
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
