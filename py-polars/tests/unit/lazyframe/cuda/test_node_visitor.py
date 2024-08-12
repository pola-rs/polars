from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Callable

import polars as pl
from polars._utils.wrap import wrap_df
from polars.polars import _ir_nodes

if TYPE_CHECKING:
    import pandas as pd


def test_run_on_pandas() -> None:
    # Simple join example, missing multiple columns, slices, etc.
    def join(
        inputs: list[Callable[[], pd.DataFrame]], obj: Any, _node_traverser: Any
    ) -> Callable[[], pd.DataFrame]:
        assert len(obj.left_on) == 1
        assert len(obj.right_on) == 1
        left_on = obj.left_on[0].output_name
        right_on = obj.right_on[0].output_name

        assert len(inputs) == 2

        def run(inputs: list[Callable[[], pd.DataFrame]]) -> pd.DataFrame:
            # materialize inputs
            dataframes = [call() for call in inputs]
            return dataframes[0].merge(
                dataframes[1], left_on=left_on, right_on=right_on
            )

        return partial(run, inputs)

    # Simple scan example, missing predicates, columns pruning, slices, etc.
    def df_scan(_inputs: None, obj: Any, _: Any) -> Callable[[], pd.DataFrame]:
        assert obj.selection is None
        return lambda: wrap_df(obj.df).to_pandas()

    @lru_cache(1)
    def get_node_converters() -> (
        dict[type, Callable[[Any, Any, Any], Callable[[], pd.DataFrame]]]
    ):
        return {
            _ir_nodes.Join: join,
            _ir_nodes.DataFrameScan: df_scan,
        }

    def get_input(node_traverser: Any) -> Callable[[], pd.DataFrame]:
        current_node = node_traverser.get_node()

        inputs_callable = []
        for inp in node_traverser.get_inputs():
            node_traverser.set_node(inp)
            inputs_callable.append(get_input(node_traverser))

        node_traverser.set_node(current_node)
        ir_node = node_traverser.view_current_node()
        return get_node_converters()[ir_node.__class__](
            inputs_callable, ir_node, node_traverser
        )

    def run_on_pandas(node_traverser: Any) -> None:
        current_node = node_traverser.get_node()

        callback = get_input(node_traverser)

        def run_callback(
            columns: list[str] | None, _: Any, n_rows: int | None
        ) -> pl.DataFrame:
            assert n_rows is None
            assert columns is None

            # produce a wrong result to ensure the callback has run.
            return pl.from_pandas(callback() * 2)

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
