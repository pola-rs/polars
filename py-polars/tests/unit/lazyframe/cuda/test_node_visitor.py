from __future__ import annotations

import typing
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Callable

import polars as pl
from polars._utils.wrap import wrap_df
from polars.polars import _ir_nodes

if TYPE_CHECKING:
    import pandas as pd


@typing.no_type_check
def test_run_on_pandas() -> None:
    # Simple join example, missing multiple columns, slices, etc.
    @typing.no_type_check
    def join(inputs: list[Callable], obj: Any, _node_traverer: Any) -> Callable:
        assert len(obj.left_on) == 1
        assert len(obj.right_on) == 1
        left_on = obj.left_on[0].output_name
        right_on = obj.right_on[0].output_name

        assert len(inputs) == 2

        def run(inputs: list[Callable]):
            # materialize inputs
            inputs = [call() for call in inputs]
            return inputs[0].merge(inputs[1], left_on=left_on, right_on=right_on)

        return partial(run, inputs)

    # Simple scan example, missing predicates, columns pruning, slices, etc.
    @typing.no_type_check
    def df_scan(_inputs: None, obj: Any, _: Any) -> pd.DataFrame:
        assert obj.selection is None
        return lambda: wrap_df(obj.df).to_pandas()

    @lru_cache(1)
    @typing.no_type_check
    def get_node_converters():
        return {
            _ir_nodes.Join: join,
            _ir_nodes.DataFrameScan: df_scan,
        }

    @typing.no_type_check
    def get_input(node_traverser):
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

    @typing.no_type_check
    def run_on_pandas(node_traverser) -> None:
        current_node = node_traverser.get_node()

        callback = get_input(node_traverser)

        @typing.no_type_check
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
    assert q.collect(post_opt_callback=run_on_pandas).to_dict(as_series=False) == {
        "foo": [2],
        "bar": [4],
    }
