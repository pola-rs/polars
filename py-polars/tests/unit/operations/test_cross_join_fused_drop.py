from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
import polars._plr as plr
from polars.testing import assert_frame_equal
from tests.unit.conftest import mock_module_import
from tests.unit.lazyframe.test_query_monitoring import CLOUD_MODULE, fake_cloud_observer

if TYPE_CHECKING:
    from typing import Any

    from polars._typing import MaintainOrderJoin
    from tests.conftest import PlMonkeyPatch


def physical_plan(q: pl.LazyFrame) -> str:
    plan = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert plan is not None
    return plan


def test_join_where_cross_join_fused_drop() -> None:
    left = pl.LazyFrame({"v": [1, 1, None, 7], "lp": list("abcd")})
    right = pl.LazyFrame({"v": [2, 1, None, 8], "rp": [20, 10, 0, 80]})
    predicate = (pl.col("v") + pl.col("v_R")) % 3 == 0
    query = left.join_where(right, predicate, suffix="_R").select("rp", "lp", "v_R")

    plan = physical_plan(query)
    assert 'label="cross-join\\nfused predicate:' in plan
    assert "project 3 / 4: rp, lp, v_R" in plan
    assert 'label="filter' not in plan

    expected = (
        left.join(right, how="cross", suffix="_R")
        .collect(engine="in-memory")
        .filter(predicate)
        .select("rp", "lp", "v_R")
    )
    assert_frame_equal(
        query.collect(engine="streaming"), expected, check_row_order=False
    )


@pytest.mark.parametrize(("order", "morsel_size"), [("left", "2"), ("right", "12")])
def test_ordered_cross_join_fused_drop(
    order: MaintainOrderJoin, morsel_size: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", morsel_size)
    left = pl.LazyFrame({"a": [1, 2, 1, 0], "lp": list("abcd")})
    right = pl.LazyFrame({"b": [2, 1, None], "rp": [3, 4, 5]})
    joined = left.join(right, how="cross", maintain_order=order)
    predicate = (pl.col("a") + pl.col("b")) % 2 != 0
    query = joined.filter(predicate).select("lp", "rp")

    assert 'label="cross-join\\nfused predicate:' in physical_plan(query)
    expected = joined.collect(engine="in-memory").filter(predicate).select("lp", "rp")
    assert_frame_equal(query.collect(engine="streaming"), expected)


@pytest.mark.parametrize(
    ("output", "excluded"),
    [(("b", "a"), 3), (("rp", "a", "b", "lp"), 2)],
)
def test_cross_join_reuses_dense_operand_output(
    output: tuple[str, ...], excluded: int, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "64")
    left = pl.LazyFrame(
        {"a": [[1], [2], [3], [4]], "l": [0, 1, 2, 3], "lp": list("abcd")}
    )
    right = pl.LazyFrame(
        {"b": [[5], [6], [7], [8]], "r": [0, 1, 2, 3], "rp": list("wxyz")}
    )
    joined = left.join(right, how="cross", maintain_order="left")
    predicate = (
        pl.col("a").list.len() + pl.col("b").list.len() + pl.col("l") + pl.col("r")
    ) % 3 != excluded
    query = joined.filter(predicate).select(*output)
    assert 'label="cross-join\\nfused predicate:' in physical_plan(query)
    expected = joined.collect(engine="in-memory").filter(predicate).select(*output)
    assert_frame_equal(query.collect(engine="streaming"), expected)


@pytest.mark.parametrize(
    ("output", "morsel_size"), [("lp", "3"), ("lp", "64"), ("rp", "64")]
)
def test_cross_join_sparse_output_from_one_side(
    output: str, morsel_size: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", morsel_size)
    left = pl.LazyFrame({"a": range(4), "lp": list("abcd")})
    right = pl.LazyFrame({"b": range(0, 40, 10), "rp": list("wxyz")})
    joined = left.join(right, how="cross", maintain_order="left")
    predicate = (pl.col("a") + pl.col("b")) % 11 == 0
    query = joined.filter(predicate).select(output)
    assert 'label="cross-join\\nfused predicate:' in physical_plan(query)
    expected = joined.collect(engine="in-memory").filter(predicate).select(output)
    assert_frame_equal(query.collect(engine="streaming"), expected)


def test_cross_join_fused_predicate_in_query_observer() -> None:
    msgpack = pytest.importorskip("msgpack")
    left = pl.LazyFrame({"a": [0, 1], "lp": list("ab")})
    right = pl.LazyFrame({"b": [0, 1]})
    joined = left.join(right, how="cross", maintain_order="left")
    fused = joined.filter((pl.col("a") + pl.col("b")) % 2 == 0).select("lp")
    module, observer = fake_cloud_observer()
    try:
        with mock_module_import(CLOUD_MODULE, module, replace_if_exists=True):
            engine = pl.StreamingEngine(monitoring=True)
            for query in [joined, fused]:
                query.collect(engine=engine)
    finally:
        plr.set_query_monitoring(False)

    def cross_join_properties(payload: bytes) -> dict[str, Any]:
        (props,) = [
            node["properties"]
            for node in msgpack.unpackb(payload, raw=False)
            if node["properties"]["type"] == "CrossJoin"
        ]
        assert isinstance(props, dict)
        return props

    unfused_call, fused_call = observer.on_query_planned.call_args_list
    assert "fused_predicate" not in cross_join_properties(unfused_call.args[3])
    (predicate,) = cross_join_properties(fused_call.args[3])["fused_predicate"]
    assert 'col("a")' in predicate
    assert 'col("b")' in predicate


def test_cross_join_non_elementwise_filter_kept() -> None:
    left = pl.LazyFrame({"a": [1, 2], "lp": ["a", "b"]})
    right = pl.LazyFrame({"b": [3, 4], "rp": ["x", "y"]})
    joined = left.join(right, how="cross", maintain_order="left")
    predicate = (pl.col("a") + pl.col("b")).cum_sum() < 12
    query = joined.filter(predicate).select("lp", "rp")

    plan = physical_plan(query)
    assert 'label="cross-join"' in plan
    assert 'label="filter' in plan
    expected = joined.collect(engine="in-memory").filter(predicate).select("lp", "rp")
    assert_frame_equal(query.collect(engine="streaming"), expected)


def test_cross_join_fused_filter_error() -> None:
    left = pl.LazyFrame({"a": ["1", "x"], "lp": [4, 5]})
    right = pl.LazyFrame({"b": ["2", "3"], "rp": [6, 7]})
    predicate = pl.concat_str("a", "b").cast(pl.Int64) < 20
    query = left.join(right, how="cross").filter(predicate).select("lp", "rp")

    assert 'label="cross-join\\nfused predicate:' in physical_plan(query)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion from"):
        query.collect(engine="streaming")


def test_cross_join_fused_filter_no_matches(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "64")
    left = pl.LazyFrame({"a": range(0, 514, 2), "lp": range(257)})
    right = pl.LazyFrame({"b": range(0, 514, 2)})
    predicate = (pl.col("a") + pl.col("b")) % 2 == 1
    query = left.join_where(right, predicate).select("lp")

    assert 'label="cross-join\\nfused predicate:' in physical_plan(query)
    assert_frame_equal(
        query.collect(engine="streaming"), pl.DataFrame(schema={"lp": pl.Int64})
    )


def test_cross_join_with_shared_output() -> None:
    left = pl.LazyFrame({"a": [1, 2], "lp": ["a", "b"]})
    right = pl.LazyFrame({"b": [3, 4], "rp": ["x", "y"]})
    joined = left.join(right, how="cross", maintain_order="left").cache()
    predicate = (pl.col("a") + pl.col("b")) % 2 == 0
    query = joined.filter(predicate).select("lp", "rp")

    expected = joined.collect(engine="in-memory")
    filtered, full = pl.collect_all([query, joined], engine="streaming")
    assert_frame_equal(filtered, expected.filter(predicate).select("lp", "rp"))
    assert_frame_equal(full, expected)
