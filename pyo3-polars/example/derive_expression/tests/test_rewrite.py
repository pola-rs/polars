from __future__ import annotations

import io
from datetime import date, datetime

import polars as pl
import pytest
from polars.exceptions import ComputeError, InvalidOperationError
from polars.plugins import register_plugin_rewrite
from polars.testing import assert_frame_equal

from expression_lib._utils import LIB
from expression_lib.rewrite import is_leap_year_any, length, struct_median, to_unit


def rewrite(name: str, *args: pl.Expr | str) -> pl.Expr:
    return register_plugin_rewrite(plugin_path=LIB, function_name=name, args=args)


def test_branch_on_extension_metadata() -> None:
    df = pl.DataFrame(
        {"a": [1.0, 2.0], "b": [100.0, 250.0]},
        schema={"a": length("m"), "b": length("cm")},
    )
    out = df.select(to_unit("a", "cm"), to_unit("b", "m"))
    expected = pl.DataFrame(
        {"a": [100.0, 200.0], "b": [1.0, 2.5]},
        schema={"a": length("cm"), "b": length("m")},
    )
    assert_frame_equal(out, expected)


def test_output_dtype() -> None:
    # Polars doesn't check the dtype a rewrite returns, so the plugin tests it.
    lf = pl.LazyFrame({"a": [1.0]}, schema={"a": length("km")})
    assert lf.select(to_unit("a", "mm")).collect_schema() == {"a": length("mm")}
    assert lf.select(to_unit("a", "mm")).collect().schema == {"a": length("mm")}


def test_struct_restructure() -> None:
    lf = pl.LazyFrame(
        {"p": [{"x": 1.0, "y": 10.0}, {"x": 3.0, "y": 30.0}], "f": [1.0, 5.0]}
    )
    out = lf.select(struct_median("p"), struct_median("f")).collect()
    expected = pl.DataFrame({"p": [{"x": 2.0, "y": 20.0}], "f": [3.0]})
    assert_frame_equal(out, expected)

    plan = lf.select(struct_median("p")).explain()
    assert "struct_median" not in plan
    assert 'col("p").struct.field_by_name(x)().median()' in plan


def test_template_calls_plugin_kernel() -> None:
    df = pl.DataFrame(
        {
            "d": [date(2024, 1, 1), date(2023, 1, 1)],
            "dt": [datetime(2023, 1, 1), datetime(2024, 1, 1)],
        }
    )
    out = df.select(is_leap_year_any("d"), is_leap_year_any("dt"))
    assert_frame_equal(out, pl.DataFrame({"d": [True, False], "dt": [False, True]}))


def test_rewrite_in_rewrite_inputs() -> None:
    # The outer rewrite sees the unit set by the inner one.
    df = pl.DataFrame({"a": [1.0, 2.0]}, schema={"a": length("m")})
    out = df.select(to_unit(to_unit("a", "cm"), "mm"))
    expected = pl.DataFrame({"a": [1000.0, 2000.0]}, schema={"a": length("mm")})
    assert_frame_equal(out, expected)


def test_v1_limitations() -> None:
    lf = pl.LazyFrame({"a": [1.0], "l": [[1.0]]})
    with pytest.raises(
        InvalidOperationError,
        match="returned an expression that contains another rewrite",
    ):
        lf.select(rewrite("returns_rewrite", "a")).collect()
    with pytest.raises(InvalidOperationError, match="inside a nested evaluation"):
        lf.select(rewrite("input_in_eval", "a")).collect()
    with pytest.raises(
        InvalidOperationError,
        match="must return a single expression, but it expanded into 2",
    ):
        lf.select(rewrite("multiple_outputs", "a")).collect()


def test_errors() -> None:
    lf = pl.LazyFrame({"a": [1.0]})
    with pytest.raises(
        ComputeError, match="plugin does not export rewrite 'does_not_exist'"
    ):
        lf.select(rewrite("does_not_exist", "a")).collect()
    with pytest.raises(ComputeError, match="this rewrite always fails"):
        lf.select(rewrite("always_fails", "a")).collect()
    with pytest.raises(ComputeError, match="the plugin panicked"):
        lf.select(rewrite("panics", "a")).collect()
    with pytest.raises(ComputeError, match="rebuild the plugin against Polars"):
        lf.select(rewrite("bad_version", "a")).collect()
    with pytest.raises(
        InvalidOperationError,
        match=r"returned rewrite_input\(5\), but it only has 1 inputs",
    ):
        lf.select(rewrite("bad_input_index", "a")).collect()
    with pytest.raises(
        ComputeError, match="to_unit expects an 'expression_lib.length'"
    ):
        lf.select(to_unit("a", "m")).collect()


def test_serialize_roundtrip() -> None:
    lf = pl.LazyFrame({"a": [1.0, 2.0]}, schema={"a": length("km")}).select(
        to_unit("a", "m")
    )
    roundtripped = pl.LazyFrame.deserialize(io.BytesIO(lf.serialize()))
    expected = pl.DataFrame({"a": [1000.0, 2000.0]}, schema={"a": length("m")})
    assert_frame_equal(roundtripped.collect(), expected)


def test_streaming_matches_in_memory() -> None:
    lf = (
        pl.LazyFrame({"g": [1, 1, 2], "p": [{"x": 1.0}, {"x": 3.0}, {"x": 5.0}]})
        .group_by("g")
        .agg(struct_median("p"))
    )
    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
        check_row_order=False,
    )
