from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal


def test_placeholder_scan_basic() -> None:
    """Basic placeholder: create template, bind, collect."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = (
        pl.scan_placeholder("input", schema)
        .filter(pl.col("a") > 0)
        .select(["a", "b"])
    )

    df = pl.DataFrame({"a": [1, -2, 3], "b": ["x", "y", "z"]})
    result = template.bind({"input": df.lazy()}).collect()

    expected = pl.DataFrame({"a": [1, 3], "b": ["x", "z"]})
    assert_frame_equal(result, expected)


def test_placeholder_scan_template_reuse() -> None:
    """Same template can be bound to different data."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).filter(pl.col("a") > 0)

    # First binding
    df1 = pl.DataFrame({"a": [1, -2, 3], "b": ["x", "y", "z"]})
    result1 = template.bind({"input": df1.lazy()}).collect()
    assert result1.height == 2

    # Second binding with different data
    df2 = pl.DataFrame({"a": [-1, 5, 10, -3], "b": ["a", "b", "c", "d"]})
    result2 = template.bind({"input": df2.lazy()}).collect()
    expected2 = pl.DataFrame({"a": [5, 10], "b": ["b", "c"]})
    assert_frame_equal(result2, expected2)


def test_placeholder_scan_multi_placeholder_join() -> None:
    """Two placeholders joined together."""
    left_ph = pl.scan_placeholder("left", {"id": pl.Int64, "value": pl.Float64})
    right_ph = pl.scan_placeholder("right", {"id": pl.Int64, "name": pl.String})
    template = left_ph.join(right_ph, on="id")

    left_df = pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    right_df = pl.DataFrame({"id": [2, 3, 4], "name": ["bob", "charlie", "dave"]})

    result = template.bind({
        "left": left_df.lazy(),
        "right": right_df.lazy(),
    }).collect()

    assert result.height == 2  # id 2 and 3 match
    assert set(result.columns) == {"id", "value", "name"}


def test_placeholder_scan_unbound_collect_errors() -> None:
    """Collecting without binding should raise an error."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema)

    with pytest.raises(InvalidOperationError, match="PlaceholderScan"):
        template.collect()


def test_placeholder_scan_missing_binding_errors() -> None:
    """Binding with a missing placeholder name should error."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema)

    df = pl.DataFrame({"a": [1], "b": ["x"]})
    with pytest.raises(InvalidOperationError, match="input"):
        template.bind({"wrong_name": df.lazy()}).collect()


def test_placeholder_scan_with_projection() -> None:
    """Projection pushdown should work with placeholder."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).select("a")

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = template.bind({"input": df.lazy()}).collect()

    assert result.columns == ["a"]
    assert result.height == 3


def test_placeholder_scan_with_sort() -> None:
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).sort("a")

    df = pl.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
    result = template.bind({"input": df.lazy()}).collect()

    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    assert_frame_equal(result, expected)


def test_placeholder_scan_with_group_by() -> None:
    schema = {"a": pl.Int64, "b": pl.String}
    template = (
        pl.scan_placeholder("input", schema)
        .group_by("b")
        .agg(pl.col("a").sum())
    )

    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    result = template.bind({"input": df.lazy()}).sort("b").collect()

    expected = pl.DataFrame({"b": ["x", "y"], "a": [4, 6]})
    assert_frame_equal(result, expected)


def test_placeholder_scan_with_slice() -> None:
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).slice(0, 2)

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = template.bind({"input": df.lazy()}).collect()

    assert result.height == 2


def test_placeholder_scan_with_with_columns() -> None:
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).with_columns(
        (pl.col("a") * 2).alias("a_doubled")
    )

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = template.bind({"input": df.lazy()}).collect()

    assert result.width == 3
    expected_doubled = pl.Series("a_doubled", [2, 4, 6])
    assert_frame_equal(result.get_column("a_doubled").to_frame(), expected_doubled.to_frame())


def test_placeholder_scan_concat() -> None:
    """Placeholder in a concat (union) context."""
    schema = {"a": pl.Int64, "b": pl.String}
    ph1 = pl.scan_placeholder("part1", schema)
    ph2 = pl.scan_placeholder("part2", schema)
    template = pl.concat([ph1, ph2])

    df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pl.DataFrame({"a": [3, 4], "b": ["z", "w"]})

    result = template.bind({
        "part1": df1.lazy(),
        "part2": df2.lazy(),
    }).collect()

    assert result.height == 4


def test_placeholder_scan_chained_operations() -> None:
    """Chain multiple operations on a placeholder template."""
    schema = {"a": pl.Int64, "b": pl.String, "c": pl.Float64}
    template = (
        pl.scan_placeholder("input", schema)
        .filter(pl.col("a") > 0)
        .with_columns((pl.col("c") * 100).alias("c_pct"))
        .select(["a", "b", "c_pct"])
        .sort("a")
    )

    df = pl.DataFrame({
        "a": [3, -1, 1, 2],
        "b": ["c", "x", "a", "b"],
        "c": [0.3, 0.1, 0.1, 0.2],
    })
    result = template.bind({"input": df.lazy()}).collect()

    assert result.height == 3
    assert result.columns == ["a", "b", "c_pct"]
    assert result["a"].to_list() == [1, 2, 3]
    assert result["c_pct"].to_list() == [10.0, 20.0, 30.0]


def test_placeholder_scan_explain() -> None:
    """The explain output should mention PLACEHOLDER before binding."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).filter(pl.col("a") > 0)

    # After binding, explain should no longer mention PLACEHOLDER
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    bound = template.bind({"input": df.lazy()})
    plan = bound.explain()
    assert "PLACEHOLDER" not in plan


# ==================== Phase 2: OptimizedTemplate tests ====================


def test_optimize_template_basic() -> None:
    """Basic optimize_template + bind_and_collect."""
    schema = {"a": pl.Int64, "b": pl.String}
    lf = pl.scan_placeholder("input", schema).filter(pl.col("a") > 0)
    template = lf.optimize_template()

    df = pl.DataFrame({"a": [1, -2, 3], "b": ["x", "y", "z"]})
    result = template.bind_and_collect({"input": df.lazy()})

    assert result.shape == (2, 2)
    assert result["a"].to_list() == [1, 3]


def test_optimize_template_reuse() -> None:
    """Same template can be bound to different data multiple times."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).filter(
        pl.col("a") > 0
    ).optimize_template()

    df1 = pl.DataFrame({"a": [1, -2, 3], "b": ["x", "y", "z"]})
    r1 = template.bind_and_collect({"input": df1.lazy()})
    assert r1.shape == (2, 2)

    df2 = pl.DataFrame({"a": [-1, 5, 10, -3], "b": ["a", "b", "c", "d"]})
    r2 = template.bind_and_collect({"input": df2.lazy()})
    assert r2.shape == (2, 2)
    assert r2["a"].to_list() == [5, 10]


def test_optimize_template_projection_pushdown() -> None:
    """Projection pushdown should work through PlaceholderScan."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).select(
        "a"
    ).optimize_template()

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = template.bind_and_collect({"input": df.lazy()})

    assert result.columns == ["a"]
    assert result.shape == (3, 1)


def test_optimize_template_with_filter() -> None:
    """Predicate pushdown + optimize_template."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).filter(
        pl.col("a") > 2
    ).optimize_template()

    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    result = template.bind_and_collect({"input": df.lazy()})

    assert result.shape == (2, 2)
    assert result["a"].to_list() == [3, 4]


def test_optimize_template_multi_placeholder_join() -> None:
    """Two placeholders joined, optimized."""
    left_schema = {"id": pl.Int64, "value": pl.Float64}
    right_schema = {"id": pl.Int64, "name": pl.String}

    left = pl.scan_placeholder("left", left_schema)
    right = pl.scan_placeholder("right", right_schema)
    lf = left.join(right, on="id", how="inner")
    template = lf.optimize_template()

    left_df = pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    right_df = pl.DataFrame({"id": [2, 3, 4], "name": ["bob", "charlie", "dave"]})

    result = template.bind_and_collect({
        "left": left_df.lazy(),
        "right": right_df.lazy(),
    })
    assert result.shape[0] == 2  # id 2 and 3 match
    assert result.shape[1] == 3  # id, value, name


def test_optimize_template_bind_returns_lazyframe() -> None:
    """bind() returns a LazyFrame that can be further operated on."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).filter(
        pl.col("a") > 0
    ).optimize_template()

    df = pl.DataFrame({"a": [1, -2, 3], "b": ["x", "y", "z"]})
    result = template.bind({"input": df.lazy()}).collect()
    assert result.shape == (2, 2)


def test_optimize_template_placeholder_names() -> None:
    """placeholder_names() returns all placeholder names."""
    left = pl.scan_placeholder("left", {"id": pl.Int64})
    right = pl.scan_placeholder("right", {"id": pl.Int64})
    lf = left.join(right, on="id", how="inner")
    template = lf.optimize_template()

    names = template.placeholder_names()
    assert sorted(names) == ["left", "right"]


def test_optimize_template_missing_binding_errors() -> None:
    """Missing binding should raise an error."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).optimize_template()

    df = pl.DataFrame({"a": [1], "b": ["x"]})
    with pytest.raises(Exception, match="input"):
        template.bind_and_collect({"wrong_name": df.lazy()})


def test_optimize_template_schema_mismatch_errors() -> None:
    """Schema mismatch should raise an error."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = pl.scan_placeholder("input", schema).optimize_template()

    df = pl.DataFrame({"a": ["not_int"], "b": ["x"]})
    with pytest.raises(Exception):
        template.bind_and_collect({"input": df.lazy()})


def test_optimize_template_with_groupby() -> None:
    """GroupBy with optimize_template."""
    schema = {"a": pl.Int64, "b": pl.String}
    template = (
        pl.scan_placeholder("input", schema)
        .group_by("b")
        .agg(pl.col("a").sum())
        .optimize_template()
    )

    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    result = template.bind_and_collect({"input": df.lazy()}).sort("b")
    assert result["a"].to_list() == [4, 6]
