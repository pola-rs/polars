import pytest

import polars as pl
from polars.testing import assert_series_equal

inf = float("inf")


def test_qcut() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    out = s.qcut([0.25, 0.50])

    expected = pl.Series(
        "a",
        [
            "(-inf, -1]",
            "(-inf, -1]",
            "(-1, 0]",
            "(0, inf]",
            "(0, inf]",
        ],
        dtype=pl.Categorical,
    )
    assert_series_equal(out, expected, categorical_as_str=True)


def test_qcut_n() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    out = s.qcut(2, labels=["x", "y"], left_closed=True)

    expected = pl.Series("a", ["x", "x", "y", "y", "y"], dtype=pl.Categorical)
    assert_series_equal(out, expected, categorical_as_str=True)


def test_qcut_include_breaks() -> None:
    s = pl.int_range(-2, 3, eager=True).alias("a")

    out = s.qcut([0.0, 0.25, 0.75], labels=["a", "b", "c", "d"], include_breaks=True)

    expected = pl.DataFrame(
        {
            "break_point": [-2.0, -1.0, 1.0, 1.0, inf],
            "category": ["a", "b", "c", "c", "d"],
        },
        schema_overrides={"category": pl.Categorical},
    ).to_struct("a")
    assert_series_equal(out, expected, categorical_as_str=True)


def test_qcut_null_values() -> None:
    s = pl.Series([-1.0, None, 1.0, 2.0, None, 8.0, 4.0])

    result = s.qcut([0.2, 0.3], labels=["a", "b", "c"])

    expected = pl.Series(["a", None, "b", "c", None, "c", "c"], dtype=pl.Categorical)
    assert_series_equal(result, expected, categorical_as_str=True)


def test_qcut_allow_duplicates() -> None:
    s = pl.Series([1, 2, 2, 3])

    with pytest.raises(pl.DuplicateError):
        s.qcut([0.50, 0.51])

    result = s.qcut([0.50, 0.51], allow_duplicates=True)

    expected = pl.Series(
        ["(-inf, 2]", "(-inf, 2]", "(-inf, 2]", "(2, inf]"], dtype=pl.Categorical
    )
    assert_series_equal(result, expected, categorical_as_str=True)


def test_qcut_over() -> None:
    df = pl.DataFrame(
        {
            "group": ["a"] * 4 + ["b"] * 4,
            "value": range(8),
        }
    )

    out = df.select(
        pl.col("value").qcut([0.5], labels=["low", "high"]).over("group")
    ).to_series()

    expected = pl.Series(
        "value",
        ["low", "low", "high", "high", "low", "low", "high", "high"],
        dtype=pl.Categorical,
    )
    assert_series_equal(out, expected, categorical_as_str=True)


def test_qcut_deprecated_label_name() -> None:
    s = pl.Series([1.0, 2.0])
    with pytest.deprecated_call():
        s.qcut([0.1], category_label="x")
    with pytest.deprecated_call():
        s.qcut([0.1], break_point_label="x")
