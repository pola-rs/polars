from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

inf = float("inf")


def test_cut() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.cut([-1, 1])

    expected = pl.Series(
        "a",
        [
            "(-inf, -1]",
            "(-inf, -1]",
            "(-1, 1]",
            "(-1, 1]",
            "(1, inf]",
        ],
        dtype=pl.Categorical,
    )
    assert_series_equal(result, expected, categorical_as_str=True)


def test_cut_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [-2, -1, 0, 1, 2]})

    result = lf.select(pl.col("a").cut([-1, 1]))

    expected = pl.LazyFrame(
        {"a": ["(-inf, -1]", "(-inf, -1]", "(-1, 1]", "(-1, 1]", "(1, inf]"]},
        schema={"a": pl.Categorical},
    )
    assert_frame_equal(result, expected, categorical_as_str=True)


def test_cut_include_breaks() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    out = s.cut([-1.5, 0.25, 1.0], labels=["a", "b", "c", "d"], include_breaks=True)

    expected = pl.DataFrame(
        {
            "break_point": [-1.5, 0.25, 0.25, 1.0, inf],
            "category": ["a", "b", "b", "c", "d"],
        },
        schema_overrides={"category": pl.Categorical},
    ).to_struct("a")
    assert_series_equal(out, expected, categorical_as_str=True)


# https://github.com/pola-rs/polars/issues/11255
def test_cut_include_breaks_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [-2, -1, 0, 1, 2]})

    result = lf.select(
        pl.col("a").cut([-1, 1], include_breaks=True).alias("cut")
    ).unnest("cut")

    expected = pl.LazyFrame(
        {
            "brk": [-1.0, -1.0, 1.0, 1.0, inf],
            "a_bin": ["(-inf, -1]", "(-inf, -1]", "(-1, 1]", "(-1, 1]", "(1, inf]"],
        },
        schema_overrides={"a_bin": pl.Categorical},
    )
    assert_frame_equal(result, expected, categorical_as_str=True)


def test_cut_null_values() -> None:
    s = pl.Series([-1.0, None, 1.0, 2.0, None, 8.0, 4.0])

    result = s.cut([1.5, 5.0], labels=["a", "b", "c"])

    expected = pl.Series(["a", None, "a", "b", None, "c", "b"], dtype=pl.Categorical)
    assert_series_equal(result, expected, categorical_as_str=True)


def test_cut_deprecated_as_series() -> None:
    a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
    with pytest.deprecated_call():
        out = a.cut(breaks=[-1, 1], as_series=False)

    assert out.shape == (12, 3)
    assert out.filter(pl.col("break_point") < 1e9).to_dict(as_series=False) == {
        "a": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        "break_point": [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        "category": [
            "(-inf, -1]",
            "(-inf, -1]",
            "(-inf, -1]",
            "(-inf, -1]",
            "(-inf, -1]",
            "(-1, 1]",
            "(-1, 1]",
            "(-1, 1]",
            "(-1, 1]",
        ],
    }


def test_cut_deprecated_label_name() -> None:
    s = pl.Series([1.0, 2.0])
    with pytest.deprecated_call():
        s.cut([0.1], category_label="x")
    with pytest.deprecated_call():
        s.cut([0.1], break_point_label="x")


def test_cut_bin_name_in_agg_context() -> None:
    df = pl.DataFrame({"a": [1]}).select(
        cut=pl.col("a").cut([1, 2], include_breaks=True).over(1),
        qcut=pl.col("a").qcut([1], include_breaks=True).over(1),
        qcut_uniform=pl.col("a").qcut(1, include_breaks=True).over(1),
    )
    schema = pl.Struct({"brk": pl.Float64, "a_bin": pl.Categorical("physical")})
    assert df.schema == {"cut": schema, "qcut": schema, "qcut_uniform": schema}


@pytest.mark.parametrize(
    ("breaks", "expected_labels", "expected_physical", "expected_unique"),
    [
        (
            [2, 4],
            pl.Series("x", ["(-inf, 2]", "(-inf, 2]", "(2, 4]", "(2, 4]", "(4, inf]"]),
            pl.Series("x", [0, 0, 1, 1, 2], dtype=pl.UInt32),
            3,
        ),
        (
            [99, 101],
            pl.Series("x", 5 * ["(-inf, 99]"]),
            pl.Series("x", 5 * [0], dtype=pl.UInt32),
            1,
        ),
    ],
)
def test_cut_fast_unique_15981(
    breaks: list[int],
    expected_labels: pl.Series,
    expected_physical: pl.Series,
    expected_unique: int,
) -> None:
    s = pl.Series("x", [1, 2, 3, 4, 5])

    include_breaks = False
    s_cut = s.cut(breaks, include_breaks=include_breaks)

    assert_series_equal(s_cut.cast(pl.String), expected_labels)
    assert_series_equal(s_cut.to_physical(), expected_physical)
    assert s_cut.n_unique() == s_cut.to_physical().n_unique() == expected_unique
    s_cut.to_frame().group_by(s.name).len()

    include_breaks = True
    s_cut = (
        s.cut(breaks, include_breaks=include_breaks).struct.field("category").alias("x")
    )

    assert_series_equal(s_cut.cast(pl.String), expected_labels)
    assert_series_equal(s_cut.to_physical(), expected_physical)
    assert s_cut.n_unique() == s_cut.to_physical().n_unique() == expected_unique
    s_cut.to_frame().group_by(s.name).len()
