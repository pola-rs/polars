from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

inf = float("inf")


def test_cut() -> None:
    # series
    s = pl.Series("foo", [-2, -1, 0, 1, 2])
    out = s.cut([-1, 1])
    expected = pl.Series(
        "foo",
        [
            "(-inf, -1]",
            "(-inf, -1]",
            "(-1, 1]",
            "(-1, 1]",
            "(1, inf]",
        ],
        dtype=pl.Categorical,
    )
    assert_series_equal(out, expected, categorical_as_str=True)

    # expr
    df = pl.DataFrame(s)
    df_out = df.select(pl.col("foo").cut([-1, 1]))
    df_expected = pl.DataFrame(expected)
    assert_frame_equal(df_out, df_expected, categorical_as_str=True)


def test_cut_with_labels() -> None:
    # series
    s = pl.Series("foo", [-2, -1, 0, 1, 2])
    out = s.cut([-1, 1], labels=["a", "b", "c"])
    expected = pl.Series("foo", ["a", "a", "b", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(out, expected, categorical_as_str=True)

    # dataframe
    df = pl.DataFrame(s)
    df_out = df.with_columns(
        pl.col("foo").cut([-1, 1], labels=["a", "b", "c"]).alias("cut")
    )
    df_expected = pl.DataFrame(
        {
            "foo": [-2, -1, 0, 1, 2],
            "cut": pl.Series(["a", "a", "b", "b", "c"], dtype=pl.Categorical),
        }
    )
    assert_frame_equal(df_out, df_expected, categorical_as_str=True)


def test_cut_include_breaks() -> None:
    # series
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

    # dataframe
    df = pl.DataFrame(s)
    df_expected = pl.DataFrame(
        {
            "a": [-2, -1, 0, 1, 2],
            "brk": [-1.0, -1.0, 1.0, 1.0, inf],
            "a_bin": pl.Series(
                [
                    "(-inf, -1]",
                    "(-inf, -1]",
                    "(-1, 1]",
                    "(-1, 1]",
                    "(1, inf]",
                ],
                dtype=pl.Categorical,
            ),
        }
    )

    # eager
    df_out = df.with_columns(
        pl.col("a").cut([-1, 1], include_breaks=True).alias("cut")
    ).unnest("cut")
    assert df_out.schema == {"a": pl.Int64, "brk": pl.Float64, "a_bin": pl.Categorical}
    assert_frame_equal(df_out, df_expected, categorical_as_str=True)

    # lazy
    df_out = (
        df.lazy()
        .with_columns(pl.col("a").cut([-1, 1], include_breaks=True).alias("cut"))
        .unnest("cut")
        .collect()
    )
    assert df_out.schema == {"a": pl.Int64, "brk": pl.Float64, "a_bin": pl.Categorical}
    assert_frame_equal(df_out, df_expected, categorical_as_str=True)


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
    assert out.filter(pl.col("break_point") < 1e9).to_dict(False) == {
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
