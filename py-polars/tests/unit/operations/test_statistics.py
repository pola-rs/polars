from datetime import timedelta

import numpy as np

import polars as pl
from polars.testing import assert_frame_equal


def test_corr() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 4],
            "b": [-1, 23, 8],
        }
    )
    result = df.corr()
    expected = pl.DataFrame(
        {
            "a": [1.0, 0.18898223650461357],
            "b": [0.1889822365046136, 1.0],
        }
    )
    assert_frame_equal(result, expected)


def test_cut() -> None:
    a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
    out = a.cut(bins=[-1, 1])

    assert out.shape == (12, 3)
    assert out.filter(pl.col("break_point") < 1e9).to_dict(False) == {
        "a": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        "break_point": [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        "category": [
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
        ],
    }

    # test cut on integers #4939
    inf = float("inf")
    df = pl.DataFrame({"a": list(range(5))})
    ser = df.select("a").to_series()
    assert ser.cut(bins=[-1, 1]).rows() == [
        (0.0, 1.0, "(-1.0, 1.0]"),
        (1.0, 1.0, "(-1.0, 1.0]"),
        (2.0, inf, "(1.0, inf]"),
        (3.0, inf, "(1.0, inf]"),
        (4.0, inf, "(1.0, inf]"),
    ]


def test_cut_maintain_order() -> None:
    np.random.seed(1)
    a = pl.Series("a", np.random.randint(0, 10, 10))
    out = a.cut(bins=[-1, 1], maintain_order=True)
    assert out["a"].cast(int).series_equal(a)
    assert (
        str(out.to_dict(False))
        == "{'a': [5.0, 8.0, 9.0, 5.0, 0.0, 0.0, 1.0, 7.0, 6.0, 9.0], 'break_point': [inf, inf, inf, inf, 1.0, 1.0, 1.0, inf, inf, inf], 'category': ['(1.0, inf]', '(1.0, inf]', '(1.0, inf]', '(1.0, inf]', '(-1.0, 1.0]', '(-1.0, 1.0]', '(-1.0, 1.0]', '(1.0, inf]', '(1.0, inf]', '(1.0, inf]']}"
    )


def test_qcut() -> None:
    assert (
        str(pl.Series("a", range(-5, 3)).qcut([0.0, 0.25, 0.75]).to_dict(False))
        == "{'a': [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0], 'break_point': [-5.0, -3.25, 0.25, 0.25, 0.25, 0.25, inf, inf], 'category': ['(-inf, -5.0]', '(-5.0, -3.25]', '(-3.25, 0.25]', '(-3.25, 0.25]', '(-3.25, 0.25]', '(-3.25, 0.25]', '(0.25, inf]', '(0.25, inf]']}"
    )


def test_hist() -> None:
    a = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
    assert (
        str(a.hist(bin_count=4).to_dict(False))
        == "{'break_point': [0.0, 2.25, 4.5, 6.75, inf], 'category': ['(-inf, 0.0]', '(0.0, 2.25]', '(2.25, 4.5]', '(4.5, 6.75]', '(6.75, inf]'], 'a_count': [0, 3, 2, 0, 2]}"
    )


def test_cut_null_values() -> None:
    s = pl.Series([-1.0, None, 1.0, 2.0, None, 8.0, 4.0])
    assert (
        str(s.qcut([0.2, 0.3], maintain_order=True).to_dict(False))
        == "{'': [-1.0, None, 1.0, 2.0, None, 8.0, 4.0], 'break_point': [0.5999999999999996, None, 1.2000000000000002, inf, None, inf, inf], 'category': ['(-inf, 0.5999999999999996]', None, '(0.5999999999999996, 1.2000000000000002]', '(1.2000000000000002, inf]', None, '(1.2000000000000002, inf]', '(1.2000000000000002, inf]']}"
    )
    assert (
        str(s.qcut([0.2, 0.3], maintain_order=False).to_dict(False))
        == "{'': [-1.0, 1.0, 2.0, 4.0, 8.0, None, None], 'break_point': [0.5999999999999996, 1.2000000000000002, inf, inf, inf, None, None], 'category': ['(-inf, 0.5999999999999996]', '(0.5999999999999996, 1.2000000000000002]', '(1.2000000000000002, inf]', '(1.2000000000000002, inf]', '(1.2000000000000002, inf]', None, None]}"
    )


def test_median_quantile_duration() -> None:
    df = pl.DataFrame({"A": [timedelta(days=0), timedelta(days=1)]})
    assert df.select(pl.col("A").median()).to_dict(False) == {
        "A": [timedelta(seconds=43200)]
    }
    assert df.select(pl.col("A").quantile(0.5, interpolation="linear")).to_dict(
        False
    ) == {"A": [timedelta(seconds=43200)]}


def test_correlation_cast_supertype() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4.0, 5.0, 2.0]})
    df = df.with_columns(pl.col("b"))
    assert df.select(pl.corr("a", "b")).to_dict(False) == {"a": [0.5447047794019223]}
