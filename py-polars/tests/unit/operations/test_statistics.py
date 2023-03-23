import numpy as np

import polars as pl


def test_corr() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 4],
            "b": [-1, 23, 8],
        }
    )
    assert df.corr().to_dict(False) == {
        "a": [1.0, 0.18898223650461357],
        "b": [0.1889822365046136, 1.0],
    }


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
