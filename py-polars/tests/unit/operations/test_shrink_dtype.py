import pytest

import polars as pl


def test_shrink_dtype() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 2 << 32],
            "c": [-1, 2, 1 << 30],
            "d": [-112, 2, 112],
            "e": [-112, 2, 129],
            "f": ["a", "b", "c"],
            "g": [0.1, 1.32, 0.12],
            "h": [True, None, False],
            "i": pl.Series([None, None, None], dtype=pl.UInt64),
            "j": pl.Series([None, None, None], dtype=pl.Int64),
            "k": pl.Series([None, None, None], dtype=pl.Float64),
        }
    )

    with pytest.warns(DeprecationWarning):
        out = df.select(pl.all().shrink_dtype())

    assert out.dtypes == [
        pl.Int64,
        pl.Int64,
        pl.Int64,
        pl.Int64,
        pl.Int64,
        pl.String,
        pl.Float64,
        pl.Boolean,
        pl.UInt64,
        pl.Int64,
        pl.Float64,
    ]

    assert out.to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [1, 2, 8589934592],
        "c": [-1, 2, 1073741824],
        "d": [-112, 2, 112],
        "e": [-112, 2, 129],
        "f": ["a", "b", "c"],
        "g": [0.1, 1.32, 0.12],
        "h": [True, None, False],
        "i": [None, None, None],
        "j": [None, None, None],
        "k": [None, None, None],
    }
