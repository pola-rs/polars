import pytest

import polars as pl
from polars.testing import assert_frame_equal

c = pl.col


@pytest.mark.parametrize(
    ("e", "equiv"),
    [
        (
            c.a.map_batches(lambda x: str(x), pl.dtype_of("b")),
            c.a.map_batches(lambda x: str(x), pl.String),
        ),
        (
            c.a.replace_strict([1, 2, 3, 4, 5], "X", return_dtype=pl.dtype_of("b")),
            pl.repeat("X", pl.len()).alias("a"),
        ),
        (
            pl.int_range(1, 5, 1, dtype=pl.dtype_of("a")),
            pl.int_range(1, 5, 1, dtype=pl.Int64),
        ),
    ],
)
def test_expressions(e: pl.Expr, equiv: pl.Expr) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "z", "w", "u"],
        }
    )

    assert_frame_equal(
        df.select(e),
        df.select(equiv),
    )
