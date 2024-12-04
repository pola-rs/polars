import pytest

import polars as pl


@pytest.mark.parametrize(
    "f",
    [
        "min",
        "max",
        "sum",
        "mean",
    ],
)
def test_shape_mismatch_19336(f: str) -> None:
    a = pl.Series([1, 2, 3])
    b = pl.Series([1, 2])
    fn = getattr(pl, f"{f}_horizontal")

    with pytest.raises(pl.exceptions.ShapeError):
        pl.select((fn)(a, b))
