import pytest

import polars as pl


def test_raise_invalid_shape_19108() -> None:
    df = pl.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(pl.exceptions.ShapeError):
        df.select(pl.col.foo.head(0), pl.col.bar.head(1))
