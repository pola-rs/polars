from typing import Any

import pytest

import polars as pl


def test_pass_name_alias_18914() -> None:
    df = pl.DataFrame({"id": [1], "value": [2]})

    assert df.select(
        pl.all()
        .map_elements(
            lambda x: x,
            skip_nulls=False,
            pass_name=True,
            return_dtype=pl.List(pl.Int64),
        )
        .over("id")
    ).to_dict(as_series=False) == {"id": [1], "value": [2]}


@pytest.mark.parametrize(
    "dtype",
    [
        pl.String,
        pl.Int64,
        pl.Boolean,
        pl.List(pl.Int32),
        pl.Array(pl.Boolean, 2),
        pl.Struct({"a": pl.Int8}),
        pl.Enum(["a"]),
    ],
)
def test_raises_udf(dtype: pl.DataType) -> None:
    def raise_f(item: Any) -> None:
        msg = "test error"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="test error"):
        pl.select(
            pl.lit(1).map_elements(
                raise_f,
                return_dtype=dtype,
            )
        )
