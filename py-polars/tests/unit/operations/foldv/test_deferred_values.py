from __future__ import annotations

import pytest

import polars as pl


def test_foldv_rejects_nan_input_values() -> None:
    # NaN behavior remains deferred with validity semantics.
    df = pl.DataFrame({"l": [1.0, float("nan")]})
    expression = pl.foldv(lambda state: state + pl.col("l"), s=0.0)

    with pytest.raises(pl.exceptions.InvalidOperationError, match="NaN values"):
        df.select(expression)
