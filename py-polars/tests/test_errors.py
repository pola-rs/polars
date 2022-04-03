import pytest

import polars as pl


def test_error_on_empty_groupby() -> None:
    with pytest.raises(
        pl.ComputeError, match="expected keys in groupby operation, got nothing"
    ):
        pl.DataFrame(dict(x=[0, 0, 1, 1])).groupby([]).agg(pl.count())
