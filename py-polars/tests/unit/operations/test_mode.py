import pytest

import polars as pl
from polars.testing import assert_series_equal


@pytest.mark.parametrize("maintain_order", [False, True])
def test_mode(maintain_order: bool) -> None:
    assert_series_equal(
        pl.Series(["A", "B", "B", "C", "", "C"]).mode(maintain_order=maintain_order),
        pl.Series(["B", "C"]),
        check_order=maintain_order,
    )

    assert_series_equal(
        pl.Series([], dtype=pl.Int64).mode(maintain_order=maintain_order),
        pl.Series([], dtype=pl.Int64),
        check_order=maintain_order,
    )

    assert_series_equal(
        pl.Series([1, 2, 3]).mode(maintain_order=maintain_order),
        pl.Series([1, 2, 3]),
        check_order=maintain_order,
    )

    assert_series_equal(
        pl.Series([True, False, True]).mode(maintain_order=maintain_order),
        pl.Series([True]),
        check_order=maintain_order,
    )

    assert_series_equal(
        pl.Series([None, None, True]).mode(maintain_order=maintain_order),
        pl.Series([None], dtype=pl.Boolean),
        check_order=maintain_order,
    )
