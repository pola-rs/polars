import polars as pl
from polars.testing import assert_frame_equal


def test_single_row_literal_ambiguity_8481() -> None:
    df = pl.DataFrame(
        {
            "store_id": [1],
            "cost_price": [2.0],
        }
    )

    inverse_cost_price = 1.0 / pl.col("cost_price")
    result = df.with_columns(
        (inverse_cost_price / inverse_cost_price.sum()).over("store_id").alias("result")
    )
    # exceptions.ComputeError: cannot aggregate a literal

    expected = pl.DataFrame(
        {
            "store_id": [1],
            "cost_price": [2.0],
            "result": [1.0],
        }
    )
    assert_frame_equal(result, expected)
