# obscure setups meant to initiate obscure bugs
import polars as pl


def test_join_with_nulls_and_bit_offset() -> None:
    n = 15
    left_df = pl.DataFrame(
        {"join": range(n), "with_nulls": [None if i % 2 == 0 else 0 for i in range(n)]}
    )
    i = 1
    length = 8
    r = range(i, i + length)
    right_df = pl.DataFrame(
        {
            "join": r,
            "with_nulls": [None if i % 3 == 0 else 1 for i in r],
        }
    )

    assert left_df.join(right_df, on=["join", "with_nulls"], how="inner").shape == (
        1,
        2,
    )
