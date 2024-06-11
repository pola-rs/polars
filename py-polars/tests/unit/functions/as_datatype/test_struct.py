import polars as pl


def test_resolved_names_15442() -> None:
    df = pl.DataFrame(
        {
            "x": [206.0],
            "y": [225.0],
        }
    )
    center = pl.struct(
        x=pl.col("x"),
        y=pl.col("y"),
    )

    left = 0
    right = 1000
    in_x = (left < center.struct.field("x")) & (center.struct.field("x") <= right)
    assert df.lazy().filter(in_x).collect().shape == (1, 2)
