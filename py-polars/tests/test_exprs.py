import polars as pl


def test_horizontal_agg(fruits_cars):
    df = fruits_cars
    out = df.select(pl.max([pl.col("A"), pl.col("B")]))
    out[:, 0].to_list() == [5, 4, 3, 4, 5]

    out = df.select(pl.min([pl.col("A"), pl.col("B")]))
    out[:, 0].to_list() == [1, 2, 3, 2, 1]
