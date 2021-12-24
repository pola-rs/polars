import polars as pl
from polars import testing


def test_horizontal_agg(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(pl.max([pl.col("A"), pl.col("B")]))
    assert out[:, 0].to_list() == [5, 4, 3, 4, 5]

    out = df.select(pl.min([pl.col("A"), pl.col("B")]))
    assert out[:, 0].to_list() == [1, 2, 3, 2, 1]


def test_suffix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().suffix("_reverse")])
    assert out.columns == ["A_reverse", "fruits_reverse", "B_reverse", "cars_reverse"]


def test_prefix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().prefix("reverse_")])
    assert out.columns == ["reverse_A", "reverse_fruits", "reverse_B", "reverse_cars"]


def test_cumcount() -> None:
    df = pl.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], columns=["A"])

    out = df.groupby("A", maintain_order=True).agg(
        [pl.col("A").cumcount(reverse=False).alias("foo")]
    )

    assert out["foo"][0].to_list() == [0, 1, 2, 3]
    assert out["foo"][1].to_list() == [0, 1]


def test_filter_where() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 1, 2, 3], "b": [4, 5, 6, 7, 8, 9]})
    result_where = df.groupby("a", maintain_order=True).agg(
        pl.col("b").where(pl.col("b") > 4).alias("c")
    )
    result_filter = df.groupby("a", maintain_order=True).agg(
        pl.col("b").filter(pl.col("b") > 4).alias("c")
    )
    expected = pl.DataFrame({"a": [1, 2, 3], "c": [[7], [5, 8], [6, 9]]})
    assert result_where.frame_equal(expected)
    assert result_filter.frame_equal(expected)


def test_flatten_explode() -> None:
    df = pl.Series("a", ["Hello", "World"])
    expected = pl.Series("a", ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"])

    result: pl.Series = df.to_frame().select(pl.col("a").flatten())[:, 0]  # type: ignore
    testing.assert_series_equal(result, expected)

    result: pl.Series = df.to_frame().select(pl.col("a").explode())[:, 0]  # type: ignore
    testing.assert_series_equal(result, expected)
