from test_series import verify_series_and_expr_api

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


def test_min_nulls_consistency() -> None:
    df = pl.DataFrame({"a": [None, 2, 3], "b": [4, None, 6], "c": [7, 5, 0]})
    out = df.select([pl.min(["a", "b", "c"])]).to_series()
    expected = pl.Series("min", [4, 2, 0])
    testing.assert_series_equal(out, expected)

    out = df.select([pl.max(["a", "b", "c"])]).to_series()
    expected = pl.Series("max", [7, 5, 6])
    testing.assert_series_equal(out, expected)


def test_list_join_strings() -> None:
    s = pl.Series("a", [["ab", "c", "d"], ["e", "f"], ["g"], []])
    expected = pl.Series("a", ["ab-c-d", "e-f", "g", ""])
    verify_series_and_expr_api(s, expected, "arr.join", "-")


def test_count_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 3, 3], "b": ["a", "a", "b", "a", "a"]})

    out = df.select(pl.count())
    assert out.shape == (1, 1)
    assert out[0, 0] == 5

    out = df.groupby("b", maintain_order=True).agg(pl.count())
    assert out["b"].to_list() == ["a", "b"]
    assert out["count"].to_list() == [4, 1]


def test_sample() -> None:
    a = pl.Series("a", range(0, 20))
    out = pl.select(pl.lit(a).sample(0.5, False, 1)).to_series()
    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)


def test_map_alias() -> None:
    out = pl.DataFrame({"foo": [1, 2, 3]}).select(
        (pl.col("foo") * 2).map_alias(lambda name: f"{name}{name}")
    )

    expected = pl.DataFrame({"foofoo": [2, 4, 6]})
    assert out.frame_equal(expected)


def test_unique_stable() -> None:
    a = pl.Series("a", [1, 1, 1, 1, 2, 2, 2, 3, 3])
    expected = pl.Series("a", [1, 2, 3])

    verify_series_and_expr_api(a, expected, "unique", True)
