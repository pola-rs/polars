import polars as pl


def test_simplify_expression_lit_true_4376() -> None:
    df = pl.DataFrame([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    assert df.lazy().filter(pl.lit(True) | (pl.col("column_0") == 1)).collect(
        simplify_expression=True
    ).rows() == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert df.lazy().filter((pl.col("column_0") == 1) | pl.lit(True)).collect(
        simplify_expression=True
    ).rows() == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]


def test_melt_values_predicate_pushdown() -> None:
    lf = pl.DataFrame(
        {
            "id": [1],
            "asset_key_1": ["123"],
            "asset_key_2": ["456"],
            "asset_key_3": ["abc"],
        }
    ).lazy()

    assert (
        lf.melt("id", ["asset_key_1", "asset_key_2", "asset_key_3"])
        .filter(pl.col("value") == pl.lit("123"))
        .collect()
    ).to_dict(False) == {"id": [1], "variable": ["asset_key_1"], "value": ["123"]}


def test_filter_is_in_4572() -> None:
    df = pl.DataFrame({"id": [1, 2, 1, 2], "k": ["a"] * 2 + ["b"] * 2})
    expected = (
        df.groupby("id").agg(pl.col("k").filter(pl.col("k") == "a").list()).sort("id")
    )
    assert (
        df.groupby("id")
        .agg(pl.col("k").filter(pl.col("k").is_in(["a"])).list())
        .sort("id")
        .frame_equal(expected)
    )
    assert (
        df.sort("id")
        .groupby("id")
        .agg(pl.col("k").filter(pl.col("k").is_in(["a"])).list())
        .frame_equal(expected)
    )


def test_filter_aggregation_any() -> None:
    assert pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "group": [1, 2, 1, 1],
            "pred_a": [False, True, False, False],
            "pred_b": [False, False, True, True],
        }
    ).groupby("group").agg(
        [
            pl.any(["pred_a", "pred_b"]),
            pl.col("id").filter(pl.any(["pred_a", "pred_b"])).alias("filtered"),
        ]
    ).sort(
        "group"
    ).to_dict(
        False
    ) == {
        "group": [1, 2],
        "any": [[False, True, True], [True]],
        "filtered": [[3, 4], [2]],
    }


def test_is_in_bool() -> None:
    bool_value_to_filter_on = [True, None]
    df = pl.DataFrame({"A": [True, False, None]})
    assert df.filter(pl.col("A").is_in(bool_value_to_filter_on)).to_dict(False) == {
        "A": [True, False]
    }


def test_predicate_order_explode_5950() -> None:
    df = pl.from_dict(
        {
            "i": [[0, 1], [1, 2]],
            "n": [0, None],
        }
    )

    assert (
        df.lazy()
        .explode("i")
        .filter(pl.col("n").count().over(["i"]) == 2)
        .filter(pl.col("n").is_not_null())
    ).collect().to_dict(False) == {"i": [1], "n": [0]}


def test_binary_simplification_5971() -> None:
    df = pl.DataFrame(pl.Series("a", [1, 2, 3, 4]))
    assert df.select((pl.col("a") > 2) | pl.lit(False))["a"].to_list() == [
        False,
        False,
        True,
        True,
    ]


def test_categorical_string_comparison_6283() -> None:
    scores = pl.DataFrame(
        {
            "zone": pl.Series(
                [
                    "North",
                    "North",
                    "North",
                    "South",
                    "South",
                    "East",
                    "East",
                    "East",
                    "East",
                ]
            ).cast(pl.Categorical),
            "funding": pl.Series(
                ["yes", "yes", "no", "yes", "no", "no", "no", "yes", "yes"]
            ).cast(pl.Categorical),
            "score": [78, 39, 76, 56, 67, 89, 100, 55, 80],
        }
    )

    assert scores.filter(scores["zone"] == "North").to_dict(False) == {
        "zone": ["North", "North", "North"],
        "funding": ["yes", "yes", "no"],
        "score": [78, 39, 76],
    }
