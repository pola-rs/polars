import pytest

import polars as pl
from polars import PolarsDataType
from polars.testing import assert_frame_equal


def test_simplify_expression_lit_true_4376() -> None:
    df = pl.DataFrame([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    assert df.lazy().filter(pl.lit(True) | (pl.col("column_0") == 1)).collect(
        simplify_expression=True
    ).rows() == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    assert df.lazy().filter((pl.col("column_0") == 1) | pl.lit(True)).collect(
        simplify_expression=True
    ).rows() == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]


def test_filter_contains_nth_11205() -> None:
    df = pl.DataFrame({"x": [False]})
    assert df.filter(pl.first()).is_empty()


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
    ).to_dict(as_series=False) == {
        "id": [1],
        "variable": ["asset_key_1"],
        "value": ["123"],
    }


def test_group_by_filter_all_true() -> None:
    df = pl.DataFrame(
        {
            "name": ["a", "a", "b", "b"],
            "type": [None, 1, 1, None],
            "order": [1, 2, 3, 4],
        }
    )
    out = (
        df.group_by("name")
        .agg([pl.col("order").filter(pl.col("type") == 1).n_unique().alias("n_unique")])
        .select("n_unique")
    )
    assert out.to_dict(as_series=False) == {"n_unique": [1, 1]}


def test_filter_is_in_4572() -> None:
    df = pl.DataFrame({"id": [1, 2, 1, 2], "k": ["a"] * 2 + ["b"] * 2})
    expected = (
        df.group_by("id")
        .agg(pl.col("k").filter(pl.col("k") == "a").implode())
        .sort("id")
    )
    result = (
        df.group_by("id")
        .agg(pl.col("k").filter(pl.col("k").is_in(["a"])).implode())
        .sort("id")
    )
    assert_frame_equal(result, expected)
    result = (
        df.sort("id")
        .group_by("id")
        .agg(pl.col("k").filter(pl.col("k").is_in(["a"])).implode())
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", [pl.Int32, pl.Boolean, pl.Utf8, pl.Binary, pl.List(pl.Int64), pl.Object]
)
def test_filter_on_empty(dtype: PolarsDataType) -> None:
    df = pl.DataFrame({"a": []}, schema={"a": dtype})
    out = df.filter(pl.col("a").is_null())
    assert out.is_empty()


def test_filter_aggregation_any() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "group": [1, 2, 1, 1],
            "pred_a": [False, True, False, False],
            "pred_b": [False, False, True, True],
        }
    )

    result = (
        df.group_by("group")
        .agg(
            pl.any_horizontal("pred_a", "pred_b"),
            pl.col("id")
            .filter(pl.any_horizontal("pred_a", "pred_b"))
            .alias("filtered"),
        )
        .sort("group")
    )

    assert result.to_dict(as_series=False) == {
        "group": [1, 2],
        "any": [[False, True, True], [True]],
        "filtered": [[3, 4], [2]],
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
    ).collect().to_dict(as_series=False) == {"i": [1], "n": [0]}


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

    assert scores.filter(scores["zone"] == "North").to_dict(as_series=False) == {
        "zone": ["North", "North", "North"],
        "funding": ["yes", "yes", "no"],
        "score": [78, 39, 76],
    }


def test_clear_window_cache_after_filter_10499() -> None:
    df = pl.from_dict(
        {
            "a": [None, None, 3, None, 5, 0, 0, 0, 9, 10],
            "b": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        }
    )

    assert df.lazy().filter((pl.col("a").null_count() < pl.count()).over("b")).filter(
        ((pl.col("a") == 0).sum() < pl.count()).over("b")
    ).collect().to_dict(as_series=False) == {
        "a": [3, None, 5, 0, 9, 10],
        "b": [2, 2, 3, 3, 5, 5],
    }


def test_agg_function_of_filter_10565() -> None:
    df_int = pl.DataFrame(data={"a": []}, schema={"a": pl.Int16})
    assert df_int.filter(pl.col("a").n_unique().over("a") == 1).to_dict(
        as_series=False
    ) == {"a": []}

    df_str = pl.DataFrame(data={"a": []}, schema={"a": pl.Utf8})
    assert df_str.filter(pl.col("a").n_unique().over("a") == 1).to_dict(
        as_series=False
    ) == {"a": []}

    assert df_str.lazy().filter(pl.col("a").n_unique().over("a") == 1).collect(
        predicate_pushdown=False
    ).to_dict(as_series=False) == {"a": []}
