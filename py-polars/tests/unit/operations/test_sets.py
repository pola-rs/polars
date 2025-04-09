from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import CategoricalRemappingWarning
from polars.testing import assert_series_equal


def test_set_intersection_13765() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([[1], [1]], dtype=pl.List(pl.UInt32)),
            "f": pl.Series([1, 2], dtype=pl.UInt32),
        }
    )

    df = df.join(df, how="cross", suffix="_other")
    df = df.filter(pl.col("f") == 1)

    df.select(pl.col("a").list.set_intersection("a_other")).to_dict(as_series=False)


def test_set_intersection_st_17129() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [2, 2, 4]})

    assert df.with_columns(
        pl.col("b")
        .over("a", mapping_strategy="join")
        .list.set_intersection([4, 8])
        .alias("intersect")
    ).to_dict(as_series=False) == {
        "a": [1, 2, 2],
        "b": [2, 2, 4],
        "intersect": [[], [4], [4]],
    }


@pytest.mark.parametrize(
    ("set_operation", "outcome"),
    [
        ("set_difference", [{"z1", "z"}, {"z"}, set(), {"z", "x2"}, {"z", "x3"}]),
        ("set_intersection", [{"x", "y"}, {"y"}, {"y", "x"}, {"x", "y"}, set()]),
        (
            "set_symmetric_difference",
            [{"z1", "z"}, {"x", "z"}, set(), {"z", "x2"}, {"x", "y", "z", "x3"}],
        ),
    ],
)
@pytest.mark.may_fail_auto_streaming
def test_set_operations_cats(set_operation: str, outcome: list[set[str]]) -> None:
    with pytest.warns(CategoricalRemappingWarning):
        df = pl.DataFrame(
            {
                "a": [
                    ["z1", "x", "y", "z"],
                    ["y", "z"],
                    ["x", "y"],
                    ["x", "y", "z", "x2"],
                    ["z", "x3"],
                ]
            },
            schema={"a": pl.List(pl.Categorical)},
        )
        df = df.with_columns(
            getattr(pl.col("a").list, set_operation)(["x", "y"]).alias("b")
        )
        assert df.get_column("b").dtype == pl.List(pl.Categorical)
        assert [set(el) for el in df["b"].to_list()] == outcome


def test_set_invalid_types() -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 3, 3], "b": [2, 2, 4, 7, 8]})

    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.with_columns(
            pl.col("b")
            .implode()
            .implode()
            .over("a", mapping_strategy="join")
            .list.set_intersection([1])
        )


@pytest.mark.parametrize("input", [[], [1, 2], [1, None]])
@pytest.mark.parametrize(
    "set_op",
    [
        "set_union",
        "set_intersection",
        "set_difference",
        "set_symmetric_difference",
    ],
)
def test_set_opts_set_input(input: list[list[int | None]], set_op: str) -> None:
    a = pl.Series([[1, 2, 3], [], [None, 3], [5, 6, 7]])
    op = getattr(a.list, set_op)
    assert_series_equal(op(input).list.sort(), op(set(input)).list.sort())
