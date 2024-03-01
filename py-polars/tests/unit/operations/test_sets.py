from __future__ import annotations

import pytest

import polars as pl


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
def test_set_operations_cats(set_operation: str, outcome: list[set[str]]) -> None:
    with pytest.warns(pl.CategoricalRemappingWarning):
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
