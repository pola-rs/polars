import re
from datetime import date
from tempfile import NamedTemporaryFile

import pytest

import polars as pl


def test_cse_rename_cross_join_5405() -> None:
    right = pl.DataFrame({"A": [1, 2], "B": [3, 4], "D": [5, 6]}).lazy()

    left = pl.DataFrame({"C": [3, 4]}).lazy().join(right.select("A"), how="cross")

    out = left.join(right.rename({"B": "C"}), on=["A", "C"], how="left")

    assert out.collect(common_subplan_elimination=True).to_dict(False) == {
        "C": [3, 3, 4, 4],
        "A": [1, 2, 1, 2],
        "D": [5, None, None, 6],
    }


def test_union_duplicates() -> None:
    n_dfs = 10
    df_lazy = pl.DataFrame({}).lazy()
    lazy_dfs = [df_lazy for _ in range(n_dfs)]
    assert (
        len(
            re.findall(
                r".*CACHE\[id: .*, count: 9].*",
                pl.concat(lazy_dfs).explain(),
                flags=re.MULTILINE,
            )
        )
        == 10
    )


def test_cse_schema_6081() -> None:
    df = pl.DataFrame(
        data=[
            [date(2022, 12, 12), 1, 1],
            [date(2022, 12, 12), 1, 2],
            [date(2022, 12, 13), 5, 2],
        ],
        schema=["date", "id", "value"],
        orient="row",
    ).lazy()

    min_value_by_group = df.groupby(["date", "id"]).agg(
        pl.col("value").min().alias("min_value")
    )

    result = df.join(min_value_by_group, on=["date", "id"], how="left")
    assert result.collect(
        common_subplan_elimination=True, projection_pushdown=True
    ).to_dict(False) == {
        "date": [date(2022, 12, 12), date(2022, 12, 12), date(2022, 12, 13)],
        "id": [1, 1, 5],
        "value": [1, 2, 2],
        "min_value": [1, 1, 2],
    }


def test_cse_9630() -> None:
    df1 = pl.DataFrame(
        {
            "key": [1],
            "x": [1],
        }
    ).lazy()

    df2 = pl.DataFrame(
        {
            "key": [1],
            "y": [2],
        }
    ).lazy()

    joined_df2 = df1.join(df2, on="key")

    all_subsections = (
        pl.concat(
            [
                df1.select("key", pl.col("x").alias("value")),
                joined_df2.select("key", pl.col("y").alias("value")),
            ]
        )
        .groupby("key")
        .agg(
            [
                pl.col("value"),
            ]
        )
    )

    intersected_df1 = all_subsections.join(df1, on="key")
    intersected_df2 = all_subsections.join(df2, on="key")

    assert intersected_df1.join(intersected_df2, on=["key"], how="left").collect(
        common_subplan_elimination=True
    ).to_dict(False) == {
        "key": [1],
        "value": [[1, 2]],
        "x": [1],
        "value_right": [[1, 2]],
        "y": [2],
    }


@pytest.mark.write_disk()
def test_schema_row_count_cse() -> None:
    csv_a = NamedTemporaryFile()
    csv_a.write(
        b"""
    A,B
    Gr1,A
    Gr1,B
    """.strip()
    )
    csv_a.seek(0)

    df_a = pl.scan_csv(csv_a.name).with_row_count("Idx")
    assert df_a.join(df_a, on="B").groupby(
        "A", maintain_order=True
    ).all().collect().to_dict(False) == {
        "A": ["Gr1"],
        "Idx": [[0, 1]],
        "B": [["A", "B"]],
        "Idx_right": [[0, 1]],
        "A_right": [["Gr1", "Gr1"]],
    }
    csv_a.close()
