import re

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
                pl.concat(lazy_dfs).describe_optimized_plan(),
                flags=re.MULTILINE,
            )
        )
        == 10
    )
