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
