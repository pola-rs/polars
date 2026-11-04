import polars as pl
from polars.testing import assert_frame_equal

# `sort` sets the sorted flag on x.
df = pl.DataFrame({"g": [1, 1, 1], "x": [1, 2, 3]}).sort("x")
df_unflagged = pl.DataFrame({"g": [1, 1, 1], "x": [1, 2, 3]})
assert df["x"].flags["SORTED_ASC"]
assert not df_unflagged["x"].flags["SORTED_ASC"]

aggs = {
    "rev_max": pl.col("x").reverse().max(),
    "rev_min": pl.col("x").reverse().min(),
    "top2_min": pl.col("x").sort_by("x", descending=True).head(2).min(),
    "sort_desc_max": pl.col("x").sort(descending=True).max(),
    "rev_arg_max": pl.col("x").reverse().arg_max(),
    "rev_arg_min": pl.col("x").reverse().arg_min(),
    "rev_str_max": pl.col("x").cast(pl.String).reverse().max(),
    "rev_bool_max": (pl.col("x") > 2).reverse().max(),
    "rev_max_by": pl.col("x").reverse().max_by(pl.col("x").reverse()),
}

for name, expr in aggs.items():
    got = df.group_by("g").agg(expr.alias(name))
    expected = df_unflagged.group_by("g").agg(expr.alias(name))
    ok = got.equals(expected)
    print(
        f"{name:14s} got={got[name][0]!r:6} expected={expected[name][0]!r:6} {'OK' if ok else 'WRONG'}"
    )

over = df.select(pl.col("x").min().over("g", order_by=-pl.col("x")))
over_expected = df_unflagged.select(pl.col("x").min().over("g", order_by=-pl.col("x")))
print(
    f"over min       got={over['x'].to_list()} expected={over_expected['x'].to_list()}"
)

out = df.group_by("g").agg(
    rev_max=pl.col("x").reverse().max(),
    top2_min=pl.col("x").sort_by("x", descending=True).head(2).min(),
)
assert out.row(0, named=True) == {"g": 1, "rev_max": 3, "top2_min": 2}, out.row(
    0, named=True
)
assert over["x"].to_list() == [1, 1, 1], over["x"].to_list()
