import polars as pl


def test_profile_columns() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    # profile lazyframe operation/plan
    lazy = ldf.group_by("a").agg(pl.implode("b"))
    profiling_info = lazy.profile()
    # ┌──────────────┬───────┬─────┐
    # │ node         ┆ start ┆ end │
    # │ ---          ┆ ---   ┆ --- │
    # │ str          ┆ u64   ┆ u64 │
    # ╞══════════════╪═══════╪═════╡
    # │ optimization ┆ 0     ┆ 69  │
    # │ group_by(a)  ┆ 69    ┆ 342 │
    # └──────────────┴───────┴─────┘
    assert len(profiling_info) == 2
    assert profiling_info[1].columns == ["node", "start", "end"]


def test_profile_with_cse() -> None:
    df = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Float32, "y": pl.Float32})

    x = pl.col("x")
    y = pl.col("y")

    assert df.lazy().with_columns(
        pl.when(x.is_null())
        .then(None)
        .otherwise(pl.when(y == 0).then(None).otherwise(x + y))
    ).profile(comm_subexpr_elim=True)[1].shape == (2, 3)
