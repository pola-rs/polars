import polars as pl


def test_sort_by_bools() -> None:
    # tests dispatch
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    out = df.with_column((pl.col("foo") % 2 == 1).alias("foo_odd")).sort(
        by=["foo", "foo_odd"]
    )
    assert out.shape == (3, 4)


def test_type_coercion_when_then_otherwise_2806() -> None:
    out = (
        pl.DataFrame({"names": ["foo", "spam", "spam"], "nrs": [1, 2, 3]})
        .select(
            [
                pl.when((pl.col("names") == "spam"))
                .then((pl.col("nrs") * 2))
                .otherwise(pl.lit("other"))
                .alias("new_col"),
            ]
        )
        .to_series()
    )
    expected = pl.Series("new_col", ["other", "4", "6"])
    assert out.to_list() == expected.to_list()

    # test it remains float32
    assert (
        pl.Series("a", [1.0, 2.0, 3.0], dtype=pl.Float32)
        .to_frame()
        .select(pl.when(pl.col("a") > 2.0).then(pl.col("a")).otherwise(0.0))
    ).to_series().dtype == pl.Float32


def test_repeat_expansion_in_groupby() -> None:
    out = (
        pl.DataFrame({"g": [1, 2, 2, 3, 3, 3]})
        .groupby("g", maintain_order=True)
        .agg(pl.repeat(1, pl.count()).cumsum())
        .to_dict()
    )
    assert out == {"g": [1, 2, 3], "literal": [[1], [1, 2], [1, 2, 3]]}


def test_agg_after_head() -> None:
    a = [1, 1, 1, 2, 2, 3, 3, 3, 3]

    df = pl.DataFrame({"a": a, "b": pl.arange(1, len(a) + 1, eager=True)})

    expected = pl.DataFrame({"a": [1, 2, 3], "b": [6, 9, 21]})

    for maintain_order in [True, False]:
        out = df.groupby("a", maintain_order=maintain_order).agg(
            [pl.col("b").head(3).sum()]
        )

        if not maintain_order:
            out = out.sort("a")

        assert out.frame_equal(expected)


def test_overflow_uint16_agg_mean() -> None:
    assert (
        pl.DataFrame(
            {
                "col1": ["A" for _ in range(1025)],
                "col3": [64 for i in range(1025)],
            }
        )
        .with_columns(
            [
                pl.col("col3").cast(pl.UInt16),
            ]
        )
        .groupby(["col1"])
        .agg(pl.col("col3").mean())
        .to_dict(False)
        == {"col1": ["A"], "col3": [64.0]}
    )
