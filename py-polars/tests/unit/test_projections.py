import typing

import numpy as np

import polars as pl


def test_projection_on_semi_join_4789() -> None:
    lfa = pl.DataFrame({"a": [1], "p": [1]}).lazy()

    lfb = pl.DataFrame({"seq": [1], "p": [1]}).lazy()

    ab = lfa.join(lfb, on="p", how="semi").inspect()

    intermediate_agg = (ab.groupby("a").agg([pl.col("a").alias("seq")])).select(
        ["a", "seq"]
    )

    q = ab.join(intermediate_agg, on="a")

    assert q.collect().to_dict(False) == {"a": [1], "p": [1], "seq": [[1]]}


def test_melt_projection_pd_block_4997() -> None:
    assert (
        pl.DataFrame({"col1": ["a"], "col2": ["b"]})
        .with_row_count()
        .lazy()
        .melt(id_vars="row_nr")
        .groupby("row_nr")
        .agg(pl.col("variable").alias("result"))
        .collect()
    ).to_dict(False) == {"row_nr": [0], "result": [["col1", "col2"]]}


def test_double_projection_pushdown() -> None:
    assert (
        "PROJECT 2/3 COLUMNS"
        in (
            pl.DataFrame({"c0": [], "c1": [], "c2": []})
            .lazy()
            .select(["c0", "c1", "c2"])
            .select(["c0", "c1"])
        ).describe_optimized_plan()
    )


def test_groupby_projection_pushdown() -> None:
    assert (
        "PROJECT 2/3 COLUMNS"
        in (
            pl.DataFrame({"c0": [], "c1": [], "c2": []})
            .lazy()
            .groupby("c0")
            .agg(
                [
                    pl.col("c1").sum().alias("sum(c1)"),
                    pl.col("c2").mean().alias("mean(c2)"),
                ]
            )
            .select(["sum(c1)"])
        ).describe_optimized_plan()
    )


def test_unnest_projection_pushdown() -> None:
    lf = pl.DataFrame({"x|y|z": [1, 2], "a|b|c": [2, 3]}).lazy()

    mlf = (
        lf.melt()
        .with_columns(pl.col("variable").str.split_exact("|", 2))
        .unnest("variable")
    )
    mlf = mlf.select(
        [
            pl.col("field_1").cast(pl.Categorical).alias("row"),
            pl.col("field_2").cast(pl.Categorical).alias("col"),
            pl.col("value"),
        ]
    )
    out = mlf.collect().to_dict(False)
    assert out == {
        "row": ["y", "y", "b", "b"],
        "col": ["z", "z", "c", "c"],
        "value": [1, 2, 2, 3],
    }


def test_unnest_columns_available() -> None:
    df = pl.DataFrame(
        {
            "title": ["Avatar", "spectre", "King Kong"],
            "content_rating": ["PG-13"] * 3,
            "genres": [
                "Action|Adventure|Fantasy|Sci-Fi",
                "Action|Adventure|Thriller",
                "Action|Adventure|Drama|Romance",
            ],
        }
    ).lazy()

    q = df.with_columns(
        pl.col("genres")
        .str.split("|")
        .arr.to_struct(
            n_field_strategy="max_width", name_generator=lambda i: f"genre{i+1}"
        )
    ).unnest("genres")

    out = q.collect()
    assert out.to_dict(False) == {
        "title": ["Avatar", "spectre", "King Kong"],
        "content_rating": ["PG-13", "PG-13", "PG-13"],
        "genre1": ["Action", "Action", "Action"],
        "genre2": ["Adventure", "Adventure", "Adventure"],
        "genre3": ["Fantasy", "Thriller", "Drama"],
        "genre4": ["Sci-Fi", None, "Romance"],
    }


def test_streaming_duplicate_cols_5537() -> None:
    assert pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).lazy().with_columns(
        [(pl.col("a") * 2).alias("foo"), (pl.col("a") * 3)]
    ).collect(streaming=True).to_dict(False) == {
        "a": [3, 6, 9],
        "b": [1, 2, 3],
        "foo": [2, 4, 6],
    }


def test_double_projection_union() -> None:
    lf1 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 3, 4, 5],
            "c": [1, 1, 2, 2],
            "d": [1, 2, 2, 1],
        }
    ).lazy()

    lf2 = pl.DataFrame(
        {
            "a": [5, 6, 7, 8],
            "b": [6, 7, 8, 9],
            "c": [1, 2, 1, 3],
        }
    ).lazy()

    # in this query the groupby projects only 2 columns, that's one
    # less than the upstream projection so the union will fail if
    # the select node does not prune one column
    q = lf1.select(["a", "b", "c"])

    q = pl.concat([q, lf2])

    q = q.groupby("c", maintain_order=True).agg([pl.col("a")])
    assert q.collect().to_dict(False) == {
        "c": [1, 2, 3],
        "a": [[1, 2, 5, 7], [3, 4, 6], [8]],
    }


@typing.no_type_check
def test_asof_join_projection_() -> None:
    lf1 = pl.DataFrame(
        {
            "m": np.linspace(0, 5, 7),
            "a": np.linspace(0, 5, 7),
            "b": np.linspace(0, 5, 7),
            "c": pl.Series(np.linspace(0, 5, 7)).cast(str),
            "d": np.linspace(0, 5, 7),
        }
    ).lazy()
    lf2 = (
        pl.DataFrame(
            {
                "group": [0, 2, 3, 0, 1, 2, 3],
                "val": [0, 2.5, 2.6, 2.7, 3.4, 4, 5],
                "c": ["x", "x", "x", "y", "y", "y", "y"],
            }
        )
        .with_columns(pl.col("val").alias("b"))
        .lazy()
    )

    joined = lf1.join_asof(
        lf2,
        on="b",
        by=["c"],
        strategy="backward",
    )

    expressions = [
        "m",
        "a",
        "b",
        "c",
        "d",
        pl.lit(0).alias("group"),
        pl.lit(0.1).alias("val"),
    ]
    dirty_lf1 = lf1.select(expressions)

    concatted = pl.concat([joined, dirty_lf1])
    assert concatted.select(["b", "a"]).collect().to_dict(False) == {
        "b": [
            0.0,
            0.8333333333333334,
            1.6666666666666667,
            2.5,
            3.3333333333333335,
            4.166666666666667,
            5.0,
            0.0,
            0.8333333333333334,
            1.6666666666666667,
            2.5,
            3.3333333333333335,
            4.166666666666667,
            5.0,
        ],
        "a": [
            0.0,
            0.8333333333333334,
            1.6666666666666667,
            2.5,
            3.3333333333333335,
            4.166666666666667,
            5.0,
            0.0,
            0.8333333333333334,
            1.6666666666666667,
            2.5,
            3.3333333333333335,
            4.166666666666667,
            5.0,
        ],
    }
