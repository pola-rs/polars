import numpy as np

import polars as pl
from polars.testing import assert_frame_equal


def test_projection_on_semi_join_4789() -> None:
    lfa = pl.DataFrame({"a": [1], "p": [1]}).lazy()

    lfb = pl.DataFrame({"seq": [1], "p": [1]}).lazy()

    ab = lfa.join(lfb, on="p", how="semi").inspect()

    intermediate_agg = (ab.group_by("a").agg([pl.col("a").alias("seq")])).select(
        ["a", "seq"]
    )

    q = ab.join(intermediate_agg, on="a")

    assert q.collect().to_dict(as_series=False) == {"a": [1], "p": [1], "seq": [[1]]}


def test_melt_projection_pd_block_4997() -> None:
    assert (
        pl.DataFrame({"col1": ["a"], "col2": ["b"]})
        .with_row_index()
        .lazy()
        .melt(id_vars="row_number")
        .group_by("row_number")
        .agg(pl.col("variable").alias("result"))
        .collect()
    ).to_dict(as_series=False) == {"row_number": [0], "result": [["col1", "col2"]]}


def test_double_projection_pushdown() -> None:
    assert (
        "PROJECT 2/3 COLUMNS"
        in (
            pl.DataFrame({"c0": [], "c1": [], "c2": []})
            .lazy()
            .select(["c0", "c1", "c2"])
            .select(["c0", "c1"])
        ).explain()
    )


def test_group_by_projection_pushdown() -> None:
    assert (
        "PROJECT 2/3 COLUMNS"
        in (
            pl.DataFrame({"c0": [], "c1": [], "c2": []})
            .lazy()
            .group_by("c0")
            .agg(
                [
                    pl.col("c1").sum().alias("sum(c1)"),
                    pl.col("c2").mean().alias("mean(c2)"),
                ]
            )
            .select(["sum(c1)"])
        ).explain()
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
    out = mlf.collect().to_dict(as_series=False)
    assert out == {
        "row": ["y", "y", "b", "b"],
        "col": ["z", "z", "c", "c"],
        "value": [1, 2, 2, 3],
    }


def test_hconcat_projection_pushdown() -> None:
    lf1 = pl.LazyFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
    lf2 = pl.LazyFrame({"c": [6, 7, 8], "d": [9, 10, 11]})
    query = pl.concat([lf1, lf2], how="horizontal").select(["a", "d"])

    explanation = query.explain()
    assert explanation.count("PROJECT 1/2 COLUMNS") == 2

    out = query.collect()
    expected = pl.DataFrame({"a": [0, 1, 2], "d": [9, 10, 11]})
    assert_frame_equal(out, expected)


def test_hconcat_projection_pushdown_length_maintained() -> None:
    # We can't eliminate the second input completely as this affects
    # the length of the result, even though no columns are used.
    lf1 = pl.LazyFrame({"a": [0, 1], "b": [2, 3]})
    lf2 = pl.LazyFrame({"c": [4, 5, 6, 7], "d": [8, 9, 10, 11]})
    query = pl.concat([lf1, lf2], how="horizontal").select(["a"])

    explanation = query.explain()
    assert "PROJECT 1/2 COLUMNS" in explanation

    out = query.collect()
    expected = pl.DataFrame({"a": [0, 1, None, None]})
    assert_frame_equal(out, expected)


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
        .list.to_struct(n_field_strategy="max_width", fields=lambda i: f"genre{i+1}")
    ).unnest("genres")

    out = q.collect()
    assert out.to_dict(as_series=False) == {
        "title": ["Avatar", "spectre", "King Kong"],
        "content_rating": ["PG-13", "PG-13", "PG-13"],
        "genre1": ["Action", "Action", "Action"],
        "genre2": ["Adventure", "Adventure", "Adventure"],
        "genre3": ["Fantasy", "Thriller", "Drama"],
        "genre4": ["Sci-Fi", None, "Romance"],
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

    # in this query the group_by projects only 2 columns, that's one
    # less than the upstream projection so the union will fail if
    # the select node does not prune one column
    q = lf1.select(["a", "b", "c"])

    q = pl.concat([q, lf2])

    q = q.group_by("c", maintain_order=True).agg([pl.col("a")])
    assert q.collect().to_dict(as_series=False) == {
        "c": [1, 2, 3],
        "a": [[1, 2, 5, 7], [3, 4, 6], [8]],
    }


def test_asof_join_projection_() -> None:
    lf1 = (
        pl.DataFrame(
            {
                "m": np.linspace(0, 5, 7),
                "a": np.linspace(0, 5, 7),
                "b": np.linspace(0, 5, 7),
                "c": pl.Series(np.linspace(0, 5, 7)).cast(str),
                "d": np.linspace(0, 5, 7),
            }
        )
        .lazy()
        .set_sorted("b")
    )
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
        .set_sorted("b")
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
    assert concatted.select(["b", "a"]).collect().to_dict(as_series=False) == {
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


def test_merge_sorted_projection_pd() -> None:
    lf = pl.LazyFrame(
        {
            "foo": [1, 2, 3, 4],
            "bar": ["patrick", "lukas", "onion", "afk"],
        }
    ).sort("foo")

    lf2 = pl.LazyFrame({"foo": [5, 6], "bar": ["nice", "false"]}).sort("foo")

    assert (
        lf.merge_sorted(lf2, key="foo").reverse().select(["bar"])
    ).collect().to_dict(as_series=False) == {
        "bar": ["false", "nice", "afk", "onion", "lukas", "patrick"]
    }


def test_distinct_projection_pd_7578() -> None:
    df = pl.DataFrame(
        {
            "foo": ["0", "1", "2", "1", "2"],
            "bar": ["a", "a", "a", "b", "b"],
        }
    )

    q = df.lazy().unique().group_by("bar").agg(pl.count())
    assert q.collect().sort("bar").to_dict(as_series=False) == {
        "bar": ["a", "b"],
        "count": [3, 2],
    }


def test_join_suffix_collision_9562() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    other_df = pl.DataFrame(
        {
            "apple": ["x", "y", "z"],
            "ham": ["a", "b", "d"],
        }
    )
    df.join(other_df, on="ham")
    assert df.lazy().join(
        other_df.lazy(), how="inner", left_on="ham", right_on="ham", suffix="m"
    ).select("ham").collect().to_dict(as_series=False) == {"ham": ["a", "b"]}


def test_projection_join_names_9955() -> None:
    batting = pl.DataFrame(
        {
            "playerID": ["abercda01"],
            "yearID": [1871],
            "lgID": ["NA"],
        }
    ).lazy()

    awards_players = pl.DataFrame(
        {
            "playerID": ["bondto01"],
            "yearID": [1877],
            "lgID": ["NL"],
        }
    ).lazy()

    right = awards_players.filter(pl.col("lgID") == "NL").select("playerID")

    q = batting.join(
        right,
        left_on=[pl.col("playerID")],
        right_on=[pl.col("playerID")],
        how="inner",
    )

    q = q.select(batting.columns)

    assert q.collect().schema == {
        "playerID": pl.String,
        "yearID": pl.Int64,
        "lgID": pl.String,
    }


def test_projection_rename_10595() -> None:
    lf = pl.LazyFrame(schema={"a": pl.Float32, "b": pl.Float32})
    result = lf.select("a", "b").rename({"b": "a", "a": "b"}).select("a")
    assert result.collect().schema == {"a": pl.Float32}


def test_projection_count_11841() -> None:
    pl.LazyFrame({"x": 1}).select(records=pl.count()).select(
        pl.lit(1).alias("x"), pl.all()
    ).collect()


def test_schema_outer_join_projection_pd_13287() -> None:
    lf = pl.LazyFrame({"a": [1, 1], "b": [2, 3]})
    lf2 = pl.LazyFrame({"a": [1, 1], "c": [2, 3]})

    assert lf.join(
        lf2,
        how="outer",
        left_on="a",
        right_on="c",
    ).with_columns(
        pl.col("a").fill_null(pl.col("c")),
    ).select("a").collect().to_dict(as_series=False) == {"a": [2, 3, 1, 1]}
