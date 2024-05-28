# Tests for the optimization pass cluster WITH_COLUMNS

import polars as pl


def test_basic_cwc() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(pl.col("a").alias("c") * 3)
        .with_columns(pl.col("a").alias("d") * 4)
    )

    assert (
        """[[(col("a")) * (2)].alias("b"), [(col("a")) * (3)].alias("c"), [(col("a")) * (4)].alias("d")]"""
        in df.explain()
    )


def test_disable_cwc() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(pl.col("a").alias("c") * 3)
        .with_columns(pl.col("a").alias("d") * 4)
    )

    explain = df.explain(cluster_with_columns=False)

    assert """[[(col("a")) * (2)].alias("b")]""" in explain
    assert """[[(col("a")) * (3)].alias("c")]""" in explain
    assert """[[(col("a")) * (4)].alias("d")]""" in explain


def test_refuse_with_deps() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(pl.col("b").alias("c") * 3)
        .with_columns(pl.col("c").alias("d") * 4)
    )

    explain = df.explain()

    assert """[[(col("a")) * (2)].alias("b")]""" in explain
    assert """[[(col("b")) * (3)].alias("c")]""" in explain
    assert """[[(col("c")) * (4)].alias("d")]""" in explain


def test_partial_deps() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(
            pl.col("a").alias("c") * 3,
            pl.col("b").alias("d") * 4,
            pl.col("a").alias("e") * 5,
        )
        .with_columns(pl.col("b").alias("f") * 6)
    )

    explain = df.explain()

    assert (
        """[[(col("b")) * (4)].alias("d"), [(col("b")) * (6)].alias("f")]""" in explain
    )
    assert (
        """[[(col("a")) * (2)].alias("b"), [(col("a")) * (3)].alias("c"), [(col("a")) * (5)].alias("e")]"""
        in explain
    )


def test_swap_remove() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(
            pl.col("b").alias("f") * 6,
            pl.col("a").alias("c") * 3,
            pl.col("b").alias("d") * 4,
            pl.col("b").alias("e") * 5,
        )
    )

    explain = df.explain()
    assert df.collect().equals(
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": [2, 4],
                "f": [12, 24],
                "c": [3, 6],
                "d": [8, 16],
                "e": [10, 20],
            }
        )
    )

    assert (
        """[[(col("b")) * (6)].alias("f"), [(col("b")) * (4)].alias("d"), [(col("b")) * (5)].alias("e")]"""
        in explain
    )
    assert (
        """[[(col("a")) * (2)].alias("b"), [(col("a")) * (3)].alias("c")]""" in explain
    )
    assert """simple π""" in explain


def test_try_remove_simple_project() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(pl.col("a").alias("d") * 4, pl.col("b").alias("c") * 3)
    )

    explain = df.explain()

    assert (
        """[[(col("a")) * (2)].alias("b"), [(col("a")) * (4)].alias("d")]""" in explain
    )
    assert """[[(col("b")) * (3)].alias("c")]""" in explain
    assert """simple π""" not in explain

    df = (
        pl.LazyFrame({"a": [1, 2]})
        .with_columns(pl.col("a").alias("b") * 2)
        .with_columns(pl.col("b").alias("c") * 3, pl.col("a").alias("d") * 4)
    )

    explain = df.explain()

    assert (
        """[[(col("a")) * (2)].alias("b"), [(col("a")) * (4)].alias("d")]""" in explain
    )
    assert """[[(col("b")) * (3)].alias("c")]""" in explain
    assert """simple π""" in explain


def test_cwc_with_internal_aliases() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .with_columns(pl.any_horizontal((pl.col("a") == 2).alias("b")).alias("c"))
        .with_columns(pl.col("b").alias("d") * 3)
    )

    explain = df.explain()

    assert (
        """[[(col("a")) == (2)].cast(Boolean).alias("c"), [(col("b")) * (3)].alias("d")]"""
        in explain
    )


def test_read_of_pushed_column_16436() -> None:
    df = pl.DataFrame(
        {
            "x": [1.12, 2.21, 4.2, 3.21],
            "y": [2.11, 3.32, 2.1, 6.12],
        }
    )

    df = (
        df.lazy()
        .with_columns((pl.col("y") / pl.col("x")).alias("z"))
        .with_columns(
            pl.when(pl.col("z").is_infinite()).then(0).otherwise(pl.col("z")).alias("z")
        )
        .fill_nan(0)
        .collect()
    )


def test_multiple_simple_projections_16435() -> None:
    df = pl.DataFrame({"a": [1]}).lazy()

    df = (
        df.with_columns(b=pl.col("a"))
        .with_columns(c=pl.col("b"))
        .with_columns(l2a=pl.lit(2))
        .with_columns(l2b=pl.col("l2a"))
        .with_columns(m=pl.lit(3))
    )

    df.collect()


def test_reverse_order() -> None:
    df = pl.LazyFrame({"a": [1], "b": [2]})

    df = (
        df.with_columns(a=pl.col("a"), b=pl.col("b"), c=pl.col("a") * pl.col("b"))
        .with_columns(x=pl.col("a"), y=pl.col("b"))
        .with_columns(b=pl.col("a"), a=pl.col("b"))
    )

    df.collect()


def test_realias_of_unread_column_16530() -> None:
    df = (
        pl.LazyFrame({"x": [True]})
        .with_columns(x=pl.lit(False))
        .with_columns(y=~pl.col("x"))
        .with_columns(y=pl.lit(False))
    )

    explain = df.explain()

    assert explain.count("WITH_COLUMNS") == 1
    assert df.collect().equals(pl.DataFrame({"x": [False], "y": [False]}))


def test_realias_with_dependencies() -> None:
    df = (
        pl.LazyFrame({"x": [True]})
        .with_columns(x=pl.lit(False))
        .with_columns(y=~pl.col("x"))
        .with_columns(y=pl.lit(False), z=pl.col("y") | True)
    )

    explain = df.explain()

    assert explain.count("WITH_COLUMNS") == 3
    assert df.collect().equals(pl.DataFrame({"x": [False], "y": [False], "z": [True]}))


def test_refuse_pushdown_with_aliases() -> None:
    df = (
        pl.LazyFrame({"x": [True]})
        .with_columns(x=pl.lit(False))
        .with_columns(y=pl.lit(True))
        .with_columns(y=pl.lit(False), z=pl.col("y") | True)
    )

    explain = df.explain()

    assert explain.count("WITH_COLUMNS") == 2
    assert df.collect().equals(pl.DataFrame({"x": [False], "y": [False], "z": [True]}))


def test_neighbour_live_expr() -> None:
    df = (
        pl.LazyFrame({"x": [True]})
        .with_columns(y=pl.lit(False))
        .with_columns(x=pl.lit(False), z=pl.col("x") | False)
    )

    explain = df.explain()

    assert explain.count("WITH_COLUMNS") == 1
    assert df.collect().equals(pl.DataFrame({"x": [False], "y": [False], "z": [True]}))
