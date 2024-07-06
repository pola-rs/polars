import polars as pl
from polars.testing import assert_frame_equal


def test_rename_invalidate_cache_15884() -> None:
    assert (
        pl.LazyFrame({"a": [1], "b": [1]})
        .rename({"b": "b1"})  # to cache schema
        .with_columns(
            c=pl.col("b1").drop_nulls(), d=pl.col("b1").drop_nulls()
        )  # to trigger CSE
        .select("c", "d")  # to trigger project push down
    ).collect().to_dict(as_series=False) == {"c": [1], "d": [1]}


def test_rename_lf() -> None:
    ldf = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    out = ldf.rename({"a": "foo", "b": "bar"}).collect()
    assert out.columns == ["foo", "bar", "c"]


def test_with_column_renamed_lf(fruits_cars: pl.DataFrame) -> None:
    res = fruits_cars.lazy().rename({"A": "C"}).collect()
    assert res.columns[0] == "C"


def test_rename_lf_lambda() -> None:
    ldf = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    out = ldf.rename(
        lambda col: "foo" if col == "a" else "bar" if col == "b" else col
    ).collect()
    assert out.columns == ["foo", "bar", "c"]


def test_with_column_renamed() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.rename({"b": "c"})
    expected = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
    assert_frame_equal(result, expected)


def test_rename_swap() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        }
    )

    out = df.rename({"a": "b", "b": "a"})
    expected = pl.DataFrame(
        {
            "b": [1, 2, 3, 4, 5],
            "a": [5, 4, 3, 2, 1],
        }
    )
    assert_frame_equal(out, expected)

    # 6195
    ldf = pl.DataFrame(
        {
            "weekday": [
                1,
            ],
            "priority": [
                2,
            ],
            "roundNumber": [
                3,
            ],
            "flag": [
                4,
            ],
        }
    ).lazy()

    # Rename some columns (note: swapping two columns)
    rename_dict = {
        "weekday": "priority",
        "priority": "weekday",
        "roundNumber": "round_number",
    }
    ldf = ldf.rename(rename_dict)

    # Select some columns
    ldf = ldf.select(["priority", "weekday", "round_number"])

    assert ldf.collect().to_dict(as_series=False) == {
        "priority": [1],
        "weekday": [2],
        "round_number": [3],
    }


def test_rename_same_name() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4, 5],
            "groups": ["A", "A", "B", "C", "B"],
        }
    ).lazy()
    df = df.rename({"groups": "groups"})
    df = df.select(["groups"])
    assert df.collect().to_dict(as_series=False) == {
        "groups": ["A", "A", "B", "C", "B"]
    }
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4, 5],
            "groups": ["A", "A", "B", "C", "B"],
            "test": [1, 2, 3, 4, 5],
        }
    ).lazy()
    df = df.rename({"nrs": "nrs", "groups": "groups"})
    df = df.select(["groups"])
    df.collect()
    assert df.collect().to_dict(as_series=False) == {
        "groups": ["A", "A", "B", "C", "B"]
    }


def test_rename_df(df: pl.DataFrame) -> None:
    out = df.rename({"strings": "bars", "int": "foos"})
    # check if we can select these new columns
    _ = out[["foos", "bars"]]


def test_rename_df_lambda() -> None:
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    out = df.rename(lambda col: "foo" if col == "a" else "bar" if col == "b" else col)
    assert out.columns == ["foo", "bar", "c"]


def test_rename_schema_order_6660() -> None:
    df = pl.DataFrame(
        {
            "a": [],
            "b": [],
            "c": [],
            "d": [],
        }
    )

    mapper = {"a": "1", "b": "2", "c": "3", "d": "4"}

    renamed = df.lazy().rename(mapper)

    computed = renamed.select([pl.all(), pl.col("4").alias("computed")])

    assert renamed.collect_schema() == renamed.collect().schema
    assert computed.collect_schema() == computed.collect().schema


def test_rename_schema_17427() -> None:
    assert (
        pl.LazyFrame({"A": [1]})
        .with_columns(B=2)
        .select(["A", "B"])
        .rename({"A": "C", "B": "A"})
        .select(["C", "A"])
        .collect()
    ).to_dict(as_series=False) == {"C": [1], "A": [2]}
