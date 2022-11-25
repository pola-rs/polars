import typing
from datetime import date, datetime, timedelta

import polars as pl


def test_predicate_4906() -> None:
    one_day = timedelta(days=1)

    ldf = pl.DataFrame(
        {
            "dt": [
                date(2022, 9, 1),
                date(2022, 9, 10),
                date(2022, 9, 20),
            ]
        }
    ).lazy()

    assert ldf.filter(
        pl.min([(pl.col("dt") + one_day), date(2022, 9, 30)]) > date(2022, 9, 10)
    ).collect().to_dict(False) == {"dt": [date(2022, 9, 10), date(2022, 9, 20)]}


def test_when_then_implicit_none() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B", "C"],
            "points": [11, 8, 10, 6, 6, 5],
        }
    )

    assert df.select(
        [
            pl.when(pl.col("points") > 7).then("Foo"),
            pl.when(pl.col("points") > 7).then("Foo").alias("bar"),
        ]
    ).to_dict(False) == {
        "literal": ["Foo", "Foo", "Foo", None, None, None],
        "bar": ["Foo", "Foo", "Foo", None, None, None],
    }


def test_predicate_null_block_asof_join() -> None:
    left = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "timestamp": [
                datetime(2022, 1, 1, 10, 0),
                datetime(2022, 1, 1, 10, 1),
                datetime(2022, 1, 1, 10, 2),
                datetime(2022, 1, 1, 10, 3),
            ],
        }
    ).lazy()

    right = pl.DataFrame(
        {
            "id": [1, 2, 3] * 2,
            "timestamp": [
                datetime(2022, 1, 1, 9, 59, 50),
                datetime(2022, 1, 1, 10, 0, 50),
                datetime(2022, 1, 1, 10, 1, 50),
                datetime(2022, 1, 1, 8, 0, 0),
                datetime(2022, 1, 1, 8, 0, 0),
                datetime(2022, 1, 1, 8, 0, 0),
            ],
            "value": ["a", "b", "c"] * 2,
        }
    ).lazy()

    assert left.join_asof(right, by="id", on="timestamp").filter(
        pl.col("value").is_not_null()
    ).collect().to_dict(False) == {
        "id": [1, 2, 3],
        "timestamp": [
            datetime(2022, 1, 1, 10, 0),
            datetime(2022, 1, 1, 10, 1),
            datetime(2022, 1, 1, 10, 2),
        ],
        "value": ["a", "b", "c"],
    }


@typing.no_type_check
def test_streaming_empty_df() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", ["a", "b", "c", "b", "a", "a"], dtype=pl.Categorical()),
            pl.Series("b", ["b", "c", "c", "b", "a", "c"], dtype=pl.Categorical()),
        ]
    )

    assert df.lazy().join(df.lazy(), on="a", how="inner").filter(
        2 == 1  # noqa: SIM300
    ).collect(allow_streaming=True).to_dict(False) == {"a": [], "b": [], "b_right": []}


def test_when_then_empty_list_5547() -> None:
    out = pl.DataFrame({"a": []}).select([pl.when(pl.col("a") > 1).then([1])])
    assert out.shape == (0, 1)
    assert out.dtypes == [pl.List(pl.Int64)]
