from datetime import date, datetime, timedelta

import numpy as np

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
        pl.min_horizontal((pl.col("dt") + one_day), date(2022, 9, 30))
        > date(2022, 9, 10)
    ).collect().to_dict(False) == {"dt": [date(2022, 9, 10), date(2022, 9, 20)]}


def test_predicate_null_block_asof_join() -> None:
    left = (
        pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "timestamp": [
                    datetime(2022, 1, 1, 10, 0),
                    datetime(2022, 1, 1, 10, 1),
                    datetime(2022, 1, 1, 10, 2),
                    datetime(2022, 1, 1, 10, 3),
                ],
            }
        )
        .lazy()
        .set_sorted("timestamp")
    )

    right = (
        pl.DataFrame(
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
        )
        .lazy()
        .set_sorted("timestamp")
    )

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


def test_predicate_strptime_6558() -> None:
    assert (
        pl.DataFrame({"date": ["2022-01-03", "2020-01-04", "2021-02-03", "2019-01-04"]})
        .lazy()
        .select(pl.col("date").str.strptime(pl.Date, format="%F"))
        .filter((pl.col("date").dt.year() == 2022) & (pl.col("date").dt.month() == 1))
        .collect()
    ).to_dict(False) == {"date": [date(2022, 1, 3)]}


def test_predicate_arr_first_6573() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [6, 5, 4, 3, 2, 1],
        }
    )

    assert (
        df.lazy()
        .with_columns(pl.col("a").implode())
        .with_columns(pl.col("a").list.first())
        .filter(pl.col("a") == pl.col("b"))
        .collect()
    ).to_dict(False) == {"a": [1], "b": [1]}


def test_fast_path_comparisons() -> None:
    s = pl.Series(np.sort(np.random.randint(0, 50, 100)))

    assert (s > 25).series_equal(s.set_sorted() > 25)
    assert (s >= 25).series_equal(s.set_sorted() >= 25)
    assert (s < 25).series_equal(s.set_sorted() < 25)
    assert (s <= 25).series_equal(s.set_sorted() <= 25)


def test_predicate_pushdown_block_8661() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 1, 2, 2, 2, 2],
            "t": [1, 2, 3, 4, 4, 3, 2, 1],
            "x": [10, 20, 30, 40, 10, 20, 30, 40],
        }
    )
    assert df.lazy().sort(["g", "t"]).filter(
        (pl.col("x").shift() > 20).over("g")
    ).collect().to_dict(False) == {"g": [1, 2, 2], "t": [4, 2, 3], "x": [40, 30, 20]}


def test_predicate_pushdown_with_context_11014() -> None:
    df1 = pl.LazyFrame(
        {
            "df1_c1": [1, 2, 3],
            "df1_c2": [2, 3, 4],
        }
    )

    df2 = pl.LazyFrame(
        {
            "df2_c1": [2, 3, 4],
            "df2_c2": [3, 4, 5],
        }
    )

    out = (
        df1.with_context(df2)
        .filter(pl.col("df1_c1").is_in(pl.col("df2_c1")))
        .collect(predicate_pushdown=True)
    )

    assert out.to_dict(False) == {"df1_c1": [2, 3], "df1_c2": [3, 4]}


def test_predicate_pushdown_cumsum_9566() -> None:
    df = pl.DataFrame({"A": range(10), "B": ["b"] * 5 + ["a"] * 5})

    q = df.lazy().sort(["B", "A"]).filter(pl.col("A").is_in([8, 2]).cumsum() == 1)

    assert q.collect()["A"].to_list() == [8, 9, 0, 1]


def test_predicate_pushdown_join_fill_null_10058() -> None:
    ids = pl.LazyFrame({"id": [0, 1, 2]})
    filters = pl.LazyFrame({"id": [0, 1], "filter": [True, False]})

    assert (
        ids.join(filters, how="left", on="id")
        .filter(pl.col("filter").fill_null(True))
        .collect()
        .to_dict(False)["id"]
    ) == [0, 2]


def test_is_in_join_blocked() -> None:
    df1 = pl.DataFrame(
        {"Groups": ["A", "B", "C", "D", "E", "F"], "values0": [1, 2, 3, 4, 5, 6]}
    ).lazy()

    df2 = pl.DataFrame(
        {"values22": [1, 2, None, 4, 5, 6], "values20": [1, 2, 3, 4, 5, 6]}
    ).lazy()

    df_all = df2.join(df1, left_on="values20", right_on="values0", how="left")
    assert df_all.filter(~pl.col("Groups").is_in(["A", "B", "F"])).collect().to_dict(
        False
    ) == {"values22": [None, 4, 5], "values20": [3, 4, 5], "Groups": ["C", "D", "E"]}


def test_predicate_pushdown_group_by_keys() -> None:
    df = pl.LazyFrame(
        {"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]}
    ).lazy()
    assert (
        'SELECTION: "None"'
        not in df.group_by("group")
        .agg([pl.count().alias("str_list")])
        .filter(pl.col("group") == 1)
        .explain()
    )
