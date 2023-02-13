from __future__ import annotations

from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal


def test_sort_dates_multiples() -> None:
    df = pl.DataFrame(
        [
            pl.Series(
                "date",
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-03 00:00:00",
                ],
            ).str.strptime(pl.Datetime, "%Y-%m-%d %T"),
            pl.Series("values", [5, 4, 3, 2, 1]),
        ]
    )

    expected = [4, 5, 2, 3, 1]

    # datetime
    out: pl.DataFrame = df.sort(["date", "values"])
    assert out["values"].to_list() == expected

    # Date
    out = df.with_columns(pl.col("date").cast(pl.Date)).sort(["date", "values"])
    assert out["values"].to_list() == expected


def test_sort_by() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 2, 2], "c": [2, 3, 1, 2, 1]}
    )

    by: list[pl.Expr | str]
    for by in [["b", "c"], [pl.col("b"), "c"]]:  # type: ignore[assignment]
        out = df.select([pl.col("a").sort_by(by)])
        assert out["a"].to_list() == [3, 1, 2, 5, 4]

    out = df.select([pl.col("a").sort_by(by, reverse=[False])])
    assert out["a"].to_list() == [3, 1, 2, 5, 4]

    out = df.select([pl.col("a").sort_by(by, reverse=[True])])
    assert out["a"].to_list() == [4, 5, 2, 1, 3]

    out = df.select([pl.col("a").sort_by(by, reverse=[True, False])])
    assert out["a"].to_list() == [5, 4, 3, 1, 2]

    # by can also be a single column
    out = df.select([pl.col("a").sort_by("b", reverse=[False])])
    assert out["a"].to_list() == [1, 2, 3, 4, 5]


def test_sort_by_exprs() -> None:
    # make sure that the expression does not overwrite columns in the dataframe
    df = pl.DataFrame({"a": [1, 2, -1, -2]})
    out = df.sort(pl.col("a").abs()).to_series()

    assert out.to_list() == [1, -1, 2, -2]


def test_arg_sort_nulls() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0, None, None])
    assert a.arg_sort(nulls_last=True).to_list() == [0, 1, 2, 4, 3]
    assert a.arg_sort(nulls_last=False).to_list() == [3, 4, 0, 1, 2]

    assert a.to_frame().sort(by="a", nulls_last=False).to_series().to_list() == [
        None,
        None,
        1.0,
        2.0,
        3.0,
    ]
    assert a.to_frame().sort(by="a", nulls_last=True).to_series().to_list() == [
        1.0,
        2.0,
        3.0,
        None,
        None,
    ]


def test_arg_sort_window_functions() -> None:
    df = pl.DataFrame({"Id": [1, 1, 2, 2, 3, 3], "Age": [1, 2, 3, 4, 5, 6]})
    out = df.select(
        [
            pl.col("Age").arg_sort().over("Id").alias("arg_sort"),
            pl.arg_sort_by("Age").over("Id").alias("arg_sort_by"),
        ]
    )

    assert (
        out["arg_sort"].to_list() == out["arg_sort_by"].to_list() == [0, 1, 0, 1, 0, 1]
    )


def test_sort_nans_3740() -> None:
    df = pl.DataFrame(
        {
            "key": [1, 2, 3, 4, 5],
            "val": [0.0, None, float("nan"), float("-inf"), float("inf")],
        }
    )
    assert df.sort("val")["key"].to_list() == [2, 4, 1, 5, 3]


def test_sort_by_exps_nulls_last() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 3, -2, None, 1],
        }
    ).with_row_count()

    assert df.sort(pl.col("a") ** 2, nulls_last=True).to_dict(False) == {
        "row_nr": [0, 4, 2, 1, 3],
        "a": [1, 1, -2, 3, None],
    }


def test_sort_aggregation_fast_paths() -> None:
    df = pl.DataFrame(
        {
            "a": [None, 3, 2, 1],
            "b": [3, 2, 1, None],
            "c": [3, None, None, None],
            "e": [None, None, None, 1],
            "f": [1, 2, 5, 1],
        }
    )

    expected = df.select(
        [
            pl.all().max().suffix("_max"),
            pl.all().min().suffix("_min"),
        ]
    )

    assert expected.to_dict(False) == {
        "a_max": [3],
        "b_max": [3],
        "c_max": [3],
        "e_max": [1],
        "f_max": [5],
        "a_min": [1],
        "b_min": [1],
        "c_min": [3],
        "e_min": [1],
        "f_min": [1],
    }

    for reverse in [True, False]:
        for null_last in [True, False]:
            out = df.select(
                [
                    pl.all()
                    .sort(reverse=reverse, nulls_last=null_last)
                    .max()
                    .suffix("_max"),
                    pl.all()
                    .sort(reverse=reverse, nulls_last=null_last)
                    .min()
                    .suffix("_min"),
                ]
            )
            assert_frame_equal(out, expected)


def test_sorted_join_and_dtypes() -> None:
    for dt in [pl.Int8, pl.Int16, pl.Int32, pl.Int16]:
        df_a = (
            pl.DataFrame({"a": [-5, -2, 3, 3, 9, 10]})
            .with_row_count()
            .with_columns(pl.col("a").cast(dt).set_sorted())
        )

    df_b = pl.DataFrame({"a": [-2, -3, 3, 10]}).with_columns(
        pl.col("a").cast(dt).set_sorted()
    )

    assert df_a.join(df_b, on="a", how="inner").to_dict(False) == {
        "row_nr": [1, 2, 3, 5],
        "a": [-2, 3, 3, 10],
    }
    assert df_a.join(df_b, on="a", how="left").to_dict(False) == {
        "row_nr": [0, 1, 2, 3, 4, 5],
        "a": [-5, -2, 3, 3, 9, 10],
    }


def test_sorted_flag_reverse() -> None:
    s = pl.arange(0, 7, eager=True)
    assert s.flags["SORTED_ASC"]
    assert s.reverse().flags["SORTED_DESC"]


def test_sorted_fast_paths() -> None:
    s = pl.Series([1, 2, 3]).sort()
    rev = s.sort(reverse=True)

    assert rev.to_list() == [3, 2, 1]
    assert s.sort().to_list() == [1, 2, 3]

    s = pl.Series([None, 1, 2, 3]).sort()
    rev = s.sort(reverse=True)
    assert rev.to_list() == [None, 3, 2, 1]
    assert rev.sort(reverse=True).to_list() == [None, 3, 2, 1]
    assert rev.sort().to_list() == [None, 1, 2, 3]


def test_arg_sort_rank_nans() -> None:
    assert (
        pl.DataFrame(
            {
                "val": [1.0, float("NaN")],
            }
        )
        .with_columns(
            [
                pl.col("val").rank().alias("rank"),
                pl.col("val").arg_sort().alias("arg_sort"),
            ]
        )
        .select(["rank", "arg_sort"])
    ).to_dict(False) == {"rank": [1.0, 2.0], "arg_sort": [0, 1]}


def test_top_k() -> None:
    s = pl.Series([3, 1, 2, 5, 8])

    assert s.top_k(3).to_list() == [8, 5, 3]
    assert s.top_k(4, reverse=True).to_list() == [1, 2, 3, 5]

    # 5886
    df = pl.DataFrame({"test": [4, 3, 2, 1]})
    assert_frame_equal(df.select(pl.col("test").top_k(10)), df)


def test_sorted_flag_unset_by_arithmetic_4937() -> None:
    df = pl.DataFrame(
        {
            "ts": [1, 1, 1, 0, 1],
            "price": [3.3, 3.0, 3.5, 3.6, 3.7],
            "mask": [1, 1, 1, 1, 0],
        }
    )

    assert df.sort("price").groupby("ts").agg(
        [
            (pl.col("price") * pl.col("mask")).max().alias("pmax"),
            (pl.col("price") * pl.col("mask")).min().alias("pmin"),
        ]
    ).sort("ts").to_dict(False) == {
        "ts": [0, 1],
        "pmax": [3.6, 3.5],
        "pmin": [3.6, 0.0],
    }


def test_unset_sorted_flag_after_extend() -> None:
    df1 = pl.DataFrame({"Add": [37, 41], "Batch": [48, 49]}).sort("Add")
    df2 = pl.DataFrame({"Add": [37], "Batch": [67]}).sort("Add")

    df1.extend(df2)
    assert not df1["Add"].flags["SORTED_ASC"]
    df = df1.groupby("Add").agg([pl.col("Batch").min()]).sort("Add")
    assert df["Add"].flags["SORTED_ASC"]
    assert df.to_dict(False) == {"Add": [37, 41], "Batch": [48, 49]}


def test_set_sorted_schema() -> None:
    assert (
        pl.DataFrame({"A": [0, 1]}).lazy().with_columns(pl.col("A").set_sorted()).schema
    ) == {"A": pl.Int64}


def test_sort_slice_fast_path_5245() -> None:
    df = pl.DataFrame(
        {
            "foo": ["f", "c", "b", "a"],
            "bar": [1, 2, 3, 4],
        }
    ).lazy()

    assert df.sort("foo").limit(1).select("foo").collect().to_dict(False) == {
        "foo": ["a"]
    }


def test_explicit_list_agg_sort_in_groupby() -> None:
    df = pl.DataFrame({"A": ["a", "a", "a", "b", "b", "a"], "B": [1, 2, 3, 4, 5, 6]})

    # this was col().list().sort() before we changed the logic
    result = df.groupby("A").agg(pl.col("B").sort(reverse=True)).sort("A")
    expected = df.groupby("A").agg(pl.col("B").sort(reverse=True)).sort("A")
    assert_frame_equal(result, expected)


def test_sorted_join_query_5406() -> None:
    df = (
        pl.DataFrame(
            {
                "Datetime": [
                    "2022-11-02 08:00:00",
                    "2022-11-02 08:00:00",
                    "2022-11-02 08:01:00",
                    "2022-11-02 07:59:00",
                    "2022-11-02 08:02:00",
                    "2022-11-02 08:02:00",
                ],
                "Group": ["A", "A", "A", "B", "B", "B"],
                "Value": [1, 2, 1, 1, 2, 1],
            }
        )
        .with_columns(pl.col("Datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
        .with_row_count("RowId")
    )

    df1 = df.sort(by=["Datetime", "RowId"])

    filter1 = (
        df1.groupby(["Datetime", "Group"])
        .agg([pl.all().sort_by("Value", reverse=True).first()])
        .sort(["Datetime", "RowId"])
    )

    out = df1.join(filter1, on="RowId", how="left").select(
        pl.exclude(["Datetime_right", "Group_right"])
    )
    assert out["Value_right"].to_list() == [1, None, 2, 1, 2, None]


def test_sort_by_in_over_5499() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2],
            "idx": pl.arange(0, 6, eager=True),
            "a": [1, 3, 2, 3, 1, 2],
        }
    )
    assert df.select(
        [
            pl.col("idx").sort_by("a").over("group").alias("sorted_1"),
            pl.col("idx").shift(1).sort_by("a").over("group").alias("sorted_2"),
        ]
    ).to_dict(False) == {
        "sorted_1": [0, 2, 1, 4, 5, 3],
        "sorted_2": [None, 1, 0, 3, 4, None],
    }


def test_merge_sorted() -> None:
    df_a = (
        pl.date_range(datetime(2022, 1, 1), datetime(2022, 12, 1), "1mo")
        .to_frame("range")
        .with_row_count()
    )

    df_b = (
        pl.date_range(datetime(2022, 1, 1), datetime(2022, 12, 1), "2mo")
        .to_frame("range")
        .with_row_count()
        .with_columns(pl.col("row_nr") * 10)
    )
    out = df_a.merge_sorted(df_b, key="range")
    assert out["range"].is_sorted()
    assert out.to_dict(False) == {
        "row_nr": [0, 0, 1, 2, 10, 3, 4, 20, 5, 6, 30, 7, 8, 40, 9, 10, 50, 11],
        "range": [
            datetime(2022, 1, 1, 0, 0),
            datetime(2022, 1, 1, 0, 0),
            datetime(2022, 2, 1, 0, 0),
            datetime(2022, 3, 1, 0, 0),
            datetime(2022, 3, 1, 0, 0),
            datetime(2022, 4, 1, 0, 0),
            datetime(2022, 5, 1, 0, 0),
            datetime(2022, 5, 1, 0, 0),
            datetime(2022, 6, 1, 0, 0),
            datetime(2022, 7, 1, 0, 0),
            datetime(2022, 7, 1, 0, 0),
            datetime(2022, 8, 1, 0, 0),
            datetime(2022, 9, 1, 0, 0),
            datetime(2022, 9, 1, 0, 0),
            datetime(2022, 10, 1, 0, 0),
            datetime(2022, 11, 1, 0, 0),
            datetime(2022, 11, 1, 0, 0),
            datetime(2022, 12, 1, 0, 0),
        ],
    }
