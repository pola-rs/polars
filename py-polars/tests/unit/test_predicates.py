from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.asserts.series import assert_series_equal


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
    ).collect().to_dict(as_series=False) == {
        "dt": [date(2022, 9, 10), date(2022, 9, 20)]
    }


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
    ).collect().to_dict(as_series=False) == {
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
    ).to_dict(as_series=False) == {"date": [date(2022, 1, 3)]}


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
    ).to_dict(as_series=False) == {"a": [1], "b": [1]}


def test_fast_path_comparisons() -> None:
    s = pl.Series(np.sort(np.random.randint(0, 50, 100)))

    assert_series_equal(s > 25, s.set_sorted() > 25)
    assert_series_equal(s >= 25, s.set_sorted() >= 25)
    assert_series_equal(s < 25, s.set_sorted() < 25)
    assert_series_equal(s <= 25, s.set_sorted() <= 25)


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
    ).collect().to_dict(as_series=False) == {
        "g": [1, 2, 2],
        "t": [4, 2, 3],
        "x": [40, 30, 20],
    }


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

    assert out.to_dict(as_series=False) == {"df1_c1": [2, 3], "df1_c2": [3, 4]}


def test_predicate_pushdown_cumsum_9566() -> None:
    df = pl.DataFrame({"A": range(10), "B": ["b"] * 5 + ["a"] * 5})

    q = df.lazy().sort(["B", "A"]).filter(pl.col("A").is_in([8, 2]).cum_sum() == 1)

    assert q.collect()["A"].to_list() == [8, 9, 0, 1]


def test_predicate_pushdown_join_fill_null_10058() -> None:
    ids = pl.LazyFrame({"id": [0, 1, 2]})
    filters = pl.LazyFrame({"id": [0, 1], "filter": [True, False]})

    assert (
        ids.join(filters, how="left", on="id")
        .filter(pl.col("filter").fill_null(True))
        .collect()
        .to_dict(as_series=False)["id"]
    ) == [0, 2]


def test_is_in_join_blocked() -> None:
    df1 = pl.DataFrame(
        {"Groups": ["A", "B", "C", "D", "E", "F"], "values0": [1, 2, 3, 4, 5, 6]}
    ).lazy()

    df2 = pl.DataFrame(
        {"values22": [1, 2, None, 4, 5, 6], "values20": [1, 2, 3, 4, 5, 6]}
    ).lazy()

    df_all = df2.join(df1, left_on="values20", right_on="values0", how="left")

    result = df_all.filter(~pl.col("Groups").is_in(["A", "B", "F"])).collect()
    expected = {
        "values22": [None, 4, 5],
        "values20": [3, 4, 5],
        "Groups": ["C", "D", "E"],
    }
    assert result.to_dict(as_series=False) == expected


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


def test_no_predicate_push_down_with_cast_and_alias_11883() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = (
        df.lazy()
        .select(pl.col("a").cast(pl.Int64).alias("b"))
        .filter(pl.col("b") == 1)
        .filter((pl.col("b") >= 1) & (pl.col("b") < 1))
    )
    assert 'SELECTION: "None"' in out.explain(predicate_pushdown=True)


@pytest.mark.parametrize(
    "predicate",
    [
        0,
        "x",
        [2, 3],
        {"x": 1},
        pl.Series([1, 2, 3]),
        None,
    ],
)
def test_invalid_filter_predicates(predicate: Any) -> None:
    df = pl.DataFrame({"colx": ["aa", "bb", "cc", "dd"]})
    with pytest.raises(ValueError, match="invalid predicate"):
        df.filter(predicate)


def test_fast_path_boolean_filter_predicates() -> None:
    df = pl.DataFrame({"colx": ["aa", "bb", "cc", "dd"]})
    assert_frame_equal(df.filter(False), pl.DataFrame(schema={"colx": pl.Utf8}))
    assert_frame_equal(df.filter(True), df)


def test_predicate_pushdown_boundary_12102() -> None:
    df = pl.DataFrame({"x": [1, 2, 4], "y": [1, 2, 4]})

    lf = (
        df.lazy()
        .filter(pl.col("y") > 1)
        .filter(pl.col("x") == pl.min("x"))
        .filter(pl.col("y") > 2)
    )

    result = lf.collect()
    result_no_ppd = lf.collect(predicate_pushdown=False)
    assert_frame_equal(result, result_no_ppd)


def test_take_can_block_predicate_pushdown() -> None:
    df = pl.DataFrame({"x": [1, 2, 4], "y": [False, True, True]})

    lf = (
        df.lazy()
        .filter(pl.col("y"))
        .filter(pl.col("x") == pl.col("x").gather(0))
        .filter(pl.col("y"))
    )
    result = lf.collect(predicate_pushdown=True)
    expected = {"x": [2], "y": [True]}
    assert result.to_dict(as_series=False) == expected


def test_literal_series_expr_predicate_pushdown() -> None:
    # No pushdown should occur in this case, because otherwise the filter will
    # attempt to filter 3 rows with a boolean mask of 2 rows.
    lf = (
        pl.LazyFrame({"x": [0, 1, 2]})
        .filter(pl.col("x") > 0)
        .filter(pl.Series([True, True]))
    )

    assert lf.collect().to_series().to_list() == [1, 2]

    # Pushdown should occur here, because the series is being used as part of
    # an `is_in`.
    lf = (
        pl.LazyFrame({"x": [0, 1, 2]})
        .filter(pl.col("x") > 0)
        .filter(pl.col("x").is_in([0, 1]))
    )

    assert "FILTER" not in lf.explain()
    assert lf.collect().to_series().to_list() == [1]


def test_multi_alias_pushdown() -> None:
    lf = pl.LazyFrame({"a": [1], "b": [1]})

    actual = lf.with_columns(m="a", n="b").filter((pl.col("m") + pl.col("n")) < 2)

    plan = actual.explain()
    assert "FILTER" not in plan
    assert r'SELECTION: "[([(col(\"a\")) + (col(\"b\"))]) < (2)]' in plan


def test_predicate_pushdown_with_window_projections_12637() -> None:
    lf = pl.LazyFrame(
        {
            "key": [1],
            "key_2": [1],
            "key_3": [1],
            "value": [1],
            "value_2": [1],
            "value_3": [1],
        }
    )

    actual = lf.with_columns(
        (pl.col("value") * 2).over("key").alias("value_2"),
        (pl.col("value") * 2).over("key").alias("value_3"),
    ).filter(pl.col("key") == 5)

    plan = actual.explain()
    assert "FILTER" not in plan
    assert r'SELECTION: "[(col(\"key\")) == (5)]"' in plan

    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", "key_2").alias("value_2"),
            (pl.col("value") * 2).over("key", "key_2").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )

    plan = actual.explain()
    assert "FILTER" not in plan
    assert (
        # hashbrown::HashMap is unordered.
        r'SELECTION: "[([(col(\"key\")) == (5)]) & ([(col(\"key_2\")) == (5)])]"'
        in plan
        or r'SELECTION: "[([(col(\"key_2\")) == (5)]) & ([(col(\"key\")) == (5)])]"'
        in plan
    )

    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", "key_2").alias("value_2"),
            (pl.col("value") * 2).over("key", "key_3").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )

    plan = actual.explain()
    assert "FILTER" in plan
    assert r'SELECTION: "[(col(\"key\")) == (5)]"' in plan

    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", pl.col("key_2") + 1).alias("value_2"),
            (pl.col("value") * 2).over("key", "key_2").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )
    plan = actual.explain()
    assert "FILTER" in plan
    assert r'SELECTION: "[(col(\"key\")) == (5)]"' in plan

    # Should block when .over() contains groups-sensitive expr
    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", pl.sum("key_2")).alias("value_2"),
            (pl.col("value") * 2).over("key", "key_2").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )

    plan = actual.explain()
    assert "FILTER" in plan
    assert 'SELECTION: "None"' in plan

    # Ensure the implementation doesn't accidentally push a window expression
    # that only refers to the common window keys.
    actual = lf.with_columns(
        (pl.col("value") * 2).over("key").alias("value_2"),
    ).filter(pl.count().over("key") == 1)

    plan = actual.explain()
    assert r'FILTER [(count().over([col("key")])) == (1)]' in plan
    assert 'SELECTION: "None"' in plan

    # Test window in filter
    actual = lf.filter(pl.count().over("key") == 1).filter(pl.col("key") == 1)
    plan = actual.explain()
    assert r'FILTER [(count().over([col("key")])) == (1)]' in plan
    assert r'SELECTION: "[(col(\"key\")) == (1)]"' in plan
