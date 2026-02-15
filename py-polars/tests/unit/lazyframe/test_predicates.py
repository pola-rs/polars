from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal
from polars.testing.asserts.series import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import PlMonkeyPatch


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

    assert_frame_equal(
        left.join_asof(right, by="id", on="timestamp")
        .filter(pl.col("value").is_not_null())
        .collect(),
        pl.DataFrame(
            {
                "id": [1, 2, 3],
                "timestamp": [
                    datetime(2022, 1, 1, 10, 0),
                    datetime(2022, 1, 1, 10, 1),
                    datetime(2022, 1, 1, 10, 2),
                ],
                "value": ["a", "b", "c"],
            }
        ),
        check_row_order=False,
    )


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


def test_predicate_pushdown_cumsum_9566() -> None:
    df = pl.DataFrame({"A": range(10), "B": ["b"] * 5 + ["a"] * 5})

    q = df.lazy().sort(["B", "A"]).filter(pl.col("A").is_in([8, 2]).cum_sum() == 1)

    assert q.collect()["A"].to_list() == [8, 9, 0, 1]


def test_predicate_pushdown_join_fill_null_10058() -> None:
    ids = pl.LazyFrame({"id": [0, 1, 2]})
    filters = pl.LazyFrame({"id": [0, 1], "filter": [True, False]})

    assert sorted(
        ids.join(filters, how="left", on="id")
        .filter(pl.col("filter").fill_null(True))
        .collect()
        .to_dict(as_series=False)["id"]
    ) == [0, 2]


def test_is_in_join_blocked() -> None:
    lf1 = pl.LazyFrame(
        {"Groups": ["A", "B", "C", "D", "E", "F"], "values0": [1, 2, 3, 4, 5, 6]}
    )
    lf2 = pl.LazyFrame(
        {"values_22": [1, 2, None, 4, 5, 6], "values_20": [1, 2, 3, 4, 5, 6]}
    )
    lf_all = lf2.join(
        lf1,
        left_on="values_20",
        right_on="values0",
        how="left",
        maintain_order="right_left",
    )

    for result in (
        lf_all.filter(~pl.col("Groups").is_in(["A", "B", "F"])),
        lf_all.remove(pl.col("Groups").is_in(["A", "B", "F"])),
    ):
        expected = pl.LazyFrame(
            {
                "values_22": [None, 4, 5],
                "values_20": [3, 4, 5],
                "Groups": ["C", "D", "E"],
            }
        )
        assert_frame_equal(result, expected)


def test_predicate_pushdown_group_by_keys() -> None:
    df = pl.LazyFrame(
        {"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]}
    ).lazy()
    q = (
        df.group_by("group")
        .agg([pl.len().alias("str_list")])
        .filter(pl.col("group") == 1)
    )
    assert not q.explain().startswith("FILTER")
    assert q.explain(
        optimizations=pl.QueryOptFlags(predicate_pushdown=False)
    ).startswith("FILTER")


def test_no_predicate_push_down_with_cast_and_alias_11883() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = (
        df.lazy()
        .select(pl.col("a").cast(pl.Int64).alias("b"))
        .filter(pl.col("b") == 1)
        .filter((pl.col("b") >= 1) & (pl.col("b") < 1))
    )
    assert (
        re.search(
            r"FILTER.*FROM\n\s*DF",
            out.explain(optimizations=pl.QueryOptFlags(predicate_pushdown=True)),
        )
        is None
    )


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
    with pytest.raises(TypeError, match="invalid predicate"):
        df.filter(predicate)


def test_fast_path_boolean_filter_predicates() -> None:
    df = pl.DataFrame({"colx": ["aa", "bb", "cc", "dd"]})
    df_empty = df.clear()

    assert_frame_equal(df.filter(False), df_empty)
    assert_frame_equal(df.filter(True), df)

    assert_frame_equal(df.remove(True), df_empty)
    assert_frame_equal(df.remove(False), df)


def test_predicate_pushdown_boundary_12102() -> None:
    df = pl.DataFrame({"x": [1, 2, 4], "y": [1, 2, 4]})

    lf = (
        df.lazy()
        .filter(pl.col("y") > 1)
        .filter(pl.col("x") == pl.min("x"))
        .filter(pl.col("y") > 2)
    )

    result = lf.collect()
    result_no_ppd = lf.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    assert_frame_equal(result, result_no_ppd)


def test_take_can_block_predicate_pushdown() -> None:
    df = pl.DataFrame({"x": [1, 2, 4], "y": [False, True, True]})
    lf = (
        df.lazy()
        .filter(pl.col("y"))
        .filter(pl.col("x") == pl.col("x").gather(0))
        .filter(pl.col("y"))
    )
    result = lf.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=True))
    assert result.to_dict(as_series=False) == {"x": [2], "y": [True]}


def test_literal_series_expr_predicate_pushdown() -> None:
    # No pushdown should occur in this case, because otherwise the filter will
    # attempt to filter 3 rows with a boolean mask of 2 rows.
    lf = pl.LazyFrame({"x": [0, 1, 2]})

    for res in (
        lf.filter(pl.col("x") > 0).filter(pl.Series([True, True])),
        lf.remove(pl.col("x") <= 0).remove(pl.Series([False, False])),
    ):
        assert res.collect().to_series().to_list() == [1, 2]

    # Pushdown should occur here; series is being used as part of an `is_in`.
    for res in (
        lf.filter(pl.col("x") > 0).filter(pl.col("x").is_in([0, 1])),
        lf.remove(pl.col("x") <= 0).remove(~pl.col("x").is_in([0, 1])),
    ):
        assert re.search(r"FILTER .*\nFROM\n\s*DF", res.explain(), re.DOTALL)
        assert res.collect().to_series().to_list() == [1]


def test_multi_alias_pushdown() -> None:
    lf = pl.LazyFrame({"a": [1], "b": [1]})

    actual = lf.with_columns(m="a", n="b").filter((pl.col("m") + pl.col("n")) < 2)
    plan = actual.explain()

    assert plan.count("FILTER") == 1
    assert re.search(r"FILTER.*FROM\n\s*DF", plan, re.DOTALL) is not None

    with pytest.warns(UserWarning, match="Comparisons with None always result in null"):
        # confirm we aren't using `eq_missing` in the query plan (denoted as " ==v ")
        assert " ==v " not in lf.select(pl.col("a").filter(a=None)).explain()


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

    assert (
        re.search(
            r'FILTER \[\(col\("key"\)\) == \(5\)\]\s*FROM\n\s*DF', plan, re.DOTALL
        )
        is not None
    )
    assert plan.count("FILTER") == 1

    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", "key_2").alias("value_2"),
            (pl.col("value") * 2).over("key", "key_2").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )

    plan = actual.explain()
    assert plan.count("FILTER") == 1
    assert re.search(r"FILTER.*FROM\n\s*DF", plan, re.DOTALL) is not None
    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", "key_2").alias("value_2"),
            (pl.col("value") * 2).over("key", "key_3").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )

    plan = actual.explain()
    assert plan.count("FILTER") == 2
    assert (
        re.search(
            r'FILTER \[\(col\("key"\)\) == \(5\)\]\s*FROM\n\s*DF', plan, re.DOTALL
        )
        is not None
    )

    actual = (
        lf.with_columns(
            (pl.col("value") * 2).over("key", pl.col("key_2") + 1).alias("value_2"),
            (pl.col("value") * 2).over("key", "key_2").alias("value_3"),
        )
        .filter(pl.col("key") == 5)
        .filter(pl.col("key_2") == 5)
    )
    plan = actual.explain()
    assert plan.count("FILTER") == 2
    assert (
        re.search(
            r'FILTER \[\(col\("key"\)\) == \(5\)\]\s*FROM\n\s*DF', plan, re.DOTALL
        )
        is not None
    )

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
    assert plan.count("FILTER") == 1
    assert "FILTER" in plan
    assert re.search(r"FILTER.*FROM\n\s*DF", plan, re.DOTALL) is None
    # Ensure the implementation doesn't accidentally push a window expression
    # that only refers to the common window keys.
    actual = lf.with_columns(
        (pl.col("value") * 2).over("key").alias("value_2"),
    ).filter(pl.len().over("key") == 1)

    plan = actual.explain()
    assert re.search(r"FILTER.*FROM\n\s*DF", plan, re.DOTALL) is None
    assert plan.count("FILTER") == 1

    # Test window in filter
    actual = lf.filter(pl.len().over("key") == 1).filter(pl.col("key") == 1)
    plan = actual.explain()
    assert plan.count("FILTER") == 2
    assert (
        re.search(
            r'FILTER \[\(len\(\).over\(\[col\("key"\)\]\)\) == \(1\)\]\s*FROM\n\s*FILTER',
            plan,
        )
        is not None
    )
    assert (
        re.search(
            r'FILTER \[\(col\("key"\)\) == \(1\)\]\s*FROM\n\s*DF', plan, re.DOTALL
        )
        is not None
    )


def test_predicate_reduction() -> None:
    # ensure we get clean reduction without casts
    lf = pl.LazyFrame({"a": [1], "b": [2]})
    for filter_frame in (lf.filter, lf.remove):
        assert (
            "cast"
            not in filter_frame(
                pl.col("a") > 1,
                pl.col("b") > 1,
            ).explain()
        )


def test_all_any_cleanup_at_single_predicate_case() -> None:
    plan = pl.LazyFrame({"a": [1], "b": [2]}).select(["a"]).drop_nulls().explain()
    assert "horizontal" not in plan
    assert "all" not in plan


def test_hconcat_predicate() -> None:
    # Predicates shouldn't be pushed down past an hconcat as we can't filter
    # across the different inputs
    lf1 = pl.LazyFrame(
        {
            "a1": [0, 1, 2, 3, 4],
            "a2": [5, 6, 7, 8, 9],
        }
    )
    lf2 = pl.LazyFrame(
        {
            "b1": [0, 1, 2, 3, 4],
            "b2": [5, 6, 7, 8, 9],
        }
    )

    query = pl.concat(
        [
            lf1.filter(pl.col("a1") < 4),
            lf2.filter(pl.col("b1") > 0),
        ],
        how="horizontal",
    ).filter(pl.col("b2") < 9)

    expected = pl.DataFrame(
        {
            "a1": [0, 1, 2],
            "a2": [5, 6, 7],
            "b1": [1, 2, 3],
            "b2": [6, 7, 8],
        }
    )
    result = query.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=True))
    assert_frame_equal(result, expected)


def test_predicate_pd_join_13300() -> None:
    # https://github.com/pola-rs/polars/issues/13300

    lf = pl.LazyFrame({"col3": range(10, 14), "new_col": range(11, 15)})
    lf_other = pl.LazyFrame({"col4": [0, 11, 2, 13]})

    lf = lf.join(lf_other, left_on="new_col", right_on="col4", how="left")
    for res in (
        lf.filter(pl.col("new_col") < 12),
        lf.remove(pl.col("new_col") >= 12),
    ):
        assert res.collect().to_dict(as_series=False) == {"col3": [10], "new_col": [11]}


def test_filter_eq_missing_13861() -> None:
    lf = pl.LazyFrame({"a": [1, None, 3], "b": ["xx", "yy", None]})
    lf_empty = lf.clear()

    with pytest.warns(UserWarning, match="Comparisons with None always result in null"):
        assert_frame_equal(lf.collect().filter(a=None), lf_empty.collect())

    with pytest.warns(UserWarning, match="Comparisons with None always result in null"):
        assert_frame_equal(lf.collect().remove(a=None), lf.collect())

    with pytest.warns(UserWarning, match="Comparisons with None always result in null"):
        lff = lf.filter(a=None)
    assert lff.collect().rows() == []
    assert " ==v " not in lff.explain()  # check no `eq_missing` op

    with pytest.warns(UserWarning, match="Comparisons with None always result in null"):
        assert_frame_equal(lf.collect().filter(a=None), lf_empty.collect())

    with pytest.warns(UserWarning, match="Comparisons with None always result in null"):
        assert_frame_equal(lf.collect().remove(a=None), lf.collect())

    for filter_expr in (
        pl.col("a").eq_missing(None),
        pl.col("a").is_null(),
    ):
        assert lf.collect().filter(filter_expr).rows() == [(None, "yy")]


@pytest.mark.parametrize("how", ["left", "inner"])
def test_predicate_pushdown_block_join(how: Any) -> None:
    q = (
        pl.LazyFrame({"a": [1]})
        .join(
            pl.LazyFrame({"a": [2], "b": [1]}),
            left_on=["a"],
            right_on=["b"],
            how=how,
        )
        .filter(pl.col("a") == 1)
    )
    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), q.collect())


def test_predicate_push_down_with_alias_15442() -> None:
    df = pl.DataFrame({"a": [1]})
    output = (
        df.lazy()
        .filter(pl.col("a").alias("x").drop_nulls() > 0)
        .collect(optimizations=pl.QueryOptFlags(predicate_pushdown=True))
    )
    assert output.to_dict(as_series=False) == {"a": [1]}


def test_predicate_slice_pushdown_list_gather_17492(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = pl.LazyFrame({"val": [[1], [1, 1]], "len": [1, 2]})

    assert_frame_equal(
        lf.filter(pl.col("len") == 2).filter(pl.col("val").list.get(1) == 1),
        lf.slice(1, 1),
    )

    # null_on_oob=True can pass

    plan = (
        lf.filter(pl.col("len") == 2)
        .filter(pl.col("val").list.get(1, null_on_oob=True) == 1)
        .explain()
    )

    assert re.search(r"FILTER.*FROM\n\s*DF", plan, re.DOTALL) is not None

    # Also check slice pushdown
    q = lf.with_columns(pl.col("val").list.get(1).alias("b")).slice(1, 1)

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "val": [[1, 1]],
                "len": pl.Series([2], dtype=pl.Int64),
                "b": pl.Series([1], dtype=pl.Int64),
            }
        ),
    )


def test_predicate_pushdown_struct_unnest_19632() -> None:
    lf = pl.LazyFrame({"a": [{"a": 1, "b": 2}]}).unnest("a")

    q = lf.filter(pl.col("a") == 1)
    plan = q.explain()

    assert "FILTER" in plan
    assert plan.index("FILTER") < plan.index("UNNEST")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": 1, "b": 2}),
    )

    # With `pl.struct()`
    lf = pl.LazyFrame({"a": 1, "b": 2}).select(pl.struct(pl.all())).unnest("a")

    q = lf.filter(pl.col("a") == 1)
    plan = q.explain()

    assert "FILTER" in plan
    assert plan.index("FILTER") < plan.index("UNNEST")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": 1, "b": 2}),
    )

    # With `value_counts()`
    lf = pl.LazyFrame({"a": [1]}).select(pl.col("a").value_counts()).unnest("a")

    q = lf.filter(pl.col("a") == 1)
    plan = q.explain()

    assert plan.index("FILTER") < plan.index("UNNEST")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {"a": 1, "count": 1}, schema={"a": pl.Int64, "count": pl.get_index_type()}
        ),
    )


@pytest.mark.parametrize(
    "predicate",
    [
        pl.col("v") == 7,
        pl.col("v") != 99,
        pl.col("v") > 0,
        pl.col("v") < 999,
        pl.col("v").is_in([7]),
        pl.col("v").cast(pl.Boolean),
        pl.col("b"),
    ],
)
@pytest.mark.parametrize("alias", [True, False])
@pytest.mark.parametrize("join_type", ["left", "right"])
def test_predicate_pushdown_join_19772(
    predicate: pl.Expr, join_type: str, alias: bool
) -> None:
    left = pl.LazyFrame({"k": [1, 2]})
    right = pl.LazyFrame({"k": [1], "v": [7], "b": True})

    if join_type == "right":
        [left, right] = [right, left]

    if alias:
        predicate = predicate.alias(":V")

    q = left.join(right, on="k", how=join_type).filter(predicate)  # type: ignore[arg-type]

    expect = pl.DataFrame({"k": 1, "v": 7, "b": True})

    if join_type == "right":
        expect = expect.select("v", "b", "k")

    assert_frame_equal(q.collect(optimizations=pl.QueryOptFlags.none()), expect)
    assert_frame_equal(q.collect(), expect)


def test_predicate_pushdown_scalar_20489() -> None:
    df = pl.DataFrame({"a": [1]})
    mask = pl.Series([False])

    assert_frame_equal(
        df.lazy().with_columns(b=pl.Series([2])).filter(mask).collect(),
        pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64}),
    )


def test_predicates_not_split_when_pushdown_disabled_20475() -> None:
    # This is important for the eager `DataFrame.filter()`, as that runs without
    # predicate pushdown enabled. Splitting the predicates in that case can
    # severely degrade performance.
    q = pl.LazyFrame({"a": 1, "b": 1, "c": 1}).filter(
        pl.col("a") > 0, pl.col("b") > 0, pl.col("c") > 0
    )
    assert (
        q.explain(optimizations=pl.QueryOptFlags(predicate_pushdown=False)).count(
            "FILTER"
        )
        == 1
    )


def test_predicate_filtering_against_nulls() -> None:
    df = pl.DataFrame({"num": [1, 2, None, 4]})

    for res in (
        df.filter(pl.col("num") > 2),
        df.filter(pl.col("num").is_in([3, 4, 5])),
    ):
        assert res["num"].to_list() == [4]

    for res in (
        df.remove(pl.col("num") <= 2),
        df.remove(pl.col("num").is_in([1, 2, 3])),
    ):
        assert res["num"].to_list() == [None, 4]

    for res in (
        df.filter(pl.col("num").ne_missing(None)),
        df.remove(pl.col("num").eq_missing(None)),
    ):
        assert res["num"].to_list() == [1, 2, 4]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            (
                pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
                .rename({"a": "A", "b": "a"})
                .select("A", "c")
                .filter(pl.col("A") == 1)
            ),
            pl.DataFrame({"A": 1, "c": 3}),
        ),
        (
            (
                pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
                .rename({"b": "a", "a": "A"})
                .select("A", "c")
                .filter(pl.col("A") == 1)
            ),
            pl.DataFrame({"A": 1, "c": 3}),
        ),
        (
            (
                pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
                .rename({"a": "b", "b": "a"})
                .select("a", "b", "c")
                .filter(pl.col("b") == 1)
            ),
            pl.DataFrame({"a": 2, "b": 1, "c": 3}),
        ),
        (
            (
                pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
                .rename({"a": "b", "b": "a"})
                .select("b", "c")
                .filter(pl.col("b") == 1)
            ),
            pl.DataFrame({"b": 1, "c": 3}),
        ),
        (
            (
                pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
                .rename({"b": "a", "a": "b"})
                .select("a", "b", "c")
                .filter(pl.col("b") == 1)
            ),
            pl.DataFrame({"a": 2, "b": 1, "c": 3}),
        ),
    ],
)
def test_predicate_pushdown_lazy_rename_22373(
    query: pl.LazyFrame,
    expected: pl.DataFrame,
) -> None:
    assert_frame_equal(
        query.collect(),
        expected,
    )

    # Ensure filter is pushed past rename
    plan = query.explain()
    assert plan.index("FILTER") > plan.index("SELECT")


@pytest.mark.parametrize(
    "base_query",
    [
        (  # Fallible expr in earlier `with_columns()`
            pl.LazyFrame({"a": [[1]]})
            .with_columns(MARKER=1)
            .with_columns(b=pl.col("a").list.get(1, null_on_oob=False))
        ),
        (  # Fallible expr in earlier `filter()`
            pl.LazyFrame({"a": [[1]]})
            .with_columns(MARKER=1)
            .filter(
                pl.col("a")
                .list.get(1, null_on_oob=False)
                .cast(pl.Boolean, strict=False)
            )
        ),
        (  # Fallible expr in earlier `select()`
            pl.LazyFrame({"a": [[1]]})
            .with_columns(MARKER=1)
            .select("a", "MARKER", b=pl.col("a").list.get(1, null_on_oob=False))
        ),
    ],
)
def test_predicate_pushdown_pushes_past_fallible(
    base_query: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Ensure baseline fails
    with pytest.raises(ComputeError, match="index is out of bounds"):
        base_query.collect()

    q = base_query.filter(pl.col("a").list.len() > 1)

    plan = q.explain()

    assert plan.index("list.len") > plan.index("MARKER")

    assert_frame_equal(q.collect(), pl.DataFrame(schema=q.collect_schema()))

    plmonkeypatch.setenv("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS", "1")

    with pytest.raises(ComputeError, match="index is out of bounds"):
        q.collect()


def test_predicate_pushdown_fallible_exprs_22284(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    q = (
        pl.LazyFrame({"a": ["xyz", "123", "456", "789"]})
        .with_columns(MARKER=1)
        .filter(pl.col.a.str.contains(r"^\d{3}$"))
        .filter(pl.col.a.cast(pl.Int64) >= 123)
    )

    plan = q.explain()

    assert (
        plan.index('FILTER [(col("a").strict_cast(Int64)) >= (123)]')
        < plan.index("MARKER")
        < plan.index(r'FILTER col("a").str.contains(["^\d{3}$"])')
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": ["123", "456", "789"],
                "MARKER": 1,
            }
        ),
    )

    lf = pl.LazyFrame(
        {
            "str_date": ["2025-01-01", "20250101"],
            "data_source": ["system_1", "system_2"],
        }
    )

    q = lf.filter(pl.col("data_source") == "system_1").filter(
        pl.col("str_date").str.to_datetime("%Y-%m-%d", strict=True)
        == datetime(2025, 1, 1)
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "str_date": ["2025-01-01"],
                "data_source": ["system_1"],
            }
        ),
    )

    q = lf.with_columns(
        pl.col("str_date").str.to_datetime("%Y-%m-%d", strict=True)
    ).filter(pl.col("data_source") == "system_1")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "str_date": [datetime(2025, 1, 1)],
                "data_source": ["system_1"],
            }
        ),
    )

    plmonkeypatch.setenv("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS", "1")

    with pytest.raises(
        InvalidOperationError, match=r"`str` to `datetime\[μs\]` failed"
    ):
        q.collect()


def test_predicate_pushdown_single_fallible() -> None:
    lf = pl.LazyFrame({"a": [0, 1]}).with_columns(MARKER=pl.lit(1, dtype=pl.Int64))

    q = lf.filter(pl.col("a").cast(pl.Boolean))

    plan = q.explain()

    assert plan.index('FILTER col("a").strict_cast(Boolean)') > plan.index("MARKER")

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1, "MARKER": 1}))


def test_predicate_pushdown_split_pushable(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = pl.LazyFrame({"a": [1, 999]}).with_columns(MARKER=pl.lit(1, dtype=pl.Int64))

    q = lf.filter(
        pl.col("a") == 1,  # pushable
        pl.col("a").cast(pl.Int8) == 1,  # fallible
    )

    plan = q.explain()

    assert (
        plan.index('FILTER [(col("a").strict_cast(Int8)) == (1)]')
        < plan.index("MARKER")
        < plan.index('FILTER [(col("a")) == (1)]')
    )

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1, "MARKER": 1}))

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS", "1")

        with pytest.raises(
            InvalidOperationError, match="conversion from `i64` to `i8` failed"
        ):
            q.collect()

    q = lf.filter(
        pl.col("a").cast(pl.UInt16) == 1,
        pl.col("a").sort() == 1,
    )

    plan = q.explain()

    assert plan.index(
        'FILTER [([(col("a").strict_cast(UInt16)) == (1)]) & ([(col("a").sort(asc)) == (1)])]'
    ) < plan.index("MARKER")

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1, "MARKER": 1}))

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_PUSHDOWN_OPT_MAINTAIN_ERRORS", "1")
        assert_frame_equal(q.collect(), pl.DataFrame({"a": 1, "MARKER": 1}))

    # Ensure it is not pushed past a join

    # Baseline
    q = lf.join(
        lf.drop("MARKER").collect().lazy(),
        on="a",
        how="inner",
        coalesce=False,
        maintain_order="left_right",
    ).filter(pl.col("a_right") == 1)

    plan = q.explain()

    assert not plan.startswith("FILTER")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": 1,
                "MARKER": 1,
                "a_right": 1,
            }
        ),
    )

    q = lf.join(
        lf.drop("MARKER").collect().lazy(),
        on="a",
        how="inner",
        coalesce=False,
        maintain_order="left_right",
    ).filter(pl.col("a_right").cast(pl.Int16) == 1)

    plan = q.explain()

    assert plan.startswith("FILTER")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": 1,
                "MARKER": 1,
                "a_right": 1,
            }
        ),
    )

    # With a select node in between

    q = (
        lf.join(
            lf.drop("MARKER").collect().lazy(),
            on="a",
            how="inner",
            coalesce=False,
            maintain_order="left_right",
        )
        .select(
            "a",
            "a_right",
            "MARKER",
        )
        .filter(pl.col("a_right").cast(pl.Int16) == 1)
    )

    plan = q.explain()

    assert plan.startswith("FILTER")

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": 1,
                "a_right": 1,
                "MARKER": 1,
            }
        ),
    )


def test_predicate_pushdown_fallible_literal_in_filter_expr() -> None:
    # Fallible operations on literals inside of the predicate expr should not
    # block pushdown.

    # Pushdown will also push any fallible expression if it's the only accumulated
    # predicate, we insert this dummy predicate to ensure the predicate is being
    # pushed solely because it is considered infallible.
    dummy_predicate = pl.lit(1) == pl.lit(1)

    lf = pl.LazyFrame(
        {"column": "2025-01-01", "column_date": datetime(2025, 1, 1), "integer": 1}
    )

    q = lf.with_columns(
        MARKER=1,
    ).filter(
        pl.col("column_date")
        == pl.lit("2025-01-01").str.to_datetime("%Y-%m-%d", strict=True),
        dummy_predicate,
    )

    plan = q.explain()

    assert plan.index("FILTER") > plan.index("MARKER")

    assert q.collect().height == 1

    q = lf.with_columns(
        MARKER=1,
    ).filter(
        pl.col("column_date") == pl.lit("2025-01-01").str.strptime(pl.Datetime),
        dummy_predicate,
    )

    plan = q.explain()

    assert plan.index("FILTER") > plan.index("MARKER")

    assert q.collect().height == 1

    q = lf.with_columns(
        MARKER=1,
    ).filter(
        pl.col("integer") == pl.lit("1").cast(pl.Int64, strict=True), dummy_predicate
    )

    plan = q.explain()

    assert plan.index("FILTER") > plan.index("MARKER")

    assert q.collect().height == 1


def test_predicate_does_not_split_barrier_expr() -> None:
    q = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .with_row_index()
        .filter(pl.col("a") > 1, pl.col("a").sort() == 3)
    )

    plan = q.explain()

    assert plan.startswith(
        'FILTER [([(col("a")) > (1)]) & ([(col("a").sort(asc)) == (3)])]'
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": 3}).with_row_index(offset=2),
    )


def test_predicate_passes_set_sorted_22397() -> None:
    plan = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .with_columns(MARKER=1, b=pl.lit(1))
        .set_sorted("a")
        .filter(pl.col("a") <= 1)
        .explain()
    )
    assert plan.index("FILTER") > plan.index("MARKER")


@pytest.mark.filterwarnings("ignore")
def test_predicate_pass() -> None:
    plan = (
        pl.LazyFrame({"a": [1, 2, 3]})
        .with_columns(MARKER=pl.col("a"))
        .filter(pl.col("a").map_elements(lambda x: x > 2, return_dtype=pl.Boolean))
        .explain()
    )
    assert plan.index("FILTER") > plan.index("MARKER")


def test_predicate_pushdown_auto_disable_strict() -> None:
    # Test that type-coercion automatically switches strict cast to
    # non-strict/overflowing for compatible types, allowing the predicate to be
    # pushed.
    lf = pl.LazyFrame(
        {"column": "2025-01-01", "column_date": datetime(2025, 1, 1), "integer": 1},
        schema={
            "column": pl.String,
            "column_date": pl.Datetime("ns"),
            "integer": pl.Int64,
        },
    )

    q = lf.with_columns(
        MARKER=1,
    ).filter(
        pl.col("column_date").cast(pl.Datetime("us")) == pl.lit(datetime(2025, 1, 1)),
        pl.col("integer") == 1,
    )

    plan = q.explain()
    assert plan.index("FILTER") > plan.index("MARKER")

    q = lf.with_columns(
        MARKER=1,
    ).filter(
        pl.col("column_date").cast(pl.Datetime("us"), strict=False)
        == pl.lit(datetime(2025, 1, 1)),
        pl.col("integer").cast(pl.Int128, strict=True) == 1,
    )

    plan = q.explain()
    assert plan.index("FILTER") > plan.index("MARKER")


@pytest.mark.may_fail_auto_streaming  # IO plugin validate=False schema mismatch
def test_predicate_pushdown_map_elements_io_plugin_22860() -> None:
    def generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        df = pl.DataFrame({"row_nr": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 1]})
        assert predicate is not None
        yield df.filter(predicate)

    q = register_io_source(
        io_source=generator, schema={"x": pl.Int64, "y": pl.Int64}
    ).filter(pl.col("y").map_elements(bool, return_dtype=pl.Boolean))

    plan = q.explain()
    assert plan.index("SELECTION") > plan.index("PYTHON SCAN")

    assert_frame_equal(q.collect(), pl.DataFrame({"row_nr": [2, 4, 5], "y": [1, 1, 1]}))


def test_duplicate_filter_removal_23243() -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3]})

    q = lf.filter(pl.col("x") == 2, pl.col("x") == 2)

    expect = pl.DataFrame({"x": [2]})

    plan = q.explain()

    assert plan.split("\n", 1)[0] == 'FILTER [(col("x")) == (2)]'

    assert_frame_equal(q.collect(), expect)


@pytest.mark.parametrize("maintain_order", [True, False])
def test_no_predicate_pushdown_on_modified_groupby_keys_21439(
    maintain_order: bool,
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    q = (
        df.lazy()
        .group_by(pl.col.a + 1, maintain_order=maintain_order)
        .agg()
        .filter(pl.col.a <= 3)
    )
    expected = pl.DataFrame({"a": [2, 3]})
    assert_frame_equal(q.collect(), expected, check_row_order=maintain_order)

    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    q = (
        df.lazy()
        .group_by([(pl.col.a + 1).alias("b"), pl.col.b.alias("a")], maintain_order=True)
        .agg()
        .filter(pl.col.b <= 2)
        .select(pl.col.b)
    )
    expected = pl.DataFrame({"b": [2]})
    assert_frame_equal(q.collect(), expected, check_row_order=maintain_order)


def test_no_predicate_pushdown_on_modified_groupby_keys_21439b() -> None:
    df = pl.DataFrame(
        {
            "time": pl.datetime_range(
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                timedelta(minutes=15),
                eager=True,
            )
        }
    )
    eager = (
        df.group_by(pl.col("time").dt.hour())
        .agg()
        .filter(pl.col("time").is_between(0, 10))
    )
    lazy = (
        df.lazy()
        .group_by(pl.col("time").dt.hour())
        .agg()
        .filter(pl.col("time").is_between(0, 10))
        .collect()
    )
    assert_frame_equal(eager, lazy, check_row_order=False)


def test_no_predicate_pushdown_unpivot() -> None:
    data = {"a": [5, 2, 8, 2], "b": [99, 33, 77, 44]}

    for index, pred in [("a", pl.col.a == 2), (["b", "a"], pl.col.b != 33)]:
        lf = pl.LazyFrame(data).unpivot(on="b", index=index).filter(pred)
        plan = lf.explain()
        assert plan.index("FILTER") > plan.index("UNPIVOT")


def test_replace_strict_predicate_merging() -> None:
    df = pl.LazyFrame({"x": [True, True, True, False]})
    out = (
        df.filter(pl.col("x")).filter(pl.col("x").replace_strict(True, True)).collect()
    )
    assert out.height == 3


def test_temporal_literal_narrowing_predicate_pushdown() -> None:
    """Temporal type mismatches should be resolved by casting the literal, not column.

    This avoids Cast nodes that block predicate pushdown to I/O.
    """
    # Date column vs Datetime literal (midnight = lossless)
    df = pl.LazyFrame(
        {
            "d": [date(2024, 1, 10), date(2024, 1, 15), date(2024, 1, 20)],
        }
    )

    lf = df.filter(pl.col("d") < datetime(2024, 1, 16))
    plan = lf.explain()
    # The plan should NOT contain a cast on the date column.
    assert "cast" not in plan.lower(), f"Unexpected cast in plan:\n{plan}"
    result = lf.collect()
    assert result["d"].to_list() == [date(2024, 1, 10), date(2024, 1, 15)]

    # Date column vs Datetime literal (non-midnight = lossy, operator adjusted)
    lf_lossy = df.filter(pl.col("d") < datetime(2024, 1, 15, 12, 30))
    result_lossy = lf_lossy.collect()
    # date(2024,1,15) < datetime(2024,1,15,12:30) is true
    assert result_lossy["d"].to_list() == [date(2024, 1, 10), date(2024, 1, 15)]

    lf_lossy_gt = df.filter(pl.col("d") >= datetime(2024, 1, 15, 12, 30))
    result_lossy_gt = lf_lossy_gt.collect()
    # date(2024,1,15) >= datetime(2024,1,15,12:30) is false (date is midnight)
    assert result_lossy_gt["d"].to_list() == [date(2024, 1, 20)]

    # Datetime column vs Date literal (upcast path should not abort optimization).
    lf_upcast = pl.LazyFrame(
        {
            "dt": [datetime(2024, 1, 10), datetime(2024, 1, 15), datetime(2024, 1, 20)],
        }
    ).filter(pl.col("dt") > date(2024, 1, 15))
    plan_upcast = lf_upcast.explain()
    assert "cast" not in plan_upcast.lower(), f"Unexpected cast in plan:\n{plan_upcast}"
    assert lf_upcast.collect()["dt"].to_list() == [datetime(2024, 1, 20)]

    # Datetime(us) column vs Datetime(ns) literal
    df_dt = pl.LazyFrame(
        {
            "dt": pl.Series(
                [
                    datetime(2024, 1, 10),
                    datetime(2024, 1, 15),
                    datetime(2024, 1, 20),
                ]
            ).cast(pl.Datetime("us")),
        }
    )
    ns_lit = pl.lit(datetime(2024, 1, 16)).cast(pl.Datetime("ns"))
    lf_dt = df_dt.filter(pl.col("dt") < ns_lit)
    plan_dt = lf_dt.explain()
    assert "cast" not in plan_dt.lower(), f"Unexpected cast in plan:\n{plan_dt}"
    result_dt = lf_dt.collect()
    assert result_dt.height == 2

    # Duration(us) column vs Duration(ns) literal
    df_dur = pl.LazyFrame(
        {
            "dur": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
                dtype=pl.Duration("us"),
            ),
        }
    )
    ns_dur_lit = pl.lit(timedelta(days=2, hours=12)).cast(pl.Duration("ns"))
    lf_dur = df_dur.filter(pl.col("dur") < ns_dur_lit)
    plan_dur = lf_dur.explain()
    assert "cast" not in plan_dur.lower(), f"Unexpected cast in plan:\n{plan_dur}"
    result_dur = lf_dur.collect()
    assert result_dur.height == 2  # 1 day and 2 days are < 2.5 days


def test_temporal_literal_lossy_comparison_correctness() -> None:
    """Lossy temporal comparisons should produce results identical to eager.

    Non-midnight datetime vs date triggers operator adjustment.
    """
    dates = [date(2024, 1, d) for d in range(1, 32)]
    df = pl.DataFrame({"d": dates})
    lf = df.lazy()

    # Non-midnight datetime literal (lossy)
    dt_lit = datetime(2024, 1, 15, 12, 30)

    for _, op_fn in [
        ("lt", _op_lt),
        ("le", _op_le),
        ("gt", _op_gt),
        ("ge", _op_ge),
    ]:
        lazy_result = lf.filter(op_fn(pl.col("d"), dt_lit)).collect()
        eager_result = df.filter(op_fn(pl.col("d"), dt_lit))
        assert_frame_equal(
            lazy_result,
            eager_result,
            check_exact=True,
        )


def test_temporal_literal_narrowing_no_timezone() -> None:
    """Temporal literal narrowing should NOT apply with timezone-aware types.

    This avoids incorrect results from DST offsets in Date<->Datetime conversion.
    """
    df = pl.LazyFrame(
        {
            "d": [date(2024, 1, 10), date(2024, 1, 15), date(2024, 1, 20)],
        }
    )
    tz_lit = pl.lit(datetime(2024, 1, 16)).cast(pl.Datetime("us", "UTC"))
    lf = df.filter(pl.col("d") < tz_lit)
    # Should still produce correct results (via normal cast path)
    result = lf.collect()
    assert result["d"].to_list() == [date(2024, 1, 10), date(2024, 1, 15)]


def test_temporal_literal_narrowing_equal_timezone_datetime() -> None:
    """Narrowing should apply for Datetime<->Datetime with equal timezone."""
    dt_df = pl.DataFrame(
        {
            "dt": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),
                None,
            ]
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("ns", "UTC")))
    dt_lf = dt_df.lazy()

    low = datetime(1500, 1, 1, tzinfo=timezone.utc)
    high = datetime(3000, 1, 1, tzinfo=timezone.utc)

    # Out-of-range for ns, representable in us: should be rewritten, not null-cast.
    plan_hi = dt_lf.filter(pl.col("dt") < high).explain().lower()
    plan_lo = dt_lf.filter(pl.col("dt") > low).explain().lower()
    assert "(null)" not in plan_hi
    assert "(null)" not in plan_lo
    assert "cast" not in plan_hi
    assert "cast" not in plan_lo

    op_fns = [_op_lt, _op_le, _op_gt, _op_ge, _op_eq, _op_ne]
    for lit in [low, high]:
        for op_fn in op_fns:
            _assert_filter_matches_baseline(
                dt_df,
                op_fn(pl.col("dt"), lit),
                op_fn(pl.col("dt").cast(pl.Datetime("us", "UTC")), lit),
            )
            _assert_filter_matches_baseline(
                dt_df,
                op_fn(pl.lit(lit), pl.col("dt")),
                op_fn(pl.lit(lit), pl.col("dt").cast(pl.Datetime("us", "UTC"))),
            )


def _assert_filter_matches_baseline(
    df: pl.DataFrame, predicate: pl.Expr, baseline_predicate: pl.Expr
) -> None:
    assert_frame_equal(
        df.lazy().filter(predicate).collect(),
        df.filter(baseline_predicate),
        check_exact=True,
    )


def _assert_select_matches_baseline(
    df: pl.DataFrame, expr: pl.Expr, baseline_expr: pl.Expr
) -> None:
    assert_frame_equal(
        df.lazy().select(expr.alias("out")).collect(),
        df.select(baseline_expr.alias("out")),
        check_exact=True,
    )


def _op_lt(lhs: pl.Expr, rhs: Any) -> pl.Expr:
    return cast("pl.Expr", lhs < rhs)


def _op_le(lhs: pl.Expr, rhs: Any) -> pl.Expr:
    return cast("pl.Expr", lhs <= rhs)


def _op_gt(lhs: pl.Expr, rhs: Any) -> pl.Expr:
    return cast("pl.Expr", lhs > rhs)


def _op_ge(lhs: pl.Expr, rhs: Any) -> pl.Expr:
    return cast("pl.Expr", lhs >= rhs)


def _op_eq(lhs: pl.Expr, rhs: Any) -> pl.Expr:
    return cast("pl.Expr", lhs == rhs)


def _op_ne(lhs: pl.Expr, rhs: Any) -> pl.Expr:
    return cast("pl.Expr", lhs != rhs)


def test_temporal_literal_narrowing_out_of_range_correctness() -> None:
    """Out-of-range temporal literal narrowing must preserve comparison semantics."""
    op_fns = [_op_lt, _op_le, _op_gt, _op_ge, _op_eq, _op_ne]

    # Datetime(ns) column: Python datetime literals are Datetime(us).
    # 1500/3000 are out of Datetime(ns) range but representable in Datetime(us).
    dt_df = pl.DataFrame(
        {"dt": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]}
    ).with_columns(pl.col("dt").cast(pl.Datetime("ns")))
    dt_lf = dt_df.lazy()
    assert "(null)" not in dt_lf.filter(pl.col("dt") < datetime(3000, 1, 1)).explain()
    assert "(null)" not in dt_lf.filter(pl.col("dt") > datetime(1500, 1, 1)).explain()
    plan_ne_hi = dt_lf.filter(pl.col("dt") != datetime(3000, 1, 1)).explain().lower()
    plan_eq_lo = dt_lf.filter(pl.col("dt") == datetime(1500, 1, 1)).explain().lower()
    assert "(null)" not in plan_ne_hi
    assert "(null)" not in plan_eq_lo
    assert "cast" not in plan_ne_hi
    assert "cast" not in plan_eq_lo

    for dt_lit in [datetime(1500, 1, 1), datetime(3000, 1, 1)]:
        for op_fn in op_fns:
            _assert_filter_matches_baseline(
                dt_df,
                op_fn(pl.col("dt"), dt_lit),
                op_fn(pl.col("dt").cast(pl.Datetime("us")), dt_lit),
            )
            _assert_filter_matches_baseline(
                dt_df,
                op_fn(pl.lit(dt_lit), pl.col("dt")),
                op_fn(pl.lit(dt_lit), pl.col("dt").cast(pl.Datetime("us"))),
            )

    # Duration(ns) column with Duration(us) literal out of Duration(ns) range.
    dur_df = pl.DataFrame(
        {
            "dur": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
                dtype=pl.Duration("ns"),
            )
        }
    )
    dur_lf = dur_df.lazy()
    dur_literals = [
        pl.lit(-(10**18)).cast(pl.Duration("us")),
        pl.lit(10**18).cast(pl.Duration("us")),
    ]
    assert "(null)" not in dur_lf.filter(pl.col("dur") < dur_literals[1]).explain()
    assert "(null)" not in dur_lf.filter(pl.col("dur") > dur_literals[0]).explain()

    for dur_lit in dur_literals:
        for op_fn in op_fns:
            _assert_filter_matches_baseline(
                dur_df,
                op_fn(pl.col("dur"), dur_lit),
                op_fn(pl.col("dur").cast(pl.Duration("us")), dur_lit),
            )
            _assert_filter_matches_baseline(
                dur_df,
                op_fn(dur_lit, pl.col("dur")),
                op_fn(dur_lit, pl.col("dur").cast(pl.Duration("us"))),
            )


def test_temporal_literal_narrowing_out_of_range_null_semantics() -> None:
    """Out-of-range rewrites must preserve null-comparison semantics."""
    df = pl.DataFrame({"dt": [datetime(2024, 1, 1), None]}).with_columns(
        pl.col("dt").cast(pl.Datetime("ns"))
    )
    lf = df.lazy()

    high = datetime(3000, 1, 1)
    low = datetime(1500, 1, 1)

    exprs = [
        (pl.col("dt") < high, pl.col("dt").cast(pl.Datetime("us")) < high),
        (pl.col("dt") != high, pl.col("dt").cast(pl.Datetime("us")) != high),
        (pl.col("dt") > low, pl.col("dt").cast(pl.Datetime("us")) > low),
        (
            pl.lit(high) >= pl.col("dt"),
            pl.lit(high) >= pl.col("dt").cast(pl.Datetime("us")),
        ),
        (
            pl.lit(low) == pl.col("dt"),
            pl.lit(low) == pl.col("dt").cast(pl.Datetime("us")),
        ),
    ]

    for lazy_expr, eager_expr in exprs:
        _assert_select_matches_baseline(df, lazy_expr, eager_expr)


def test_temporal_literal_narrowing_negative_lossy_floor_duration() -> None:
    """Lossy ns->us narrowing must floor correctly for negative values."""
    df = pl.DataFrame(
        {
            "dur": pl.Series(
                [-2, -1, 0, 1, 2],
                dtype=pl.Int64,
            ).cast(pl.Duration("us"))
        }
    )
    lf = df.lazy()

    lit = pl.lit(-1).cast(
        pl.Duration("ns")
    )  # -1ns would truncate to 0us without floor fix.
    assert "(null)" not in lf.filter(pl.col("dur") < lit).explain()

    op_fns = [_op_lt, _op_le, _op_gt, _op_ge]
    for op_fn in op_fns:
        _assert_filter_matches_baseline(
            df,
            op_fn(pl.col("dur"), lit),
            op_fn(pl.col("dur").cast(pl.Duration("ns")), lit),
        )
        _assert_filter_matches_baseline(
            df,
            op_fn(lit, pl.col("dur")),
            op_fn(lit, pl.col("dur").cast(pl.Duration("ns"))),
        )


def test_temporal_literal_narrowing_pre_1970_datetime() -> None:
    """Pre-1970 datetime comparisons should preserve semantics under narrowing."""
    df = pl.DataFrame(
        {
            "dt": [
                datetime(1969, 12, 31, 23, 59, 59, 999999),
                datetime(1970, 1, 1, 0, 0, 0),
                datetime(1970, 1, 1, 0, 0, 0, 1),
            ]
        }
    ).with_columns(pl.col("dt").cast(pl.Datetime("ns")))

    lit = datetime(1970, 1, 1, 0, 0, 0, 1)
    for op_fn in (_op_lt, _op_le, _op_gt, _op_ge):
        _assert_filter_matches_baseline(
            df,
            op_fn(pl.col("dt"), lit),
            op_fn(pl.col("dt").cast(pl.Datetime("us")), lit),
        )
        _assert_filter_matches_baseline(
            df,
            op_fn(pl.lit(lit), pl.col("dt")),
            op_fn(pl.lit(lit), pl.col("dt").cast(pl.Datetime("us"))),
        )
