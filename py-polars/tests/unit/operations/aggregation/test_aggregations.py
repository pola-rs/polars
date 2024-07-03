from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    import numpy.typing as npt

    from polars._typing import PolarsDataType


def test_quantile_expr_input() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0.0, 0.0, 0.3, 0.2, 0.0]})

    assert_frame_equal(
        df.select([pl.col("a").quantile(pl.col("b").sum() + 0.1)]),
        df.select(pl.col("a").quantile(0.6)),
    )


def test_boolean_aggs() -> None:
    df = pl.DataFrame({"bool": [True, False, None, True]})

    aggs = [
        pl.mean("bool").alias("mean"),
        pl.std("bool").alias("std"),
        pl.var("bool").alias("var"),
    ]
    assert df.select(aggs).to_dict(as_series=False) == {
        "mean": [0.6666666666666666],
        "std": [0.5773502691896258],
        "var": [0.33333333333333337],
    }

    assert df.group_by(pl.lit(1)).agg(aggs).to_dict(as_series=False) == {
        "literal": [1],
        "mean": [0.6666666666666666],
        "std": [0.5773502691896258],
        "var": [0.33333333333333337],
    }


def test_duration_aggs() -> None:
    df = pl.DataFrame(
        {
            "time1": pl.datetime_range(
                start=datetime(2022, 12, 12),
                end=datetime(2022, 12, 18),
                interval="1d",
                eager=True,
            ),
            "time2": pl.datetime_range(
                start=datetime(2023, 1, 12),
                end=datetime(2023, 1, 18),
                interval="1d",
                eager=True,
            ),
        }
    )

    df = df.with_columns((pl.col("time2") - pl.col("time1")).alias("time_difference"))

    assert df.select("time_difference").mean().to_dict(as_series=False) == {
        "time_difference": [timedelta(days=31)]
    }
    assert df.group_by(pl.lit(1)).agg(pl.mean("time_difference")).to_dict(
        as_series=False
    ) == {
        "literal": [1],
        "time_difference": [timedelta(days=31)],
    }


def test_mean_horizontal_with_str_column() -> None:
    assert pl.DataFrame(
        {"int": [1, 2, 3], "bool": [True, True, None], "str": ["a", "b", "c"]}
    ).mean_horizontal().to_list() == [1.0, 1.5, 3.0]


def test_list_aggregation_that_filters_all_data_6017() -> None:
    out = (
        pl.DataFrame({"col_to_group_by": [2], "flt": [1672740910.967138], "col3": [1]})
        .group_by("col_to_group_by")
        .agg((pl.col("flt").filter(col3=0).diff() * 1000).diff().alias("calc"))
    )

    assert out.schema == {"col_to_group_by": pl.Int64, "calc": pl.List(pl.Float64)}
    assert out.to_dict(as_series=False) == {"col_to_group_by": [2], "calc": [[]]}


def test_median() -> None:
    s = pl.Series([1, 2, 3])
    assert s.median() == 2


def test_single_element_std() -> None:
    s = pl.Series([1])
    assert s.std(ddof=1) is None
    assert s.std(ddof=0) == 0.0


def test_quantile() -> None:
    s = pl.Series([1, 2, 3])
    assert s.quantile(0.5, "nearest") == 2
    assert s.quantile(0.5, "lower") == 2
    assert s.quantile(0.5, "higher") == 2


@pytest.mark.slow()
@pytest.mark.parametrize("tp", [int, float])
@pytest.mark.parametrize("n", [1, 2, 10, 100])
def test_quantile_vs_numpy(tp: type, n: int) -> None:
    a: np.ndarray[Any, Any] = np.random.randint(0, 50, n).astype(tp)
    np_result: npt.ArrayLike | None = np.median(a)
    # nan check
    if np_result != np_result:
        np_result = None
    median = pl.Series(a).median()
    if median is not None:
        assert np.isclose(median, np_result)  # type: ignore[arg-type]
    else:
        assert np_result is None

    q = np.random.sample()
    try:
        np_result = np.quantile(a, q)
    except IndexError:
        np_result = None
    if np_result:
        # nan check
        if np_result != np_result:
            np_result = None
        assert np.isclose(
            pl.Series(a).quantile(q, interpolation="linear"),  # type: ignore[arg-type]
            np_result,  # type: ignore[arg-type]
        )


def test_mean_overflow() -> None:
    assert np.isclose(
        pl.Series([9_223_372_036_854_775_800, 100]).mean(),  # type: ignore[arg-type]
        4.611686018427388e18,
    )


def test_mean_null_simd() -> None:
    for dtype in [int, float]:
        df = (
            pl.Series(np.random.randint(0, 100, 1000))
            .cast(dtype)
            .to_frame("a")
            .select(pl.when(pl.col("a") > 40).then(pl.col("a")))
        )

        s = df["a"]
        assert s.mean() == s.to_pandas().mean()


def test_literal_group_agg_chunked_7968() -> None:
    df = pl.DataFrame({"A": [1, 1], "B": [1, 3]})
    ser = pl.concat([pl.Series([3]), pl.Series([4, 5])], rechunk=False)

    assert_frame_equal(
        df.group_by("A").agg(pl.col("B").search_sorted(ser)),
        pl.DataFrame(
            [
                pl.Series("A", [1], dtype=pl.Int64),
                pl.Series("B", [[1, 2, 2]], dtype=pl.List(pl.UInt32)),
            ]
        ),
    )


def test_duration_function_literal() -> None:
    df = pl.DataFrame(
        {
            "A": ["x", "x", "y", "y", "y"],
            "T": pl.datetime_range(
                date(2022, 1, 1), date(2022, 5, 1), interval="1mo", eager=True
            ),
            "S": [1, 2, 4, 8, 16],
        }
    )

    result = df.group_by("A", maintain_order=True).agg(
        (pl.col("T").max() + pl.duration(seconds=1)) - pl.col("T")
    )

    # this checks if the `pl.duration` is flagged as AggState::Literal
    expected = pl.DataFrame(
        {
            "A": ["x", "y"],
            "T": [
                [timedelta(days=31, seconds=1), timedelta(seconds=1)],
                [
                    timedelta(days=61, seconds=1),
                    timedelta(days=30, seconds=1),
                    timedelta(seconds=1),
                ],
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_string_par_materialize_8207() -> None:
    df = pl.LazyFrame(
        {
            "a": ["a", "b", "d", "c", "e"],
            "b": ["P", "L", "R", "T", "a long string"],
        }
    )

    assert df.group_by(["a"]).agg(pl.min("b")).sort("a").collect().to_dict(
        as_series=False
    ) == {
        "a": ["a", "b", "c", "d", "e"],
        "b": ["P", "L", "T", "R", "a long string"],
    }


def test_online_variance() -> None:
    df = pl.DataFrame(
        {
            "id": [1] * 5,
            "no_nulls": [1, 2, 3, 4, 5],
            "nulls": [1, None, 3, None, 5],
        }
    )

    assert_frame_equal(
        df.group_by("id")
        .agg(pl.all().exclude("id").std())
        .select(["no_nulls", "nulls"]),
        df.select(pl.all().exclude("id").std()),
    )


def test_err_on_implode_and_agg() -> None:
    df = pl.DataFrame({"type": ["water", "fire", "water", "earth"]})

    # this would OOB
    with pytest.raises(
        InvalidOperationError,
        match=r"'implode' followed by an aggregation is not allowed",
    ):
        df.group_by("type").agg(pl.col("type").implode().first().alias("foo"))

    # implode + function should be allowed in group_by
    assert df.group_by("type", maintain_order=True).agg(
        pl.col("type").implode().list.head().alias("foo")
    ).to_dict(as_series=False) == {
        "type": ["water", "fire", "earth"],
        "foo": [[["water", "water"]], [["fire"]], [["earth"]]],
    }

    # but not during a window function as the groups cannot be mapped back
    with pytest.raises(
        InvalidOperationError,
        match=r"'implode' followed by an aggregation is not allowed",
    ):
        df.lazy().select(pl.col("type").implode().list.head(1).over("type")).collect()


def test_mapped_literal_to_literal_9217() -> None:
    df = pl.DataFrame({"unique_id": ["a", "b"]})
    assert df.group_by(True).agg(
        pl.struct(pl.lit("unique_id").alias("unique_id"))
    ).to_dict(as_series=False) == {
        "literal": [True],
        "unique_id": [{"unique_id": "unique_id"}],
    }


def test_sum_empty_and_null_set() -> None:
    series = pl.Series("a", [], dtype=pl.Float32)
    assert series.sum() == 0

    series = pl.Series("a", [None], dtype=pl.Float32)
    assert series.sum() == 0

    df = pl.DataFrame(
        {"a": [None, None, None], "b": [1, 1, 1]},
        schema={"a": pl.Float32, "b": pl.Int64},
    )
    assert df.select(pl.sum("a")).item() == 0.0
    assert df.group_by("b").agg(pl.sum("a"))["a"].item() == 0.0


def test_horizontal_sum_null_to_identity() -> None:
    assert pl.DataFrame({"a": [1, 5], "b": [10, None]}).select(
        pl.sum_horizontal(["a", "b"])
    ).to_series().to_list() == [11, 5]


def test_horizontal_sum_bool_dtype() -> None:
    out = pl.DataFrame({"a": [True, False]}).select(pl.sum_horizontal("a"))
    assert_frame_equal(out, pl.DataFrame({"a": pl.Series([1, 0], dtype=pl.UInt32)}))


def test_horizontal_sum_in_group_by_15102() -> None:
    nbr_records = 1000
    out = (
        pl.LazyFrame(
            {
                "x": [None, "two", None] * nbr_records,
                "y": ["one", "two", None] * nbr_records,
                "z": [None, "two", None] * nbr_records,
            }
        )
        .select(pl.sum_horizontal(pl.all().is_null()).alias("num_null"))
        .group_by("num_null")
        .len()
        .sort(by="num_null")
        .collect()
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "num_null": pl.Series([0, 2, 3], dtype=pl.UInt32),
                "len": pl.Series([nbr_records] * 3, dtype=pl.UInt32),
            }
        ),
    )


def test_first_last_unit_length_12363() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [None, None],
        }
    )

    assert df.select(
        pl.all().drop_nulls().first().name.suffix("_first"),
        pl.all().drop_nulls().last().name.suffix("_last"),
    ).to_dict(as_series=False) == {
        "a_first": [1],
        "b_first": [None],
        "a_last": [2],
        "b_last": [None],
    }


def test_binary_op_agg_context_no_simplify_expr_12423() -> None:
    expect = pl.DataFrame({"x": [1], "y": [1]}, schema={"x": pl.Int64, "y": pl.Int32})

    for simplify_expression in (True, False):
        assert_frame_equal(
            expect,
            pl.LazyFrame({"x": [1]})
            .group_by("x")
            .agg(y=pl.lit(1) * pl.lit(1))
            .collect(simplify_expression=simplify_expression),
        )


def test_nan_inf_aggregation() -> None:
    df = pl.DataFrame(
        [
            ("both nan", np.nan),
            ("both nan", np.nan),
            ("nan and 5", np.nan),
            ("nan and 5", 5),
            ("nan and null", np.nan),
            ("nan and null", None),
            ("both none", None),
            ("both none", None),
            ("both inf", np.inf),
            ("both inf", np.inf),
            ("inf and null", np.inf),
            ("inf and null", None),
        ],
        schema=["group", "value"],
        orient="row",
    )

    assert_frame_equal(
        df.group_by("group", maintain_order=True).agg(
            min=pl.col("value").min(),
            max=pl.col("value").max(),
            mean=pl.col("value").mean(),
        ),
        pl.DataFrame(
            [
                ("both nan", np.nan, np.nan, np.nan),
                ("nan and 5", 5, 5, np.nan),
                ("nan and null", np.nan, np.nan, np.nan),
                ("both none", None, None, None),
                ("both inf", np.inf, np.inf, np.inf),
                ("inf and null", np.inf, np.inf, np.inf),
            ],
            schema=["group", "min", "max", "mean"],
            orient="row",
        ),
    )


@pytest.mark.parametrize("dtype", [pl.Int16, pl.UInt16])
def test_int16_max_12904(dtype: PolarsDataType) -> None:
    s = pl.Series([None, 1], dtype=dtype)

    assert s.min() == 1
    assert s.max() == 1


def test_agg_filter_over_empty_df_13610() -> None:
    ldf = pl.LazyFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [True, True, True, True, True],
            "c": [None, None, None, None, None],
        }
    )

    out = (
        ldf.drop_nulls()
        .group_by(["a"], maintain_order=True)
        .agg(pl.col("b").filter(pl.col("b").shift(1)))
        .collect()
    )
    expected = pl.DataFrame(schema={"a": pl.Int64, "b": pl.List(pl.Boolean)})
    assert_frame_equal(out, expected)

    df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.Boolean})
    out = df.group_by("a").agg(pl.col("b").filter(pl.col("b").shift()))
    expected = pl.DataFrame(schema={"a": pl.Int64, "b": pl.List(pl.Boolean)})
    assert_frame_equal(out, expected)


@pytest.mark.slow()
def test_agg_empty_sum_after_filter_14734() -> None:
    f = (
        pl.DataFrame({"a": [1, 2], "b": [1, 2]})
        .lazy()
        .group_by("a")
        .agg(pl.col("b").filter(pl.lit(False)).sum())
        .collect
    )

    last = f()

    # We need both possible output orders, which should happen within
    # 1000 iterations (during testing it usually happens within 10).
    limit = 1000
    i = 0
    while (curr := f()).equals(last):
        i += 1
        assert i != limit

    expect = pl.Series("b", [0, 0]).to_frame()
    assert_frame_equal(expect, last.select("b"))
    assert_frame_equal(expect, curr.select("b"))


@pytest.mark.slow()
def test_grouping_hash_14749() -> None:
    n_groups = 251
    rows_per_group = 4
    assert (
        pl.DataFrame(
            {
                "grp": np.repeat(np.arange(n_groups), rows_per_group),
                "x": np.tile(np.arange(rows_per_group), n_groups),
            }
        )
        .select(pl.col("x").max().over("grp"))["x"]
        .value_counts()
    ).to_dict(as_series=False) == {"x": [3], "count": [1004]}


@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [
        (pl.Boolean, pl.Float64),
        (pl.UInt8, pl.Float64),
        (pl.UInt16, pl.Float64),
        (pl.UInt32, pl.Float64),
        (pl.UInt64, pl.Float64),
        (pl.Int8, pl.Float64),
        (pl.Int16, pl.Float64),
        (pl.Int32, pl.Float64),
        (pl.Int64, pl.Float64),
        (pl.Float32, pl.Float32),
        (pl.Float64, pl.Float64),
    ],
)
def test_horizontal_mean_single_column(
    in_dtype: PolarsDataType,
    out_dtype: PolarsDataType,
) -> None:
    out = (
        pl.LazyFrame({"a": pl.Series([1, 0]).cast(in_dtype)})
        .select(pl.mean_horizontal(pl.all()))
        .collect()
    )

    assert_frame_equal(out, pl.DataFrame({"a": pl.Series([1.0, 0.0]).cast(out_dtype)}))


def test_horizontal_mean_in_group_by_15115() -> None:
    nbr_records = 1000
    out = (
        pl.LazyFrame(
            {
                "w": [None, "one", "two", "three"] * nbr_records,
                "x": [None, None, "two", "three"] * nbr_records,
                "y": [None, None, None, "three"] * nbr_records,
                "z": [None, None, None, None] * nbr_records,
            }
        )
        .select(pl.mean_horizontal(pl.all().is_null()).alias("mean_null"))
        .group_by("mean_null")
        .len()
        .sort(by="mean_null")
        .collect()
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "mean_null": pl.Series([0.25, 0.5, 0.75, 1.0], dtype=pl.Float64),
                "len": pl.Series([nbr_records] * 4, dtype=pl.UInt32),
            }
        ),
    )


def test_group_count_over_null_column_15705() -> None:
    df = pl.DataFrame(
        {"a": [1, 1, 2, 2, 3, 3], "c": [None, None, None, None, None, None]}
    )
    out = df.group_by("a", maintain_order=True).agg(pl.col("c").count())
    assert out["c"].to_list() == [0, 0, 0]


@pytest.mark.release()
def test_min_max_2850() -> None:
    # https://github.com/pola-rs/polars/issues/2850
    df = pl.DataFrame(
        {
            "id": [
                130352432,
                130352277,
                130352611,
                130352833,
                130352305,
                130352258,
                130352764,
                130352475,
                130352368,
                130352346,
            ]
        }
    )

    minimum = 130352258
    maximum = 130352833.0

    for _ in range(10):
        permuted = df.sample(fraction=1.0, seed=0)
        computed = permuted.select(
            pl.col("id").min().alias("min"), pl.col("id").max().alias("max")
        )
        assert cast(int, computed[0, "min"]) == minimum
        assert cast(float, computed[0, "max"]) == maximum


def test_multi_arg_structify_15834() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 2, 1, 2],
            "value": [
                0.1973209146402105,
                0.13380719982405365,
                0.6152394463707009,
                0.4558767896005155,
            ],
        }
    )

    assert df.lazy().group_by("group").agg(
        pl.struct(a=1, value=pl.col("value").sum())
    ).collect().sort("group").to_dict(as_series=False) == {
        "group": [1, 2],
        "a": [
            {"a": 1, "value": 0.8125603610109114},
            {"a": 1, "value": 0.5896839894245691},
        ],
    }


def test_filter_aggregation_16642() -> None:
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(2022, 1, 1, 11, 0),
                datetime(2022, 1, 1, 11, 1),
                datetime(2022, 1, 1, 11, 2),
                datetime(2022, 1, 1, 11, 3),
                datetime(2022, 1, 1, 11, 4),
                datetime(2022, 1, 1, 11, 5),
                datetime(2022, 1, 1, 11, 6),
                datetime(2022, 1, 1, 11, 7),
                datetime(2022, 1, 1, 11, 8),
                datetime(2022, 1, 1, 11, 9, 1),
                datetime(2022, 1, 2, 11, 0),
                datetime(2022, 1, 2, 11, 1),
                datetime(2022, 1, 2, 11, 2),
                datetime(2022, 1, 2, 11, 3),
                datetime(2022, 1, 2, 11, 4),
                datetime(2022, 1, 2, 11, 5),
                datetime(2022, 1, 2, 11, 6),
                datetime(2022, 1, 2, 11, 7),
                datetime(2022, 1, 2, 11, 8),
                datetime(2022, 1, 2, 11, 9, 1),
            ],
            "alpha": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],
            "num": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    grouped = df.group_by(pl.col("datetime").dt.date())

    ts_filter = pl.col("datetime").dt.time() <= pl.time(11, 3)

    report = grouped.agg(pl.col("num").filter(ts_filter).max()).sort("datetime")
    assert report.to_dict(as_series=False) == {
        "datetime": [date(2022, 1, 1), date(2022, 1, 2)],
        "num": [3, 3],
    }


def test_sort_by_over_single_nulls_first() -> None:
    key = [0, 0, 0, 0, 1, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key": key,
            "value": [2, None, 1, 0, 2, None, 1, 0],
        }
    )
    out = df.select(
        pl.all().sort_by("value", nulls_last=False, maintain_order=True).over("key")
    )
    expected = pl.DataFrame(
        {
            "key": key,
            "value": [None, 0, 1, 2, None, 0, 1, 2],
        }
    )
    assert_frame_equal(out, expected)


def test_sort_by_over_single_nulls_last() -> None:
    key = [0, 0, 0, 0, 1, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key": key,
            "value": [2, None, 1, 0, 2, None, 1, 0],
        }
    )
    out = df.select(
        pl.all().sort_by("value", nulls_last=True, maintain_order=True).over("key")
    )
    expected = pl.DataFrame(
        {
            "key": key,
            "value": [0, 1, 2, None, 0, 1, 2, None],
        }
    )
    assert_frame_equal(out, expected)


def test_sort_by_over_multiple_nulls_first() -> None:
    key1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    key2 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [1, None, 0, 1, None, 0, 1, None, 0, None, 1, 0],
        }
    )
    out = df.select(
        pl.all()
        .sort_by("value", nulls_last=False, maintain_order=True)
        .over("key1", "key2")
    )
    expected = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [None, 0, 1, None, 0, 1, None, 0, 1, None, 0, 1],
        }
    )
    assert_frame_equal(out, expected)


def test_sort_by_over_multiple_nulls_last() -> None:
    key1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    key2 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [1, None, 0, 1, None, 0, 1, None, 0, None, 1, 0],
        }
    )
    out = df.select(
        pl.all()
        .sort_by("value", nulls_last=True, maintain_order=True)
        .over("key1", "key2")
    )
    expected = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [0, 1, None, 0, 1, None, 0, 1, None, 0, 1, None],
        }
    )
    assert_frame_equal(out, expected)
