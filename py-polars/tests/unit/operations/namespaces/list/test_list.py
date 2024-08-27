from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

import polars as pl
from polars.exceptions import (
    ComputeError,
    OutOfBoundsError,
    SchemaError,
    StructFieldNotFoundError,
)
from polars.testing import assert_frame_equal, assert_series_equal


def test_list_arr_get() -> None:
    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.list.get(0, null_on_oob=False)
    expected = pl.Series("a", [1, 4, 6])
    assert_series_equal(out, expected)
    out = a.list[0]
    expected = pl.Series("a", [1, 4, 6])
    assert_series_equal(out, expected)
    out = a.list.first()
    assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).list.first()).to_series()
    assert_series_equal(out, expected)

    out = a.list.get(-1, null_on_oob=False)
    expected = pl.Series("a", [3, 5, 9])
    assert_series_equal(out, expected)
    out = a.list.last()
    assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).list.last()).to_series()
    assert_series_equal(out, expected)

    with pytest.raises(ComputeError, match="get index is out of bounds"):
        a.list.get(3, null_on_oob=False)

    # Null index.
    out_df = a.to_frame().select(pl.col.a.list.get(pl.lit(None), null_on_oob=False))
    expected_df = pl.Series("a", [None, None, None], dtype=pl.Int64).to_frame()
    assert_frame_equal(out_df, expected_df)

    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])

    with pytest.raises(ComputeError, match="get index is out of bounds"):
        a.list.get(-3, null_on_oob=False)

    with pytest.raises(ComputeError, match="get index is out of bounds"):
        pl.DataFrame(
            {"a": [[1], [2], [3], [4, 5, 6], [7, 8, 9], [None, 11]]}
        ).with_columns(
            pl.col("a").list.get(i, null_on_oob=False).alias(f"get_{i}")
            for i in range(4)
        )

    # get by indexes where some are out of bounds
    df = pl.DataFrame({"cars": [[1, 2, 3], [2, 3], [4], []], "indexes": [-2, 1, -3, 0]})

    with pytest.raises(ComputeError, match="get index is out of bounds"):
        df.select([pl.col("cars").list.get("indexes", null_on_oob=False)]).to_dict(
            as_series=False
        )

    # exact on oob boundary
    df = pl.DataFrame(
        {
            "index": [3, 3, 3],
            "lists": [[3, 4, 5], [4, 5, 6], [7, 8, 9, 4]],
        }
    )

    with pytest.raises(ComputeError, match="get index is out of bounds"):
        df.select(pl.col("lists").list.get(3, null_on_oob=False))

    with pytest.raises(ComputeError, match="get index is out of bounds"):
        df.select(pl.col("lists").list.get(pl.col("index"), null_on_oob=False))


def test_list_arr_get_null_on_oob() -> None:
    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.list.get(0, null_on_oob=True)
    expected = pl.Series("a", [1, 4, 6])
    assert_series_equal(out, expected)
    out = a.list[0]
    expected = pl.Series("a", [1, 4, 6])
    assert_series_equal(out, expected)
    out = a.list.first()
    assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).list.first()).to_series()
    assert_series_equal(out, expected)

    out = a.list.get(-1, null_on_oob=True)
    expected = pl.Series("a", [3, 5, 9])
    assert_series_equal(out, expected)
    out = a.list.last()
    assert_series_equal(out, expected)
    out = pl.select(pl.lit(a).list.last()).to_series()
    assert_series_equal(out, expected)

    # Out of bounds index.
    out = a.list.get(3, null_on_oob=True)
    expected = pl.Series("a", [None, None, 9])
    assert_series_equal(out, expected)

    # Null index.
    out_df = a.to_frame().select(pl.col.a.list.get(pl.lit(None), null_on_oob=True))
    expected_df = pl.Series("a", [None, None, None], dtype=pl.Int64).to_frame()
    assert_frame_equal(out_df, expected_df)

    a = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    out = a.list.get(-3, null_on_oob=True)
    expected = pl.Series("a", [1, None, 7])
    assert_series_equal(out, expected)

    assert pl.DataFrame(
        {"a": [[1], [2], [3], [4, 5, 6], [7, 8, 9], [None, 11]]}
    ).with_columns(
        pl.col("a").list.get(i, null_on_oob=True).alias(f"get_{i}") for i in range(4)
    ).to_dict(as_series=False) == {
        "a": [[1], [2], [3], [4, 5, 6], [7, 8, 9], [None, 11]],
        "get_0": [1, 2, 3, 4, 7, None],
        "get_1": [None, None, None, 5, 8, 11],
        "get_2": [None, None, None, 6, 9, None],
        "get_3": [None, None, None, None, None, None],
    }

    # get by indexes where some are out of bounds
    df = pl.DataFrame({"cars": [[1, 2, 3], [2, 3], [4], []], "indexes": [-2, 1, -3, 0]})

    assert df.select([pl.col("cars").list.get("indexes", null_on_oob=True)]).to_dict(
        as_series=False
    ) == {"cars": [2, 3, None, None]}
    # exact on oob boundary
    df = pl.DataFrame(
        {
            "index": [3, 3, 3],
            "lists": [[3, 4, 5], [4, 5, 6], [7, 8, 9, 4]],
        }
    )

    assert df.select(pl.col("lists").list.get(3, null_on_oob=True)).to_dict(
        as_series=False
    ) == {"lists": [None, None, 4]}
    assert df.select(
        pl.col("lists").list.get(pl.col("index"), null_on_oob=True)
    ).to_dict(as_series=False) == {"lists": [None, None, 4]}


def test_list_categorical_get() -> None:
    df = pl.DataFrame(
        {
            "actions": pl.Series(
                [["a", "b"], ["c"], [None], None], dtype=pl.List(pl.Categorical)
            ),
        }
    )
    expected = pl.Series("actions", ["a", "c", None, None], dtype=pl.Categorical)
    assert_series_equal(
        df["actions"].list.get(0, null_on_oob=True), expected, categorical_as_str=True
    )


def test_contains() -> None:
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    out = a.list.contains(2)
    expected = pl.Series("a", [True, True, False])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).list.contains(2)).to_series()
    assert_series_equal(out, expected)


def test_list_contains_invalid_datatype() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4]]}, schema={"a": pl.Array(pl.Int8, shape=2)})
    with pytest.raises(SchemaError, match="invalid series dtype: expected `List`"):
        df.select(pl.col("a").list.contains(2))


def test_list_concat() -> None:
    df = pl.DataFrame({"a": [[1, 2], [1], [1, 2, 3]]})

    out = df.select([pl.col("a").list.concat(pl.Series([[1, 2]]))])
    assert out["a"][0].to_list() == [1, 2, 1, 2]

    out = df.select([pl.col("a").list.concat([1, 4])])
    assert out["a"][0].to_list() == [1, 2, 1, 4]

    out_s = df["a"].list.concat([4, 1])
    assert out_s[0].to_list() == [1, 2, 4, 1]


def test_list_join() -> None:
    df = pl.DataFrame(
        {
            "a": [["ab", "c", "d"], ["e", "f"], ["g"], [], None],
            "separator": ["&", None, "*", "_", "*"],
        }
    )
    out = df.select(pl.col("a").list.join("-"))
    assert out.to_dict(as_series=False) == {"a": ["ab-c-d", "e-f", "g", "", None]}
    out = df.select(pl.col("a").list.join(pl.col("separator")))
    assert out.to_dict(as_series=False) == {"a": ["ab&c&d", None, "g", "", None]}

    # test ignore_nulls argument
    df = pl.DataFrame(
        {
            "a": [["a", None, "b", None], None, [None, None], ["c", "d"], []],
            "separator": ["-", "&", " ", "@", "/"],
        }
    )
    # ignore nulls
    out = df.select(pl.col("a").list.join("-", ignore_nulls=True))
    assert out.to_dict(as_series=False) == {"a": ["a-b", None, "", "c-d", ""]}
    out = df.select(pl.col("a").list.join(pl.col("separator"), ignore_nulls=True))
    assert out.to_dict(as_series=False) == {"a": ["a-b", None, "", "c@d", ""]}
    # propagate nulls
    out = df.select(pl.col("a").list.join("-", ignore_nulls=False))
    assert out.to_dict(as_series=False) == {"a": [None, None, None, "c-d", ""]}
    out = df.select(pl.col("a").list.join(pl.col("separator"), ignore_nulls=False))
    assert out.to_dict(as_series=False) == {"a": [None, None, None, "c@d", ""]}


def test_list_arr_empty() -> None:
    df = pl.DataFrame({"cars": [[1, 2, 3], [2, 3], [4], []]})

    out = df.select(
        pl.col("cars").list.first().alias("cars_first"),
        pl.when(pl.col("cars").list.first() == 2)
        .then(1)
        .when(pl.col("cars").list.contains(2))
        .then(2)
        .otherwise(3)
        .alias("cars_literal"),
    )
    expected = pl.DataFrame(
        {"cars_first": [1, 2, 4, None], "cars_literal": [2, 1, 3, 3]},
        schema_overrides={"cars_literal": pl.Int32},  # Literals default to Int32
    )
    assert_frame_equal(out, expected)


def test_list_argminmax() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [0, 2], dtype=pl.UInt32)
    assert_series_equal(s.list.arg_min(), expected)
    expected = pl.Series("a", [1, 0], dtype=pl.UInt32)
    assert_series_equal(s.list.arg_max(), expected)


def test_list_shift() -> None:
    s = pl.Series("a", [[1, 2], [3, 2, 1]])
    expected = pl.Series("a", [[None, 1], [None, 3, 2]])
    assert s.list.shift().to_list() == expected.to_list()

    df = pl.DataFrame(
        {
            "values": [
                [1, 2, None],
                [1, 2, 3],
                [None, 1, 2],
                [None, None, None],
                [1, 2],
            ],
            "shift": [1, -2, 3, 2, None],
        }
    )
    df = df.select(pl.col("values").list.shift(pl.col("shift")))
    expected_df = pl.DataFrame(
        {
            "values": [
                [None, 1, 2],
                [3, None, None],
                [None, None, None],
                [None, None, None],
                None,
            ]
        }
    )
    assert_frame_equal(df, expected_df)


def test_list_drop_nulls() -> None:
    s = pl.Series("values", [[1, None, 2, None], [None, None], [1, 2], None])
    expected = pl.Series("values", [[1, 2], [], [1, 2], None])
    assert_series_equal(s.list.drop_nulls(), expected)

    df = pl.DataFrame({"values": [[None, 1, None, 2], [None], [3, 4]]})
    df = df.select(pl.col("values").list.drop_nulls())
    expected_df = pl.DataFrame({"values": [[1, 2], [], [3, 4]]})
    assert_frame_equal(df, expected_df)


def test_list_sample() -> None:
    s = pl.Series("values", [[1, 2, 3, None], [None, None], [1, 2], None])

    expected_sample_n = pl.Series("values", [[3, 1], [None], [2], None])
    assert_series_equal(
        s.list.sample(n=pl.Series([2, 1, 1, 1]), seed=1), expected_sample_n
    )

    expected_sample_frac = pl.Series("values", [[3, 1], [None], [1, 2], None])
    assert_series_equal(
        s.list.sample(fraction=pl.Series([0.5, 0.5, 1.0, 0.3]), seed=1),
        expected_sample_frac,
    )

    df = pl.DataFrame(
        {
            "values": [[1, 2, 3, None], [None, None], [3, 4]],
            "n": [2, 1, 2],
            "frac": [0.5, 0.5, 1.0],
        }
    )
    df = df.select(
        sample_n=pl.col("values").list.sample(n=pl.col("n"), seed=1),
        sample_frac=pl.col("values").list.sample(fraction=pl.col("frac"), seed=1),
    )
    expected_df = pl.DataFrame(
        {"sample_n": [[3, 1], [None], [3, 4]], "sample_frac": [[3, 1], [None], [3, 4]]}
    )
    assert_frame_equal(df, expected_df)


def test_list_diff() -> None:
    s = pl.Series("a", [[1, 2], [10, 2, 1]])
    expected = pl.Series("a", [[None, 1], [None, -8, -1]])
    assert s.list.diff().to_list() == expected.to_list()


def test_slice() -> None:
    vals = [[1, 2, 3, 4], [10, 2, 1]]
    s = pl.Series("a", vals)
    assert s.list.head(2).to_list() == [[1, 2], [10, 2]]
    assert s.list.tail(2).to_list() == [[3, 4], [2, 1]]
    assert s.list.tail(200).to_list() == vals
    assert s.list.head(200).to_list() == vals
    assert s.list.slice(1, 2).to_list() == [[2, 3], [2, 1]]
    assert s.list.slice(-5, 2).to_list() == [[1], []]


def test_list_eval_dtype_inference() -> None:
    grades = pl.DataFrame(
        {
            "student": ["bas", "laura", "tim", "jenny"],
            "arithmetic": [10, 5, 6, 8],
            "biology": [4, 6, 2, 7],
            "geography": [8, 4, 9, 7],
        }
    )

    rank_pct = pl.col("").rank(descending=True) / pl.col("").count().cast(pl.UInt16)

    # the .list.first() would fail if .list.eval did not correctly infer the output type
    assert grades.with_columns(
        pl.concat_list(pl.all().exclude("student")).alias("all_grades")
    ).select(
        pl.col("all_grades")
        .list.eval(rank_pct, parallel=True)
        .alias("grades_rank")
        .list.first()
    ).to_series().to_list() == [
        0.3333333333333333,
        0.6666666666666666,
        0.6666666666666666,
        0.3333333333333333,
    ]


def test_list_eval_categorical() -> None:
    df = pl.DataFrame({"test": [["a", None]]}, schema={"test": pl.List(pl.Categorical)})
    df = df.select(
        pl.col("test").list.eval(pl.element().filter(pl.element().is_not_null()))
    )
    assert_series_equal(
        df.get_column("test"), pl.Series("test", [["a"]], dtype=pl.List(pl.Categorical))
    )


def test_list_ternary_concat() -> None:
    df = pl.DataFrame(
        {
            "list1": [["123", "456"], None],
            "list2": [["789"], ["zzz"]],
        }
    )

    assert df.with_columns(
        pl.when(pl.col("list1").is_null())
        .then(pl.col("list1").list.concat(pl.col("list2")))
        .otherwise(pl.col("list2"))
        .alias("result")
    ).to_dict(as_series=False) == {
        "list1": [["123", "456"], None],
        "list2": [["789"], ["zzz"]],
        "result": [["789"], None],
    }

    assert df.with_columns(
        pl.when(pl.col("list1").is_null())
        .then(pl.col("list2"))
        .otherwise(pl.col("list1").list.concat(pl.col("list2")))
        .alias("result")
    ).to_dict(as_series=False) == {
        "list1": [["123", "456"], None],
        "list2": [["789"], ["zzz"]],
        "result": [["123", "456", "789"], ["zzz"]],
    }


def test_arr_contains_categorical() -> None:
    df = pl.DataFrame(
        {"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]}
    ).lazy()
    df = df.with_columns(pl.col("str").cast(pl.Categorical))
    df_groups = df.group_by("group").agg([pl.col("str").alias("str_list")])

    result = df_groups.filter(pl.col("str_list").list.contains("C")).collect()
    expected = {"group": [2], "str_list": [["A", "C"]]}
    assert result.to_dict(as_series=False) == expected


def test_list_eval_type_coercion() -> None:
    last_non_null_value = pl.element().fill_null(3).last()
    df = pl.DataFrame({"array_cols": [[1, None]]})

    assert df.select(
        pl.col("array_cols")
        .list.eval(last_non_null_value, parallel=False)
        .alias("col_last")
    ).to_dict(as_series=False) == {"col_last": [[3]]}


def test_list_slice() -> None:
    df = pl.DataFrame(
        {
            "lst": [[1, 2, 3, 4], [10, 2, 1]],
            "offset": [1, 2],
            "len": [3, 2],
        }
    )

    assert df.select([pl.col("lst").list.slice("offset", "len")]).to_dict(
        as_series=False
    ) == {"lst": [[2, 3, 4], [1]]}
    assert df.select([pl.col("lst").list.slice("offset", 1)]).to_dict(
        as_series=False
    ) == {"lst": [[2], [1]]}
    assert df.select([pl.col("lst").list.slice(-2, "len")]).to_dict(
        as_series=False
    ) == {"lst": [[3, 4], [2, 1]]}


def test_list_sliced_get_5186() -> None:
    # https://github.com/pola-rs/polars/issues/5186
    n = 30
    df = pl.from_dict(
        {
            "ind": pl.arange(0, n, eager=True),
            "inds": pl.Series(
                np.stack([np.arange(n), -np.arange(n)], axis=-1), dtype=pl.List
            ),
        }
    )

    exprs = [
        "ind",
        pl.col("inds").list.first().alias("first_element"),
        pl.col("inds").list.last().alias("last_element"),
    ]
    out1 = df.select(exprs)[10:20]
    out2 = df[10:20].select(exprs)
    assert_frame_equal(out1, out2)


def test_empty_eval_dtype_5546() -> None:
    # https://github.com/pola-rs/polars/issues/5546
    df = pl.DataFrame([{"a": [{"name": 1}, {"name": 2}]}])

    dtype = df.dtypes[0]

    assert (
        df.limit(0).with_columns(
            pl.col("a")
            .list.eval(pl.element().filter(pl.first().struct.field("name") == 1))
            .alias("a_filtered")
        )
    ).dtypes == [dtype, dtype]


def test_list_amortized_apply_explode_5812() -> None:
    s = pl.Series([None, [1, 3], [0, -3], [1, 2, 2]])
    assert s.list.sum().to_list() == [None, 4, -3, 5]
    assert s.list.min().to_list() == [None, 1, -3, 1]
    assert s.list.max().to_list() == [None, 3, 0, 2]
    assert s.list.arg_min().to_list() == [None, 0, 1, 0]
    assert s.list.arg_max().to_list() == [None, 1, 0, 1]


def test_list_slice_5866() -> None:
    vals = [[1, 2, 3, 4], [10, 2, 1]]
    s = pl.Series("a", vals)
    assert s.list.slice(1).to_list() == [[2, 3, 4], [2, 1]]


def test_list_gather() -> None:
    s = pl.Series("a", [[1, 2, 3], [4, 5], [6, 7, 8]])
    # mypy: we make it work, but idiomatic is `arr.get`.
    assert s.list.gather(0).to_list() == [[1], [4], [6]]  # type: ignore[arg-type]
    assert s.list.gather([0, 1]).to_list() == [[1, 2], [4, 5], [6, 7]]

    assert s.list.gather([-1, 1]).to_list() == [[3, 2], [5, 5], [8, 7]]

    # use another list to make sure negative indices are respected
    gatherer = pl.Series([[-1, 1], [-1, 1], [-1, -2]])
    assert s.list.gather(gatherer).to_list() == [[3, 2], [5, 5], [8, 7]]
    with pytest.raises(OutOfBoundsError, match=r"gather indices are out of bounds"):
        s.list.gather([1, 2])
    s = pl.Series(
        [["A", "B", "C"], ["A"], ["B"], ["1", "2"], ["e"]],
    )

    assert s.list.gather([0, 2], null_on_oob=True).to_list() == [
        ["A", "C"],
        ["A", None],
        ["B", None],
        ["1", None],
        ["e", None],
    ]
    assert s.list.gather([0, 1, 2], null_on_oob=True).to_list() == [
        ["A", "B", "C"],
        ["A", None, None],
        ["B", None, None],
        ["1", "2", None],
        ["e", None, None],
    ]
    s = pl.Series([[42, 1, 2], [5, 6, 7]])

    with pytest.raises(OutOfBoundsError, match=r"gather indices are out of bounds"):
        s.list.gather([[0, 1, 2, 3], [0, 1, 2, 3]])

    assert s.list.gather([0, 1, 2, 3], null_on_oob=True).to_list() == [
        [42, 1, 2, None],
        [5, 6, 7, None],
    ]


def test_list_eval_all_null() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, None, None]}).with_columns(
        pl.col("bar").cast(pl.List(pl.String))
    )

    assert df.select(pl.col("bar").list.eval(pl.element())).to_dict(
        as_series=False
    ) == {"bar": [None, None, None]}


def test_list_function_group_awareness() -> None:
    df = pl.DataFrame(
        {
            "a": [100, 103, 105, 106, 105, 104, 103, 106, 100, 102],
            "group": [0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
        }
    )

    assert df.group_by("group").agg(
        [
            pl.col("a").get(0).alias("get_scalar"),
            pl.col("a").gather([0]).alias("take_no_implode"),
            pl.col("a").implode().list.get(0).alias("implode_get"),
            pl.col("a").implode().list.gather([0]).alias("implode_take"),
            pl.col("a").implode().list.slice(0, 3).alias("implode_slice"),
        ]
    ).sort("group").to_dict(as_series=False) == {
        "group": [0, 1, 2],
        "get_scalar": [100, 105, 100],
        "take_no_implode": [[100], [105], [100]],
        "implode_get": [[100], [105], [100]],
        "implode_take": [[[100]], [[105]], [[100]]],
        "implode_slice": [[[100, 103]], [[105, 106, 105]], [[100, 102]]],
    }


def test_list_get_logical_types() -> None:
    df = pl.DataFrame(
        {
            "date_col": [[datetime(2023, 2, 1).date(), datetime(2023, 2, 2).date()]],
            "datetime_col": [[datetime(2023, 2, 1), datetime(2023, 2, 2)]],
        }
    )

    assert df.select(pl.all().list.get(1).name.suffix("_element_1")).to_dict(
        as_series=False
    ) == {
        "date_col_element_1": [date(2023, 2, 2)],
        "datetime_col_element_1": [datetime(2023, 2, 2, 0, 0)],
    }


def test_list_gather_logical_type() -> None:
    df = pl.DataFrame(
        {"foo": [["foo", "foo", "bar"]], "bar": [[5.0, 10.0, 12.0]]}
    ).with_columns(pl.col("foo").cast(pl.List(pl.Categorical)))

    df = pl.concat([df, df], rechunk=False)
    assert df.n_chunks() == 2
    assert df.select(pl.all().gather([0, 1])).to_dict(as_series=False) == {
        "foo": [["foo", "foo", "bar"], ["foo", "foo", "bar"]],
        "bar": [[5.0, 10.0, 12.0], [5.0, 10.0, 12.0]],
    }


def test_list_unique() -> None:
    s = pl.Series([[1, 1, 2, 2, 3], [3, 3, 3, 2, 1, 2]])
    result = s.list.unique(maintain_order=True)
    expected = pl.Series([[1, 2, 3], [3, 2, 1]])
    assert_series_equal(result, expected)


def test_list_unique2() -> None:
    s = pl.Series("a", [[2, 1], [1, 2, 2]])
    result = s.list.unique()
    assert len(result) == 2
    assert sorted(result[0]) == [1, 2]
    assert sorted(result[1]) == [1, 2]


def test_list_to_struct() -> None:
    df = pl.DataFrame({"n": [[0, 1, 2], [0, 1]]})

    assert df.select(pl.col("n").list.to_struct()).rows(named=True) == [
        {"n": {"field_0": 0, "field_1": 1, "field_2": 2}},
        {"n": {"field_0": 0, "field_1": 1, "field_2": None}},
    ]

    assert df.select(pl.col("n").list.to_struct(fields=lambda idx: f"n{idx}")).rows(
        named=True
    ) == [
        {"n": {"n0": 0, "n1": 1, "n2": 2}},
        {"n": {"n0": 0, "n1": 1, "n2": None}},
    ]

    assert df.select(pl.col("n").list.to_struct(fields=["one", "two", "three"])).rows(
        named=True
    ) == [
        {"n": {"one": 0, "two": 1, "three": 2}},
        {"n": {"one": 0, "two": 1, "three": None}},
    ]


def test_select_from_list_to_struct_11143() -> None:
    ldf = pl.LazyFrame({"some_col": [[1.0, 2.0], [1.5, 3.0]]})
    ldf = ldf.select(
        pl.col("some_col").list.to_struct(fields=["a", "b"], upper_bound=2)
    )
    df = ldf.select(pl.col("some_col").struct.field("a")).collect()
    assert df.equals(pl.DataFrame({"a": [1.0, 1.5]}))


def test_list_arr_get_8810() -> None:
    assert pl.DataFrame(pl.Series("a", [None], pl.List(pl.Int64))).select(
        pl.col("a").list.get(0, null_on_oob=True)
    ).to_dict(as_series=False) == {"a": [None]}


def test_list_tail_underflow_9087() -> None:
    assert pl.Series([["a", "b", "c"]]).list.tail(pl.lit(1, pl.UInt32)).to_list() == [
        ["c"]
    ]


def test_list_count_match_boolean_nulls_9141() -> None:
    a = pl.DataFrame({"a": [[True, None, False]]})
    assert a.select(pl.col("a").list.count_matches(True))["a"].to_list() == [1]


def test_list_count_match_categorical() -> None:
    df = pl.DataFrame(
        {"list": [["0"], ["1"], ["1", "2", "3", "2"], ["1", "2", "1"], ["4", "4"]]},
        schema={"list": pl.List(pl.Categorical)},
    )
    assert df.select(pl.col("list").list.count_matches("2").alias("number_of_twos"))[
        "number_of_twos"
    ].to_list() == [0, 0, 2, 1, 0]


def test_list_count_matches_boolean_nulls_9141() -> None:
    a = pl.DataFrame({"a": [[True, None, False]]})

    assert a.select(pl.col("a").list.count_matches(True))["a"].to_list() == [1]


def test_list_gather_oob_10079() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2, 3], [], [None, 3], [5, 6, 7]],
            "b": [["2"], ["3"], [None], ["3", "Hi"]],
        }
    )
    with pytest.raises(OutOfBoundsError, match="gather indices are out of bounds"):
        df.select(pl.col("a").gather(999))


def test_utf8_empty_series_arg_min_max_10703() -> None:
    res = pl.select(pl.lit(pl.Series("list", [["a"], []]))).with_columns(
        pl.all(),
        pl.all().list.arg_min().alias("arg_min"),
        pl.all().list.arg_max().alias("arg_max"),
    )
    assert res.to_dict(as_series=False) == {
        "list": [["a"], []],
        "arg_min": [0, None],
        "arg_max": [0, None],
    }


def test_list_len() -> None:
    s = pl.Series([[1, 2, None], [5]])
    result = s.list.len()
    expected = pl.Series([3, 1], dtype=pl.UInt32)
    assert_series_equal(result, expected)


def test_list_to_array() -> None:
    data = [[1.0, 2.0], [3.0, 4.0]]
    s = pl.Series(data, dtype=pl.List(pl.Float32))

    result = s.list.to_array(2)
    result_slice = s[1:].list.to_array(2)

    expected = pl.Series(data, dtype=pl.Array(pl.Float32, 2))
    assert_series_equal(result, expected)

    expected_slice = pl.Series([data[1]], dtype=pl.Array(pl.Float32, 2))
    assert_series_equal(result_slice, expected_slice)

    # test logical type
    df = pl.DataFrame(
        data={"duration": [[1000, 2000], None]},
        schema={
            "duration": pl.List(pl.Datetime),
        },
    ).with_columns(pl.col("duration").list.to_array(2))

    expected_df = pl.DataFrame(
        data={"duration": [[1000, 2000], None]},
        schema={
            "duration": pl.Array(pl.Datetime, 2),
        },
    )
    assert_frame_equal(df, expected_df)


def test_list_to_array_wrong_lengths() -> None:
    s = pl.Series([[1.0, 2.0], [3.0, 4.0]], dtype=pl.List(pl.Float32))
    with pytest.raises(ComputeError, match="not all elements have the specified width"):
        s.list.to_array(3)


def test_list_to_array_wrong_dtype() -> None:
    s = pl.Series([1.0, 2.0])
    with pytest.raises(ComputeError, match="expected List dtype"):
        s.list.to_array(2)


def test_list_lengths() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert_series_equal(s.list.len(), pl.Series("a", [2, 3], dtype=pl.UInt32))
    df = pl.DataFrame([s])
    assert_series_equal(
        df.select(pl.col("a").list.len())["a"], pl.Series("a", [2, 3], dtype=pl.UInt32)
    )


def test_list_arithmetic() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert_series_equal(s.list.sum(), pl.Series("a", [3, 6]))
    assert_series_equal(s.list.mean(), pl.Series("a", [1.5, 2.0]))
    assert_series_equal(s.list.max(), pl.Series("a", [2, 3]))
    assert_series_equal(s.list.min(), pl.Series("a", [1, 1]))


def test_list_ordering() -> None:
    s = pl.Series("a", [[2, 1], [1, 3, 2]])
    assert_series_equal(s.list.sort(), pl.Series("a", [[1, 2], [1, 2, 3]]))
    assert_series_equal(s.list.reverse(), pl.Series("a", [[1, 2], [2, 3, 1]]))

    # test nulls_last
    s = pl.Series([[None, 1, 2], [-1, None, 9]])
    assert_series_equal(
        s.list.sort(nulls_last=True), pl.Series([[1, 2, None], [-1, 9, None]])
    )
    assert_series_equal(
        s.list.sort(nulls_last=False), pl.Series([[None, 1, 2], [None, -1, 9]])
    )


def test_list_get_logical_type() -> None:
    s = pl.Series(
        "a",
        [
            [date(1999, 1, 1), date(2000, 1, 1)],
            [date(2001, 10, 1), None],
        ],
        dtype=pl.List(pl.Date),
    )

    out = s.list.get(0)
    expected = pl.Series(
        "a",
        [date(1999, 1, 1), date(2001, 10, 1)],
        dtype=pl.Date,
    )
    assert_series_equal(out, expected)

    out = s.list.get(pl.Series([1, -2]))
    expected = pl.Series(
        "a",
        [date(2000, 1, 1), date(2001, 10, 1)],
        dtype=pl.Date,
    )
    assert_series_equal(out, expected)


def test_list_eval_gater_every_13410() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]})
    out = df.with_columns(result=pl.col("a").list.eval(pl.element().gather_every(2)))
    expected = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]], "result": [[1, 3], [4, 6]]})
    assert_frame_equal(out, expected)


def test_list_gather_every() -> None:
    df = pl.DataFrame(
        {
            "lst": [[1, 2, 3], [], [4, 5], None, [6, 7, 8], [9, 10, 11, 12]],
            "n": [2, 2, 1, 3, None, 2],
            "offset": [None, 1, 0, 1, 2, 2],
        }
    )

    out = df.select(
        n_expr=pl.col("lst").list.gather_every(pl.col("n"), 0),
        offset_expr=pl.col("lst").list.gather_every(2, pl.col("offset")),
        all_expr=pl.col("lst").list.gather_every(pl.col("n"), pl.col("offset")),
        all_lit=pl.col("lst").list.gather_every(2, 0),
    )

    expected = pl.DataFrame(
        {
            "n_expr": [[1, 3], [], [4, 5], None, None, [9, 11]],
            "offset_expr": [None, [], [4], None, [8], [11]],
            "all_expr": [None, [], [4, 5], None, None, [11]],
            "all_lit": [[1, 3], [], [4], None, [6, 8], [9, 11]],
        }
    )

    assert_frame_equal(out, expected)


def test_list_n_unique() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 1, 2], [3, 3], [None], None, []],
        }
    )

    out = df.select(n_unique=pl.col("a").list.n_unique())
    expected = pl.DataFrame(
        {"n_unique": [2, 1, 1, None, 0]}, schema={"n_unique": pl.UInt32}
    )
    assert_frame_equal(out, expected)


def test_list_get_with_null() -> None:
    df = pl.DataFrame({"a": [None, [1, 2]], "b": [False, True]})

    # We allow two layouts of null in ListArray:
    # 1. null element are stored as arbitrary values in `value` array.
    # 2. null element are not stored in `value` array.
    out = df.select(
        # For performance reasons, when-then-otherwise produces the list with layout-1.
        layout1=pl.when(pl.col("b")).then([1, 2]).list.get(0, null_on_oob=True),
        layout2=pl.col("a").list.get(0, null_on_oob=True),
    )

    expected = pl.DataFrame(
        {
            "layout1": [None, 1],
            "layout2": [None, 1],
        }
    )

    assert_frame_equal(out, expected)


def test_list_eval_err_raise_15653() -> None:
    df = pl.DataFrame({"foo": [[]]})
    with pytest.raises(StructFieldNotFoundError):
        df.with_columns(bar=pl.col("foo").list.eval(pl.element().struct.field("baz")))


def test_list_sum_bool_schema() -> None:
    q = pl.LazyFrame({"x": [[True, True, False]]})
    assert q.select(pl.col("x").list.sum()).collect_schema()["x"] == pl.UInt32


def test_list_eval_type_cast_11188() -> None:
    df = pl.DataFrame(
        [
            {"a": None},
        ],
        schema={"a": pl.List(pl.Int64)},
    )
    assert df.select(
        pl.col("a").list.eval(pl.element().cast(pl.String)).alias("a_str")
    ).schema == {"a_str": pl.List(pl.String)}
