from __future__ import annotations

import pickle
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars import PolarsDataType


def test_dtype() -> None:
    # inferred
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    assert a.dtype == pl.List
    assert a.inner_dtype == pl.Int64
    assert a.dtype.inner == pl.Int64  # type: ignore[union-attr]
    assert a.dtype.is_(pl.List(pl.Int64))

    # explicit
    df = pl.DataFrame(
        data={
            "i": [[1, 2, 3]],
            "tm": [[time(10, 30, 45)]],
            "dt": [[date(2022, 12, 31)]],
            "dtm": [[datetime(2022, 12, 31, 1, 2, 3)]],
        },
        schema=[
            ("i", pl.List(pl.Int8)),
            ("tm", pl.List(pl.Time)),
            ("dt", pl.List(pl.Date)),
            ("dtm", pl.List(pl.Datetime)),
        ],
    )
    assert df.schema == {
        "i": pl.List(pl.Int8),
        "tm": pl.List(pl.Time),
        "dt": pl.List(pl.Date),
        "dtm": pl.List(pl.Datetime),
    }
    assert all(tp.is_nested for tp in df.dtypes)
    assert df.schema["i"].inner == pl.Int8  # type: ignore[union-attr]
    assert df.rows() == [
        (
            [1, 2, 3],
            [time(10, 30, 45)],
            [date(2022, 12, 31)],
            [datetime(2022, 12, 31, 1, 2, 3)],
        )
    ]


def test_categorical() -> None:
    # https://github.com/pola-rs/polars/issues/2038
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 1, 1, 1, 1, 1, 1]),
            pl.Series("b", [8, 2, 3, 6, 3, 6, 2, 2]),
            pl.Series("c", ["a", "b", "c", "a", "b", "c", "a", "b"]).cast(
                pl.Categorical
            ),
        ]
    )
    out = (
        df.group_by(["a", "b"])
        .agg(
            [
                pl.col("c").count().alias("num_different_c"),
                pl.col("c").alias("c_values"),
            ]
        )
        .filter(pl.col("num_different_c") >= 2)
        .to_series(3)
    )

    assert out.inner_dtype == pl.Categorical
    assert not out.inner_dtype.is_nested


def test_cast_inner() -> None:
    a = pl.Series([[1, 2]])
    for t in [bool, pl.Boolean]:
        b = a.cast(pl.List(t))
        assert b.dtype == pl.List(pl.Boolean)
        assert b.to_list() == [[True, True]]

    # this creates an inner null type
    df = pl.from_pandas(pd.DataFrame(data=[[[]], [[]]], columns=["A"]))
    assert (
        df["A"].cast(pl.List(int)).dtype.inner == pl.Int64  # type: ignore[union-attr]
    )


def test_list_unique() -> None:
    s = pl.Series("a", [[1, 2], [3], [1, 2], [4, 5], [2], [2]])
    assert s.unique(maintain_order=True).to_list() == [[1, 2], [3], [4, 5], [2]]
    assert s.arg_unique().to_list() == [0, 1, 3, 4]
    assert s.n_unique() == 4


def test_list_empty_group_by_result_3521() -> None:
    # Create a left relation where the join column contains a null value
    left = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("group_by_column"),
            pl.lit(None).cast(pl.Int32).alias("join_column"),
        ]
    )

    # Create a right relation where there is a column to count distinct on
    right = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("join_column"),
            pl.lit(1).alias("n_unique_column"),
        ]
    )

    # Calculate n_unique after dropping nulls
    # This will panic on polars version 0.13.38 and 0.13.39
    assert (
        left.join(right, on="join_column", how="left")
        .group_by("group_by_column")
        .agg(pl.col("n_unique_column").drop_nulls())
    ).to_dict(False) == {"group_by_column": [1], "n_unique_column": [[]]}


def test_list_fill_null() -> None:
    df = pl.DataFrame({"C": [["a", "b", "c"], [], [], ["d", "e"]]})
    assert df.with_columns(
        [
            pl.when(pl.col("C").list.len() == 0)
            .then(None)
            .otherwise(pl.col("C"))
            .alias("C")
        ]
    ).to_series().to_list() == [["a", "b", "c"], None, None, ["d", "e"]]


def test_list_fill_list() -> None:
    assert pl.DataFrame({"a": [[1, 2, 3], []]}).select(
        [
            pl.when(pl.col("a").list.len() == 0)
            .then([5])
            .otherwise(pl.col("a"))
            .alias("filled")
        ]
    ).to_dict(False) == {"filled": [[1, 2, 3], [5]]}


def test_empty_list_construction() -> None:
    assert pl.Series([[]]).to_list() == [[]]
    assert pl.DataFrame([{"array": [], "not_array": 1234}], orient="row").to_dict(
        False
    ) == {"array": [[]], "not_array": [1234]}

    df = pl.DataFrame(schema=[("col", pl.List)])
    assert df.schema == {"col": pl.List}
    assert df.rows() == []


def test_list_hash() -> None:
    out = pl.DataFrame({"a": [[1, 2, 3], [3, 4], [1, 2, 3]]}).with_columns(
        pl.col("a").hash().alias("b")
    )
    assert out.dtypes == [pl.List(pl.Int64), pl.UInt64]
    assert out[0, "b"] == out[2, "b"]


def test_list_diagonal_concat() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})

    df2 = pl.DataFrame({"b": [[1]]})

    assert pl.concat([df1, df2], how="diagonal").to_dict(False) == {
        "a": [1, 2, None],
        "b": [None, None, [1]],
    }


def test_inner_type_categorical_on_rechunk() -> None:
    df = pl.DataFrame({"cats": ["foo", "bar"]}).select(
        pl.col(pl.Utf8).cast(pl.Categorical).implode()
    )

    assert pl.concat([df, df], rechunk=True).dtypes == [pl.List(pl.Categorical)]


def test_group_by_list_column() -> None:
    df = (
        pl.DataFrame({"a": ["a", "b", "a"]})
        .with_columns(pl.col("a").cast(pl.Categorical))
        .group_by("a", maintain_order=True)
        .agg(pl.col("a").alias("a_list"))
    )

    assert df.group_by("a_list", maintain_order=True).first().to_dict(False) == {
        "a_list": [["a", "a"], ["b"]],
        "a": ["a", "b"],
    }


def test_group_by_multiple_keys_contains_list_column() -> None:
    df = (
        pl.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": [[1, 2], [1, 2], [3, 4, 5], [6]],
                "c": [3, 2, 1, 0],
            }
        )
        .group_by(["a", "b"], maintain_order=True)
        .agg(pl.all())
    )
    assert df.to_dict(False) == {
        "a": ["x", "y", "y"],
        "b": [[1, 2], [3, 4, 5], [6]],
        "c": [[3, 2], [1], [0]],
    }


def test_fast_explode_flag() -> None:
    df1 = pl.DataFrame({"values": [[[1, 2]]]})
    assert df1.clone().vstack(df1)["values"].flags["FAST_EXPLODE"]

    # test take that produces a null in list
    df = pl.DataFrame({"a": [1, 2, 1, 3]})
    df_b = pl.DataFrame({"a": [1, 2], "c": [["1", "2", "c"], ["1", "2", "c"]]})
    assert df_b["c"].flags["FAST_EXPLODE"]

    # join produces a null
    assert not (df.join(df_b, on=["a"], how="left").select(["c"]))["c"].flags[
        "FAST_EXPLODE"
    ]


def test_fast_explode_on_list_struct_6208() -> None:
    data = [
        {
            "label": "l",
            "tag": "t",
            "ref": 1,
            "parents": [{"ref": 1, "tag": "t", "ratio": 62.3}],
        },
        {"label": "l", "tag": "t", "ref": 1, "parents": None},
    ]

    df = pl.DataFrame(
        data,
        schema={
            "label": pl.Utf8,
            "tag": pl.Utf8,
            "ref": pl.Int64,
            "parents": pl.List(
                pl.Struct({"ref": pl.Int64, "tag": pl.Utf8, "ratio": pl.Float64})
            ),
        },
    )

    assert not df["parents"].flags["FAST_EXPLODE"]
    assert df.explode("parents").to_dict(False) == {
        "label": ["l", "l"],
        "tag": ["t", "t"],
        "ref": [1, 1],
        "parents": [
            {"ref": 1, "tag": "t", "ratio": 62.3},
            {"ref": None, "tag": None, "ratio": None},
        ],
    }


def test_flat_aggregation_to_list_conversion_6918() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [[0, 1], [2, 3], [4, 5]]})

    assert df.group_by("a", maintain_order=True).agg(
        pl.concat_list([pl.col("b").list.get(i).mean().implode() for i in range(2)])
    ).to_dict(False) == {"a": [1, 2], "b": [[[0.0, 1.0]], [[3.0, 4.0]]]}


def test_list_count_matches_deprecated() -> None:
    with pytest.deprecated_call():
        # Your test code here
        assert pl.DataFrame(
            {"listcol": [[], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]}
        ).select(pl.col("listcol").list.count_match(2).alias("number_of_twos")).to_dict(
            False
        ) == {
            "number_of_twos": [0, 0, 2, 1, 0]
        }


def test_list_count_matches() -> None:
    assert pl.DataFrame({"listcol": [[], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]}).select(
        pl.col("listcol").list.count_matches(2).alias("number_of_twos")
    ).to_dict(False) == {"number_of_twos": [0, 0, 2, 1, 0]}
    assert pl.DataFrame({"listcol": [[], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]}).select(
        pl.col("listcol").list.count_matches(2).alias("number_of_twos")
    ).to_dict(False) == {"number_of_twos": [0, 0, 2, 1, 0]}


def test_list_sum_and_dtypes() -> None:
    # ensure the dtypes of sum align with normal sum
    for dt_in, dt_out in [
        (pl.Int8, pl.Int64),
        (pl.Int16, pl.Int64),
        (pl.Int32, pl.Int32),
        (pl.Int64, pl.Int64),
        (pl.UInt8, pl.Int64),
        (pl.UInt16, pl.Int64),
        (pl.UInt32, pl.UInt32),
        (pl.UInt64, pl.UInt64),
    ]:
        df = pl.DataFrame(
            {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]},
            schema={"a": pl.List(dt_in)},
        )

        summed = df.explode("a").sum()
        assert summed.dtypes == [dt_out]
        assert summed.item() == 32
        assert df.select(pl.col("a").list.sum()).dtypes == [dt_out]

    assert df.select(pl.col("a").list.sum()).to_dict(False) == {"a": [1, 6, 10, 15]}

    # include nulls
    assert pl.DataFrame(
        {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None]}
    ).select(pl.col("a").list.sum()).to_dict(False) == {"a": [1, 6, 10, 15, None]}


def test_list_mean() -> None:
    assert pl.DataFrame({"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]}).select(
        pl.col("a").list.mean()
    ).to_dict(False) == {"a": [1.0, 2.0, 2.5, 3.0]}

    assert pl.DataFrame({"a": [[1], [1, 2, 3], [1, 2, 3, 4], None]}).select(
        pl.col("a").list.mean()
    ).to_dict(False) == {"a": [1.0, 2.0, 2.5, None]}


def test_list_all() -> None:
    assert pl.DataFrame(
        {
            "a": [
                [True],
                [False],
                [True, True],
                [True, False],
                [False, False],
                [None],
                [],
            ]
        }
    ).select(pl.col("a").list.all()).to_dict(False) == {
        "a": [True, False, True, False, False, True, True]
    }


def test_list_any() -> None:
    assert pl.DataFrame(
        {
            "a": [
                [True],
                [False],
                [True, True],
                [True, False],
                [False, False],
                [None],
                [],
            ]
        }
    ).select(pl.col("a").list.any()).to_dict(False) == {
        "a": [True, False, True, True, False, False, False]
    }


def test_list_min_max() -> None:
    for dt in pl.NUMERIC_DTYPES:
        if dt == pl.Decimal:
            continue
        df = pl.DataFrame(
            {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]},
            schema={"a": pl.List(dt)},
        )
        assert df.select(pl.col("a").list.min())["a"].series_equal(
            df.select(pl.col("a").list.first())["a"]
        )
        assert df.select(pl.col("a").list.max())["a"].series_equal(
            df.select(pl.col("a").list.last())["a"]
        )

    df = pl.DataFrame(
        {"a": [[1], [1, 5, -1, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None]},
    )
    assert df.select(pl.col("a").list.min()).to_dict(False) == {
        "a": [1, -1, 1, 1, None]
    }
    assert df.select(pl.col("a").list.max()).to_dict(False) == {"a": [1, 5, 4, 5, None]}


def test_fill_null_empty_list() -> None:
    assert pl.Series([["a"], None]).fill_null([]).to_list() == [["a"], []]


def test_nested_logical() -> None:
    assert pl.select(
        pl.lit(pl.Series(["a", "b"], dtype=pl.Categorical)).implode().implode()
    ).to_dict(False) == {"": [[["a", "b"]]]}


def test_null_list_construction_and_materialization() -> None:
    s = pl.Series([None, []])
    assert s.dtype == pl.List(pl.Null)
    assert s.to_list() == [None, []]


def test_logical_type_struct_agg_list() -> None:
    df = pl.DataFrame(
        {"cats": ["Value1", "Value2", "Value1"]},
        schema_overrides={"cats": pl.Categorical},
    )
    out = df.group_by(1).agg(pl.struct("cats"))
    assert out.dtypes == [
        pl.Int32,
        pl.List(pl.Struct([pl.Field("cats", pl.Categorical)])),
    ]
    assert out["cats"].to_list() == [
        [{"cats": "Value1"}, {"cats": "Value2"}, {"cats": "Value1"}]
    ]


def test_logical_parallel_list_collect() -> None:
    # this triggers the anonymous builder in par collect
    out = (
        pl.DataFrame(
            {
                "Group": ["GroupA", "GroupA", "GroupA"],
                "Values": ["Value1", "Value2", "Value1"],
            },
            schema_overrides={"Values": pl.Categorical},
        )
        .group_by("Group")
        .agg(pl.col("Values").value_counts(sort=True))
        .explode("Values")
        .unnest("Values")
    )
    assert out.dtypes == [pl.Utf8, pl.Categorical, pl.UInt32]
    assert out.to_dict(False) == {
        "Group": ["GroupA", "GroupA"],
        "Values": ["Value1", "Value2"],
        "counts": [2, 1],
    }


def test_list_recursive_categorical_cast() -> None:
    # go 3 deep, just to show off
    dtype = pl.List(pl.List(pl.List(pl.Categorical)))
    values = [[[["x"], ["y"]]], [[["x"]]]]
    s = pl.Series(values).cast(dtype)
    assert s.dtype == dtype
    assert s.to_list() == values


@pytest.mark.parametrize(
    ("data", "expected_data", "dtype"),
    [
        ([1, 2], [[1], [2]], pl.Int64),
        ([1.0, 2.0], [[1.0], [2.0]], pl.Float64),
        (["x", "y"], [["x"], ["y"]], pl.Utf8),
        ([True, False], [[True], [False]], pl.Boolean),
    ],
)
def test_non_nested_cast_to_list(
    data: list[Any], expected_data: list[Any], dtype: PolarsDataType
) -> None:
    s = pl.Series(data, dtype=dtype)
    casted_s = s.cast(pl.List(dtype))
    expected = pl.Series(expected_data, dtype=pl.List(dtype))
    assert_series_equal(casted_s, expected)


def test_list_new_from_index_logical() -> None:
    s = (
        pl.select(pl.struct(pl.Series("a", [date(2001, 1, 1)])).implode())
        .to_series()
        .new_from_index(0, 1)
    )
    assert s.dtype == pl.List(pl.Struct([pl.Field("a", pl.Date)]))
    assert s.to_list() == [[{"a": date(2001, 1, 1)}]]

    # empty new_from_index # 8420
    dtype = pl.List(pl.Struct({"c": pl.Boolean}))
    s = pl.Series("b", values=[[]], dtype=dtype)
    s = s.new_from_index(0, 2)
    assert s.dtype == dtype
    assert s.to_list() == [[], []]


def test_list_recursive_time_unit_cast() -> None:
    values = [[datetime(2000, 1, 1, 0, 0, 0)]]
    dtype = pl.List(pl.Datetime("ns"))
    s = pl.Series(values)
    out = s.cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == values


def test_list_null_list_categorical_cast() -> None:
    expected = pl.List(pl.Categorical)
    s = pl.Series([[]], dtype=pl.List(pl.Null)).cast(expected)
    assert s.dtype == expected
    assert s.to_list() == [[]]


def test_list_null_pickle() -> None:
    df = pl.DataFrame([{"a": None}], schema={"a": pl.List(pl.Int64)})
    assert_frame_equal(df, pickle.loads(pickle.dumps(df)))


def test_struct_with_nulls_as_list() -> None:
    df = pl.DataFrame([[{"a": 1, "b": 2}], [{"c": 3, "d": None}]])
    assert df.select(pl.concat_list(pl.all()).alias("as_list")).to_dict(False) == {
        "as_list": [
            [
                {"a": 1, "b": 2, "c": None, "d": None},
                {"a": None, "b": None, "c": 3, "d": None},
            ]
        ]
    }


def test_list_amortized_iter_clear_settings_10126() -> None:
    out = (
        pl.DataFrame({"a": [[1], [1], [2]], "b": [[1, 2], [1, 3], [4]]})
        .explode("a")
        .group_by("a")
        .agg(pl.col("b").flatten())
        .with_columns(pl.col("b").list.unique())
        .sort("a")
    )

    assert out.to_dict(False) == {"a": [1, 2], "b": [[1, 2, 3], [4]]}


def test_list_inner_cast_physical_11513() -> None:
    df = pl.DataFrame(
        {
            "date": ["foo"],
            "struct": [[]],
        },
        schema_overrides={
            "struct": pl.List(
                pl.Struct(
                    {
                        "field": pl.Struct(
                            {"subfield": pl.List(pl.Struct({"subsubfield": pl.Date}))}
                        )
                    }
                )
            )
        },
    )
    assert df.select(pl.col("struct").take(0)).to_dict(False) == {"struct": [[]]}
