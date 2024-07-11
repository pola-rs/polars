from __future__ import annotations

import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import NUMERIC_DTYPES

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_dtype() -> None:
    # inferred
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    assert a.dtype == pl.List
    assert a.dtype.inner == pl.Int64  # type: ignore[attr-defined]
    assert a.dtype.is_(pl.List(pl.Int64))

    # explicit
    u64_max = (2**64) - 1
    df = pl.DataFrame(
        data={
            "i": [[1, 2, 3]],
            "li": [[[1, 2, 3]]],
            "u": [[u64_max]],
            "tm": [[time(10, 30, 45)]],
            "dt": [[date(2022, 12, 31)]],
            "dtm": [[datetime(2022, 12, 31, 1, 2, 3)]],
        },
        schema=[
            ("i", pl.List(pl.Int8)),
            ("li", pl.List(pl.List(pl.Int8))),
            ("u", pl.List(pl.UInt64)),
            ("tm", pl.List(pl.Time)),
            ("dt", pl.List(pl.Date)),
            ("dtm", pl.List(pl.Datetime)),
        ],
    )
    assert df.schema == {
        "i": pl.List(pl.Int8),
        "li": pl.List(pl.List(pl.Int8)),
        "u": pl.List(pl.UInt64),
        "tm": pl.List(pl.Time),
        "dt": pl.List(pl.Date),
        "dtm": pl.List(pl.Datetime),
    }
    assert all(tp.is_nested() for tp in df.dtypes)
    assert df.schema["i"].inner == pl.Int8  # type: ignore[attr-defined]
    assert df.rows() == [
        (
            [1, 2, 3],
            [[1, 2, 3]],
            [u64_max],
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
            pl.col("c").count().alias("num_different_c"),
            pl.col("c").alias("c_values"),
        )
        .filter(pl.col("num_different_c") >= 2)
        .to_series(3)
    )

    assert out.dtype.inner == pl.Categorical  # type: ignore[attr-defined]
    assert out.dtype.inner.is_nested() is False  # type: ignore[attr-defined]


def test_decimal() -> None:
    input = [[Decimal("1.23"), Decimal("4.56")], [Decimal("7.89"), Decimal("10.11")]]
    s = pl.Series(input)
    assert s.dtype == pl.List(pl.Decimal)
    assert s.dtype.inner == pl.Decimal  # type: ignore[attr-defined]
    assert s.dtype.inner.is_nested() is False  # type: ignore[attr-defined]
    assert s.to_list() == input


def test_cast_inner() -> None:
    a = pl.Series([[1, 2]])
    for t in [bool, pl.Boolean]:
        b = a.cast(pl.List(t))
        assert b.dtype == pl.List(pl.Boolean)
        assert b.to_list() == [[True, True]]

    # this creates an inner null type
    df = pl.from_pandas(pd.DataFrame(data=[[[]], [[]]], columns=["A"]))
    assert (
        df["A"].cast(pl.List(int)).dtype.inner == pl.Int64  # type: ignore[attr-defined]
    )


def test_list_empty_group_by_result_3521() -> None:
    # Create a left relation where the join column contains a null value
    left = pl.DataFrame().with_columns(
        pl.lit(1).alias("group_by_column"),
        pl.lit(None).cast(pl.Int32).alias("join_column"),
    )

    # Create a right relation where there is a column to count distinct on
    right = pl.DataFrame().with_columns(
        pl.lit(1).alias("join_column"),
        pl.lit(1).alias("n_unique_column"),
    )

    # Calculate n_unique after dropping nulls
    # This will panic on polars version 0.13.38 and 0.13.39
    result = (
        left.join(right, on="join_column", how="left")
        .group_by("group_by_column")
        .agg(pl.col("n_unique_column").drop_nulls())
    )
    expected = {"group_by_column": [1], "n_unique_column": [[]]}
    assert result.to_dict(as_series=False) == expected


def test_list_fill_null() -> None:
    df = pl.DataFrame({"C": [["a", "b", "c"], [], [], ["d", "e"]]})
    assert df.with_columns(
        pl.when(pl.col("C").list.len() == 0)
        .then(None)
        .otherwise(pl.col("C"))
        .alias("C")
    ).to_series().to_list() == [["a", "b", "c"], None, None, ["d", "e"]]


def test_list_fill_list() -> None:
    assert pl.DataFrame({"a": [[1, 2, 3], []]}).select(
        pl.when(pl.col("a").list.len() == 0)
        .then([5])
        .otherwise(pl.col("a"))
        .alias("filled")
    ).to_dict(as_series=False) == {"filled": [[1, 2, 3], [5]]}


def test_empty_list_construction() -> None:
    assert pl.Series([[]]).to_list() == [[]]

    df = pl.DataFrame([{"array": [], "not_array": 1234}], orient="row")
    expected = {"array": [[]], "not_array": [1234]}
    assert df.to_dict(as_series=False) == expected

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

    assert pl.concat([df1, df2], how="diagonal").to_dict(as_series=False) == {
        "a": [1, 2, None],
        "b": [None, None, [1]],
    }


def test_inner_type_categorical_on_rechunk() -> None:
    df = pl.DataFrame({"cats": ["foo", "bar"]}).select(
        pl.col(pl.String).cast(pl.Categorical).implode()
    )

    assert pl.concat([df, df], rechunk=True).dtypes == [pl.List(pl.Categorical)]


def test_local_categorical_list() -> None:
    values = [["a", "b"], ["c"], ["a", "d", "d"]]
    s = pl.Series(values, dtype=pl.List(pl.Categorical))
    assert s.dtype == pl.List
    assert s.dtype.inner == pl.Categorical  # type: ignore[attr-defined]
    assert s.to_list() == values

    # Check that underlying physicals match
    idx_df = pl.Series([[0, 1], [2], [0, 3, 3]], dtype=pl.List(pl.UInt32))
    assert_series_equal(s.cast(pl.List(pl.UInt32)), idx_df)

    # Check if the categories array does not overlap
    assert s.list.explode().cat.get_categories().to_list() == ["a", "b", "c", "d"]


def test_group_by_list_column() -> None:
    df = (
        pl.DataFrame({"a": ["a", "b", "a"]})
        .with_columns(pl.col("a").cast(pl.Categorical))
        .group_by("a", maintain_order=True)
        .agg(pl.col("a").alias("a_list"))
    )

    assert df.group_by("a_list", maintain_order=True).first().to_dict(
        as_series=False
    ) == {
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
    assert df.to_dict(as_series=False) == {
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
            "label": pl.String,
            "tag": pl.String,
            "ref": pl.Int64,
            "parents": pl.List(
                pl.Struct({"ref": pl.Int64, "tag": pl.String, "ratio": pl.Float64})
            ),
        },
    )

    assert not df["parents"].flags["FAST_EXPLODE"]
    assert df.explode("parents").to_dict(as_series=False) == {
        "label": ["l", "l"],
        "tag": ["t", "t"],
        "ref": [1, 1],
        "parents": [{"ref": 1, "tag": "t", "ratio": 62.3}, None],
    }


def test_flat_aggregation_to_list_conversion_6918() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [[0, 1], [2, 3], [4, 5]]})

    assert df.group_by("a", maintain_order=True).agg(
        pl.concat_list([pl.col("b").list.get(i).mean().implode() for i in range(2)])
    ).to_dict(as_series=False) == {"a": [1, 2], "b": [[[0.0, 1.0]], [[3.0, 4.0]]]}


def test_list_count_matches() -> None:
    assert pl.DataFrame({"listcol": [[], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]}).select(
        pl.col("listcol").list.count_matches(2).alias("number_of_twos")
    ).to_dict(as_series=False) == {"number_of_twos": [0, 0, 2, 1, 0]}


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

    assert df.select(pl.col("a").list.sum()).to_dict(as_series=False) == {
        "a": [1, 6, 10, 15]
    }

    # include nulls
    assert pl.DataFrame(
        {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None]}
    ).select(pl.col("a").list.sum()).to_dict(as_series=False) == {
        "a": [1, 6, 10, 15, None]
    }

    # Booleans
    assert pl.DataFrame(
        {"a": [[True], [True, True], [True, False, True], [True, True, True, None]]},
    ).select(pl.col("a").list.sum()).to_dict(as_series=False) == {"a": [1, 2, 2, 3]}

    assert pl.DataFrame(
        {"a": [[False], [False, False], [False, False, False]]},
    ).select(pl.col("a").list.sum()).to_dict(as_series=False) == {"a": [0, 0, 0]}

    assert pl.DataFrame(
        {"a": [[True], [True, True], [True, True, True]]},
    ).select(pl.col("a").list.sum()).to_dict(as_series=False) == {"a": [1, 2, 3]}


def test_list_mean() -> None:
    assert pl.DataFrame({"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]}).select(
        pl.col("a").list.mean()
    ).to_dict(as_series=False) == {"a": [1.0, 2.0, 2.5, 3.0]}

    assert pl.DataFrame({"a": [[1], [1, 2, 3], [1, 2, 3, 4], None]}).select(
        pl.col("a").list.mean()
    ).to_dict(as_series=False) == {"a": [1.0, 2.0, 2.5, None]}


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
    ).select(pl.col("a").list.all()).to_dict(as_series=False) == {
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
    ).select(pl.col("a").list.any()).to_dict(as_series=False) == {
        "a": [True, False, True, True, False, False, False]
    }


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_list_min_max(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]},
        schema={"a": pl.List(dtype)},
    )
    result = df.select(pl.col("a").list.min())
    expected = df.select(pl.col("a").list.first())
    assert_frame_equal(result, expected)

    result = df.select(pl.col("a").list.max())
    expected = df.select(pl.col("a").list.last())
    assert_frame_equal(result, expected)


def test_list_min_max2() -> None:
    df = pl.DataFrame(
        {"a": [[1], [1, 5, -1, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None]},
    )
    assert df.select(pl.col("a").list.min()).to_dict(as_series=False) == {
        "a": [1, -1, 1, 1, None]
    }
    assert df.select(pl.col("a").list.max()).to_dict(as_series=False) == {
        "a": [1, 5, 4, 5, None]
    }


def test_list_mean_fast_path_empty() -> None:
    df = pl.DataFrame(
        {
            "a": [[], [1, 2, 3]],
        }
    )
    output = df.select(pl.col("a").list.mean())
    assert output.to_dict(as_series=False) == {"a": [None, 2.0]}


def test_list_min_max_13978() -> None:
    df = pl.DataFrame(
        {
            "a": [[], [1, 2, 3]],
            "b": [[1, 2], None],
            "c": [[], [None, 1, 2]],
        }
    )
    out = df.select(
        min_a=pl.col("a").list.min(),
        max_a=pl.col("a").list.max(),
        min_b=pl.col("b").list.min(),
        max_b=pl.col("b").list.max(),
        min_c=pl.col("c").list.min(),
        max_c=pl.col("c").list.max(),
    )
    expected = pl.DataFrame(
        {
            "min_a": [None, 1],
            "max_a": [None, 3],
            "min_b": [1, None],
            "max_b": [2, None],
            "min_c": [None, 1],
            "max_c": [None, 2],
        }
    )
    assert_frame_equal(out, expected)


def test_fill_null_empty_list() -> None:
    assert pl.Series([["a"], None]).fill_null([]).to_list() == [["a"], []]


def test_nested_logical() -> None:
    assert pl.select(
        pl.lit(pl.Series(["a", "b"], dtype=pl.Categorical)).implode().implode()
    ).to_dict(as_series=False) == {"": [[["a", "b"]]]}


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
    assert out.dtypes == [pl.String, pl.Categorical, pl.UInt32]
    assert out.to_dict(as_series=False) == {
        "Group": ["GroupA", "GroupA"],
        "Values": ["Value1", "Value2"],
        "count": [2, 1],
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
        ([None, 1, 2], [None, [1], [2]], pl.Int64),
        ([None, 1.0, 2.0], [None, [1.0], [2.0]], pl.Float64),
        ([None, "x", "y"], [None, ["x"], ["y"]], pl.String),
        ([None, True, False], [None, [True], [False]], pl.Boolean),
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
    result = df.select(pl.concat_list(pl.all()).alias("as_list"))
    assert result.to_dict(as_series=False) == {
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

    assert out.to_dict(as_series=False) == {"a": [1, 2], "b": [[1, 2, 3], [4]]}


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
    assert df.select(pl.col("struct").gather(0)).to_dict(as_series=False) == {
        "struct": [[]]
    }


@pytest.mark.parametrize(
    ("dtype", "expected"), [(pl.List, True), (pl.Struct, True), (pl.String, False)]
)
def test_datatype_is_nested(dtype: PolarsDataType, expected: bool) -> None:
    assert dtype.is_nested() is expected


def test_list_series_construction_with_dtype_11849_11878() -> None:
    s = pl.Series([[1, 2], [3.3, 4.9]], dtype=pl.List(pl.Float64))
    assert s.to_list() == [[1, 2], [3.3, 4.9]]

    s1 = pl.Series([[1, 2], [3.0, 4.0]], dtype=pl.List(pl.Float64))
    s2 = pl.Series([[1, 2], [3.0, 4.9]], dtype=pl.List(pl.Float64))
    assert_series_equal(s1 == s2, pl.Series([True, False]))

    s = pl.Series(
        "groups",
        [[{"1": "A", "2": None}], [{"1": "B", "2": "C"}, {"1": "D", "2": "E"}]],
        dtype=pl.List(pl.Struct([pl.Field("1", pl.String), pl.Field("2", pl.String)])),
    )

    assert s.to_list() == [
        [{"1": "A", "2": None}],
        [{"1": "B", "2": "C"}, {"1": "D", "2": "E"}],
    ]


def test_as_list_logical_type() -> None:
    df = pl.select(timestamp=pl.date(2000, 1, 1), value=0)
    assert df.group_by(True).agg(
        pl.col("timestamp").gather(pl.col("value").arg_max())
    ).to_dict(as_series=False) == {"literal": [True], "timestamp": [[date(2000, 1, 1)]]}


@pytest.fixture()
def data_dispersion() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "int": [[1, 2, 3, 4, 5]],
            "float": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "duration": [[1000, 2000, 3000, 4000, 5000]],
        },
        schema={
            "int": pl.List(pl.Int64),
            "float": pl.List(pl.Float64),
            "duration": pl.List(pl.Duration),
        },
    )


def test_list_var(data_dispersion: pl.DataFrame) -> None:
    df = data_dispersion

    result = df.select(
        pl.col("int").list.var().name.suffix("_var"),
        pl.col("float").list.var().name.suffix("_var"),
        pl.col("duration").list.var().name.suffix("_var"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("int_var", [2.5], dtype=pl.Float64),
            pl.Series("float_var", [2.5], dtype=pl.Float64),
            pl.Series(
                "duration_var",
                [timedelta(microseconds=2000)],
                dtype=pl.Duration(time_unit="ms"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_list_std(data_dispersion: pl.DataFrame) -> None:
    df = data_dispersion

    result = df.select(
        pl.col("int").list.std().name.suffix("_std"),
        pl.col("float").list.std().name.suffix("_std"),
        pl.col("duration").list.std().name.suffix("_std"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("int_std", [1.5811388300841898], dtype=pl.Float64),
            pl.Series("float_std", [1.5811388300841898], dtype=pl.Float64),
            pl.Series(
                "duration_std",
                [timedelta(microseconds=1581)],
                dtype=pl.Duration(time_unit="us"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_list_median(data_dispersion: pl.DataFrame) -> None:
    df = data_dispersion

    result = df.select(
        pl.col("int").list.median().name.suffix("_median"),
        pl.col("float").list.median().name.suffix("_median"),
        pl.col("duration").list.median().name.suffix("_median"),
    )

    expected = pl.DataFrame(
        [
            pl.Series("int_median", [3.0], dtype=pl.Float64),
            pl.Series("float_median", [3.0], dtype=pl.Float64),
            pl.Series(
                "duration_median",
                [timedelta(microseconds=3000)],
                dtype=pl.Duration(time_unit="us"),
            ),
        ]
    )

    assert_frame_equal(result, expected)


def test_list_gather_null_struct_14927() -> None:
    df = pl.DataFrame(
        [
            {
                "index": 0,
                "col_0": [{"field_0": 1.0}],
            },
            {
                "index": 1,
                "col_0": None,
            },
        ]
    )

    expected = pl.DataFrame(
        {"index": [1], "col_0": [None], "field_0": [None]},
        schema={**df.schema, "field_0": pl.Float64},
    )
    expr = pl.col("col_0").list.get(0, null_on_oob=True).struct.field("field_0")
    out = df.filter(pl.col("index") > 0).with_columns(expr)
    assert_frame_equal(out, expected)


def test_list_of_series_with_nulls() -> None:
    inner_series = pl.Series("inner", [1, 2, 3])
    s = pl.Series("a", [inner_series, None])
    assert_series_equal(s, pl.Series("a", [[1, 2, 3], None]))


def test_take_list_15719() -> None:
    schema = pl.List(pl.List(pl.Int64))
    df = pl.DataFrame(
        {"a": [None, None], "b": [None, [[1, 2]]]}, schema={"a": schema, "b": schema}
    )
    df = df.select(
        a_explode=pl.col("a").explode(),
        a_get=pl.col("a").list.get(0, null_on_oob=True),
        b_explode=pl.col("b").explode(),
        b_get=pl.col("b").list.get(0, null_on_oob=True),
    )

    expected_schema = pl.List(pl.Int64)
    expected = pl.DataFrame(
        {
            "a_explode": [None, None],
            "a_get": [None, None],
            "b_explode": [None, [1, 2]],
            "b_get": [None, [1, 2]],
        },
        schema={
            "a_explode": expected_schema,
            "a_get": expected_schema,
            "b_explode": expected_schema,
            "b_get": expected_schema,
        },
    )

    assert_frame_equal(df, expected)


def test_list_str_sum_exception_12935() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series(["foo", "bar"]).sum()


def test_list_list_sum_exception_12935() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series([[1], [2]]).sum()


def test_null_list_categorical_16405() -> None:
    df = pl.DataFrame(
        [(None, "foo")],
        schema={
            "match": pl.List(pl.Categorical),
            "what": pl.Categorical,
        },
        orient="row",
    )

    df = df.select(
        pl.col("match")
        .list.set_intersection(pl.concat_list(pl.col("what")))
        .alias("result")
    )

    expected = pl.DataFrame([None], schema={"result": pl.List(pl.Categorical)})
    assert_frame_equal(df, expected)
