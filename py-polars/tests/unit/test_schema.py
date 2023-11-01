from __future__ import annotations

from collections import OrderedDict
from datetime import date, timedelta
from typing import Any, Iterator, Mapping

import pytest

import polars as pl
from polars.testing import assert_frame_equal


class CustomSchema(Mapping[str, Any]):
    """Dummy schema object for testing compatibility with Mapping."""

    _entries: dict[str, Any]

    def __init__(self, **named_entries: Any) -> None:
        self._items = OrderedDict(named_entries.items())

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        yield from self._items


def test_custom_schema() -> None:
    df = pl.DataFrame(schema=CustomSchema(bool=pl.Boolean, misc=pl.UInt8))
    assert df.schema == OrderedDict([("bool", pl.Boolean), ("misc", pl.UInt8)])

    with pytest.raises(ValueError):
        pl.DataFrame(schema=CustomSchema(bool="boolean", misc="unsigned int"))


def test_schema_on_agg() -> None:
    df = pl.DataFrame({"a": ["x", "x", "y", "n"], "b": [1, 2, 3, 4]})

    assert (
        df.lazy()
        .group_by("a")
        .agg(
            [
                pl.col("b").min().alias("min"),
                pl.col("b").max().alias("max"),
                pl.col("b").sum().alias("sum"),
                pl.col("b").first().alias("first"),
                pl.col("b").last().alias("last"),
            ]
        )
    ).schema == {
        "a": pl.Utf8,
        "min": pl.Int64,
        "max": pl.Int64,
        "sum": pl.Int64,
        "first": pl.Int64,
        "last": pl.Int64,
    }


def test_fill_null_minimal_upcast_4056() -> None:
    df = pl.DataFrame({"a": [-1, 2, None]})
    df = df.with_columns(pl.col("a").cast(pl.Int8))
    assert df.with_columns(pl.col(pl.Int8).fill_null(-1)).dtypes[0] == pl.Int8
    assert df.with_columns(pl.col(pl.Int8).fill_null(-1000)).dtypes[0] == pl.Int32


def test_pow_dtype() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5],
        }
    ).lazy()

    df = df.with_columns([pl.col("foo").cast(pl.UInt32)]).with_columns(
        [
            (pl.col("foo") * 2**2).alias("scaled_foo"),
            (pl.col("foo") * 2**2.1).alias("scaled_foo2"),
        ]
    )
    assert df.collect().dtypes == [pl.UInt32, pl.UInt32, pl.Float64]


def test_bool_numeric_supertype() -> None:
    df = pl.DataFrame({"v": [1, 2, 3, 4, 5, 6]})
    for dt in [
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
    ]:
        assert (
            df.select([(pl.col("v") < 3).sum().cast(dt) / pl.count()]).item()
            - 0.3333333
            <= 0.00001
        )


def test_with_context() -> None:
    df_a = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "c", None]}).lazy()
    df_b = pl.DataFrame({"c": ["foo", "ham"]})

    assert (
        df_a.with_context(df_b.lazy()).select([pl.col("b") + pl.col("c").first()])
    ).collect().to_dict(as_series=False) == {"b": ["afoo", "cfoo", None]}

    with pytest.raises(pl.ComputeError):
        (df_a.with_context(df_b.lazy()).select(["a", "c"])).collect()


def test_from_dicts_nested_nulls() -> None:
    assert pl.from_dicts([{"a": [None, None]}, {"a": [1, 2]}]).to_dict(
        as_series=False
    ) == {"a": [[None, None], [1, 2]]}


def test_group_schema_err() -> None:
    df = pl.DataFrame({"foo": [None, 1, 2], "bar": [1, 2, 3]}).lazy()
    with pytest.raises(pl.ColumnNotFoundError):
        df.group_by("not-existent").agg(pl.col("bar").max().alias("max_bar")).schema


def test_schema_inference_from_rows() -> None:
    # these have to upcast to float
    assert pl.from_records([[1, 2.1, 3], [4, 5, 6.4]]).to_dict(as_series=False) == {
        "column_0": [1.0, 2.1, 3.0],
        "column_1": [4.0, 5.0, 6.4],
    }
    assert pl.from_dicts([{"a": 1, "b": 2}, {"a": 3.1, "b": 4.5}]).to_dict(
        as_series=False
    ) == {
        "a": [1.0, 3.1],
        "b": [2.0, 4.5],
    }


def test_lazy_map_schema() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    # identity
    assert_frame_equal(df.lazy().map_batches(lambda x: x).collect(), df)

    def custom(df: pl.DataFrame) -> pl.Series:
        return df["a"]

    with pytest.raises(
        pl.ComputeError,
        match="Expected 'LazyFrame.map' to return a 'DataFrame', got a",
    ):
        df.lazy().map_batches(custom).collect()  # type: ignore[arg-type]

    def custom2(
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        # changes schema
        return df.select(pl.all().cast(pl.Utf8))

    with pytest.raises(
        pl.ComputeError,
        match="The output schema of 'LazyFrame.map' is incorrect. Expected",
    ):
        df.lazy().map_batches(custom2).collect()

    assert df.lazy().map_batches(
        custom2, validate_output_schema=False
    ).collect().to_dict(as_series=False) == {"a": ["1", "2", "3"], "b": ["a", "b", "c"]}


def test_join_as_of_by_schema() -> None:
    a = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    b = pl.DataFrame({"a": [1], "b": [2], "d": [4]}).lazy()
    q = a.join_asof(b, on=pl.col("a").set_sorted(), by="b")
    assert q.collect().columns == q.columns


def test_unknown_map_elements() -> None:
    df = pl.DataFrame(
        {"Amount": [10, 1, 1, 5], "Flour": ["1000g", "100g", "50g", "75g"]}
    )

    q = df.lazy().select(
        [
            pl.col("Amount"),
            pl.col("Flour").map_elements(lambda x: 100.0) / pl.col("Amount"),
        ]
    )

    assert q.collect().to_dict(as_series=False) == {
        "Amount": [10, 1, 1, 5],
        "Flour": [10.0, 100.0, 100.0, 20.0],
    }
    assert q.dtypes == [pl.Int64, pl.Unknown]


def test_remove_redundant_mapping_4668() -> None:
    df = pl.DataFrame([["a"]] * 2, ["A", "B "]).lazy()
    clean_name_dict = {x: " ".join(x.split()) for x in df.columns}
    df = df.rename(clean_name_dict)
    assert df.columns == ["A", "B"]


def test_fold_all_schema() -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
            "optional": [28, 300, None, 2, -30],
        }
    )
    # divide because of overflow
    result = df.select(pl.sum_horizontal(pl.all().hash(seed=1) // int(1e8)))
    assert result.dtypes == [pl.UInt64]


def test_fill_null_static_schema_4843() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": [1, None, 4],
        }
    ).lazy()

    df2 = df1.select([pl.col(pl.Int64).fill_null(0)])
    df3 = df2.select(pl.col(pl.Int64))
    assert df3.schema == {"a": pl.Int64, "b": pl.Int64}


def test_shrink_dtype() -> None:
    out = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 2 << 32],
            "c": [-1, 2, 1 << 30],
            "d": [-112, 2, 112],
            "e": [-112, 2, 129],
            "f": ["a", "b", "c"],
            "g": [0.1, 1.32, 0.12],
            "h": [True, None, False],
            "i": pl.Series([None, None, None], dtype=pl.UInt64),
            "j": pl.Series([None, None, None], dtype=pl.Int64),
            "k": pl.Series([None, None, None], dtype=pl.Float64),
        }
    ).select(pl.all().shrink_dtype())
    assert out.dtypes == [
        pl.Int8,
        pl.Int64,
        pl.Int32,
        pl.Int8,
        pl.Int16,
        pl.Utf8,
        pl.Float32,
        pl.Boolean,
        pl.UInt8,
        pl.Int8,
        pl.Float32,
    ]

    assert out.to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [1, 2, 8589934592],
        "c": [-1, 2, 1073741824],
        "d": [-112, 2, 112],
        "e": [-112, 2, 129],
        "f": ["a", "b", "c"],
        "g": [0.10000000149011612, 1.3200000524520874, 0.11999999731779099],
        "h": [True, None, False],
        "i": [None, None, None],
        "j": [None, None, None],
        "k": [None, None, None],
    }


def test_diff_duration_dtype() -> None:
    data = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-03"]
    df = pl.Series("date", data).str.to_date("%Y-%m-%d").to_frame()

    result = df.select(pl.col("date").diff() < pl.duration(days=1))

    expected = pl.Series("date", [None, False, False, True]).to_frame()
    assert_frame_equal(result, expected)


def test_schema_owned_arithmetic_5669() -> None:
    df = (
        pl.DataFrame({"A": [1, 2, 3]})
        .lazy()
        .filter(pl.col("A") >= 3)
        .with_columns(-pl.col("A").alias("B"))
        .collect()
    )
    assert df.columns == ["A", "B"]
    assert df.rows() == [(3, -3)]


def test_fill_null_f32_with_lit() -> None:
    # ensure the literal integer does not upcast the f32 to an f64
    df = pl.DataFrame({"a": [1.1, 1.2]}, schema=[("a", pl.Float32)])
    assert df.fill_null(value=0).dtypes == [pl.Float32]


def test_lazy_rename() -> None:
    df = pl.DataFrame({"x": [1], "y": [2]})

    assert (
        df.lazy().rename({"y": "x", "x": "y"}).select(["x", "y"]).collect()
    ).to_dict(as_series=False) == {"x": [2], "y": [1]}


def test_all_null_cast_5826() -> None:
    df = pl.DataFrame(data=[pl.Series("a", [None], dtype=pl.Utf8)])
    out = df.with_columns(pl.col("a").cast(pl.Boolean))
    assert out.dtypes == [pl.Boolean]
    assert out.item() is None


def test_empty_list_eval_schema_5734() -> None:
    df = pl.DataFrame({"a": [[{"b": 1, "c": 2}]]})
    assert df.filter(False).select(
        pl.col("a").list.eval(pl.element().struct.field("b"))
    ).schema == {"a": pl.List(pl.Int64)}


def test_schema_true_divide_6643() -> None:
    df = pl.DataFrame({"a": [1]})
    a = pl.col("a")
    assert df.lazy().select(a / 2).select(pl.col(pl.Int64)).collect().shape == (0, 0)


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

    assert renamed.schema == renamed.collect().schema
    assert computed.schema == computed.collect().schema


def test_from_dicts_all_cols_6716() -> None:
    dicts = [{"a": None} for _ in range(20)] + [{"a": "crash"}]

    with pytest.raises(
        pl.ComputeError, match="make sure that all rows have the same schema"
    ):
        pl.from_dicts(dicts, infer_schema_length=20)
    assert pl.from_dicts(dicts, infer_schema_length=None).dtypes == [pl.Utf8]


def test_from_dicts_empty() -> None:
    with pytest.raises(pl.NoDataError, match="no data, cannot infer schema"):
        pl.from_dicts([])


def test_duration_division_schema() -> None:
    df = pl.DataFrame({"a": [1]})
    q = (
        df.lazy()
        .with_columns(pl.col("a").cast(pl.Duration))
        .select(pl.col("a") / pl.col("a"))
    )

    assert q.schema == {"a": pl.Float64}
    assert q.collect().to_dict(as_series=False) == {"a": [1.0]}


def test_int_operator_stability() -> None:
    for dt in pl.datatypes.INTEGER_DTYPES:
        s = pl.Series(values=[10], dtype=dt)
        assert pl.select(pl.lit(s) // 2).dtypes == [dt]
        assert pl.select(pl.lit(s) + 2).dtypes == [dt]
        assert pl.select(pl.lit(s) - 2).dtypes == [dt]
        assert pl.select(pl.lit(s) * 2).dtypes == [dt]
        assert pl.select(pl.lit(s) / 2).dtypes == [pl.Float64]


def test_deep_subexpression_f32_schema_7129() -> None:
    df = pl.DataFrame({"a": [1.1, 2.3, 3.4, 4.5]}, schema={"a": pl.Float32()})
    assert df.with_columns(pl.col("a") - pl.col("a").median()).dtypes == [pl.Float32]
    assert df.with_columns(
        (pl.col("a") - pl.col("a").mean()) / (pl.col("a").std() + 0.001)
    ).dtypes == [pl.Float32]


def test_absence_off_null_prop_8224() -> None:
    # a reminder to self to not do null propagation
    # it is inconsistent and makes output dtype
    # dependent of the data, big no!

    def sub_col_min(column: str, min_column: str) -> pl.Expr:
        return pl.col(column).sub(pl.col(min_column).min())

    df = pl.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "vals_num": [10.0, 11.0, 12.0, 13.0],
            "vals_partial": [None, None, 12.0, 13.0],
            "vals_null": [None, None, None, None],
        }
    )

    q = (
        df.lazy()
        .group_by("group")
        .agg(
            [
                sub_col_min("vals_num", "vals_num").alias("sub_num"),
                sub_col_min("vals_num", "vals_partial").alias("sub_partial"),
                sub_col_min("vals_num", "vals_null").alias("sub_null"),
            ]
        )
    )

    assert q.collect().dtypes == [
        pl.Int64,
        pl.List(pl.Float64),
        pl.List(pl.Float64),
        pl.List(pl.Float64),
    ]


@pytest.mark.parametrize(
    ("data", "expr", "expected_select", "expected_gb"),
    [
        (
            {"x": ["x"], "y": ["y"]},
            pl.coalesce(pl.col("x"), pl.col("y")),
            {"x": pl.Utf8},
            {"x": pl.List(pl.Utf8)},
        ),
        (
            {"x": [True]},
            pl.col("x").sum(),
            {"x": pl.UInt32},
            {"x": pl.UInt32},
        ),
    ],
)
def test_schemas(
    data: dict[str, list[Any]],
    expr: pl.Expr,
    expected_select: dict[str, pl.PolarsDataType],
    expected_gb: dict[str, pl.PolarsDataType],
) -> None:
    df = pl.DataFrame(data)

    # test selection schema
    schema = df.select(expr).schema
    for key, dtype in expected_select.items():
        assert schema[key] == dtype

    # test group_by schema
    schema = df.group_by(pl.lit(1)).agg(expr).schema
    for key, dtype in expected_gb.items():
        assert schema[key] == dtype


def test_list_null_constructor_schema() -> None:
    expected = pl.List(pl.Null)
    assert pl.Series([[]]).dtype == expected
    assert pl.Series([[]], dtype=pl.List).dtype == expected
    assert pl.DataFrame({"a": [[]]}).dtypes[0] == expected
    assert pl.DataFrame(schema={"a": pl.List}).dtypes[0] == expected


def test_schema_ne_missing_9256() -> None:
    df = pl.DataFrame({"a": [0, 1, None], "b": [True, False, True]})

    assert df.select(pl.col("a").ne_missing(0).or_(pl.col("b")))["a"].all()


def test_concat_vertically_relaxed() -> None:
    a = pl.DataFrame(
        data={"a": [1, 2, 3], "b": [True, False, None]},
        schema={"a": pl.Int8, "b": pl.Boolean},
    )
    b = pl.DataFrame(
        data={"a": [43, 2, 3], "b": [32, 1, None]},
        schema={"a": pl.Int16, "b": pl.Int64},
    )
    out = pl.concat([a, b], how="vertical_relaxed")
    assert out.schema == {"a": pl.Int16, "b": pl.Int64}
    assert out.to_dict(as_series=False) == {
        "a": [1, 2, 3, 43, 2, 3],
        "b": [1, 0, None, 32, 1, None],
    }
    out = pl.concat([b, a], how="vertical_relaxed")
    assert out.schema == {"a": pl.Int16, "b": pl.Int64}
    assert out.to_dict(as_series=False) == {
        "a": [43, 2, 3, 1, 2, 3],
        "b": [32, 1, None, 1, 0, None],
    }

    c = pl.DataFrame({"a": [1, 2], "b": [2, 1]})
    d = pl.DataFrame({"a": [1.0, 0.2], "b": [None, 0.1]})

    out = pl.concat([c, d], how="vertical_relaxed")
    assert out.schema == {"a": pl.Float64, "b": pl.Float64}
    assert out.to_dict(as_series=False) == {
        "a": [1.0, 2.0, 1.0, 0.2],
        "b": [2.0, 1.0, None, 0.1],
    }
    out = pl.concat([d, c], how="vertical_relaxed")
    assert out.schema == {"a": pl.Float64, "b": pl.Float64}
    assert out.to_dict(as_series=False) == {
        "a": [1.0, 0.2, 1.0, 2.0],
        "b": [None, 0.1, 2.0, 1.0],
    }


def test_lit_iter_schema() -> None:
    df = pl.DataFrame(
        {
            "key": ["A", "A", "A", "A"],
            "dates": [
                date(1970, 1, 1),
                date(1970, 1, 1),
                date(1970, 1, 2),
                date(1970, 1, 3),
            ],
        }
    )

    result = df.group_by("key").agg(pl.col("dates").unique() + timedelta(days=1))
    expected = {
        "key": ["A"],
        "dates": [[date(1970, 1, 2), date(1970, 1, 3), date(1970, 1, 4)]],
    }
    assert result.to_dict(as_series=False) == expected
