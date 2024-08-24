from __future__ import annotations

import io
from datetime import date, datetime, time, tzinfo
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.datatypes.convert import dtype_to_py_type
from polars.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    InvalidOperationError,
    OutOfBoundsError,
    PanicException,
    SchemaError,
    SchemaFieldNotFoundError,
    StructFieldNotFoundError,
)
from tests.unit.conftest import TEMPORAL_DTYPES

if TYPE_CHECKING:
    from polars._typing import ConcatMethod


def test_error_on_empty_group_by() -> None:
    with pytest.raises(
        ComputeError, match="at least one key is required in a group_by operation"
    ):
        pl.DataFrame({"x": [0, 0, 1, 1]}).group_by([]).agg(pl.len())


def test_error_on_reducing_map() -> None:
    df = pl.DataFrame(
        {"id": [0, 0, 0, 1, 1, 1], "t": [2, 4, 5, 10, 11, 14], "y": [0, 1, 1, 2, 3, 4]}
    )
    with pytest.raises(
        InvalidOperationError,
        match=(
            r"output length of `map` \(1\) must be equal to "
            r"the input length \(6\); consider using `apply` instead"
        ),
    ):
        df.group_by("id").agg(pl.map_batches(["t", "y"], np.mean))

    df = pl.DataFrame({"x": [1, 2, 3, 4], "group": [1, 2, 1, 2]})
    with pytest.raises(
        InvalidOperationError,
        match=(
            r"output length of `map` \(1\) must be equal to "
            r"the input length \(4\); consider using `apply` instead"
        ),
    ):
        df.select(
            pl.col("x")
            .map_batches(
                lambda x: x.cut(breaks=[1, 2, 3], include_breaks=True).struct.unnest(),
                is_elementwise=True,
            )
            .over("group")
        )


def test_error_on_invalid_by_in_asof_join() -> None:
    df1 = pl.DataFrame(
        {
            "a": ["a", "b", "a"],
            "b": [1, 2, 3],
            "c": ["a", "b", "a"],
        }
    ).set_sorted("b")

    df2 = df1.with_columns(pl.col("a").cast(pl.Categorical))
    with pytest.raises(ComputeError):
        df1.join_asof(df2, on="b", by=["a", "c"])


@pytest.mark.parametrize("dtype", TEMPORAL_DTYPES)
def test_error_on_invalid_series_init(dtype: pl.DataType) -> None:
    py_type = dtype_to_py_type(dtype)
    with pytest.raises(
        TypeError,
        match=f"'float' object cannot be interpreted as a {py_type.__name__!r}",
    ):
        pl.Series([1.5, 2.0, 3.75], dtype=dtype)


def test_error_on_invalid_series_init2() -> None:
    with pytest.raises(TypeError, match="unexpected value"):
        pl.Series([1.5, 2.0, 3.75], dtype=pl.Int32)


def test_error_on_invalid_struct_field() -> None:
    with pytest.raises(StructFieldNotFoundError):
        pl.struct(
            [pl.Series("a", [1, 2]), pl.Series("b", ["a", "b"])], eager=True
        ).struct.field("z")


def test_not_found_error() -> None:
    csv = "a,b,c\n2,1,1"
    df = pl.read_csv(io.StringIO(csv))
    with pytest.raises(ColumnNotFoundError):
        df.select("d")


def test_string_numeric_comp_err() -> None:
    with pytest.raises(ComputeError, match="cannot compare string with numeric type"):
        pl.DataFrame({"a": [1.1, 21, 31, 21, 51, 61, 71, 81]}).select(pl.col("a") < "9")


def test_panic_error() -> None:
    with pytest.raises(
        PanicException,
        match="unit: 'k' not supported",
    ):
        pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 3),
            interval="99k",
            eager=True,
        )


def test_join_lazy_on_df() -> None:
    df_left = pl.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Names": ["A", "B", "C", "D"],
        }
    )
    df_right = pl.DataFrame({"Id": [1, 3], "Tags": ["xxx", "yyy"]})

    with pytest.raises(
        TypeError,
        match="expected `other` .* to be a LazyFrame.* not a 'DataFrame'",
    ):
        df_left.lazy().join(df_right, on="Id")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="expected `other` .* to be a LazyFrame.* not a 'DataFrame'",
    ):
        df_left.lazy().join_asof(df_right, on="Id")  # type: ignore[arg-type]


def test_projection_update_schema_missing_column() -> None:
    with pytest.raises(
        ColumnNotFoundError,
        match='unable to find column "colC"',
    ):
        (
            pl.DataFrame({"colA": ["a", "b", "c"], "colB": [1, 2, 3]})
            .lazy()
            .filter(~pl.col("colC").is_null())
            .group_by(["colA"])
            .agg([pl.col("colB").sum().alias("result")])
            .collect()
        )


def test_not_found_on_rename() -> None:
    df = pl.DataFrame({"exists": [1, 2, 3]})

    err_type = (SchemaFieldNotFoundError, ColumnNotFoundError)
    with pytest.raises(err_type):
        df.rename({"does_not_exist": "exists"})

    with pytest.raises(err_type):
        df.select(pl.col("does_not_exist").alias("new_name"))


def test_getitem_errs() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match=r"cannot select columns using key of type 'set': {'some'}",
    ):
        df[{"some"}]  # type: ignore[call-overload]

    with pytest.raises(
        TypeError,
        match=r"cannot select elements using key of type 'set': {'strange'}",
    ):
        df["a"][{"strange"}]  # type: ignore[call-overload]

    with pytest.raises(
        TypeError,
        match=r"cannot use `__setitem__` on DataFrame with key {'some'} of type 'set' and value 'foo' of type 'str'",
    ):
        df[{"some"}] = "foo"  # type: ignore[index]


def test_err_bubbling_up_to_lit() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1)], "value": [42]})

    with pytest.raises(TypeError):
        df.filter(pl.col("date") == pl.Date("2020-01-01"))  # type: ignore[call-arg,operator]


def test_error_on_double_agg() -> None:
    for e in [
        "mean",
        "max",
        "min",
        "sum",
        "std",
        "var",
        "n_unique",
        "last",
        "first",
        "median",
        "skew",  # this one is comes from Apply
    ]:
        with pytest.raises(ComputeError, match="the column is already aggregated"):
            (
                pl.DataFrame(
                    {
                        "a": [1, 1, 1, 2, 2],
                        "b": [1, 2, 3, 4, 5],
                    }
                )
                .group_by("a")
                .agg([getattr(pl.col("b").min(), e)()])
            )


def test_filter_not_of_type_bool() -> None:
    df = pl.DataFrame({"json_val": ['{"a":"hello"}', None, '{"a":"world"}']})
    with pytest.raises(
        ComputeError, match="filter predicate must be of type `Boolean`, got"
    ):
        df.filter(pl.col("json_val").str.json_path_match("$.a"))


def test_is_nan_on_non_boolean() -> None:
    with pytest.raises(InvalidOperationError):
        pl.Series(["1", "2", "3"]).fill_nan("2")  # type: ignore[arg-type]


def test_window_expression_different_group_length() -> None:
    try:
        pl.DataFrame({"groups": ["a", "a", "b", "a", "b"]}).select(
            pl.col("groups").map_elements(lambda _: pl.Series([1, 2])).over("groups")
        )
    except ComputeError as exc:
        msg = str(exc)
        assert (
            "the length of the window expression did not match that of the group" in msg
        )
        assert "group:" in msg
        assert "group length:" in msg
        assert "output: 'shape:" in msg


def test_invalid_concat_type_err() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2],
            "bar": [6, 7],
            "ham": ["a", "b"],
        }
    )
    with pytest.raises(
        ValueError,
        match="DataFrame `how` must be one of {'vertical', 'vertical_relaxed', 'diagonal', 'diagonal_relaxed', 'horizontal', 'align'}, got 'sausage'",
    ):
        pl.concat([df, df], how="sausage")  # type: ignore[arg-type]


@pytest.mark.parametrize("how", ["horizontal", "diagonal"])
def test_series_concat_err(how: ConcatMethod) -> None:
    s = pl.Series([1, 2, 3])
    with pytest.raises(
        ValueError,
        match="Series only supports 'vertical' concat strategy",
    ):
        pl.concat([s, s], how=how)


def test_invalid_sort_by() -> None:
    df = pl.DataFrame(
        {
            "a": ["bill", "bob", "jen", "allie", "george"],
            "b": ["M", "M", "F", "F", "M"],
            "c": [32, 40, 20, 19, 39],
        }
    )

    # `select a where b order by c desc`
    with pytest.raises(
        ComputeError,
        match=r"`sort_by` produced different length \(5\) than the Series that has to be sorted \(3\)",
    ):
        df.select(pl.col("a").filter(pl.col("b") == "M").sort_by("c", descending=True))


def test_epoch_time_type() -> None:
    with pytest.raises(
        InvalidOperationError,
        match="`timestamp` operation not supported for dtype `time`",
    ):
        pl.Series([time(0, 0, 1)]).dt.epoch("s")


def test_duplicate_columns_arg_csv() -> None:
    f = io.BytesIO()
    pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).write_csv(f)
    f.seek(0)
    with pytest.raises(
        ValueError, match=r"`columns` arg should only have unique values"
    ):
        pl.read_csv(f, columns=["x", "x", "y"])


def test_datetime_time_add_err() -> None:
    with pytest.raises(SchemaError, match="failed to determine supertype"):
        pl.Series([datetime(1970, 1, 1, 0, 0, 1)]) + pl.Series([time(0, 0, 2)])


def test_invalid_dtype() -> None:
    with pytest.raises(
        TypeError,
        match=r"cannot parse input of type 'str' into Polars data type \(given: 'mayonnaise'\)",
    ):
        pl.Series([1, 2], dtype="mayonnaise")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="cannot parse input <class 'datetime.tzinfo'> into Polars data type",
    ):
        pl.Series([None], dtype=tzinfo)  # type: ignore[arg-type]


def test_arr_eval_named_cols() -> None:
    df = pl.DataFrame({"A": ["a", "b"], "B": [["a", "b"], ["c", "d"]]})

    with pytest.raises(
        ComputeError,
    ):
        df.select(pl.col("B").list.eval(pl.element().append(pl.col("A"))))


def test_alias_in_join_keys() -> None:
    df = pl.DataFrame({"A": ["a", "b"], "B": [["a", "b"], ["c", "d"]]})
    with pytest.raises(
        ComputeError,
        match=r"'alias' is not allowed in a join key, use 'with_columns' first",
    ):
        df.join(df, on=pl.col("A").alias("foo"))


def test_sort_by_different_lengths() -> None:
    df = pl.DataFrame(
        {
            "group": ["a"] * 3 + ["b"] * 3,
            "col1": [1, 2, 3, 300, 200, 100],
            "col2": [1, 2, 3, 300, 1, 1],
        }
    )
    with pytest.raises(
        ComputeError,
        match=r"the expression in `sort_by` argument must result in the same length",
    ):
        df.group_by("group").agg(
            [
                pl.col("col1").sort_by(pl.col("col2").unique()),
            ]
        )

    with pytest.raises(
        ComputeError,
        match=r"the expression in `sort_by` argument must result in the same length",
    ):
        df.group_by("group").agg(
            [
                pl.col("col1").sort_by(pl.col("col2").arg_unique()),
            ]
        )


def test_err_filter_no_expansion() -> None:
    # df contains floats
    df = pl.DataFrame(
        {
            "a": [0.1, 0.2],
        }
    )

    with pytest.raises(
        ComputeError, match=r"The predicate expanded to zero expressions"
    ):
        # we filter by ints
        df.filter(pl.col(pl.Int16).min() < 0.1)


@pytest.mark.parametrize(
    ("e"),
    [
        pl.col("date") > "2021-11-10",
        pl.col("date") < "2021-11-10",
    ],
)
def test_date_string_comparison(e: pl.Expr) -> None:
    df = pl.DataFrame(
        {
            "date": [
                "2022-11-01",
                "2022-11-02",
                "2022-11-05",
            ],
        }
    ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    with pytest.raises(
        InvalidOperationError,
        match=r"cannot compare 'date/datetime/time' to a string value",
    ):
        df.select(e)


def test_err_on_multiple_column_expansion() -> None:
    # this would be a great feature :)
    with pytest.raises(
        ComputeError, match=r"expanding more than one `col` is not allowed"
    ):
        pl.DataFrame(
            {
                "a": [1],
                "b": [2],
                "c": [3],
                "d": [4],
            }
        ).select([pl.col(["a", "b"]) + pl.col(["c", "d"])])


def test_compare_different_len() -> None:
    df = pl.DataFrame(
        {
            "idx": list(range(5)),
        }
    )

    s = pl.Series([2, 5, 8])
    with pytest.raises(
        ComputeError, match=r"cannot evaluate two Series of different lengths"
    ):
        df.filter(pl.col("idx") == s)


def test_take_negative_index_is_oob() -> None:
    df = pl.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(OutOfBoundsError):
        df["value"].gather(-4)


def test_string_numeric_arithmetic_err() -> None:
    df = pl.DataFrame({"s": ["x"]})
    with pytest.raises(
        InvalidOperationError, match=r"arithmetic on string and numeric not allowed"
    ):
        df.select(pl.col("s") + 1)


def test_ambiguous_filter_err() -> None:
    df = pl.DataFrame({"a": [None, "2", "3"], "b": [None, None, "z"]})
    with pytest.raises(
        ComputeError,
        match=r"The predicate passed to 'LazyFrame.filter' expanded to multiple expressions",
    ):
        df.filter(pl.col(["a", "b"]).is_null())


def test_with_column_duplicates() -> None:
    df = pl.DataFrame({"a": [0, None, 2, 3, None], "b": [None, 1, 2, 3, None]})
    with pytest.raises(
        ComputeError,
        match=r"the name 'same' passed to `LazyFrame.with_columns` is duplicate.*",
    ):
        assert df.with_columns([pl.all().alias("same")]).columns == ["a", "b", "same"]


def test_skip_nulls_err() -> None:
    df = pl.DataFrame({"foo": [None, None]})

    with pytest.raises(
        ComputeError,
        match=r"The output type of the 'map_elements' function cannot be determined",
    ):
        df.with_columns(pl.col("foo").map_elements(lambda x: x, skip_nulls=True))


@pytest.mark.parametrize(
    ("test_df", "type", "expected_message"),
    [
        pytest.param(
            pl.DataFrame({"A": [1, 2, 3], "B": ["1", "2", "help"]}),
            pl.UInt32,
            "conversion .* failed",
            id="Unsigned integer",
        )
    ],
)
def test_cast_err_column_value_highlighting(
    test_df: pl.DataFrame, type: pl.DataType, expected_message: str
) -> None:
    with pytest.raises(InvalidOperationError, match=expected_message):
        test_df.with_columns(pl.all().cast(type))


def test_lit_agg_err() -> None:
    with pytest.raises(ComputeError, match=r"cannot aggregate a literal"):
        pl.DataFrame({"y": [1]}).with_columns(pl.lit(1).sum().over("y"))


def test_invalid_group_by_arg() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(
        TypeError, match="specifying aggregations as a dictionary is not supported"
    ):
        df.group_by(1).agg({"a": "sum"})


def test_overflow_msg() -> None:
    with pytest.raises(
        ComputeError,
        match=r"could not append value: 2147483648 of type: i64 to the builder",
    ):
        pl.DataFrame([[2**31]], [("a", pl.Int32)], orient="row")


def test_sort_by_err_9259() -> None:
    df = pl.DataFrame(
        {"a": [1, 1, 1], "b": [3, 2, 1], "c": [1, 1, 2]},
        schema={"a": pl.Float32, "b": pl.Float32, "c": pl.Float32},
    )
    with pytest.raises(ComputeError):
        df.lazy().group_by("c").agg(
            [pl.col("a").sort_by(pl.col("b").filter(pl.col("b") > 100)).sum()]
        ).collect()


def test_empty_inputs_error() -> None:
    df = pl.DataFrame({"col1": [1]})
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="expected at least 1 input"
    ):
        df.select(pl.sum_horizontal(pl.exclude("col1")))


@pytest.mark.parametrize(
    ("colname", "values", "expected"),
    [
        ("a", [2], [False, True, False]),
        ("a", [True, False], None),
        ("a", ["2", "3", "4"], None),
        ("b", [Decimal("3.14")], None),
        ("c", [-2, -1, 0, 1, 2], None),
        (
            "d",
            pl.datetime_range(
                datetime.now(),
                datetime.now(),
                interval="2345ns",
                time_unit="ns",
                eager=True,
            ),
            None,
        ),
        ("d", [time(10, 30)], None),
        ("e", [datetime(1999, 12, 31, 10, 30)], None),
        ("f", ["xx", "zz"], None),
    ],
)
def test_invalid_is_in_dtypes(
    colname: str, values: list[Any], expected: list[Any] | None
) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [-2.5, 0.0, 2.5],
            "c": [True, None, False],
            "d": [datetime(2001, 10, 30), None, datetime(2009, 7, 5)],
            "e": [date(2029, 12, 31), date(1999, 12, 31), None],
            "f": [b"xx", b"yy", b"zz"],
        }
    )
    if expected is None:
        with pytest.raises(
            InvalidOperationError,
            match="'is_in' cannot check for .*? values in .*? data",
        ):
            df.select(pl.col(colname).is_in(values))
    else:
        assert df.select(pl.col(colname).is_in(values))[colname].to_list() == expected


def test_sort_by_error() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3, 3, 3],
            "number": [1, 3, 2, 1, 2, 2, 1, 3],
            "type": ["A", "B", "A", "B", "B", "A", "B", "C"],
            "cost": [10, 25, 20, 25, 30, 30, 50, 100],
        }
    )

    with pytest.raises(
        ComputeError,
        match="expressions in 'sort_by' produced a different number of groups",
    ):
        df.group_by("id", maintain_order=True).agg(
            pl.col("cost").filter(pl.col("type") == "A").sort_by("number")
        )


def test_non_existent_expr_inputs_in_lazy() -> None:
    with pytest.raises(ColumnNotFoundError):
        pl.LazyFrame().filter(pl.col("x") == 1).explain()  # tests: 12074

    lf = pl.LazyFrame({"foo": [1, 1, -2, 3]})

    with pytest.raises(ColumnNotFoundError):
        (
            lf.select(pl.col("foo").cum_sum().alias("bar"))
            .filter(pl.col("bar") == pl.col("foo"))
            .explain()
        )


def test_error_list_to_array() -> None:
    with pytest.raises(ComputeError, match="not all elements have the specified width"):
        pl.DataFrame(
            data={"a": [[1, 2], [3, 4, 5]]}, schema={"a": pl.List(pl.Int8)}
        ).with_columns(array=pl.col("a").list.to_array(2))


def test_raise_not_found_in_simplify_14974() -> None:
    df = pl.DataFrame()
    with pytest.raises(ColumnNotFoundError):
        df.select(1 / (1 + pl.col("a")))


def test_invalid_product_type() -> None:
    with pytest.raises(
        InvalidOperationError,
        match="`product` operation not supported for dtype",
    ):
        pl.Series([[1, 2, 3]]).product()


def test_fill_null_invalid_supertype() -> None:
    df = pl.DataFrame({"date": [date(2022, 1, 1), None]})
    with pytest.raises(InvalidOperationError, match="got invalid or ambiguous"):
        df.select(pl.col("date").fill_null(1.0))


def test_raise_array_of_cats() -> None:
    with pytest.raises(InvalidOperationError, match="is not yet supported"):
        pl.Series([["a", "b"], ["a", "c"]], dtype=pl.Array(pl.Categorical, 2))


def test_raise_invalid_arithmetic() -> None:
    df = pl.Series("a", [object()]).to_frame()

    with pytest.raises(InvalidOperationError):
        df.select(pl.col("a") - pl.col("a"))


def test_raise_on_sorted_multi_args() -> None:
    with pytest.raises(TypeError):
        pl.DataFrame({"a": [1], "b": [1]}).set_sorted(
            ["a", "b"]  # type: ignore[arg-type]
        )


def test_err_invalid_comparison() -> None:
    with pytest.raises(
        SchemaError,
        match="could not evaluate comparison between series 'a' of dtype: date and series 'b' of dtype: bool",
    ):
        _ = pl.Series("a", [date(2020, 1, 1)]) == pl.Series("b", [True])

    with pytest.raises(
        InvalidOperationError,
        match="could not apply comparison on series of dtype 'object; operand names: 'a', 'b'",
    ):
        _ = pl.Series("a", [object()]) == pl.Series("b", [object])


def test_no_panic_pandas_nat() -> None:
    # we don't want to support pd.nat, but don't want to panic.
    with pytest.raises(Exception):  # noqa: B017
        pl.DataFrame({"x": [pd.NaT]})


def test_list_to_struct_invalid_type() -> None:
    with pytest.raises(pl.exceptions.SchemaError):
        pl.DataFrame({"a": 1}).select(pl.col("a").list.to_struct())
