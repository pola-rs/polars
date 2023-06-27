from __future__ import annotations

import io
import typing
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

import polars as pl
from polars.datatypes.convert import dtype_to_py_type


def test_error_on_empty_groupby() -> None:
    with pytest.raises(
        pl.ComputeError, match="at least one key is required in a groupby operation"
    ):
        pl.DataFrame({"x": [0, 0, 1, 1]}).groupby([]).agg(pl.count())


def test_error_on_reducing_map() -> None:
    df = pl.DataFrame(
        {"id": [0, 0, 0, 1, 1, 1], "t": [2, 4, 5, 10, 11, 14], "y": [0, 1, 1, 2, 3, 4]}
    )
    with pytest.raises(
        pl.InvalidOperationError,
        match=(
            r"output length of `map` \(6\) must be equal to "
            r"the input length \(1\); consider using `apply` instead"
        ),
    ):
        df.groupby("id").agg(pl.map(["t", "y"], np.trapz))

    df = pl.DataFrame({"x": [1, 2, 3, 4], "group": [1, 2, 1, 2]})
    with pytest.raises(
        pl.InvalidOperationError,
        match=(
            r"output length of `map` \(4\) must be equal to "
            r"the input length \(1\); consider using `apply` instead"
        ),
    ):
        df.select(
            pl.col("x")
            .map(lambda x: x.cut(bins=[1, 2, 3], maintain_order=True))
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
    with pytest.raises(pl.ComputeError):
        df1.join_asof(df2, on="b", by=["a", "c"])


def test_error_on_invalid_series_init() -> None:
    for dtype in pl.TEMPORAL_DTYPES:
        py_type = dtype_to_py_type(dtype)
        with pytest.raises(
            TypeError,
            match=f"'float' object cannot be interpreted as a {py_type.__name__}",
        ):
            pl.Series([1.5, 2.0, 3.75], dtype=dtype)

    with pytest.raises(
        TypeError, match="'float' object cannot be interpreted as an integer"
    ):
        pl.Series([1.5, 2.0, 3.75], dtype=pl.Int32)


def test_error_on_invalid_struct_field() -> None:
    with pytest.raises(pl.StructFieldNotFoundError):
        pl.struct(
            [pl.Series("a", [1, 2]), pl.Series("b", ["a", "b"])], eager=True
        ).struct.field("z")


def test_not_found_error() -> None:
    csv = "a,b,c\n2,1,1"
    df = pl.read_csv(io.StringIO(csv))
    with pytest.raises(pl.ColumnNotFoundError):
        df.select("d")


def test_string_numeric_comp_err() -> None:
    with pytest.raises(pl.ComputeError, match="cannot compare utf-8 with numeric data"):
        pl.DataFrame({"a": [1.1, 21, 31, 21, 51, 61, 71, 81]}).select(pl.col("a") < "9")


def test_panic_error() -> None:
    with pytest.raises(
        pl.PolarsPanicError,
        match="""dimensions cannot be empty""",
    ):
        pl.Series("a", [1, 2, 3]).reshape(())


@typing.no_type_check
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
        match="Expected 'other' .* to be a LazyFrame.* not a DataFrame",
    ):
        df_left.lazy().join(df_right, on="Id")

    with pytest.raises(
        TypeError,
        match="Expected 'other' .* to be a LazyFrame.* not a DataFrame",
    ):
        df_left.lazy().join_asof(df_right, on="Id")


def test_projection_update_schema_missing_column() -> None:
    with pytest.raises(
        pl.ComputeError, match="column 'colC' not available in schema Schema:*"
    ):
        (
            pl.DataFrame({"colA": ["a", "b", "c"], "colB": [1, 2, 3]})
            .lazy()
            .filter(~pl.col("colC").is_null())
            .groupby(["colA"])
            .agg([pl.col("colB").sum().alias("result")])
            .collect()
        )


def test_not_found_on_rename() -> None:
    df = pl.DataFrame({"exists": [1, 2, 3]})

    err_type = (pl.SchemaFieldNotFoundError, pl.ColumnNotFoundError)
    with pytest.raises(err_type):
        df.rename({"does_not_exist": "exists"})

    with pytest.raises(err_type):
        df.select(pl.col("does_not_exist").alias("new_name"))


@typing.no_type_check
def test_getitem_errs() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match=r"Cannot __getitem__ on DataFrame with item: "
        r"'{'some'}' of type: '<class 'set'>'.",
    ):
        df[{"some"}]

    with pytest.raises(
        ValueError,
        match=r"Cannot __getitem__ on Series of dtype: "
        r"'Int64' with argument: "
        r"'{'strange'}' of type: '<class 'set'>'.",
    ):
        df["a"][{"strange"}]

    with pytest.raises(
        ValueError,
        match=r"Cannot __setitem__ on "
        r"DataFrame with key: '{'some'}' of "
        r"type: '<class 'set'>' and value: "
        r"'foo' of type: '<class 'str'>'",
    ):
        df[{"some"}] = "foo"


def test_err_bubbling_up_to_lit() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1)], "value": [42]})

    with pytest.raises(ValueError):
        df.filter(pl.col("date") == pl.Date("2020-01-01"))


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
        with pytest.raises(pl.ComputeError, match="the column is already aggregated"):
            (
                pl.DataFrame(
                    {
                        "a": [1, 1, 1, 2, 2],
                        "b": [1, 2, 3, 4, 5],
                    }
                )
                .groupby("a")
                .agg([getattr(pl.col("b").min(), e)()])
            )


def test_unique_on_list_df() -> None:
    with pytest.raises(pl.InvalidOperationError):
        pl.DataFrame({"a": [1, 2, 3, 4], "b": [[1, 1], [2], [3], [4, 4]]}).unique()


def test_filter_not_of_type_bool() -> None:
    df = pl.DataFrame({"json_val": ['{"a":"hello"}', None, '{"a":"world"}']})
    with pytest.raises(
        pl.ComputeError, match="filter predicate must be of type `Boolean`, got"
    ):
        df.filter(pl.col("json_val").str.json_path_match("$.a"))


def test_err_asof_join_null_values() -> None:
    n = 5
    start_time = datetime(2021, 9, 30)

    df_coor = pl.DataFrame(
        {
            "vessel_id": [1] * n + [2] * n,
            "timestamp": [start_time + timedelta(hours=h) for h in range(n)]
            + [start_time + timedelta(hours=h) for h in range(n)],
        }
    )

    df_voyages = pl.DataFrame(
        {
            "vessel_id": [1, None],
            "voyage_id": [1, None],
            "voyage_start": [datetime(2022, 1, 1), None],
            "voyage_end": [datetime(2022, 1, 20), None],
        }
    )
    with pytest.raises(
        pl.ComputeError, match=".sof join must not have null values in 'on' argument"
    ):
        (
            df_coor.sort("timestamp").join_asof(
                df_voyages.sort("voyage_start"),
                right_on="voyage_start",
                left_on="timestamp",
                by="vessel_id",
                strategy="backward",
            )
        )


def test_is_nan_on_non_boolean() -> None:
    with pytest.raises(pl.InvalidOperationError):
        pl.Series(["1", "2", "3"]).fill_nan("2")  # type: ignore[arg-type]


def test_window_expression_different_group_length() -> None:
    try:
        pl.DataFrame({"groups": ["a", "a", "b", "a", "b"]}).select(
            [pl.col("groups").apply(lambda _: pl.Series([1, 2])).over("groups")]
        )
    except pl.ComputeError as exc:
        msg = str(exc)
        assert (
            "the length of the window expression did not match that of the group" in msg
        )
        assert "group:" in msg
        assert "group length:" in msg
        assert "output: 'shape:" in msg


@typing.no_type_check
def test_lazy_concat_err() -> None:
    df1 = pl.DataFrame(
        {
            "foo": [1, 2],
            "bar": [6, 7],
            "ham": ["a", "b"],
        }
    )
    df2 = pl.DataFrame(
        {
            "foo": [3, 4],
            "ham": ["c", "d"],
            "bar": [8, 9],
        }
    )
    with pytest.raises(
        ValueError,
        match="'LazyFrame' only allows {'vertical','vertical_relaxed','diagonal','align'} concat strategies.",
    ):
        pl.concat([df1.lazy(), df2.lazy()], how="horizontal").collect()


@typing.no_type_check
def test_series_concat_err() -> None:
    s = pl.Series([1, 2, 3])
    for how in ("horizontal", "diagonal"):
        with pytest.raises(
            ValueError,
            match="'Series' only allows {'vertical'} concat strategy.",
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
        pl.ComputeError,
        match=r"`sort_by` produced different length: 5 than the series that has to be sorted: 3",
    ):
        df.select(pl.col("a").filter(pl.col("b") == "M").sort_by("c", descending=True))


def test_epoch_time_type() -> None:
    with pytest.raises(
        pl.InvalidOperationError,
        match="`timestamp` operation not supported for dtype `time`",
    ):
        pl.Series([time(0, 0, 1)]).dt.epoch("s")


def test_duplicate_columns_arg_csv() -> None:
    f = io.BytesIO()
    pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).write_csv(f)
    f.seek(0)
    with pytest.raises(
        ValueError, match=r"'columns' arg should only have unique values"
    ):
        pl.read_csv(f, columns=["x", "x", "y"])


def test_datetime_time_add_err() -> None:
    with pytest.raises(pl.ComputeError):
        pl.Series([datetime(1970, 1, 1, 0, 0, 1)]) + pl.Series([time(0, 0, 2)])


@typing.no_type_check
def test_invalid_dtype() -> None:
    with pytest.raises(
        ValueError,
        match=r"Given dtype: 'mayonnaise' is not a valid Polars data type and cannot be converted into one",
    ):
        pl.Series([1, 2], dtype="mayonnaise")


def test_arr_eval_named_cols() -> None:
    df = pl.DataFrame({"A": ["a", "b"], "B": [["a", "b"], ["c", "d"]]})

    with pytest.raises(
        pl.ComputeError,
    ):
        df.select(pl.col("B").list.eval(pl.element().append(pl.col("A"))))


def test_alias_in_join_keys() -> None:
    df = pl.DataFrame({"A": ["a", "b"], "B": [["a", "b"], ["c", "d"]]})
    with pytest.raises(
        pl.ComputeError,
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
        pl.ComputeError,
        match=r"the expression in `sort_by` argument must result in the same length",
    ):
        df.groupby("group").agg(
            [
                pl.col("col1").sort_by(pl.col("col2").unique()),
            ]
        )

    with pytest.raises(
        pl.ComputeError,
        match=r"the expression in `sort_by` argument must result in the same length",
    ):
        df.groupby("group").agg(
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
        pl.ComputeError, match=r"The predicate expanded to zero expressions"
    ):
        # we filter by ints
        df.filter(pl.col(pl.Int16).min() < 0.1)


def test_date_string_comparison() -> None:
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
        pl.ComputeError, match=r"cannot compare 'date/datetime/time' to a string value"
    ):
        df.select(pl.col("date") > "2021-11-10")


def test_err_on_multiple_column_expansion() -> None:
    # this would be a great feature :)
    with pytest.raises(
        pl.ComputeError, match=r"expanding more than one `col` is not allowed"
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
        pl.ComputeError, match=r"cannot evaluate two series of different lengths"
    ):
        df.filter(pl.col("idx") == s)


def test_take_negative_index_is_oob() -> None:
    df = pl.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(pl.ComputeError, match=r"index out of bounds"):
        df["value"].take(-1)


def test_string_numeric_arithmetic_err() -> None:
    df = pl.DataFrame({"s": ["x"]})
    with pytest.raises(
        pl.ComputeError, match=r"arithmetic on string and numeric not allowed"
    ):
        df.select(pl.col("s") + 1)


def test_file_path_truncate_err() -> None:
    content = "lskdfj".join(str(i) for i in range(25))
    with pytest.raises(
        FileNotFoundError,
        match=r"\.\.\.lskdfj14lskdfj15lskdfj16lskdfj17lskdfj18lskdfj19lskdfj20lskdfj21lskdfj22lskdfj23lskdfj24",
    ):
        pl.read_csv(content)


def test_ambiguous_filter_err() -> None:
    df = pl.DataFrame({"a": [None, "2", "3"], "b": [None, None, "z"]})
    with pytest.raises(
        pl.ComputeError,
        match=r"The predicate passed to 'LazyFrame.filter' expanded to multiple expressions",
    ):
        df.filter(pl.col(["a", "b"]).is_null())


def test_with_column_duplicates() -> None:
    df = pl.DataFrame({"a": [0, None, 2, 3, None], "b": [None, 1, 2, 3, None]})
    with pytest.raises(
        pl.ComputeError,
        match=r"The name: 'same' passed to `LazyFrame.with_columns` is duplicate",
    ):
        assert df.with_columns([pl.all().alias("same")]).columns == ["a", "b", "same"]


def test_skip_nulls_err() -> None:
    df = pl.DataFrame({"foo": [None, None]})

    with pytest.raises(
        pl.ComputeError, match=r"The output type of 'apply' function cannot determined"
    ):
        df.with_columns(pl.col("foo").apply(lambda x: x, skip_nulls=True))


def test_err_on_time_datetime_cast() -> None:
    s = pl.Series([time(10, 0, 0), time(11, 30, 59)])
    with pytest.raises(pl.ComputeError, match=r"cannot cast `Time` to `Datetime`"):
        s.cast(pl.Datetime)


def test_invalid_inner_type_cast_list() -> None:
    s = pl.Series([[-1, 1]])
    with pytest.raises(
        pl.ComputeError, match=r"cannot cast List inner type: 'Int64' to Categorical"
    ):
        s.cast(pl.List(pl.Categorical))


@pytest.mark.parametrize(
    ("every", "match"),
    [
        ("-1i", r"'every' argument must be positive"),
        (
            "2h",
            r"you cannot combine time durations like '2h' with integer durations like '3i'",
        ),
    ],
)
def test_groupby_dynamic_validation(every: str, match: str) -> None:
    df = pl.DataFrame(
        {
            "index": [0, 0, 1, 1],
            "group": ["banana", "pear", "banana", "pear"],
            "weight": [2, 3, 5, 7],
        }
    )

    with pytest.raises(pl.ComputeError, match=match):
        df.groupby_dynamic("index", by="group", every=every, period="2i").agg(
            pl.col("weight")
        )


def test_lit_agg_err() -> None:
    with pytest.raises(pl.ComputeError, match=r"cannot aggregate a literal"):
        pl.DataFrame({"y": [1]}).with_columns(pl.lit(1).sum().over("y"))


def test_window_size_validation() -> None:
    df = pl.DataFrame({"x": [1.0]})

    with pytest.raises(ValueError, match=r"'window_size' should be positive"):
        df.with_columns(trailing_min=pl.col("x").rolling_min(window_size=-3))


@typing.no_type_check
def test_invalid_getitem_key_err() -> None:
    df = pl.DataFrame({"x": [1.0], "y": [1.0]})

    with pytest.raises(KeyError, match=r"('x', 'y')"):
        df["x", "y"]


def test_invalid_groupby_arg() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(
        ValueError, match="specifying aggregations as a dictionary is not supported"
    ):
        df.groupby(1).agg({"a": "sum"})


def test_no_sorted_err() -> None:
    df = pl.DataFrame(
        {
            "dt": [datetime(2001, 1, 1), datetime(2001, 1, 2)],
        }
    )
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"argument in operation 'groupby_dynamic' is not explicitly sorted",
    ):
        df.groupby_dynamic("dt", every="1h").agg(pl.all().count().suffix("_foo"))


def test_serde_validation() -> None:
    f = io.StringIO(
        """
    {
      "columns": [
        {
          "name": "a",
          "datatype": "Int64",
          "values": [
            1,
            2
          ]
        },
        {
          "name": "b",
          "datatype": "Int64",
          "values": [
            1
          ]
        }
      ]
    }
    """
    )
    with pytest.raises(
        pl.ComputeError,
        match=r"lengths don't match",
    ):
        pl.read_json(f)


def test_transpose_categorical_cached() -> None:
    with pytest.raises(
        pl.ComputeError,
        match=r"'transpose' of categorical can only be done if all are from the same global string cache",
    ):
        pl.DataFrame(
            {"b": pl.Series(["a", "b", "c"], dtype=pl.Categorical)}
        ).transpose()


def test_overflow_msg() -> None:
    with pytest.raises(
        pl.ComputeError,
        match=r"could not append value: 2147483648 of type: i64 to the builder",
    ):
        pl.DataFrame([[2**31]], [("a", pl.Int32)], orient="row")


def test_sort_by_err_9259() -> None:
    df = pl.DataFrame(
        {"a": [1, 1, 1], "b": [3, 2, 1], "c": [1, 1, 2]},
        schema={"a": pl.Float32, "b": pl.Float32, "c": pl.Float32},
    )
    with pytest.raises(pl.ComputeError):
        df.lazy().groupby("c").agg(
            [pl.col("a").sort_by(pl.col("b").filter(pl.col("b") > 100)).sum()]
        ).collect()
