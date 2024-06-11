from __future__ import annotations

from datetime import date, timedelta

import pytest

import polars as pl
from polars.testing import assert_frame_equal


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
        return df.select(pl.all().cast(pl.String))

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
        {
            "Amount": [10, 1, 1, 5],
            "Flour": ["1000g", "100g", "50g", "75g"],
        }
    )

    q = df.lazy().select(
        pl.col("Amount"),
        pl.col("Flour").map_elements(lambda x: 100.0) / pl.col("Amount"),
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


def test_nested_binary_literal_super_type_12227() -> None:
    # The `.alias` is important here to trigger the bug.
    assert (
        pl.select(x=1).select((pl.lit(0) + ((pl.col("x") > 0) * 0.1)).alias("x")).item()
        == 0.1
    )
    assert (
        pl.select(
            (pl.lit(0) + (pl.lit(0) == pl.lit(0)) * pl.lit(0.1)) + pl.lit(0)
        ).item()
        == 0.1
    )


def test_alias_prune_in_fold_15438() -> None:
    df = pl.DataFrame({"x": [1, 2], "expected_result": ["first", "second"]}).select(
        actual_result=pl.fold(
            acc=pl.lit("other", dtype=pl.Utf8),
            function=lambda acc, x: pl.when(x).then(pl.lit(x.name)).otherwise(acc),  # type: ignore[arg-type, return-value]
            exprs=[
                (pl.col("x") == 1).alias("first"),
                (pl.col("x") == 2).alias("second"),
            ],
        )
    )
    expected = pl.DataFrame(
        {
            "actual_result": ["first", "second"],
        }
    )
    assert_frame_equal(df, expected)


@pytest.mark.parametrize("op", ["and_", "or_"])
def test_bitwise_integral_schema(op: str) -> None:
    df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    q = df.select(getattr(pl.col("a"), op)(pl.col("b")))
    assert q.schema["a"] == df.schema["a"]
