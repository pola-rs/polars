from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, ShapeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_map_groups() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    result = df.group_by("a").map_groups(lambda df: df[["c"]].sum())

    expected = pl.DataFrame({"c": [10, 10, 1]})
    assert_frame_equal(result, expected, check_row_order=False)


def test_map_groups_lazy() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 3.0]})

    schema = {"a": pl.Float64, "b": pl.Float64}
    result = lf.group_by("a").map_groups(lambda df: df * 2.0, schema=schema)

    expected = pl.LazyFrame({"a": [6.0, 2.0, 2.0], "b": [6.0, 2.0, 4.0]})
    assert_frame_equal(result, expected, check_row_order=False)
    assert result.collect_schema() == expected.collect_schema()


def test_map_groups_rolling() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
        }
    ).set_sorted("a")

    def function(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col("a").min(),
            pl.col("b").max(),
        )

    result = df.rolling("a", period="2i").map_groups(function, schema=df.schema)

    expected = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 2, 3, 4], dtype=pl.Int64),
            pl.Series("b", [1, 2, 3, 4, 5], dtype=pl.Int64),
        ]
    )
    assert_frame_equal(result, expected)


def test_map_groups_empty() -> None:
    df = pl.DataFrame(schema={"x": pl.Int64})
    with pytest.raises(
        ComputeError, match=r"cannot group_by \+ apply on empty 'DataFrame'"
    ):
        df.group_by("x").map_groups(lambda x: x)


def test_map_groups_none() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2, 5],
            "a": [2, 4, 5, 190, 1, 4, 1],
            "b": [1, 3, 2, 1, 43, 3, 1],
        }
    )

    out = (
        df.group_by("g", maintain_order=True).agg(
            pl.map_groups(
                exprs=["a", pl.col("b") ** 4, pl.col("a") / 4],
                function=lambda x: x[0] * x[1] + x[2].sum(),
                return_dtype=pl.Float64,
                returns_scalar=False,
            ).alias("multiple")
        )
    )["multiple"]
    assert out[0].to_list() == [4.75, 326.75, 82.75]
    assert out[1].to_list() == [238.75, 3418849.75, 372.75]

    out_df = df.select(pl.map_batches(exprs=["a", "b"], function=lambda s: s[0] * s[1]))
    assert out_df["a"].to_list() == (df["a"] * df["b"]).to_list()

    # check if we can return None
    def func(s: Sequence[pl.Series]) -> pl.Series | None:
        if s[0][0] == 190:
            return None
        else:
            return s[0].implode()

    out = (
        df.group_by("g", maintain_order=True).agg(
            pl.map_groups(
                exprs=["a", pl.col("b") ** 4, pl.col("a") / 4],
                function=func,
                return_dtype=pl.self_dtype().wrap_in_list(),
                returns_scalar=True,
            ).alias("multiple")
        )
    )["multiple"]
    assert out[1] is None


def test_map_groups_object_output() -> None:
    df = pl.DataFrame(
        {
            "names": ["foo", "ham", "spam", "cheese", "egg", "foo"],
            "dates": ["1", "1", "2", "3", "3", "4"],
            "groups": ["A", "A", "B", "B", "B", "C"],
        }
    )

    class Foo:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

    result = df.group_by("groups").agg(
        pl.map_groups(
            [pl.col("dates"), pl.col("names")],
            lambda s: Foo(dict(zip(s[0], s[1]))),
            return_dtype=pl.Object,
            returns_scalar=True,
        )
    )

    assert result.dtypes == [pl.String, pl.Object]


def test_map_groups_numpy_output_3057() -> None:
    df = pl.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1],
            "t": [2.0, 4.3, 5, 10, 11, 14],
            "y": [0.0, 1, 1.3, 2, 3, 4],
        }
    )

    result = df.group_by("id", maintain_order=True).agg(
        pl.map_groups(
            ["y", "t"],
            lambda lst: np.mean([lst[0], lst[1]]),
            returns_scalar=True,
            return_dtype=pl.self_dtype(),
        ).alias("result")
    )

    expected = pl.DataFrame({"id": [0, 1], "result": [2.266666, 7.333333]})
    assert_frame_equal(result, expected)


def test_map_groups_return_all_null_15260() -> None:
    def foo(x: Sequence[pl.Series]) -> pl.Series:
        return pl.Series([x[0][0]], dtype=x[0].dtype)

    assert_frame_equal(
        pl.DataFrame({"key": [0, 0, 1], "a": [None, None, None]})
        .group_by("key")
        .agg(
            pl.map_groups(
                exprs=["a"],
                function=foo,
                returns_scalar=True,
                return_dtype=pl.self_dtype(),
            )
        )
        .sort("key"),
        pl.DataFrame({"key": [0, 1], "a": [None, None]}),
    )


@pytest.mark.parametrize(
    ("func", "result"),
    [
        (lambda n: n[0] + n[1], [[85], [85]]),
        (lambda _: pl.Series([1, 2, 3]), [[1, 2, 3], [1, 2, 3]]),
    ],
)
@pytest.mark.parametrize("maintain_order", [True, False])
def test_map_groups_multiple_all_literal(
    func: Any, result: list[int], maintain_order: bool
) -> None:
    df = pl.DataFrame({"g": [10, 10, 20], "a": [1, 2, 3], "b": [2, 3, 4]})

    q = (
        df.lazy()
        .group_by(pl.col("g"), maintain_order=maintain_order)
        .agg(
            pl.map_groups(
                exprs=[pl.lit(42).cast(pl.Int64), pl.lit(43).cast(pl.Int64)],
                function=func,
                return_dtype=pl.Int64,
            ).alias("out")
        )
    )
    out = q.collect()
    expected = pl.DataFrame({"g": [10, 20], "out": result})
    assert_frame_equal(out, expected, check_row_order=maintain_order)


@pytest.mark.may_fail_auto_streaming  # reason: alternate error message
def test_map_groups_multiple_all_literal_elementwise_raises() -> None:
    df = pl.DataFrame({"g": [10, 10, 20], "a": [1, 2, 3], "b": [2, 3, 4]})
    q = (
        df.lazy()
        .group_by(pl.col("g"))
        .agg(
            pl.map_groups(
                exprs=[pl.lit(42), pl.lit(43)],
                function=lambda _: pl.Series([1, 2, 3]),
                return_dtype=pl.Int64,
                is_elementwise=True,
            ).alias("out")
        )
    )
    msg = "elementwise expression dyn int: 42.python_udf([dyn int: 43]) must return exactly 1 value on literals, got 3"
    with pytest.raises(ComputeError, match=re.escape(msg)):
        q.collect(engine="in-memory")

    # different error message in streaming, not specific to the problem
    with pytest.raises(ShapeError):
        q.collect(engine="streaming")


def test_nested_query_with_streaming_dispatch_25172() -> None:
    def simple(_: Any) -> pl.Series:
        import io

        pl.LazyFrame({}).sink_parquet(
            pl.PartitionMaxSize("", file_path=lambda _: io.BytesIO(), max_size=1),
            engine="in-memory",
        )
        return pl.Series([1])

    assert_frame_equal(
        pl.LazyFrame({"a": ["A", "B"] * 1000, "b": [1] * 2000})
        .group_by("a")
        .agg(pl.map_groups(["b"], simple, pl.Int64(), returns_scalar=True))
        .collect(engine="in-memory")
        .sort("a"),
        pl.DataFrame({"a": ["A", "B"], "b": [1, 1]}, schema_overrides={"b": pl.Int64}),
    )


@pytest.mark.skipif(
    os.getenv("POLARS_AUTO_NEW_STREAMING") == "1",
    reason="List(Object) not yet supported in new streaming engine",
)
def test_map_groups_object_dtype_display_22814() -> None:
    class CustomObject:
        pass

    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "objects": [CustomObject(), CustomObject(), CustomObject(), CustomObject()],
        }
    )

    result = df.group_by("id").agg(
        pl.map_groups(
            [pl.col("objects")], lambda x: x[0], return_dtype=pl.Object
        ).alias("first_object")
    )

    result_str = str(result)
    assert "first_object" in result_str
    assert result.shape == (4, 2)


@pytest.mark.skipif(
    os.getenv("POLARS_AUTO_NEW_STREAMING") == "1",
    reason="List(Object) not yet supported in new streaming engine",
)
def test_list_object_multiple_objects() -> None:
    class Marker:
        def __init__(self, value: int) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f"M{self.value}"

    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "objects": [Marker(i) for i in range(5)],
        }
    )

    result = df.group_by("id", maintain_order=True).agg(
        pl.map_groups(
            [pl.col("objects")], lambda x: x[0], return_dtype=pl.Object
        ).alias("first_obj"),
        pl.map_groups(
            [pl.col("objects")], lambda x: x[-1], return_dtype=pl.Object
        ).alias("last_obj"),
    )

    result_str = str(result)
    assert "first_obj" in result_str
    assert "last_obj" in result_str

    # Verify we can access the lists
    assert len(result) == 2
    assert isinstance(result["first_obj"][0][0], Marker)
    assert isinstance(result["last_obj"][0][0], Marker)


@pytest.mark.skipif(
    os.getenv("POLARS_AUTO_NEW_STREAMING") == "1",
    reason="List(Object) not yet supported in new streaming engine",
)
def test_list_object_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "objects": [object(), None, object()],
        }
    )

    result = df.group_by("id").agg(
        pl.map_groups(
            [pl.col("objects")], lambda x: x[0], return_dtype=pl.Object
        ).alias("obj")
    )

    result_str = str(result)
    assert "obj" in result_str


@pytest.mark.skipif(
    os.getenv("POLARS_AUTO_NEW_STREAMING") == "1",
    reason="List(Object) not yet supported in new streaming engine",
)
def test_list_object_iteration() -> None:
    class Value:
        def __init__(self, x: int) -> None:
            self.x = x

        def __repr__(self) -> str:
            return f"V{self.x}"

    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "val": [Value(i) for i in range(5)],
        }
    )

    result = df.group_by("id").agg(
        pl.map_groups([pl.col("val")], lambda x: x[0], return_dtype=pl.Object).alias(
            "values"
        )
    )

    # Test that iteration doesn't panic
    count = 0
    for _ in result["values"]:
        count += 1
    assert count == 5

    # Test formatting the entire column doesn't panic
    result_str = str(result["values"])
    assert "values" in result_str or "V" in result_str


@pytest.mark.skipif(
    os.getenv("POLARS_AUTO_NEW_STREAMING") == "1",
    reason="List(Object) not yet supported in new streaming engine",
)
def test_list_object_mixed_types() -> None:
    from pathlib import Path

    class Wrapper:
        def __init__(self, value: Any) -> None:
            self.value = value

    df = pl.DataFrame(
        {
            "group": [1, 2, 3],
            "mixed": [Wrapper({"a": 1}), Path("test"), object()],
        }
    )

    result = df.group_by("group").agg(
        pl.map_groups([pl.col("mixed")], lambda x: x[0], return_dtype=pl.Object).alias(
            "objects"
        )
    )

    result_str = str(result)
    assert "objects" in result_str

    obj_values = [item[0] for item in result["objects"]]
    assert any(isinstance(obj, Wrapper) for obj in obj_values)
    assert any(isinstance(obj, Path) for obj in obj_values)


@pytest.mark.skipif(
    os.getenv("POLARS_AUTO_NEW_STREAMING") == "1",
    reason="List(Object) not yet supported in new streaming engine",
)
def test_list_object_formatting_many_items() -> None:
    class Num:
        def __init__(self, n: int) -> None:
            self.n = n

        def __repr__(self) -> str:
            return f"N{self.n}"

    # Multiple rows to test iteration and formatting
    df = pl.DataFrame({"id": list(range(10)), "obj": [Num(i) for i in range(10)]})

    result = df.group_by("id").agg(
        pl.map_groups([pl.col("obj")], lambda x: x[0], return_dtype=pl.Object).alias(
            "nums"
        )
    )

    result_str = str(result)
    assert "nums" in result_str
    assert len(result) == 10

    for i in range(len(result)):
        obj_list = result["nums"][i]
        assert len(obj_list) == 1
        assert isinstance(obj_list[0], Num)
