from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

import polars as pl
from polars.exceptions import ArgumentRemovedError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import EngineType


@pytest.fixture(scope="module")
def str_mapping() -> dict[str | None, str]:
    return {
        "CA": "Canada",
        "DE": "Germany",
        "FR": "France",
        None: "Not specified",
    }


def test_replace_str_to_str(str_mapping: dict[str | None, str]) -> None:
    df = pl.DataFrame({"country_code": ["FR", None, "ES", "DE"]})
    result = df.select(replaced=pl.col("country_code").replace(str_mapping))
    expected = pl.DataFrame({"replaced": ["France", "Not specified", "ES", "Germany"]})
    assert_frame_equal(result, expected)


def test_replace_enum() -> None:
    dtype = pl.Enum(["a", "b", "c", "d"])
    s = pl.Series(["a", "b", "c"], dtype=dtype)
    old = ["a", "b"]
    new = pl.Series(["c", "d"], dtype=dtype)

    result = s.replace(old, new)

    expected = pl.Series(["c", "d", "c"], dtype=dtype)
    assert_series_equal(result, expected)


def test_replace_enum_to_str() -> None:
    dtype = pl.Enum(["a", "b", "c", "d"])
    s = pl.Series(["a", "b", "c"], dtype=dtype)

    result = s.replace({"a": "c", "b": "d"})

    expected = pl.Series(["c", "d", "c"], dtype=dtype)
    assert_series_equal(result, expected)


def test_replace_cat_to_cat(str_mapping: dict[str | None, str]) -> None:
    lf = pl.LazyFrame(
        {"country_code": ["FR", None, "ES", "DE"]},
        schema={"country_code": pl.Categorical},
    )
    old = pl.Series(["CA", "DE", "FR", None], dtype=pl.Categorical)
    new = pl.Series(
        ["Canada", "Germany", "France", "Not specified"], dtype=pl.Categorical
    )

    result = lf.select(replaced=pl.col("country_code").replace(old, new))

    expected = pl.LazyFrame(
        {"replaced": ["France", "Not specified", "ES", "Germany"]},
        schema_overrides={"replaced": pl.Categorical},
    )
    assert_frame_equal(result, expected)


CATEGORIES = [
    pytest.param(pl.Categories.random, id="own"),
    pytest.param(lambda: pl.Categories(f"named-{uuid4()}"), id="named"),
    pytest.param(pl.Categories, id="global"),
    pytest.param(lambda: pl.Categories.random(physical=pl.UInt8), id="uint8"),
]


@pytest.mark.parametrize("make_categories", CATEGORIES)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_replace_categorical_absent_old_adds_no_categories(
    make_categories: Callable[[], pl.Categories], engine: EngineType
) -> None:
    cats = make_categories()
    dtype = pl.Categorical(cats)
    lf = pl.LazyFrame({"s": ["a", "b"]}, schema={"s": dtype})
    # A `new` value paired with an absent `old` value can't be output either.
    absent, unused = f"absent-{uuid4()}", f"unused-{uuid4()}"
    for expr, expected in [
        (pl.col("s").replace(absent, "a"), ["a", "b"]),
        (pl.col("s").replace(absent, unused), ["a", "b"]),
        (pl.col("s").replace([absent, "b"], "a"), ["a", "a"]),
        (pl.col("s").replace([absent, "a"], [unused, "b"]), ["b", "b"]),
        (pl.col("s").replace({absent: unused, "b": "a"}), ["a", "a"]),
        (pl.col("s").replace({absent: unused, f"{absent}-2": "a"}), ["a", "b"]),
    ]:
        result = lf.select(expr).collect(engine=engine)
        assert_frame_equal(result, pl.DataFrame({"s": expected}, schema={"s": dtype}))
    s = lf.collect()["s"]
    assert_series_equal(s.replace(absent, unused), s)
    with pytest.raises(InvalidOperationError, match="must have the same length"):
        lf.select(pl.col("s").replace([absent, "a"], ["x", "y", "z"])).collect(
            engine=engine
        )
    assert absent not in cats
    assert unused not in cats


def test_replace_categorical_absent_old_does_not_fill_categories() -> None:
    cats = pl.Categories.random(physical=pl.UInt8)
    s = pl.Series([f"c{i}" for i in range(250)], dtype=pl.Categorical(cats))
    # Each call used to add both values, so the 3rd call overflowed the 255 allowed.
    for i in range(10):
        assert_series_equal(s.replace(f"absent-{i}", f"new-{i}"), s)
    assert cats.to_series().len() == 250


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    ("values", "mapping", "expected"),
    [
        (["a"] * 8, {"a": "b", "b": "c"}, "b"),
        (["a"] * 8, {None: "b", "a": "c"}, "c"),
        ([None] * 8, {None: "b", "b": "c"}, "b"),
    ],
)
def test_replace_categorical_only_adds_values_for_present_keys(
    values: list[str | None],
    mapping: dict[str | None, str],
    expected: str,
    engine: EngineType,
) -> None:
    cats = pl.Categories.random(physical=pl.UInt8)
    dtype = pl.Categorical(cats)
    seed = pl.Series(["a", *(f"c{i}" for i in range(253))], dtype=dtype)
    lf = pl.LazyFrame({"s": values}, schema={"s": dtype})

    with pl.Config(streaming_chunk_size=1):
        result = lf.select(pl.col("s").replace(mapping)).collect(engine=engine)

    assert_frame_equal(result, pl.DataFrame({"s": [expected] * 8}, schema={"s": dtype}))
    assert cats.to_series().len() == seed.len() + 1


@pytest.mark.parametrize("make_categories", CATEGORIES)
def test_replace_categorical_adds_new_values(
    make_categories: Callable[[], pl.Categories],
) -> None:
    cats = make_categories()
    s = pl.Series(["a", "b"], dtype=pl.Categorical(cats))
    new = f"new-{uuid4()}"
    assert s.replace("a", new).to_list() == [new, "b"]
    assert new in cats


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_replace_enum_absent_old_raises(engine: EngineType) -> None:
    lf = pl.LazyFrame({"s": ["a", "b"]}, schema={"s": pl.Enum(["a", "b", "c"])})
    for expr in [
        pl.col("s").replace("absent", "a"),
        pl.col("s").replace(["absent", "b"], "a"),
        pl.col("s").replace({"absent": "a", "b": "a"}),
    ]:
        with pytest.raises(
            InvalidOperationError, match="conversion from `str` to `enum` failed"
        ):
            lf.select(expr).collect(engine=engine)


def test_replace_invalid_old_dtype() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    mapping = {"a": 10, "b": 20}
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `i64` failed"
    ):
        lf.select(pl.col("a").replace(mapping)).collect()


def test_replace_int_to_int() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: 5, 3: 7}
    result = df.select(replaced=pl.col("int").replace(mapping))
    expected = pl.DataFrame(
        {"replaced": [None, 5, None, 7]}, schema={"replaced": pl.Int16}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_int_keep_dtype() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    old = [1, 3]
    new = pl.Series([5, 7], dtype=pl.Int16)

    result = df.select(replaced=pl.col("int").replace(old, new))
    expected = pl.DataFrame(
        {"replaced": [None, 5, None, 7]}, schema={"replaced": pl.Int16}
    )
    assert_frame_equal(result, expected)


def test_replace_int_to_str() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: "b", 3: "d"}
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `i16` failed"
    ):
        df.select(replaced=pl.col("int").replace(mapping))


def test_replace_int_to_str_with_null() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping = {1: "b", 3: "d", None: "e"}
    with pytest.raises(
        InvalidOperationError, match="conversion from `str` to `i16` failed"
    ):
        df.select(replaced=pl.col("int").replace(mapping))


def test_replace_empty_mapping() -> None:
    df = pl.DataFrame({"int": [None, 1, None, 3]}, schema={"int": pl.Int16})
    mapping: dict[Any, Any] = {}
    result = df.select(pl.col("int").replace(mapping))
    assert_frame_equal(result, df)


def test_replace_mapping_different_dtype_str_int() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1: "b", 3: "d"}

    result = df.select(pl.col("int").replace(mapping))
    expected = pl.DataFrame({"int": [None, "b", None, "d"]})
    assert_frame_equal(result, expected)


def test_replace_mapping_different_dtype_map_none() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1: "b", 3: "d", None: "e"}
    result = df.select(pl.col("int").replace(mapping))
    expected = pl.DataFrame({"int": ["e", "b", "e", "d"]})
    assert_frame_equal(result, expected)


def test_replace_mapping_different_dtype_str_float() -> None:
    df = pl.DataFrame({"int": [None, "1", None, "3"]})
    mapping = {1.0: "b", 3.0: "d"}

    result = df.select(pl.col("int").replace(mapping))
    assert_frame_equal(result, df)


# https://github.com/pola-rs/polars/issues/7132
def test_replace_str_to_str_replace_all() -> None:
    df = pl.DataFrame({"text": ["abc"]})
    mapping = {"abc": "123"}
    result = df.select(pl.col("text").replace(mapping).str.replace_all("1", "-"))
    expected = pl.DataFrame({"text": ["-23"]})
    assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def int_mapping() -> dict[int, int]:
    return {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}


def test_replace_int_to_int1(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5])
    result = s.replace(int_mapping)
    expected = pl.Series([-1, 22, None, 44, -5])
    assert_series_equal(result, expected)


def test_replace_int_to_int4(int_mapping: dict[int, int]) -> None:
    s = pl.Series([-1, 22, None, 44, -5])
    result = s.replace(int_mapping)
    expected = pl.Series([-1, 22, None, 44, -5])
    assert_series_equal(result, expected)


# https://github.com/pola-rs/polars/issues/12728
def test_replace_str_to_int2() -> None:
    s = pl.Series(["a", "b"])
    mapping = {"a": 1, "b": 2}
    result = s.replace(mapping)
    expected = pl.Series(["1", "2"])
    assert_series_equal(result, expected)


def test_replace_str_to_bool_without_default() -> None:
    s = pl.Series(["True", "False", "False", None])
    mapping = {"True": True, "False": False}
    result = s.replace(mapping)
    expected = pl.Series(["true", "false", "false", None])
    assert_series_equal(result, expected)


def test_replace_old_new() -> None:
    s = pl.Series([1, 2, 2, 3])
    result = s.replace(2, 9)
    expected = s = pl.Series([1, 9, 9, 3])
    assert_series_equal(result, expected)


def test_replace_old_new_many_to_one() -> None:
    s = pl.Series([1, 2, 2, 3])
    result = s.replace([2, 3], 9)
    expected = s = pl.Series([1, 9, 9, 9])
    assert_series_equal(result, expected)


def test_replace_old_new_mismatched_lengths() -> None:
    s = pl.Series([1, 2, 2, 3, 4])
    with pytest.raises(InvalidOperationError):
        s.replace([2, 3, 4], [8, 9])


def test_replace_fast_path_one_to_one() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})
    result = lf.select(pl.col("a").replace(2, 100))
    expected = pl.LazyFrame({"a": [1, 100, 100, 3]})
    assert_frame_equal(result, expected)


def test_replace_fast_path_one_null_to_one() -> None:
    # https://github.com/pola-rs/polars/issues/13391
    lf = pl.LazyFrame({"a": [1, None]})
    result = lf.select(pl.col("a").replace(None, 100))
    expected = pl.LazyFrame({"a": [1, 100]})
    assert_frame_equal(result, expected)


def test_replace_fast_path_many_with_null_to_one() -> None:
    lf = pl.LazyFrame({"a": [1, 2, None]})
    result = lf.select(pl.col("a").replace([1, None], 100))
    expected = pl.LazyFrame({"a": [100, 2, 100]})
    assert_frame_equal(result, expected)


def test_replace_fast_path_many_to_one() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 3]})
    result = lf.select(pl.col("a").replace([2, 3], 100))
    expected = pl.LazyFrame({"a": [1, 100, 100, 100]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ([2, 2], 100),
        ([2, 2], [100, 200]),
        ([2, 2], [100, 100]),
    ],
)
def test_replace_duplicates_old(old: list[int], new: int | list[int]) -> None:
    s = pl.Series([1, 2, 3, 2, 3])
    with pytest.raises(
        InvalidOperationError,
        match="`old` input for `replace` must not contain duplicates",
    ):
        s.replace(old, new)


def test_replace_duplicates_new() -> None:
    s = pl.Series([1, 2, 3, 2, 3])
    result = s.replace([1, 2], [100, 100])
    expected = s = pl.Series([100, 100, 3, 100, 3])
    assert_series_equal(result, expected)


def test_replace_return_dtype_removed() -> None:
    def msg(qualname: str) -> str:
        return (
            re.escape(
                f"the argument 'return_dtype' for '{qualname}' was deprecated in "
            )
            + ".*"
            + re.escape("Use `replace_strict` instead to set a return data type while")
        )

    s = pl.Series([1, 2, 3])
    with pytest.raises(ArgumentRemovedError, match=msg("Series.replace")):
        s.replace(1, 10, return_dtype=pl.Int8)  # type: ignore[call-arg]

    e = pl.col("a")
    with pytest.raises(ArgumentRemovedError, match=msg("Expr.replace")):
        e.replace(1, 10, return_dtype=pl.Int8)  # type: ignore[call-arg]


def test_replace_default_removed() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.raises(
        ArgumentRemovedError,
        match=re.escape("Use `replace_strict` instead to set a default while "),
    ):
        s.replace(1, 10, default=None)  # type: ignore[call-arg]


def test_replace_single_argument_not_mapping() -> None:
    df = pl.DataFrame({"a": ["a", "b", "c"]})
    with pytest.raises(
        TypeError,
        match="`new` argument is required if `old` argument is not a Mapping type",
    ):
        df.select(pl.col("a").replace("b"))


@pytest.mark.parametrize("height", [1, 3])
@pytest.mark.parametrize(
    "values",
    [
        pl.col("l"),
        pl.col("l").head(1),
        pl.lit(pl.Series([[1]])),
        pl.lit(pl.Series([[1]])).list.drop_nulls(),
    ],
)
def test_replace_list_of_values_per_row_raises(height: int, values: pl.Expr) -> None:
    # Even a list that has one row at runtime must be a scalar in the plan.
    df = pl.DataFrame({"a": [1] * height, "l": [[1]] * height})
    with pytest.raises(ShapeError, match="'old' must be a scalar value in `replace`"):
        df.lazy().select(pl.col("a").replace(values, 9)).collect()
    with pytest.raises(
        ShapeError, match="'new' must be a scalar value in `replace_strict`"
    ):
        df.lazy().select(pl.col("a").replace_strict(1, values)).collect()


def test_replace_list_input_with_column_values() -> None:
    df = pl.DataFrame(
        {"x": [[1], [2], [1]], "y": [[1], [5], [7]], "z": [[10], [50], [70]]}
    )
    out = df.lazy().select(
        r=pl.col("x").replace(pl.col("y"), pl.col("z")),
        s=pl.col("x").replace_strict(
            pl.col("y"), pl.col("z"), default=None, return_dtype=pl.List(pl.Int64)
        ),
    )
    expected = pl.DataFrame(
        {"r": [[10], [2], [10]], "s": [[10], None, [10]]},
        schema={"r": pl.List(pl.Int64), "s": pl.List(pl.Int64)},
    )
    assert_frame_equal(out.collect(), expected)


def test_replace_strict_list_input_infers_dtype_from_new() -> None:
    lf = pl.LazyFrame({"x": [[1], [2], [1]]})
    old = pl.lit([[1], [2]])
    out = lf.select(
        a=pl.col("x").replace_strict(old, pl.lit([10, 20])),
        b=pl.col("x").replace_strict(old, pl.lit(10, dtype=pl.Int64)),
    )
    expected = pl.DataFrame({"a": [10, 20, 10], "b": [10, 10, 10]})
    assert out.collect_schema() == expected.schema
    assert_frame_equal(out.collect(), expected)
