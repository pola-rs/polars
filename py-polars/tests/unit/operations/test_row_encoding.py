from __future__ import annotations

from decimal import Decimal as D
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from hypothesis import given

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import dataframes, series
from polars.testing.parametric.strategies.dtype import dtypes
from tests.unit.conftest import FLOAT_DTYPES, INTEGER_DTYPES

if TYPE_CHECKING:
    from typing import Any

    from polars._typing import ArrayLike, EngineType, PolarsDataType

FIELD_COMBS = [
    (descending, nulls_last, False)
    for descending in [False, True]
    for nulls_last in [False, True]
] + [(None, None, True)]

FIELD_COMBS_ARGS = [
    {
        "unordered": unordered,
        "descending": descending,
        "nulls_last": nulls_last,
    }
    for descending, nulls_last, unordered in FIELD_COMBS
]


def roundtrip_re(
    df: pl.DataFrame,
    *,
    unordered: bool = False,
    descending: list[bool] | None = None,
    nulls_last: list[bool] | None = None,
) -> None:
    row_encoded = df._row_encode(
        unordered=unordered,
        descending=descending,
        nulls_last=nulls_last,
    )

    if unordered:
        return

    names = df.columns
    dtypes = df.dtypes
    result = row_encoded._row_decode(
        names, dtypes, unordered=unordered, descending=descending, nulls_last=nulls_last
    ).struct.unnest()

    assert_frame_equal(df, result)


def roundtrip_series_re(
    values: ArrayLike,
    dtype: PolarsDataType,
    *,
    unordered: bool = False,
    descending: bool | None = None,
    nulls_last: bool | None = False,
) -> None:
    descending_lst = None if descending is None else [descending]
    nulls_last_lst = None if nulls_last is None else [nulls_last]

    roundtrip_re(
        pl.Series("series", values, dtype).to_frame(),
        unordered=unordered,
        descending=descending_lst,
        nulls_last=nulls_last_lst,
    )


@given(
    df=dataframes(
        excluded_dtypes=[
            pl.Categorical,
            pl.Decimal,  # Bug: see https://github.com/pola-rs/polars/issues/20308
        ]
    )
)
@pytest.mark.parametrize(("descending", "nulls_last", "unordered"), FIELD_COMBS)
def test_row_encoding_parametric(
    df: pl.DataFrame,
    unordered: bool,
    descending: bool | None,
    nulls_last: bool | None,
) -> None:
    roundtrip_re(
        df,
        unordered=unordered,
        descending=None if descending is None else [descending] * df.width,
        nulls_last=None if nulls_last is None else [nulls_last] * df.width,
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_nulls(field: Any) -> None:
    roundtrip_series_re([], pl.Null, **field)
    roundtrip_series_re([None], pl.Null, **field)
    roundtrip_series_re([None] * 2, pl.Null, **field)
    roundtrip_series_re([None] * 13, pl.Null, **field)
    roundtrip_series_re([None] * 42, pl.Null, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_bool(field: Any) -> None:
    roundtrip_series_re([], pl.Boolean, **field)
    roundtrip_series_re([False], pl.Boolean, **field)
    roundtrip_series_re([True], pl.Boolean, **field)
    roundtrip_series_re([False, True], pl.Boolean, **field)
    roundtrip_series_re([True, False], pl.Boolean, **field)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_int(dtype: pl.DataType, field: Any) -> None:
    min = pl.select(x=dtype.min()).item()  # type: ignore[attr-defined]
    max = pl.select(x=dtype.max()).item()  # type: ignore[attr-defined]

    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([0], dtype, **field)
    roundtrip_series_re([min], dtype, **field)
    roundtrip_series_re([max], dtype, **field)

    roundtrip_series_re([1, 2, 3], dtype, **field)
    roundtrip_series_re([0, 1, 2, 3], dtype, **field)
    roundtrip_series_re([min, 0, max], dtype, **field)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_float(dtype: pl.DataType, field: Any) -> None:
    inf = float("inf")
    inf_b = float("-inf")

    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([0.0], dtype, **field)
    roundtrip_series_re([inf], dtype, **field)
    roundtrip_series_re([-inf_b], dtype, **field)

    roundtrip_series_re([1.0, 2.0, 3.0], dtype, **field)
    roundtrip_series_re([0.0, 1.0, 2.0, 3.0], dtype, **field)
    roundtrip_series_re([inf, 0, -inf_b], dtype, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_str(field: Any) -> None:
    dtype = pl.String
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([""], dtype, **field)

    roundtrip_series_re(["a", "b", "c"], dtype, **field)
    roundtrip_series_re(["", "a", "b", "c"], dtype, **field)

    roundtrip_series_re(
        ["different", "length", "strings"],
        dtype,
        **field,
    )
    roundtrip_series_re(
        ["different", "", "length", "", "strings"],
        dtype,
        **field,
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_struct(field: Any) -> None:
    dtype = pl.Struct({})
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([{}], dtype, **field)
    roundtrip_series_re([{}, {}, {}], dtype, **field)
    roundtrip_series_re([{}, None, {}], dtype, **field)

    dtype = pl.Struct({"x": pl.Int32})
    roundtrip_series_re([{"x": 1}], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([{"x": 1}] * 3, dtype, **field)
    roundtrip_series_re([{"x": 1}, {"x": None}, None], dtype, **field)

    dtype = pl.Struct({"x": pl.Int32, "y": pl.Int32})
    roundtrip_series_re(
        [{"x": 1}, {"y": 2}],
        dtype,
        **field,
    )
    roundtrip_series_re([None], dtype, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_list(field: Any) -> None:
    dtype = pl.List(pl.Int32)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([[]], dtype, **field)
    roundtrip_series_re([[1], [2]], dtype, **field)
    roundtrip_series_re([[1, 2], [3]], dtype, **field)
    roundtrip_series_re([[1, 2], [], [3]], dtype, **field)
    roundtrip_series_re([None, [1, 2], None, [], [3]], dtype, **field)

    dtype = pl.List(pl.String)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([[]], dtype, **field)
    roundtrip_series_re([[""], [""]], dtype, **field)
    roundtrip_series_re([["abc"], ["xyzw"]], dtype, **field)
    roundtrip_series_re([["x", "yx"], ["abc"]], dtype, **field)
    roundtrip_series_re([["wow", "this is"], [], ["cool"]], dtype, **field)
    roundtrip_series_re(
        [None, ["very", "very"], None, [], ["cool"]],
        dtype,
        **field,
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_nested_categorical(field: Any) -> None:
    values = [["b", None], ["a"], None, []]
    roundtrip_series_re(values, pl.List(pl.Categorical), **field)
    roundtrip_series_re(
        [["b", None], ["a", "c"], None], pl.Array(pl.Categorical, 2), **field
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_array(field: Any) -> None:
    dtype = pl.Array(pl.Int32, 0)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([[]], dtype, **field)
    roundtrip_series_re([None, [], None], dtype, **field)
    roundtrip_series_re([None], dtype, **field)

    dtype = pl.Array(pl.Int32, 2)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([[5, 6]], dtype, **field)
    roundtrip_series_re([[1, 2], [2, 3]], dtype, **field)
    roundtrip_series_re([[1, 2], [3, 7]], dtype, **field)
    roundtrip_series_re([[1, 2], [13, 11], [3, 7]], dtype, **field)
    roundtrip_series_re(
        [None, [1, 2], None, [13, 11], [5, 7]],
        dtype,
        **field,
    )

    dtype = pl.Array(pl.String, 2)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([["a", "b"]], dtype, **field)
    roundtrip_series_re([["", ""], ["", "a"]], dtype, **field)
    roundtrip_series_re([["abc", "def"], ["ghi", "xyzw"]], dtype, **field)
    roundtrip_series_re([["x", "yx"], ["abc", "xxx"]], dtype, **field)
    roundtrip_series_re(
        [["wow", "this is"], ["soo", "so"], ["veryyy", "cool"]],
        dtype,
        **field,
    )
    roundtrip_series_re(
        [None, ["very", "very"], None, [None, None], ["verryy", "cool"]],
        dtype,
        **field,
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
@pytest.mark.parametrize("precision", range(1, 38))
def test_decimal(field: Any, precision: int) -> None:
    dtype = pl.Decimal(precision=precision, scale=0)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([D("1")], dtype, **field)
    roundtrip_series_re([D("-1")], dtype, **field)
    roundtrip_series_re([D("9" * precision)], dtype, **field)
    roundtrip_series_re([D("-" + "9" * precision)], dtype, **field)
    roundtrip_series_re([None, D("-1"), None], dtype, **field)
    roundtrip_series_re([D("-1"), D("0"), D("1")], dtype, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_enum(field: Any) -> None:
    dtype = pl.Enum([])

    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([None, None], dtype, **field)

    dtype = pl.Enum(["a", "x", "b"])

    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re(["a"], dtype, **field)
    roundtrip_series_re(["x"], dtype, **field)
    roundtrip_series_re(["b"], dtype, **field)
    roundtrip_series_re(["b", "x", "a"], dtype, **field)
    roundtrip_series_re([None, "b", None], dtype, **field)
    roundtrip_series_re([None, "a", None], dtype, **field)


@pytest.mark.parametrize("size", [127, 128, 255, 256, 2**15, 2**15 + 1])
@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
@pytest.mark.slow
def test_large_enum(size: int, field: Any) -> None:
    dtype = pl.Enum([str(i) for i in range(size)])
    roundtrip_series_re([None, "1"], dtype, **field)
    roundtrip_series_re(["1", None], dtype, **field)

    roundtrip_series_re(
        [str(i) for i in range(3, size, int(7 * size / (2**8)))], dtype, **field
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_list_arr(field: Any) -> None:
    dtype = pl.List(pl.Array(pl.String, 2))
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([[None]], dtype, **field)
    roundtrip_series_re([[[None, None]]], dtype, **field)
    roundtrip_series_re([[["a", "b"]]], dtype, **field)
    roundtrip_series_re([[["a", "b"], ["xyz", "wowie"]]], dtype, **field)
    roundtrip_series_re([[["a", "b"]], None, [None, None]], dtype, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_list_struct_arr(field: Any) -> None:
    dtype = pl.List(
        pl.Struct({"x": pl.Array(pl.String, 2), "y": pl.Array(pl.Int64, 3)})
    )
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([[None]], dtype, **field)
    roundtrip_series_re([[{"x": None, "y": None}]], dtype, **field)
    roundtrip_series_re([[{"x": ["a", None], "y": [1, None, 3]}]], dtype, **field)
    roundtrip_series_re([[{"x": ["a", "xyz"], "y": [1, 7, 3]}]], dtype, **field)
    roundtrip_series_re([[{"x": ["a", "xyz"], "y": [1, 7, 3]}], []], dtype, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_list_nulls(field: Any) -> None:
    dtype = pl.List(pl.Null)
    roundtrip_series_re([], dtype, **field)
    roundtrip_series_re([[]], dtype, **field)
    roundtrip_series_re([None], dtype, **field)
    roundtrip_series_re([[None]], dtype, **field)
    roundtrip_series_re([[None, None, None]], dtype, **field)
    roundtrip_series_re([[None], [None, None], [None, None, None]], dtype, **field)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
def test_masked_out_list_20151(field: Any) -> None:
    dtype = pl.List(pl.Int64())

    values = [[1, 2], None, [4, 5], [None, 3]]

    array_series = pl.Series(values, dtype=pl.Array(pl.Int64(), 2))
    list_from_array_series = array_series.cast(dtype)

    roundtrip_series_re(list_from_array_series, dtype, **field)


def test_int_after_null() -> None:
    roundtrip_re(
        pl.DataFrame(
            [
                pl.Series("a", [None], pl.Null),
                pl.Series("b", [None], pl.Int8),
            ]
        ),
        nulls_last=[True, True],
    )


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
@given(s=series(allow_null=False, allow_chunks=False, excluded_dtypes=[pl.Categorical]))
def test_optional_eq_non_optional_20320(field: Any, s: pl.Series) -> None:
    with_null = s.extend(pl.Series([None], dtype=s.dtype))

    re_without_null = s._row_encode(**field)
    re_with_null = with_null._row_encode(**field)

    re_without_null = re_without_null.cast(pl.Binary)
    re_with_null = re_with_null.cast(pl.Binary)

    assert_series_equal(re_with_null.head(s.len()), re_without_null)


@pytest.mark.parametrize("field", FIELD_COMBS_ARGS)
@given(dtype=dtypes(excluded_dtypes=[pl.Categorical]))
def test_null(
    field: Any,
    dtype: pl.DataType,
) -> None:
    s = pl.Series("a", [None], dtype)

    assert_series_equal(
        s._row_encode(**field)
        ._row_decode(
            ["a"],
            [dtype],
            descending=None if field["descending"] is None else [field["descending"]],
            nulls_last=None if field["nulls_last"] is None else [field["nulls_last"]],
            unordered=field["unordered"],
        )
        .struct.unnest()
        .to_series(),
        s,
    )


@pytest.mark.parametrize(
    ("dtype", "vs"),
    [
        (pl.List(pl.String), [[None], ["A"], ["B"]]),
        (pl.Array(pl.String, 1), [[None], ["A"], ["B"]]),
        (pl.Struct({"x": pl.String}), [{"x": None}, {"x": "A"}, {"x": "B"}]),
        (pl.Array(pl.String, 2), [[None, "Z"], ["A", "C"], ["B", "B"]]),
    ],
)
def test_nested_sorting_22557(dtype: pl.DataType, vs: list[Any]) -> None:
    s = pl.Series("a", [vs[1], None, vs[0], vs[2]], dtype)

    assert_series_equal(
        s.sort(descending=False, nulls_last=False), pl.Series("a", [None] + vs, dtype)
    )
    assert_series_equal(
        s.sort(descending=False, nulls_last=True), pl.Series("a", vs + [None], dtype)
    )
    assert_series_equal(
        s.sort(descending=True, nulls_last=False),
        pl.Series("a", [None] + vs[::-1], dtype),
    )
    assert_series_equal(
        s.sort(descending=True, nulls_last=True),
        pl.Series("a", vs[::-1] + [None], dtype),
    )

    roundtrip_series_re(vs, dtype, descending=False, nulls_last=False)
    roundtrip_series_re(vs, dtype, descending=False, nulls_last=True)
    roundtrip_series_re(vs, dtype, descending=True, nulls_last=False)
    roundtrip_series_re(vs, dtype, descending=True, nulls_last=True)

    assert_series_equal(
        s._row_encode(descending=False, nulls_last=False).arg_sort(),
        pl.Series("a", [1, 2, 0, 3], pl.get_index_type()),
        check_names=False,
    )
    assert_series_equal(
        s._row_encode(descending=False, nulls_last=True).arg_sort(),
        pl.Series("a", [2, 0, 3, 1], pl.get_index_type()),
        check_names=False,
    )
    assert_series_equal(
        s._row_encode(descending=True, nulls_last=False).arg_sort(),
        pl.Series("a", [1, 3, 0, 2], pl.get_index_type()),
        check_names=False,
    )
    assert_series_equal(
        s._row_encode(descending=True, nulls_last=True).arg_sort(),
        pl.Series("a", [3, 0, 2, 1], pl.get_index_type()),
        check_names=False,
    )


def test_row_encoding_null_chunks() -> None:
    lf1 = pl.select(a=pl.lit(1, pl.Int64)).lazy()
    lf2 = pl.select(a=None).lazy()

    lf = pl.concat([lf1, lf2]).select(pl.col.a._row_encode())

    out = (
        lf.select(cs.all()._row_decode(["a"], [pl.Int64]))
        .unnest(cs.all())
        .collect(engine="streaming")
    )

    assert_frame_equal(
        pl.concat([lf1, lf2]).collect(),
        out,
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_row_encoding_group_context(engine: EngineType) -> None:
    lf = pl.LazyFrame({"g": [1, 2, 2, 1], "a": [1, 2, 3, None]})
    out = lf.select(
        pl.col("a")._row_encode().over("g")._row_decode(["a"], [pl.Int64])
    ).unnest(cs.all())
    assert_frame_equal(out.collect(engine=engine), lf.select("a").collect())

    out = (
        lf.group_by("g")
        .agg(pl.col("a")._row_encode())
        .select("g", n=pl.col("a").list.len())
        .sort("g")
    )
    expected = pl.DataFrame(
        {"g": [1, 2], "n": [2, 2]}, schema_overrides={"n": pl.UInt32}
    )
    assert_frame_equal(out.collect(engine=engine), expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_row_encoding_distinct(engine: EngineType) -> None:
    lf = pl.LazyFrame({"a": ["x", "y", "x", None]})
    k = pl.col("a")._row_encode()
    out = lf.select(
        n_unique=k.n_unique(),
        unique=k.unique().len(),
        first=k.is_first_distinct().implode(),
        last=k.is_last_distinct().implode(),
    )
    expected = pl.DataFrame(
        {
            "n_unique": [3],
            "unique": [3],
            "first": [[True, True, False, True]],
            "last": [[False, True, True, True]],
        },
        schema_overrides={
            "n_unique": pl.get_index_type(),
            "unique": pl.get_index_type(),
        },
    )
    assert_frame_equal(out.collect(engine=engine), expected)

    out = (
        lf.group_by(k)
        .agg(pl.len())
        .select(pl.col("a")._row_decode(["a"], [pl.String]), "len")
        .unnest("a")
        .sort("a")
    )
    expected = pl.DataFrame(
        {"a": [None, "x", "y"], "len": [1, 2, 1]},
        schema_overrides={"len": pl.get_index_type()},
    )
    assert_frame_equal(out.collect(engine=engine), expected)


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        ([[1, 2, 3], [4, 5, 6]], (2, 3)),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (2, 2, 2)),
    ],
)
def test_null_array_row_hash_after_gather_30022(
    value: list[Any], shape: tuple[int, ...]
) -> None:
    s = pl.Series("a", [None, value], dtype=pl.Array(pl.Int32, shape))
    out = pl.concat([s, s.gather([0, 1])])
    assert out.hash()[0] == out.hash()[2]
    assert out.n_unique() == 2
    assert out.unique().len() == 2
    assert out.to_frame().group_by("a").len().height == 2


def _array_2d(hidden: bool) -> pa.Array:
    # [None, [[1, 2, 3], [4, 5, 6]]]; the null row's leaves are valid zeros if hidden.
    leaves = pa.array(
        [0] * 6 + [1, 2, 3, 4, 5, 6],
        pa.int32(),
        mask=pa.array([not hidden] * 6 + [False] * 6),
    )
    inner = pa.FixedSizeListArray.from_arrays(
        leaves, 3, mask=pa.array([True, True, False, False])
    )
    return pa.FixedSizeListArray.from_arrays(inner, 2, mask=pa.array([True, False]))


def _hidden_values(case: str, hidden: bool) -> pa.Array:
    x = _array_2d(hidden)
    if case == "array":
        return x
    if case == "list_null_row":
        offsets = pa.array([0, 0, 2], pa.int64())
        return pa.LargeListArray.from_arrays(offsets, x, mask=pa.array([True, False]))
    if case == "list_no_validity":
        return pa.LargeListArray.from_arrays(pa.array([0, 2], pa.int64()), x)
    if case == "struct_all_valid":
        return pa.StructArray.from_arrays([x], ["x"], mask=pa.array([False, False]))
    if case == "struct_masked_field":
        y = pa.array([None, 7], pa.int32())
        return pa.StructArray.from_arrays(
            [y, x], ["y", "x"], mask=pa.array([True, False])
        )
    if case == "struct_field_with_nulls":
        a = pa.array([1, 5, 9], pa.int32())
        b = pa.array([1 if hidden else None, 7, None], pa.int32())
        return pa.StructArray.from_arrays(
            [a, b], ["a", "b"], mask=pa.array([True, False, True])
        )
    raise ValueError(case)


@pytest.mark.parametrize(
    "case",
    [
        "array",
        "list_null_row",
        "list_no_validity",
        "struct_all_valid",
        "struct_masked_field",
        "struct_field_with_nulls",
    ],
)
def test_hidden_values_under_nulls_hash_equal_30022(case: str) -> None:
    hidden = pl.from_arrow(_hidden_values(case, hidden=True))
    clean = pl.from_arrow(_hidden_values(case, hidden=False))
    assert isinstance(hidden, pl.Series)
    assert isinstance(clean, pl.Series)
    assert_series_equal(hidden, clean)
    assert hidden.hash().to_list() == clean.hash().to_list()
    assert pl.concat([hidden, clean]).n_unique() == clean.n_unique()


OUTPUT_DF = pl.DataFrame(
    {
        "b": [True, False, None, True, False, None, True, False],
        "bn": [True, False, False, True, True, False, True, False],
        "u8": pl.Series([0, 255, None, 7, 255, 0, 7, None], dtype=pl.UInt8),
        "u8n": pl.Series([3, 0, 255, 3, 1, 0, 255, 3], dtype=pl.UInt8),
        "i8": pl.Series([-128, 127, -1, None, 0, -1, 127, None], dtype=pl.Int8),
        "i8n": pl.Series([0, -1, 1, -128, 127, 0, -1, 1], dtype=pl.Int8),
        "i32": pl.Series(
            [None, -(2**31), 2**31 - 1, 0, -1, 1, None, -1], dtype=pl.Int32
        ),
        "i32n": pl.Series([5, -5, 0, 5, -(2**31), 2**31 - 1, 0, -5], dtype=pl.Int32),
        "i64": pl.Series(
            [None, -(2**63), 2**63 - 1, 0, -1, 1, 0, None], dtype=pl.Int64
        ),
        "i64n": pl.Series([1, 1, -(2**63), 2**63 - 1, 0, 0, -1, 1], dtype=pl.Int64),
        "i128n": pl.Series(
            [0, -(2**127), 2**127 - 1, -1, 1, 0, -1, 1], dtype=pl.Int128
        ),
        "u128": pl.Series(
            [0, 2**128 - 1, None, 1, 0, 2**128 - 1, None, 1], dtype=pl.UInt128
        ),
        "f32": pl.Series(
            [1.5, float("-inf"), float("nan"), None, float("inf"), -2.0, 1.5, 0.0],
            dtype=pl.Float32,
        ),
        "f64": pl.Series(
            [float("nan"), -1e300, None, 1e300, -0.5, 0.0, float("nan"), None],
            dtype=pl.Float64,
        ),
        "dec": pl.Series(
            [D("1.50"), D("-99999999.99"), None, D("0.00")]
            + [D("99999999.99"), D("-0.01"), D("1.50"), None],
            dtype=pl.Decimal(10, 2),
        ),
        "decn": pl.Series(
            [D("1.50"), D("-999999999999.99"), D("0.00"), D("999999999999.99")]
            + [D("-0.01"), D("1.50"), D("0.01"), D("-1.50")],
            dtype=pl.Decimal(14, 2),
        ),
        "enum": pl.Series(
            ["b", "a", None, "c", "a", "b", None, "c"], dtype=pl.Enum(["c", "a", "b"])
        ),
        "enumn": pl.Series(
            ["b", "a", "a", "c", "c", "b", "a", "c"], dtype=pl.Enum(["c", "a", "b"])
        ),
        "s": ["x", "", None, "a string that is too long to inline", "x", None, "", "y"],
        "nul": pl.Series([None] * 8, dtype=pl.Null),
        "stn": pl.Series(
            [
                {"x": 1, "y": "a"},
                {"x": None, "y": None},
                {"x": -1, "y": ""},
                {"x": 1, "y": None},
                {"x": None, "y": "a"},
                {"x": 0, "y": "b"},
                {"x": 1, "y": "a"},
                {"x": -1, "y": "a string that is too long to inline"},
            ],
            dtype=pl.Struct({"x": pl.Int32, "y": pl.String}),
        ),
        "arrn": pl.Series(
            [
                [1, 2],
                [None, 3],
                [-1, None],
                [0, 0],
                [None, None],
                [5, -5],
                [1, 2],
                [3, 4],
            ],
            dtype=pl.Array(pl.Int16, 2),
        ),
        "stfn": pl.Series(
            [{"x": i % 3 - 1, "d": None if i % 4 == 0 else D(i)} for i in range(8)],
            dtype=pl.Struct({"x": pl.Int64, "d": pl.Decimal(10, 2)}),
        ),
        "arrsn": pl.Series(
            [
                ["a"],
                [None],
                [""],
                ["b"],
                ["a"],
                [None],
                ["too long to inline here"],
                [""],
            ],
            dtype=pl.Array(pl.String, 1),
        ),
    }
)

# The columns, which of them encode nulls and the expected output of
# `output="optimal"`: a dtype, or "bytes" for rows in the byte format.
OPTIMAL_CASES: list[tuple[list[str], list[bool] | None, Any]] = [
    (["bn"], [False], pl.Boolean),
    (["nul"], None, pl.Boolean),
    (["b"], None, pl.UInt8),
    (["u8n", "i8n"], [False, False], pl.UInt16),
    (["u8n", "i8n"], None, pl.UInt32),
    (["i32n"], [False], pl.UInt32),
    (["i32", "b"], None, pl.UInt64),
    (["i64n"], [False], pl.UInt64),
    (["dec", "enum"], None, pl.UInt64),
    (["i64", "i32"], None, pl.UInt128),
    (["f32", "f64"], None, pl.UInt128),
    (["i128n"], [False], pl.UInt128),
    (["i128n"], None, "bytes"),
    (["i64", "i64n"], None, "bytes"),
    (["u128", "i64", "i32", "u8", "b", "nul"], None, "bytes"),
    (["s", "i8"], None, "bytes"),
    (["decn", "enumn"], [False, False], pl.UInt64),
    (["decn", "enumn"], None, pl.UInt64),
    (["i128n", "i64n", "decn"], None, "bytes"),
    (["stn"], None, "bytes"),
]

# The columns and which of them encode nulls, for the byte format.
BYTE_FORMAT_ENCODE_NULLS_CASES: list[tuple[list[str], list[bool]]] = [
    (["i128n", "i64n", "decn"], [False, False, False]),
    (["stn"], [False]),
    (["arrn", "i32n"], [False, False]),
    (["stfn", "arrsn"], [False, False]),
    (["arrsn", "stfn"], [False, True]),
    (["stn", "arrn", "u8n", "i64n", "bn", "enumn"], [False] * 4 + [True, False]),
    (["stn", "b", "arrn", "bn", "s"], [False, True, False, False, True]),
]

ORDER_COMBS: list[Any] = [*FIELD_COMBS, "alternating"]


def order_args(order: Any, num_cols: int) -> dict[str, Any]:
    if order == "alternating":
        return {
            "descending": [i % 2 == 0 for i in range(num_cols)],
            "nulls_last": [i % 3 == 0 for i in range(num_cols)],
        }
    descending, nulls_last, unordered = order
    if unordered:
        return {"unordered": True}
    return {
        "descending": [descending] * num_cols,
        "nulls_last": [nulls_last] * num_cols,
    }


def check_round_trip_and_order(
    df: pl.DataFrame,
    output: Any,
    encode_nulls: list[bool] | None,
    args: dict[str, Any],
) -> pl.Expr:
    """Check that the rows decode to `df` and, if ordered, sort like `df`."""
    encoded = pl.functions._row_encode(
        df.columns, output=output, encode_nulls=encode_nulls, **args
    )
    decoded = df.select(
        encoded._row_decode(
            df.columns, df.dtypes, output=output, encode_nulls=encode_nulls, **args
        ).struct.unnest()
    )
    assert_frame_equal(decoded, df)

    if not args.get("unordered"):
        indexed = df.with_row_index()
        assert_frame_equal(
            indexed.sort(encoded, maintain_order=True),
            indexed.sort(df.columns, **args, maintain_order=True),
        )
    return encoded


@pytest.mark.parametrize(("cols", "encode_nulls", "expected"), OPTIMAL_CASES)
@pytest.mark.parametrize("order", ORDER_COMBS)
def test_row_encode_optimal(
    cols: list[str], encode_nulls: list[bool] | None, expected: Any, order: Any
) -> None:
    df = OUTPUT_DF.select(cols)
    args = order_args(order, len(cols))
    encoded = check_round_trip_and_order(df, "optimal", encode_nulls, args)

    if expected == "bytes":
        assert df.select(encoded).to_series().to_arrow().type == pa.large_binary()
        byte_format = pl.functions._row_encode(cols, **args)
        out = df.select(
            optimal=encoded.cast(pl.Binary), byte_format=byte_format.cast(pl.Binary)
        )
        assert_series_equal(out["optimal"], out["byte_format"], check_names=False)
    else:
        assert df.select(encoded).to_series().dtype == expected


@pytest.mark.parametrize(
    "s",
    [
        pl.Series("a", ["x", "", None, "a string that is too long to inline", None]),
        pl.Series("a", [b"x", None, b"", b"\x00\xff", b"a binary that is too long"]),
    ],
)
@pytest.mark.parametrize("order", ORDER_COMBS)
def test_row_encode_optimal_single_binary(s: pl.Series, order: Any) -> None:
    df = s.to_frame()
    args = order_args(order, 1)
    encoded = check_round_trip_and_order(df, "optimal", None, args)
    rows = df.select(encoded).to_series()

    ascending_nulls_first = args.get("unordered") or not (
        args["descending"][0] or args["nulls_last"][0]
    )
    if ascending_nulls_first:
        assert_series_equal(rows, s.cast(pl.Binary), check_names=False)
    else:
        assert rows.to_arrow().type == pa.large_binary()
        assert rows.null_count() == 0


@pytest.mark.parametrize("unordered", [False, True])
def test_row_encode_optimal_single_binary_offset(unordered: bool) -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    inner = pl.functions._row_encode(["a", "b"])
    outer = pl.functions._row_encode([inner], output="optimal", unordered=unordered)
    out = df.select(inner=inner, outer=outer)
    assert out["outer"].to_arrow().type == pa.large_binary()
    assert_series_equal(
        out["outer"].cast(pl.Binary), out["inner"].cast(pl.Binary), check_names=False
    )


@pytest.mark.parametrize("output", ["binary_offset", "binary"])
@pytest.mark.parametrize("order", ORDER_COMBS)
def test_row_encode_byte_format(output: Any, order: Any) -> None:
    args = order_args(order, OUTPUT_DF.width)
    encoded = check_round_trip_and_order(OUTPUT_DF, output, None, args)
    if output == "binary":
        assert OUTPUT_DF.select(encoded).to_series().dtype == pl.Binary


@pytest.mark.parametrize(("cols", "encode_nulls"), BYTE_FORMAT_ENCODE_NULLS_CASES)
@pytest.mark.parametrize("output", ["binary_offset", "binary"])
@pytest.mark.parametrize("order", ORDER_COMBS)
def test_row_encode_byte_format_nulls_not_encoded(
    cols: list[str], encode_nulls: list[bool], output: Any, order: Any
) -> None:
    df = OUTPUT_DF.select(cols)
    args = order_args(order, len(cols))
    encoded = check_round_trip_and_order(df, output, encode_nulls, args)
    all_nulls = pl.functions._row_encode(cols, output=output, **args)
    out = df.select(
        encoded=encoded.cast(pl.Binary), all_nulls=all_nulls.cast(pl.Binary)
    )

    # Each column without nulls encoded is one byte shorter, except booleans.
    num_shorter = sum(
        not e and df[c].dtype != pl.Boolean
        for c, e in zip(cols, encode_nulls, strict=True)
    )
    size_diff = out["all_nulls"].bin.size() - out["encoded"].bin.size()
    assert (size_diff == num_shorter).all()


@pytest.mark.parametrize(
    "s",
    [
        pl.Series("a", [1, None]),
        pl.Series("a", ["x", None]),
        pl.Series("a", [{"x": 1}, None]),
        pl.Series("a", [[1, 2], None], dtype=pl.Array(pl.Int64, 2)),
        pl.Series("a", [[1, 2], None, [3]]),
    ],
)
@pytest.mark.parametrize("output", ["binary_offset", "binary", "optimal"])
def test_row_encode_nulls_not_encoded(s: pl.Series, output: Any) -> None:
    df = s.to_frame()
    with pytest.raises(pl.exceptions.ComputeError, match="nulls are not encoded"):
        df.select(pl.functions._row_encode(["a"], output=output, encode_nulls=[False]))


@pytest.mark.parametrize("output", ["binary_offset", "binary", "optimal"])
@pytest.mark.parametrize("encode_nulls", [None, [False, False]])
@pytest.mark.parametrize("unordered", [False, True])
def test_row_encode_propagate_nulls(
    output: Any, encode_nulls: list[bool] | None, unordered: bool
) -> None:
    df = pl.DataFrame({"a": [1, None, 3, 4], "b": ["x", "y", None, "x"]})
    rows = df.select(
        pl.functions._row_encode(
            ["a", "b"],
            unordered=unordered,
            output=output,
            encode_nulls=encode_nulls,
            propagate_nulls=True,
        )
    ).to_series()
    assert rows.is_null().to_list() == [False, True, True, False]
    assert rows[0] != rows[3]


def test_row_decode_propagated_nulls_raises() -> None:
    df = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    rows = df._row_encode(propagate_nulls=True)
    with pytest.raises(pl.exceptions.InvalidOperationError, match="null rows"):
        rows._row_decode(["a", "b"], [pl.Int64, pl.String])
