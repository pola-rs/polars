from __future__ import annotations

import io
import math
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import accumulate
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError, InvalidOperationError, SchemaError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import IO

MAP = pl.Map(pl.String, pl.Int64)
FLOAT_MAP = pl.Map(pl.String, pl.Float64)
ENTRIES = pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))

MAP_EXTENSION_NAME = "testing.map_container"
pl.register_extension_type(MAP_EXTENSION_NAME, pl.Extension)


class _CustomMapping(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any]) -> None:
        # Not `self.values`: that shadows `Mapping.values()`.
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_map_dtype_init() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    assert dtype.key == pl.String
    assert dtype.value == pl.Int64
    assert repr(dtype) == "Map(String, Int64)"
    assert str(dtype) == "Map(String, Int64)"


def test_map_dtype_init_parses_python_types() -> None:
    assert pl.Map(str, int) == pl.Map(pl.String, pl.Int64)


def test_map_dtype_equality() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    # A bare class is not specific about its inner types, so it compares equal.
    assert dtype == pl.Map
    assert dtype == pl.Map(pl.String, pl.Int64)
    assert dtype != pl.Map(pl.String, pl.Int32)
    assert dtype != pl.Map(pl.Int64, pl.Int64)
    assert dtype != pl.List(pl.Int64)


def test_map_dtype_hash() -> None:
    assert len({pl.Map, pl.Map(pl.String, pl.Int64), pl.Map(pl.String, pl.Int32)}) == 3


def test_map_dtype_is_nested() -> None:
    assert pl.Map(pl.String, pl.Int64).is_nested()
    assert pl.Map in pl.datatypes.group.NESTED_DTYPES


def test_map_dtype_to_py_type() -> None:
    assert pl.datatypes.convert.dtype_to_py_type(pl.Map) is dict


def test_map_unpack_dtypes() -> None:
    dtype = pl.Map(pl.String, pl.List(pl.Int64))
    assert pl.datatypes.unpack_dtypes(dtype) == {pl.String, pl.Int64}
    assert pl.datatypes.unpack_dtypes(dtype, include_compound=True) == {
        dtype,
        pl.String,
        pl.List(pl.Int64),
        pl.Int64,
    }


def test_map_series_from_dicts() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    s = pl.Series("m", [{"a": 1, "b": 2}, {"x": 9}, None], dtype=dtype)

    assert s.dtype == dtype
    assert s.len() == 3
    assert s.null_count() == 1
    assert s.to_list() == [{"a": 1, "b": 2}, {"x": 9}, None]


def test_map_series_repr() -> None:
    s = pl.Series("m", [{"a": 1}], dtype=pl.Map(pl.String, pl.Int64))
    assert "map[str, i64]" in repr(s)
    assert '{"a": 1}' in repr(s)


def test_map_dtype_in_schema() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    df = pl.DataFrame({"m": pl.Series([{"a": 1}], dtype=dtype)})
    assert df.schema == pl.Schema({"m": dtype})


def test_map_series_requires_instantiated_dtype() -> None:
    with pytest.raises(TypeError, match="requires a key and a value type"):
        pl.Series("m", [{"a": 1}], dtype=pl.Map)


def test_map_cast_value_dtype() -> None:
    s = pl.Series("m", [{"a": 1}], dtype=pl.Map(pl.String, pl.Int64))
    out = s.cast(pl.Map(pl.String, pl.Float64))
    assert out.dtype == pl.Map(pl.String, pl.Float64)
    assert out.to_list() == [{"a": 1.0}]


def test_map_cast_to_entries_list_and_back() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    s = pl.Series("m", [{"a": 1, "b": 2}, {}, None], dtype=dtype)

    entries = s.cast(ENTRIES)
    assert entries.to_list() == [
        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
        [],
        None,
    ]
    assert_series_equal(entries.cast(dtype), s)


def test_map_cast_key_dtype_is_rejected() -> None:
    s = pl.Series("m", [{"a": 1}], dtype=pl.Map(pl.String, pl.Int64))
    with pytest.raises(InvalidOperationError, match="cannot cast Map key"):
        s.cast(pl.Map(pl.Int64, pl.Int64))


def test_map_cast_from_entries_canonicalizes_duplicate_keys() -> None:
    # Duplicates keep the first position and the last value.
    s = pl.Series(
        "m",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "a", "value": 3},
            ]
        ],
    )
    assert s.cast(pl.Map(pl.String, pl.Int64)).to_list() == [{"a": 3, "b": 2}]


def test_map_strict_cast_failure_reports_value_column() -> None:
    s = pl.Series("m", [{"a": "nope"}], dtype=pl.Map(pl.String, pl.String))
    with pytest.raises(InvalidOperationError, match="conversion from `str` to `i64`"):
        s.cast(pl.Map(pl.String, pl.Int64), strict=True)


def test_map_concat_requires_matching_dtypes() -> None:
    left = pl.Series("m", [{"a": 1}], dtype=pl.Map(pl.String, pl.Int32))
    right = pl.Series("m", [{"b": 2}], dtype=pl.Map(pl.String, pl.Int64))
    with pytest.raises(SchemaError):
        pl.concat([left, right])


def test_map_concat_relaxed_merges_value_dtype() -> None:
    left = pl.DataFrame({"m": pl.Series([{"a": 1}], dtype=pl.Map(pl.String, pl.Int32))})
    right = pl.DataFrame(
        {"m": pl.Series([{"b": 2}], dtype=pl.Map(pl.String, pl.Int64))}
    )

    out = pl.concat([left, right], how="vertical_relaxed")
    assert out.schema == pl.Schema({"m": pl.Map(pl.String, pl.Int64)})
    assert out["m"].to_list() == [{"a": 1}, {"b": 2}]


def test_map_group_by_and_sort() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    df = pl.DataFrame({"m": pl.Series([{"a": 1}, {"b": 2}, {"a": 1}], dtype=dtype)})

    out = df.group_by("m").len().sort("m")
    assert out["m"].to_list() == [{"a": 1}, {"b": 2}]
    assert out["len"].to_list() == [2, 1]


def test_map_map_elements_receives_dict() -> None:
    s = pl.Series("m", [{"a": 1, "b": 2}, {"x": 9}], dtype=pl.Map(pl.String, pl.Int64))
    out = s.map_elements(lambda d: len(d), return_dtype=pl.Int64)
    assert out.to_list() == [2, 1]


def test_map_map_elements_returns_dict() -> None:
    s = pl.Series("x", [1, 2])
    assert_series_equal(
        s.map_elements(lambda i: {"a": i}, return_dtype=MAP),
        pl.Series("x", [{"a": 1}, {"a": 2}], dtype=MAP),
    )

    # As for construction, the return dtype has to reach the conversion at every depth.
    nested = pl.Map(pl.Int64, pl.Map(pl.Int64, pl.String))
    out = s.map_elements(lambda i: {i: {2: "x"}}, return_dtype=nested)
    assert out.dtype == nested
    assert out.to_list() == [{1: {2: "x"}}, {2: {2: "x"}}]

    out = s.map_elements(lambda i: [{"a": i}], return_dtype=pl.List(MAP))
    assert out.to_list() == [[{"a": 1}], [{"a": 2}]]


def test_map_map_elements_returns_dict_in_expression() -> None:
    df = pl.DataFrame({"x": [1, 2]})
    out = df.select(pl.col("x").map_elements(lambda i: {"a": i}, return_dtype=MAP))
    assert_series_equal(out["x"], pl.Series("x", [{"a": 1}, {"a": 2}], dtype=MAP))


def test_map_map_rows_returns_dict() -> None:
    df = pl.DataFrame({"x": [1, 2]})
    out = df.map_rows(lambda row: {"a": row[0]}, return_dtype=MAP)
    assert_series_equal(
        out.to_series(), pl.Series("map", [{"a": 1}, {"a": 2}], dtype=MAP)
    )


@pytest.mark.parametrize(
    ("key_dtype", "key"),
    [
        (pl.Int64, 1),
        (pl.Float64, 1.5),
        (pl.Boolean, True),
        (pl.Date, date(2020, 1, 1)),
        (pl.String, "a"),
        (pl.Categorical, "a"),
    ],
)
def test_map_arbitrary_dict_key_types(key_dtype: pl.DataType, key: Any) -> None:
    # A dict is read as map entries, not as a Struct, so keys are values and not
    # field names -- any key dtype Polars accepts works.
    dtype = pl.Map(key_dtype, pl.String)
    s = pl.Series("m", [{key: "x"}], dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == [{key: "x"}]


def test_map_dict_keys_are_not_stringified() -> None:
    # Would have been {"1": ...} if keys still round-tripped through field names.
    s = pl.Series("m", [{1: "x", 2: "y"}], dtype=pl.Map(pl.Int64, pl.String))
    assert s.to_list() == [{1: "x", 2: "y"}]


def test_map_dict_key_dtype_mismatch_is_strict() -> None:
    with pytest.raises((TypeError, SchemaError)):
        pl.Series("m", [{"a": 1}], dtype=pl.Map(pl.Int64, pl.Int64))


def test_map_from_dicts_row_oriented_dataframe() -> None:
    dtype = pl.Map(pl.Int64, pl.String)
    df = pl.DataFrame([{"m": {1: "x"}}], schema={"m": dtype})
    assert df.schema == pl.Schema({"m": dtype})
    assert df["m"].to_list() == [{1: "x"}]


def test_map_nested_as_value() -> None:
    # Non-string keys nest too, which the dict-as-Struct reading could not express.
    dtype = pl.Map(pl.Int64, pl.Map(pl.Int64, pl.String))
    s = pl.Series("m", [{1: {2: "x"}}], dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == [{1: {2: "x"}}]


ENTRY_CASES = [
    pytest.param([{"key": "a", "value": 1}], True, id="canonical"),
    pytest.param([{"value": 1, "key": "a"}], True, id="reversed-order"),
    pytest.param([{"k": "a", "v": 1}], False, id="wrong-names"),
    pytest.param([{"key": "a", "value": 1, "extra": 9}], False, id="extra-field"),
    pytest.param([{"key": "a"}], False, id="missing-value-field"),
]


@pytest.mark.parametrize(("entries", "ok"), ENTRY_CASES)
def test_map_entries_matched_by_name_on_construction(
    entries: list[dict[str, Any]], ok: bool
) -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    if ok:
        assert pl.Series("m", [entries], dtype=dtype).to_list() == [{"a": 1}]
    else:
        with pytest.raises((TypeError, InvalidOperationError), match="named `key`"):
            pl.Series("m", [entries], dtype=dtype)


@pytest.mark.parametrize(("entries", "ok"), ENTRY_CASES)
def test_map_entries_matched_by_name_on_cast(
    entries: list[dict[str, Any]], ok: bool
) -> None:
    # Construction and casting must agree: only Arrow and Parquet match positionally.
    dtype = pl.Map(pl.String, pl.Int64)
    s = pl.Series("m", [entries])
    if ok:
        assert s.cast(dtype).to_list() == [{"a": 1}]
    else:
        with pytest.raises(InvalidOperationError, match="named `key`"):
            s.cast(dtype)


@pytest.mark.parametrize(("entries", "ok"), ENTRY_CASES)
def test_map_entries_matched_by_name_on_list_to_map(
    entries: list[dict[str, Any]], ok: bool
) -> None:
    s = pl.Series("m", [entries])
    if ok:
        assert s.list.to_map().to_list() == [{"a": 1}]
    else:
        with pytest.raises(InvalidOperationError, match="named `key`"):
            s.list.to_map()


@pytest.mark.parametrize("values", [[[]], [[], None]])
def test_map_empty_entries_both_routes(values: list[Any]) -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    expected: list[Any] = [{} if v is not None else None for v in values]
    assert pl.Series("m", values, dtype=dtype).to_list() == expected
    assert pl.Series("m", values).cast(dtype).to_list() == expected


def test_map_lit_cast_does_not_panic() -> None:
    # `should_cast_column` used to hit `debug_assert!(!target_dtype.is_nested())`.
    dtype = pl.Map(pl.String, pl.Int64)
    s = pl.Series("m", [{"a": 1}], dtype=dtype)
    assert pl.select(pl.lit(s).alias("m")).to_dicts() == [{"m": {"a": 1}}]


def test_map_strict_construction_does_not_coerce() -> None:
    # Must match `pl.Series([1.5], dtype=pl.Int64)`, which raises.
    for value in ({"a": 1.5}, {"a": "1"}):
        with pytest.raises((TypeError, SchemaError)):
            pl.Series("m", [value], dtype=pl.Map(pl.String, pl.Int64))
    with pytest.raises((TypeError, SchemaError)):
        pl.Series("m", [{1.5: 1}], dtype=pl.Map(pl.Int64, pl.Int64))


def test_map_non_strict_construction_coerces() -> None:
    s = pl.Series("m", [{"a": "1"}], dtype=pl.Map(pl.String, pl.Int64), strict=False)
    assert s.to_list() == [{"a": 1}]


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (pl.Map(pl.Duration("ns"), pl.Int64), {timedelta(seconds=1): 1}),
        (pl.Map(pl.Datetime("ns"), pl.Int64), {datetime(2020, 1, 1): 1}),
        (pl.Map(pl.String, pl.Duration("ms")), {"a": timedelta(seconds=1)}),
        (
            pl.Map(pl.String, pl.Struct({"d": pl.Duration("ns")})),
            {"x": {"d": timedelta(seconds=1)}},
        ),
    ],
)
def test_map_temporal_unit_is_converted(dtype: pl.Map, value: Any) -> None:
    # A Python `timedelta`/`datetime` has a fixed resolution, so it cannot arrive at the
    # target unit; the plain constructor converts it, and a Map child must agree.
    s = pl.Series("m", [value], dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == [value]


@pytest.mark.parametrize(
    ("child", "value"),
    [
        (pl.List(pl.Int64), ["1"]),
        (pl.Array(pl.Int64, 2), ["1", "2"]),
        (pl.Struct({"a": pl.Int64}), {"a": "1"}),
        (pl.Int64, "1"),
        (pl.Int64, 1.5),
        (pl.Duration("ns"), timedelta(seconds=1)),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_map_child_matches_plain_constructor(
    child: pl.DataType, value: Any, strict: bool
) -> None:
    # A Map child must accept and reject exactly what the plain constructor does.
    # Whether a container coerces is Polars' business; agreeing with it is ours.
    def outcome(build: Any) -> Any:
        try:
            return build()
        except Exception as e:
            return type(e).__name__

    plain = outcome(
        lambda: pl.Series("c", [value], dtype=child, strict=strict).to_list()
    )
    nested = outcome(
        lambda: pl.Series(
            "m", [{"x": value}], dtype=pl.Map(pl.String, child), strict=strict
        ).to_list()
    )
    expected = plain if isinstance(plain, str) else [{"x": plain[0]}]
    assert nested == expected


DEPTH_CASES = [
    (pl.Map(pl.String, pl.Int64), {"a": 1}),
    (pl.List(pl.Map(pl.String, pl.Int64)), [{"a": 1}]),
    (pl.Array(pl.Map(pl.String, pl.Int64), 2), [{"a": 1}, {"b": 2}]),
    (pl.Struct({"m": pl.Map(pl.String, pl.Int64)}), {"m": {"a": 1}}),
    (pl.List(pl.List(pl.Map(pl.String, pl.Int64))), [[{"a": 1}]]),
    (
        pl.Map(pl.String, pl.Array(pl.Map(pl.String, pl.Int64), 2)),
        {"x": [{"a": 1}, {"b": 2}]},
    ),
    (pl.Map(pl.String, pl.List(pl.Map(pl.String, pl.Int64))), {"x": [{"a": 1}]}),
]


@pytest.mark.parametrize(("dtype", "value"), DEPTH_CASES)
def test_map_dtype_hint_reaches_every_depth(dtype: pl.DataType, value: Any) -> None:
    # A dict is a Struct unless the target dtype says otherwise, so the hint has to be
    # threaded through every enclosing container, not just the outermost one.
    s = pl.Series("m", [value], dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == [value]


def test_map_nested_key_dtype_cannot_convert_to_python() -> None:
    entries = pl.Series("m", [[{"key": [1, 2], "value": "x"}]])
    s = entries.cast(pl.Map(pl.List(pl.Int64), pl.String))
    assert s.dtype == pl.Map(pl.List(pl.Int64), pl.String)
    with pytest.raises(TypeError, match="not hashable"):
        s.to_list()


_MAP_KEY_DTYPE = pl.Map(pl.Map(pl.String, pl.Int64), pl.Int64)
_MAP_KEY_ENTRIES = [{"key": {"a": 1}, "value": 7}]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: pl.Series("c", [_MAP_KEY_ENTRIES], dtype=_MAP_KEY_DTYPE),
            id="series",
        ),
        pytest.param(
            lambda: pl.DataFrame(
                {"c": [_MAP_KEY_ENTRIES]}, schema={"c": _MAP_KEY_DTYPE}
            )["c"],
            id="df-column-oriented",
        ),
        pytest.param(
            lambda: pl.DataFrame(
                [{"c": _MAP_KEY_ENTRIES}], schema={"c": _MAP_KEY_DTYPE}
            )["c"],
            id="df-row-oriented",
        ),
        pytest.param(
            lambda: pl.DataFrame(
                [{"c": _MAP_KEY_ENTRIES}], schema_overrides={"c": _MAP_KEY_DTYPE}
            )["c"],
            id="df-schema-overrides",
        ),
    ],
)
def test_map_as_key_dtype_is_constructible(build: Callable[[], pl.Series]) -> None:
    # A dict key is unhashable, so the entries form is the only way to spell a Map key.
    # It therefore needs the dtype hint as much as the mapping form does.
    s = build()
    assert s.dtype == _MAP_KEY_DTYPE
    entries = s.cast(
        pl.List(pl.Struct({"key": pl.Map(pl.String, pl.Int64), "value": pl.Int64}))
    )
    assert entries.to_list() == [_MAP_KEY_ENTRIES]


def test_map_as_key_dtype_nests() -> None:
    dtype = pl.Map(_MAP_KEY_DTYPE, pl.Int64)
    s = pl.Series("c", [[{"key": _MAP_KEY_ENTRIES, "value": 1}]], dtype=dtype)
    assert s.dtype == dtype


@pytest.mark.parametrize(("dtype", "value"), DEPTH_CASES)
@pytest.mark.parametrize("via", ["schema", "schema_overrides"])
def test_map_dtype_hint_at_every_depth_row_oriented(
    dtype: pl.DataType, value: Any, via: str
) -> None:
    # Row-oriented construction calls `py_object_to_any_value` with the column dtype
    # directly, so it gets none of the Series constructor's per-row recursion.
    df = pl.DataFrame([{"c": value}], **{via: {"c": dtype}})  # type: ignore[arg-type]
    assert df.schema == pl.Schema({"c": dtype})
    assert df["c"].to_list() == [value]


@pytest.mark.parametrize(("dtype", "value"), DEPTH_CASES)
def test_map_dtype_hint_at_every_depth_mapping_rows(
    dtype: pl.DataType, value: Any
) -> None:
    # `mappings_to_rows` is a separate path from `dicts_to_rows`.
    df = pl.DataFrame([_CustomMapping({"c": value})], schema={"c": dtype})
    assert df["c"].to_list() == [value]


def test_map_lit_requires_explicit_dtype() -> None:
    # A dict infers as a Struct, so comparing a Map column to a bare `pl.lit(dict)`
    # is a dtype mismatch, exactly as for any other pair of unrelated dtypes.
    assert pl.select(pl.lit({"a": 1})).schema == pl.Schema(
        {"literal": pl.Struct({"a": pl.Int64})}
    )
    lf = pl.LazyFrame({"m": pl.Series([{"a": 1}], dtype=pl.Map(pl.String, pl.Int64))})
    with pytest.raises(SchemaError):
        lf.filter(pl.col("m") == pl.lit({"a": 1})).collect()


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (pl.Map(pl.String, pl.Int64), {"a": 1}),
        (pl.Map(pl.Int64, pl.String), {1: "x"}),
        (pl.Map(pl.String, pl.Map(pl.String, pl.Int64)), {"x": {"a": 1}}),
    ],
)
def test_map_lit_with_dtype(dtype: pl.Map, value: Any) -> None:
    # `lit` otherwise builds the Series and casts into the dtype, which cannot reach a
    # Map because `Struct -> Map` is not a cast.
    out = pl.select(pl.lit(value, dtype=dtype))
    assert out.schema == pl.Schema({"literal": dtype})
    assert out.to_dicts() == [{"literal": value}]


def test_map_filter_against_literal() -> None:
    dtype = pl.Map(pl.String, pl.Int64)
    lf = pl.LazyFrame({"m": pl.Series([{"a": 1}, {"b": 2}], dtype=dtype), "i": [1, 2]})

    for lit in (
        pl.lit({"a": 1}, dtype=dtype),
        pl.lit(pl.Series([{"a": 1}], dtype=dtype)),
    ):
        assert lf.filter(pl.col("m") == lit).collect()["i"].to_list() == [1]
        assert lf.filter(pl.col("m") != lit).collect()["i"].to_list() == [2]


ARROW_SHAPES = [
    pytest.param(pl.Map(pl.String, pl.Int64), [{"a": 1}, None, {}], id="map"),
    pytest.param(pl.Map(pl.Int64, pl.String), [{1: "x"}], id="map-int-keys"),
    pytest.param(
        pl.Map(pl.String, pl.Map(pl.String, pl.Int64)),
        [{"x": {"a": 1}}],
        id="map-of-map",
    ),
    pytest.param(pl.List(pl.Map(pl.String, pl.Int64)), [[{"a": 1}]], id="list-of-map"),
    pytest.param(
        pl.Array(pl.Map(pl.String, pl.Int64), 1), [[{"a": 1}]], id="array-of-map"
    ),
    pytest.param(
        pl.Struct({"m": pl.Map(pl.String, pl.Int64)}),
        [{"m": {"a": 1}}],
        id="struct-of-map",
    ),
    pytest.param(
        pl.Map(pl.Datetime("ms"), pl.Duration("us")),
        [{datetime(2020, 1, 1): timedelta(seconds=1)}],
        id="map-temporal",
    ),
]


@pytest.mark.parametrize(("dtype", "values"), ARROW_SHAPES)
def test_map_arrow_roundtrip(dtype: pl.DataType, values: list[Any]) -> None:
    s = pl.Series("c", values, dtype=dtype)
    back = pl.from_arrow(s.to_frame().to_arrow())
    assert isinstance(back, pl.DataFrame)
    assert back.schema == pl.Schema({"c": dtype})
    assert back["c"].to_list() == values


@pytest.mark.parametrize(("dtype", "values"), ARROW_SHAPES)
@pytest.mark.parametrize("stream", [False, True])
def test_map_ipc_roundtrip(dtype: pl.DataType, values: list[Any], stream: bool) -> None:
    df = pl.Series("c", values, dtype=dtype).to_frame()
    buf = io.BytesIO()
    if stream:
        df.write_ipc_stream(buf)
    else:
        df.write_ipc(buf)
    buf.seek(0)
    back = pl.read_ipc_stream(buf) if stream else pl.read_ipc(buf)
    assert back.schema == pl.Schema({"c": dtype})
    assert back["c"].to_list() == values


def test_map_arrow_export_is_a_map_type() -> None:
    pa = pytest.importorskip("pyarrow")
    dtype = pl.Map(pl.String, pl.Int64)
    field = (
        pl.Series("m", [{"a": 1}], dtype=dtype).to_frame().to_arrow().schema.field("m")
    )
    assert field.type == pa.map_(pa.large_string(), pa.int64())
    # Arrow requires non-null keys.
    assert not field.type.key_field.nullable


def test_map_arrow_import_matches_entries_positionally() -> None:
    pa = pytest.importorskip("pyarrow")
    # Only Arrow and Parquet do this. The names carry no meaning there, so they are
    # normalized to `key`/`value` on the way in -- Map equality compares them.
    map_type = pa.map_(
        pa.field("k", pa.string(), nullable=False), pa.field("v", pa.int64())
    )
    tbl = pa.table({"m": pa.array([[("a", 1)]], type=map_type)})
    s = pl.from_arrow(tbl)["m"]  # type: ignore[index]
    assert s.dtype == pl.Map(pl.String, pl.Int64)
    assert s.to_list() == [{"a": 1}]
    assert s.cast(ENTRIES).to_list() == [[{"key": "a", "value": 1}]]


def test_map_arrow_import_keeps_duplicate_keys() -> None:
    pa = pytest.importorskip("pyarrow")
    # Key uniqueness is not validated on Arrow import, we just trust it blindly
    tbl = pa.table(
        {"m": pa.array([[("a", 1), ("a", 2)]], type=pa.map_(pa.string(), pa.int64()))}
    )
    s = pl.from_arrow(tbl)["m"]  # type: ignore[index]
    assert s.dtype == pl.Map(pl.String, pl.Int64)
    assert s.cast(ENTRIES).to_list() == [
        [{"key": "a", "value": 1}, {"key": "a", "value": 2}]
    ]
    # Eventually, we build a Python dict, which keeps only the last value
    assert s.to_list() == [{"a": 2}]


def _ipc_buffer(s: pl.Series) -> IO[bytes]:
    buf = io.BytesIO()
    s.to_frame().write_ipc(buf)
    buf.seek(0)
    return buf


def test_map_scan_unifies_value_across_files() -> None:
    # Follow the same unification rules as List, applied to the Map values only.
    ordered = pl.Map(pl.String, pl.Struct({"a": pl.Int64, "b": pl.String}))
    swapped = pl.Map(pl.String, pl.Struct({"b": pl.String, "a": pl.Int64}))
    rows: list[Any] = [
        {"k": {"a": 1, "b": "x"}},
        None,
        {},
        {"p": {"a": 3, "b": "z"}, "q": {"a": 4, "b": "w"}},
    ]
    rows_swapped: list[Any] = [None, {}, {"k": {"b": "y", "a": 2}}]

    with pytest.raises(InvalidOperationError, match="field name mismatch"):
        pl.Series("m", rows_swapped, swapped).cast(ordered)

    sources = [
        _ipc_buffer(pl.Series("m", rows, ordered)),
        _ipc_buffer(pl.Series("m", rows_swapped, swapped)),
    ]
    out = pl.scan_ipc(sources).collect()
    assert out.schema == {"m": ordered}
    assert out["m"].to_list() == [*rows, *rows_swapped]


def test_map_scan_unifies_nested_value_across_files() -> None:
    ordered = pl.Map(pl.String, pl.List(pl.Struct({"a": pl.Int64, "b": pl.String})))
    swapped = pl.Map(pl.String, pl.List(pl.Struct({"b": pl.String, "a": pl.Int64})))
    sources = [
        _ipc_buffer(pl.Series("m", [{"k": [{"a": 1, "b": "x"}]}], ordered)),
        _ipc_buffer(pl.Series("m", [{"k": [{"b": "y", "a": 2}]}], swapped)),
    ]

    out = pl.scan_ipc(sources).collect()
    assert out.schema == {"m": ordered}
    assert out["m"].to_list() == [
        {"k": [{"a": 1, "b": "x"}]},
        {"k": [{"a": 2, "b": "y"}]},
    ]


def test_map_scan_refuses_key_change_across_files() -> None:
    # If this used regular casting during unification,
    # the key dtype would be promoted to String.
    sources = [
        _ipc_buffer(pl.Series("m", [{"k": 1}], pl.Map(pl.String, pl.Int64))),
        _ipc_buffer(pl.Series("m", [{7: 2}], pl.Map(pl.Int32, pl.Int64))),
    ]

    with pytest.raises(SchemaError, match="data type mismatch for column m"):
        pl.scan_ipc(sources).collect()


def test_map_scan_canonicalizes_after_a_key_cast() -> None:
    # Decimal rescale is the one admitted key cast that is not injective, so the cast
    # has to merge the entries it collapses -- first key position, last value.
    hi = pl.Map(pl.Decimal(10, 2), pl.Int64)
    lo = pl.Map(pl.Decimal(10, 1), pl.Int64)
    sources = [
        _ipc_buffer(pl.Series("m", [{Decimal("9.9"): 9}], lo)),
        _ipc_buffer(
            pl.Series("m", [{Decimal("1.01"): 1, Decimal("1.02"): 2}], hi),
        ),
    ]

    out = pl.scan_ipc(sources).collect()
    assert out.schema == {"m": lo}
    entries = pl.List(pl.Struct({"key": pl.Decimal(10, 1), "value": pl.Int64}))
    assert out["m"].cast(entries).to_list() == [
        [{"key": Decimal("9.9"), "value": 9}],
        [{"key": Decimal("1.0"), "value": 2}],
    ]


def _parquet_buffer(s: pl.Series) -> IO[bytes]:
    buf = io.BytesIO()
    s.to_frame().write_parquet(buf)
    buf.seek(0)
    return buf


@pytest.mark.parametrize(("dtype", "values"), ARROW_SHAPES)
def test_map_parquet_roundtrip(dtype: pl.DataType, values: list[Any]) -> None:
    s = pl.Series("c", values, dtype=dtype)
    back = pl.read_parquet(_parquet_buffer(s))
    assert back.schema == {"c": dtype}
    assert_series_equal(back["c"], s)


def test_map_scan_as_entries_schema_override() -> None:
    dtype = pl.Map(pl.Int32, pl.String)
    entries = pl.List(pl.Struct({"key": pl.Int32, "value": pl.String}))
    s = pl.Series("x", [{1: "a", 2: "b"}, None, {}], dtype=dtype)

    assert pl.scan_parquet(_parquet_buffer(s)).collect_schema() == {"x": dtype}
    assert pl.scan_parquet(_parquet_buffer(s), schema={"x": entries}).collect()[
        "x"
    ].to_list() == [
        [{"key": 1, "value": "a"}, {"key": 2, "value": "b"}],
        None,
        [],
    ]

    # An unrelated target is still refused.
    with pytest.raises(SchemaError, match="data type mismatch"):
        pl.scan_parquet(_parquet_buffer(s), schema={"x": pl.List(pl.Int64)}).collect()


def test_map_scan_from_entries_schema_override() -> None:
    entries = pl.List(pl.Struct({"key": pl.Int32, "value": pl.String}))
    dtype = pl.Map(pl.Int32, pl.String)
    s = pl.Series("x", [[{"key": 1, "value": "a"}, {"key": 1, "value": "b"}]], entries)

    assert pl.scan_parquet(_parquet_buffer(s)).collect_schema() == {"x": entries}
    out = pl.scan_parquet(_parquet_buffer(s), schema={"x": dtype}).collect()
    assert out.schema == {"x": dtype}
    # We deduplicate the map entries, so this is not lossless
    assert out["x"].to_list() == [{1: "b"}]


def test_map_parquet_entries_satisfy_a_map_schema() -> None:
    # A `List(Struct {key, value})` column can be written under a user-provided
    # arrow Map schema, never being a Map.
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    df = pl.DataFrame(
        {
            "m": [
                [{"key": "a", "value": 1}],
                [{"key": "a", "value": 2}, {"key": "b", "value": 3}],
            ]
        }
    )
    assert df.schema == {"m": pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))}

    buf = io.BytesIO()
    schema = pa.schema([pa.field("m", pa.map_(pa.large_string(), pa.int64()))])
    df.write_parquet(buf, arrow_schema=schema)

    buf.seek(0)
    assert pq.read_schema(buf).field("m").type == schema.field("m").type

    buf.seek(0)
    assert pl.scan_parquet(buf).collect_schema() == {"m": pl.Map(pl.String, pl.Int64)}
    buf.seek(0)
    assert pl.read_parquet(buf)["m"].to_list() == [{"a": 1}, {"a": 2, "b": 3}]


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(pl.Map(pl.Null, pl.Int64), id="null-key"),
        pytest.param(pl.Map(pl.Object, pl.Int64), id="object-key"),
    ],
)
def test_map_invalid_key_dtype_is_rejected_without_data(dtype: pl.Map) -> None:
    for build in (
        lambda: pl.Series("m", [], dtype=dtype),
        lambda: pl.Series("m", [None], dtype=dtype),
        lambda: pl.DataFrame(schema={"m": dtype}),
        lambda: pl.select(pl.lit(None).cast(dtype)),
        # Reaches `Series::full_null` with the dtype and never calls the function.
        lambda: pl.Series("m", [None]).map_elements(lambda v: v, return_dtype=dtype),
    ):
        with pytest.raises((InvalidOperationError, TypeError), match="Map key dtype"):
            build()


def test_map_valid_key_dtypes_still_construct_empty() -> None:
    # A `Null` *behind a container* is fine: it can be materialized later.
    for dtype in (
        pl.Map(pl.String, pl.Int64),
        pl.Map(pl.List(pl.Null), pl.Int64),
        pl.Map(pl.Map(pl.String, pl.Int64), pl.Int64),
    ):
        assert pl.Series("m", [], dtype=dtype).dtype == dtype
        assert pl.DataFrame(schema={"m": dtype}).schema == {"m": dtype}


def test_map_entries_expr_and_series() -> None:
    # Deliberately unsorted: entry order is preserved, not normalized.
    s = pl.Series("m", [{"b": 1, "a": 2}, {}, None], dtype=MAP)

    expected = pl.Series(
        "m",
        [[{"key": "b", "value": 1}, {"key": "a", "value": 2}], [], None],
        dtype=ENTRIES,
    )
    assert_series_equal(s.map.entries(), expected)

    df = pl.DataFrame({"m": s})
    assert_series_equal(df.select(pl.col("m").map.entries())["m"], expected)


def arrow_map_retaining_entries(
    keys: list[str], values: list[int], offsets: list[int], valid: list[bool]
) -> pl.Series:
    """Use PyArrow's standard constructor to keep entries under null rows."""
    pa = pytest.importorskip("pyarrow")
    arr = pa.MapArray.from_arrays(
        pa.array(offsets, pa.int32()),
        pa.array(keys, pa.large_string()),
        pa.array(values, pa.int64()),
        mask=pa.array([not v for v in valid]),
    )
    s = pl.from_arrow(arr)
    assert isinstance(s, pl.Series)
    return s.rename("m")


def retaining_null_row_map() -> pl.Series:
    """Rows `null` and `{b: 2, c: 3}`, with entry `a` kept under the null row."""
    return arrow_map_retaining_entries(
        ["a", "b", "c"], [1, 2, 3], [0, 1, 3], [False, True]
    )


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("skip", [0, 1, 3, 9])
@pytest.mark.parametrize("pattern", ["valid", "null", "alternating", "runs"])
def test_map_null_row_export_compaction_validity_runs(
    n: int, skip: int, pattern: str
) -> None:
    pa = pytest.importorskip("pyarrow")
    lengths = [i % 3 for i in range(n + skip)]
    offsets = list(accumulate(lengths, initial=0))
    valid = [
        pattern == "valid"
        or (pattern == "alternating" and i % 2 == 0)
        or (pattern == "runs" and (i // 17) % 2 == 0)
        for i in range(n + skip)
    ]
    keys = [str(i) for i in range(offsets[-1])]
    values = list(range(offsets[-1]))
    arr = pa.MapArray.from_arrays(
        pa.array(offsets, pa.int32()),
        pa.array(keys, pa.large_string()),
        pa.array(values, pa.int64()),
        mask=pa.array([not v for v in valid], pa.bool_()),
    )
    # Slice before import to exercise nonzero bitmap and list offsets.
    result = pl.from_arrow(arr.slice(skip, n))
    assert isinstance(result, pl.Series)
    expected = [
        {keys[j]: values[j] for j in range(offsets[i], offsets[i + 1])}
        if valid[i]
        else None
        for i in range(skip, skip + n)
    ]
    assert result.to_list() == expected
    exported = result.to_arrow()
    exported.validate(full=True)
    # Export compacts and rebases, so the entries are exactly the live ones.
    assert exported.offsets.to_pylist() == list(
        accumulate((len(row) if row is not None else 0 for row in expected), initial=0)
    )
    assert exported.keys.to_pylist() == [key for row in expected if row for key in row]


def test_map_export_compacts_a_null_row_that_spans_entries() -> None:
    s = retaining_null_row_map()
    assert s.to_list() == [None, {"b": 2, "c": 3}]
    # The import left entry `a` where the producer put it; the export drops it.
    assert s.to_arrow().offsets.to_pylist() == [0, 0, 2]
    assert s.to_arrow().values.to_pylist() == [
        {"key": "b", "value": 2},
        {"key": "c", "value": 3},
    ]

    expected = [None, [{"key": "b", "value": 2}, {"key": "c", "value": 3}]]
    for out in (s.map.entries(), s.cast(ENTRIES)):
        assert out.to_list() == expected
        assert out.to_arrow().offsets.to_pylist() == [0, 0, 2]
    assert s.map.entries().list.to_map().to_list() == s.to_list()
    assert s.cast(pl.Map(pl.String, pl.Float64)).to_list() == [
        None,
        {"b": 2.0, "c": 3.0},
    ]


def test_map_null_row_strict_cast_inside_a_container() -> None:
    # Entries a null row hides must not reach a strict-cast validity check.
    s = retaining_null_row_map()
    entries = [None, [{"key": "b", "value": 2}, {"key": "c", "value": 3}]]

    assert s.implode().cast(pl.List(ENTRIES)).to_list() == [entries]
    assert s.to_frame().select(pl.struct("m")).to_series().cast(
        pl.Struct({"m": ENTRIES})
    ).to_list() == [{"m": entries[0]}, {"m": entries[1]}]
    assert s.reshape((1, 2)).cast(pl.Array(ENTRIES, 2)).to_list() == [entries]
    assert s.implode().cast(pl.List(pl.Map(pl.String, pl.Float64))).to_list() == [
        [None, {"b": 2.0, "c": 3.0}]
    ]


def test_map_null_row_strict_cast_reaches_a_nested_map() -> None:
    pa = pytest.importorskip("pyarrow")
    # Inner maps of `{p: null, q: {b: 2, c: 3}}`, with `a` under `p` on input.
    inner = retaining_null_row_map().rename("value").to_arrow()
    keys = pa.array(["p", "q"], pa.large_string())
    outer_entries = pa.StructArray.from_arrays([keys, inner], names=["key", "value"])
    outer_type = pa.map_(pa.field("key", pa.large_string(), nullable=False), inner.type)
    outer = pa.Array.from_buffers(
        outer_type,
        1,
        [None, pa.array([0, 2], pa.int32()).buffers()[1]],
        children=[outer_entries],
    )
    s = pl.from_arrow(outer)
    assert isinstance(s, pl.Series)
    assert s.to_list() == [{"p": None, "q": {"b": 2, "c": 3}}]

    assert s.cast(
        pl.List(pl.Struct({"key": pl.String, "value": ENTRIES}))
    ).to_list() == [
        [
            {"key": "p", "value": None},
            {"key": "q", "value": [{"key": "b", "value": 2}, {"key": "c", "value": 3}]},
        ]
    ]
    assert s.cast(pl.Map(pl.String, pl.Map(pl.String, pl.Float64))).to_list() == [
        {"p": None, "q": {"b": 2.0, "c": 3.0}}
    ]


def test_map_null_row_masking_survives_slicing() -> None:
    # Rows 0 and 2 are null while still spanning entries `a` and `d`.
    s = arrow_map_retaining_entries(
        ["a", "b", "c", "d", "e"],
        [1, 2, 3, 4, 5],
        [0, 1, 3, 4, 5],
        [False, True, False, True],
    )
    assert s.to_list() == [None, {"b": 2, "c": 3}, None, {"e": 5}]

    # Include slices with nonzero offsets.
    for offset, length in [(0, 4), (1, 3), (2, 2), (1, 1)]:
        sliced = s.slice(offset, length)
        expected = [
            None if row is None else [{"key": k, "value": v} for k, v in row.items()]
            for row in sliced.to_list()
        ]
        assert sliced.map.entries().to_list() == expected
        assert sliced.cast(ENTRIES).to_list() == expected


def test_map_dsl_round_trip() -> None:
    s = pl.Series("m", [{"b": 1, "a": 2}, {}, None], dtype=MAP)
    df = pl.DataFrame({"m": s})
    round_tripped = df.select(pl.col("m").map.entries().list.to_map())
    assert_series_equal(round_tripped["m"], s)


def test_list_to_map_rejects_null_keys() -> None:
    s = pl.Series("m", [[{"key": None, "value": 1}]], dtype=ENTRIES)
    with pytest.raises(InvalidOperationError, match="null"):
        s.list.to_map()


def test_map_entries_requires_map_dtype() -> None:
    df = pl.DataFrame({"m": [[1, 2]]})
    with pytest.raises(InvalidOperationError, match=r"`map\.entries` requires a Map"):
        df.select(pl.col("m").map.entries())


# Entry field names are covered by `ENTRY_CASES`; these two shapes are only
# reachable through the DSL, where the target dtype is derived rather than given.
@pytest.mark.parametrize(
    ("data", "dtype", "match"),
    [
        pytest.param([1], pl.Int64, "requires a List dtype", id="not-a-list"),
        pytest.param([[1]], pl.List(pl.Int64), "must be `Struct", id="not-a-struct"),
    ],
)
def test_list_to_map_invalid_input(
    data: list[Any], dtype: pl.DataType, match: str
) -> None:
    df = pl.DataFrame({"m": pl.Series(data, dtype=dtype)})
    with pytest.raises(InvalidOperationError, match=match):
        df.select(pl.col("m").list.to_map())


def test_map_dsl_resolves_schema_without_data() -> None:
    lf = pl.LazyFrame(schema={"m": MAP})
    assert lf.select(pl.col("m").map.entries()).collect_schema() == {"m": ENTRIES}

    lf = pl.LazyFrame(schema={"m": ENTRIES})
    assert lf.select(pl.col("m").list.to_map()).collect_schema() == {"m": MAP}


def test_map_dsl_on_nested_value_dtype() -> None:
    dtype = pl.Map(pl.String, pl.Struct({"x": pl.Int64}))
    s = pl.Series("m", [{"a": {"x": 1}}], dtype=dtype)

    entries = s.map.entries()
    assert entries.dtype == pl.List(
        pl.Struct({"key": pl.String, "value": pl.Struct({"x": pl.Int64})})
    )
    assert_series_equal(entries.list.to_map(), s)


def test_map_float_keys_are_canonicalized() -> None:
    # Row encoding has a single zero and a single NaN, so these keys collapse.
    entries = pl.List(pl.Struct({"key": pl.Float64, "value": pl.Int64}))
    dtype = pl.Map(pl.Float64, pl.Int64)

    signed_zeros = pl.Series(
        "m", [[{"key": -0.0, "value": 1}, {"key": 0.0, "value": 2}]], dtype=entries
    )
    assert signed_zeros.list.to_map().to_list() == [{-0.0: 2}]

    # A Python dict can hold two NaN keys, since NaN is not equal to itself.
    nan = float("nan")
    row = {nan: 1, float("nan"): 2}
    assert len(row) == 2
    (collapsed,) = pl.Series("m", [row], dtype=dtype).to_list()
    (key,) = collapsed
    assert math.isnan(key)
    assert collapsed[key] == 2

    # Grouping agrees with construction.
    df = pl.DataFrame({"m": pl.Series([row, {nan: 2}], dtype=dtype)})
    assert df.group_by("m").len()["len"].to_list() == [2]


def test_map_equality_is_entry_order_sensitive() -> None:
    ordered = pl.Series("m", [{"a": 1, "b": 2}], dtype=MAP)
    swapped = pl.Series("m", [{"b": 2, "a": 1}], dtype=MAP)

    assert not ordered.equals(swapped)
    assert pl.select(eq=pl.lit(ordered) == pl.lit(swapped))["eq"].to_list() == [False]

    df = pl.DataFrame({"m": pl.concat([ordered, swapped])})
    assert df["m"].n_unique() == 2
    assert df.group_by("m").len().height == 2


def test_map_join_on_map_keys() -> None:
    left = pl.DataFrame({"m": pl.Series([{"a": 1}, {"b": 2}], dtype=MAP), "l": [1, 2]})
    right = pl.DataFrame(
        {"m": pl.Series([{"b": 2}, {"a": 1}], dtype=MAP), "r": [10, 20]}
    )

    out = left.join(right, on="m", how="inner").sort("l")
    assert out.to_dicts() == [
        {"m": {"a": 1}, "l": 1, "r": 20},
        {"m": {"b": 2}, "l": 2, "r": 10},
    ]


def test_map_join_on_map_keys_is_entry_order_sensitive() -> None:
    left = pl.DataFrame({"m": pl.Series([{"a": 1, "b": 2}], dtype=MAP)})
    right = pl.DataFrame({"m": pl.Series([{"b": 2, "a": 1}], dtype=MAP), "r": [9]})
    assert left.join(right, on="m", how="inner").height == 0


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(lambda lf: lf, [{"a": 2}, {"b": 3}], id="plain"),
        pytest.param(lambda lf: lf.select("m"), [{"a": 2}, {"b": 3}], id="projection"),
        pytest.param(lambda lf: lf.filter(pl.col("i") == 1), [{"a": 2}], id="filter"),
        pytest.param(
            lambda lf: lf.filter(pl.col("i") == 2).select("m"),
            [{"b": 3}],
            id="filter-and-projection",
        ),
    ],
)
def test_map_parquet_duplicate_keys_recovered_in_every_read_path(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], expected: list[Any]
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    # pyarrow permits duplicate keys; we canonicalize on read.
    tbl = pa.table(
        {
            "m": pa.array(
                [[("a", 1), ("a", 2)], [("b", 3)]],
                type=pa.map_(pa.string(), pa.int64()),
            ),
            "i": pa.array([1, 2]),
        }
    )
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    buf.seek(0)

    out = query(pl.scan_parquet(buf)).collect()["m"]
    assert out.dtype == MAP
    assert out.to_list() == expected


def test_map_parquet_duplicate_keys_recovered_when_nested() -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    tbl = pa.table(
        {
            "n": pa.array(
                [[[("a", 7), ("a", 8)]], [[("c", 9)]]],
                type=pa.list_(pa.map_(pa.string(), pa.int64())),
            )
        }
    )
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    buf.seek(0)

    out = pl.scan_parquet(buf).collect()["n"]
    assert out.dtype == pl.List(MAP)
    assert out.to_list() == [[{"a": 8}], [{"c": 9}]]


@pytest.mark.parametrize("first", [{"a": None}, {}, None])
def test_map_construction_tolerates_uninformative_first_row(first: Any) -> None:
    # A dict never *infers* as a Map, so the dtype is always known here; a first row
    # that carries no value dtype must not narrow it.
    s = pl.Series("m", [first, {"a": 1}], dtype=MAP)
    assert s.dtype == MAP
    assert s.to_list() == [first, {"a": 1}]


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        pytest.param(
            pl.Map(pl.String, pl.Datetime("us")),
            {"a": datetime(2020, 1, 1)},
            id="datetime-value",
        ),
        pytest.param(
            pl.Map(pl.String, pl.Date), {"a": date(2020, 1, 1)}, id="date-value"
        ),
        pytest.param(
            pl.Map(pl.String, pl.Duration("ms")),
            {"a": timedelta(days=1)},
            id="duration-value",
        ),
        pytest.param(
            pl.Map(pl.String, pl.Decimal(10, 2)),
            {"a": Decimal("1.50")},
            id="decimal-value",
        ),
        pytest.param(
            pl.Map(pl.String, pl.Categorical), {"a": "x"}, id="categorical-value"
        ),
        pytest.param(pl.Map(pl.String, pl.Enum(["x"])), {"a": "x"}, id="enum-value"),
        pytest.param(
            pl.Map(pl.Datetime("us"), pl.Int64),
            {datetime(2020, 1, 1): 1},
            id="datetime-key",
        ),
        pytest.param(pl.Map(pl.Categorical, pl.Int64), {"x": 1}, id="categorical-key"),
    ],
)
def test_map_group_by_agg_list_keeps_logical_children(
    dtype: pl.Map, value: dict[Any, Any]
) -> None:
    # `Map::to_physical` recurses into the children, so the aggregated list must not be
    # relabelled as if the storage were fully physical.
    s = pl.Series("m", [value, None], dtype=dtype)
    df = pl.DataFrame({"g": [1, 1], "m": s})

    out = df.group_by("g").agg("m")
    assert out["m"].dtype == pl.List(dtype)
    assert out["m"].to_list() == [[value, None]]


def test_map_shift_with_fill_value() -> None:
    s = pl.Series("m", [{"a": 1}, {"b": 2}, {"c": 3}], dtype=MAP)
    df = pl.DataFrame({"m": s})

    out = df.select(pl.col("m").shift(1, fill_value=pl.col("m").last()))["m"]
    assert_series_equal(out, pl.Series("m", [{"c": 3}, {"a": 1}, {"b": 2}], dtype=MAP))

    out = df.select(pl.col("m").shift(-1, fill_value=pl.col("m").first()))["m"]
    assert_series_equal(out, pl.Series("m", [{"b": 2}, {"c": 3}, {"a": 1}], dtype=MAP))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(pl.Map(pl.String, pl.Object), id="object-value"),
        pytest.param(pl.Map(pl.String, pl.List(pl.Object)), id="nested-object-value"),
    ],
)
def test_map_object_value_dtype_is_rejected(dtype: pl.Map) -> None:
    # A `Struct` cannot hold objects, and map entries are a struct.
    for build in (
        lambda: pl.Series("m", [], dtype=dtype),
        lambda: pl.Series("m", [None], dtype=dtype),
        lambda: pl.DataFrame(schema={"m": dtype}),
        lambda: pl.select(pl.lit(None).cast(dtype)),
    ):
        with pytest.raises(InvalidOperationError, match="Map value dtype"):
            build()


def test_bare_map_class_is_not_a_dtype() -> None:
    # `Map(Null, _)` is not a valid dtype, so there is no bare stand-in the way `List`
    # has `List(Null)`.
    for build in (
        lambda: pl.Series("m", [None], dtype=pl.Map),
        lambda: pl.DataFrame(schema={"m": pl.Map}),
        lambda: pl.select(pl.lit(None).cast(pl.Map)),
    ):
        with pytest.raises(TypeError, match="Map requires a key and a value type"):
            build()


def test_bare_map_class_selects_every_map_column() -> None:
    df = pl.DataFrame(
        {
            "m": pl.Series([{"a": 1}], dtype=MAP),
            "n": pl.Series([{1: "a"}], dtype=pl.Map(pl.Int64, pl.String)),
            "l": pl.Series([[1]], dtype=pl.List(pl.Int64)),
        }
    )

    assert df.select(pl.col(pl.Map)).columns == ["m", "n"]
    assert df.select(cs.map()).columns == ["m", "n"]
    assert df.select(cs.by_dtype(pl.Map)).columns == ["m", "n"]
    assert df.select(~cs.map()).columns == ["l"]

    # A parametrized Map still matches exactly.
    assert df.select(pl.col(MAP)).columns == ["m"]


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda df: df.write_json(), id="json"),
        pytest.param(lambda df: df.write_ndjson(), id="ndjson"),
    ],
)
def test_map_json_write_errors(write: Callable[[pl.DataFrame], str]) -> None:
    df = pl.DataFrame({"m": pl.Series([{"a": 1}], dtype=MAP)})
    with pytest.raises(ComputeError, match="cannot write 'Map' datatype to json"):
        write(df)

    nested = pl.DataFrame({"m": pl.Series([[{"a": 1}]], dtype=pl.List(MAP))})
    with pytest.raises(ComputeError, match="cannot write 'Map' datatype to json"):
        write(nested)


def test_map_json_read_errors() -> None:
    with pytest.raises(ComputeError):
        pl.read_json(io.BytesIO(b'[{"m": {"a": 1}}]'), schema={"m": MAP})
    with pytest.raises(ComputeError):
        pl.read_ndjson(io.BytesIO(b'{"m": {"a": 1}}\n'), schema={"m": MAP})


def test_map_is_first_and_last_distinct() -> None:
    s = pl.Series("m", [{"a": 1}, {"b": 2}, {"a": 1}, None, None], dtype=MAP)
    assert s.is_first_distinct().to_list() == [True, True, False, True, False]
    assert s.is_last_distinct().to_list() == [False, True, True, False, True]

    # Nested keys and values are row-encoded, unlike the plain `List` path.
    nested = (
        pl.Series(
            "m",
            [[{"key": [1], "value": {"x": 1}}], [{"key": [1], "value": {"x": 1}}]],
            dtype=pl.List(
                pl.Struct(
                    {"key": pl.List(pl.Int64), "value": pl.Struct({"x": pl.Int64})}
                )
            ),
        )
        .to_frame()
        .select(pl.first().list.to_map())["m"]
    )
    assert nested.is_first_distinct().to_list() == [True, False]
    assert nested.is_last_distinct().to_list() == [False, True]


def test_map_inequality_comparison_names_the_map_dtype() -> None:
    s = pl.Series("m", [{"a": 1}], dtype=MAP)
    with pytest.raises(InvalidOperationError, match=r"dtype: map\[str, i64\]"):
        pl.select(pl.lit(s) < pl.lit(s))


def test_map_sort_by_multiple_keys() -> None:
    s = pl.Series("m", [{"b": 2}, {"a": 1}, {"a": 1}], dtype=MAP)
    df = pl.DataFrame({"m": s, "x": [1, 2, 3], "y": [3, 4, 2]})

    # A Map has no ordering to compare row by row, so a multi-key sort goes through the
    # row encoding and must agree with `DataFrame.sort`.
    assert df.sort("m", "y")["x"].to_list() == [3, 2, 1]
    assert df.select(pl.col("x").sort_by("m", "y"))["x"].to_list() == [3, 2, 1]
    assert df.select(pl.col("x").sort_by("y", "m"))["x"].to_list() == [3, 1, 2]
    assert df.select(pl.col("x").sort_by("m", "y", descending=[True, False]))[
        "x"
    ].to_list() == [1, 3, 2]


def test_map_sort_by_multiple_keys_in_group_by() -> None:
    s = pl.Series("m", [{"b": 2}, {"a": 1}, {"a": 1}], dtype=MAP)
    df = pl.DataFrame({"g": [1, 1, 1], "m": s, "x": [1, 2, 3], "y": [3, 4, 2]})
    out = df.group_by("g").agg(pl.col("x").sort_by("m", "y"))
    assert out["x"].to_list() == [[3, 2, 1]]


def test_map_to_numpy_and_rows() -> None:
    s = pl.Series("m", [{"a": 1}, None], dtype=MAP)
    assert s.to_numpy().tolist() == [{"a": 1}, None]
    assert s.to_frame().to_numpy().tolist() == [[{"a": 1}], [None]]
    assert s.to_frame().rows() == [({"a": 1},), (None,)]
    assert s.to_frame().row(0) == ({"a": 1},)
    assert s.to_frame().to_dicts() == [{"m": {"a": 1}}, {"m": None}]


@pytest.mark.parametrize(
    "convert",
    [
        lambda s: s.to_list(),
        lambda s: s.to_numpy(),
        lambda s: s.to_frame().to_numpy(),
        lambda s: pl.select(pl.lit(s).implode()).to_series().to_numpy(),
        lambda s: pl.select(pl.struct(pl.lit(s))).to_series().to_numpy(),
        lambda s: s.to_frame().rows(),
        lambda s: s.to_frame().row(0),
        lambda s: s.to_frame().to_dicts(),
        lambda s: s.to_frame().map_rows(lambda row: row),
        lambda s: s.to_frame().map_rows(lambda row: row, return_dtype=pl.Int64),
    ],
)
def test_map_nested_key_conversion_is_an_error_not_a_panic(
    convert: Callable[[pl.Series], Any],
) -> None:
    entries = pl.Series("m", [[{"key": [1, 2], "value": "x"}]])
    s = entries.cast(pl.Map(pl.List(pl.Int64), pl.String))
    # A nested key is not hashable, so it has no Python dict equivalent. That is a
    # supported error case, so it must not turn into a panic.
    with pytest.raises(TypeError, match="not hashable"):
        convert(s)


def _map_with_null_row_keeping_its_entries() -> pl.Series:
    """An Arrow Map whose null row still spans an entry on input."""
    pa = pytest.importorskip("pyarrow")

    keys = pa.array([Decimal("1.50"), Decimal("2.50")], type=pa.decimal128(10, 2))
    values = pa.array([1, 2], type=pa.int64())
    # The mask nulls row 0 while the offsets keep pointing at its entry.
    arr = pa.MapArray.from_arrays(
        pa.array([0, 1, 2], type=pa.int32()),
        keys,
        values,
        mask=pa.array([True, False]),
    )
    return pl.from_arrow(pa.table({"m": arr}))["m"]  # type: ignore[index]


DEC_MAP = pl.Map(pl.Decimal(10, 2), pl.Int64)
RESCALED_MAP = pl.Map(pl.Decimal(12, 3), pl.Int64)
NULL_ROW = None
LIVE_ROW = {Decimal("2.500"): 2}


def test_map_null_row_keeping_its_entries() -> None:
    # Arrow may keep entries under null rows; polars keeps them too and hides them.
    s = _map_with_null_row_keeping_its_entries()
    assert s.dtype == DEC_MAP
    assert s.to_arrow().offsets.to_pylist() == [0, 0, 1]
    assert s.to_list() == [None, {Decimal("2.50"): 2}]

    # Rescaling a Decimal key is the cast that revalidates the entries.
    assert s.cast(RESCALED_MAP).to_list() == [None, LIVE_ROW]

    # Value casts remain exportable.
    widened = s.cast(pl.Map(pl.Decimal(10, 2), pl.Float64))
    assert widened.to_list() == [None, {Decimal("2.50"): 2.0}]
    assert widened.to_arrow().values.null_count == 0

    # Row encoding also propagates first.
    assert s.to_frame().group_by("m").len().height == 2
    assert s.to_frame().sort("m")["m"].to_list() == [None, {Decimal("2.50"): 2}]


def test_map_sliced_null_row_leaves_nothing_reachable() -> None:
    # The null row's entry is hidden, so slicing the row away exposes nothing.
    sliced = _map_with_null_row_keeping_its_entries().slice(1, 1)
    assert sliced.to_list() == [{Decimal("2.50"): 2}]

    exported = sliced.cast(pl.Map(pl.Decimal(10, 2), pl.Float64)).to_arrow()
    assert exported.offsets.to_pylist() == [0, 1]
    assert exported.keys.to_pylist() == [Decimal("2.50")]
    assert sliced.cast(RESCALED_MAP).to_list() == [LIVE_ROW]


def test_map_slicing_leaves_live_entries_outside_the_window() -> None:
    # Slicing past a live row may leave its entries outside the offsets, as with List.
    s = pl.Series("m", [{Decimal("1.50"): 1}, {Decimal("2.50"): 2}], dtype=DEC_MAP)
    sliced = s.slice(1, 1)
    assert sliced.to_list() == [{Decimal("2.50"): 2}]

    for out in (
        sliced.cast(pl.Map(pl.Decimal(10, 2), pl.Float64)),
        sliced.cast(RESCALED_MAP),
    ):
        exported = out.to_arrow()
        assert exported.offsets.to_pylist() == [0, 1]
        assert exported.keys.to_pylist() == [Decimal("2.50")]
    assert sliced.map.entries().to_arrow().offsets.to_pylist() == [0, 1]


def masked_null_row_map() -> pl.Series:
    """Rows `{a: 1, b: 2}`, null, `{d: 4, e: 5}`, with `c` hidden under the null row."""
    s = pl.Series("m", [{"a": 1, "b": 2}, {"c": 3}, {"d": 4, "e": 5}], dtype=MAP)
    df = pl.DataFrame({"m": s, "keep": [True, False, True]})
    return df.select(
        pl.when(pl.col("keep")).then(pl.col("m")).otherwise(None)
    ).to_series()


def test_map_flat_accessors_skip_entries_hidden_by_a_null_row() -> None:
    s = masked_null_row_map()
    assert s.to_list() == [{"a": 1, "b": 2}, None, {"d": 4, "e": 5}]

    # `entries` and the `List(Struct)` cast both expose live entries only.
    expected = [
        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
        None,
        [{"key": "d", "value": 4}, {"key": "e", "value": 5}],
    ]
    for out in (s.map.entries(), s.cast(ENTRIES)):
        assert out.to_list() == expected
        assert out.to_arrow().offsets.to_pylist() == [0, 2, 2, 4]

    # A value cast pairs `values()` with `with_values()`, so both must skip `c`.
    widened = s.cast(FLOAT_MAP)
    assert widened.to_list() == [{"a": 1.0, "b": 2.0}, None, {"d": 4.0, "e": 5.0}]
    assert widened.to_arrow().items.to_pylist() == [1.0, 2.0, 4.0, 5.0]


def test_map_strict_cast_ignores_values_hidden_by_a_null_row() -> None:
    pa = pytest.importorskip("pyarrow")
    # Only the hidden entry's value is unparsable, so a strict cast of it would fail.
    keys = pa.array(["a", "b"], pa.large_string())
    values = pa.array(["xyz", "7"], pa.large_string())
    mask = pa.array([True, False])
    arr = pa.MapArray.from_arrays(
        pa.array([0, 1, 2], pa.int32()), keys, values, mask=mask
    )
    s = pl.from_arrow(arr)
    assert isinstance(s, pl.Series)
    assert s.dtype == pl.Map(pl.String, pl.String)
    assert s.cast(MAP).to_list() == [None, {"b": 7}]

    # The same payload as a `List(Struct)`, where propagation nulls the hidden entry.
    entries = pa.StructArray.from_arrays([keys, values], names=["key", "value"])
    lst = pa.ListArray.from_arrays(pa.array([0, 1, 2], pa.int32()), entries, mask=mask)
    from_list = pl.from_arrow(lst)
    assert isinstance(from_list, pl.Series)
    assert from_list.cast(MAP).to_list() == [None, {"b": 7}]


def test_map_sliced_export_rebases_offsets() -> None:
    pytest.importorskip("pyarrow")
    rows = [{"a": 1}, {"b": 2, "c": 3}, {"d": 4}]
    s = pl.Series("m", rows, dtype=MAP)

    for offset, length in [(0, 3), (1, 2), (2, 1), (1, 1), (3, 0)]:
        exported = s.slice(offset, length).to_arrow()
        exported.validate(full=True)
        assert exported.offsets.to_pylist()[0] == 0
        assert exported.to_pylist() == [
            list(row.items()) for row in rows[offset : offset + length]
        ]


@pytest.mark.parametrize("container", ["list", "struct"])
def test_map_hidden_entries_under_a_nulled_container_row_export_valid_arrow(
    container: str,
) -> None:
    pytest.importorskip("pyarrow")
    # Nulling the outer row propagates nulls into the hidden Map entries, which the
    # export must drop rather than hand to Arrow's non-nullable entries field.
    df = pl.DataFrame({"m": masked_null_row_map().cast(FLOAT_MAP)})
    if container == "list":
        nested = _outer_nulled_container(df["m"].head(2))
        dtype: pl.DataType = pl.List(FLOAT_MAP)
        expected: list[Any] = [None, [None]]
    else:
        nested = df.select(
            pl.when(pl.Series([False, True, True])).then(pl.struct("m")).otherwise(None)
        ).to_series()
        dtype = pl.Struct({"m": FLOAT_MAP})
        expected = [None, {"m": None}, {"m": [("d", 4.0), ("e", 5.0)]}]

    exported = nested.cast(dtype).to_arrow()
    exported.validate(full=True)
    assert exported.to_pylist() == expected
    inner = exported.field("m") if container == "struct" else exported.values
    assert inner.values.null_count == 0
    assert inner.keys.null_count == 0


def test_map_null_rows_written_in_place_are_compacted_on_export() -> None:
    # Several paths null a Map row without touching its offsets. These are the ones an
    # expression can reach; `deposit` and the empty-group aggregation of a scalar
    # column are covered by the Rust tests in `logical::map`.
    s = pl.Series("m", [{"a": 1, "b": 2}, {"c": 3}, {"d": 4, "e": 5}], dtype=MAP)
    df = pl.DataFrame({"m": s, "keep": [True, False, True]})

    masked = df.select(
        pl.when(pl.col("keep")).then(pl.col("m")).otherwise(None)
    ).to_series()
    assert masked.to_list() == [{"a": 1, "b": 2}, None, {"d": 4, "e": 5}]
    assert masked.to_arrow().offsets.to_pylist() == [0, 2, 2, 4]

    shifted = s.shift(1)
    assert shifted.to_list() == [None, {"a": 1, "b": 2}, {"c": 3}]
    assert shifted.to_arrow().offsets.to_pylist() == [0, 0, 2, 3]

    # A broadcast scalar gathered with null indices nulls rows of the materialized copy.
    gathered = pl.DataFrame({"i": [0, None, 0, None]}).select(
        pl.lit(s.slice(0, 1)).first().gather(pl.col("i"))
    )["m"]
    assert gathered.to_list() == [{"a": 1, "b": 2}, None, {"a": 1, "b": 2}, None]
    assert gathered.to_arrow().offsets.to_pylist() == [0, 2, 2, 4, 4]


def test_map_null_rows_in_a_container_are_compacted_on_export() -> None:
    s = pl.Series("m", [{"a": 1.0, "b": 2.0}, {"c": 3.0}], dtype=FLOAT_MAP)
    df = pl.DataFrame({"m": s, "keep": [False, True]})

    # `to_arrow` alone does not propagate nulls into the container's child; the cast
    # does.
    nulled = df.select(
        pl.when(pl.col("keep")).then(pl.col("m")).otherwise(None).implode()
    ).to_series()
    exported = nulled.cast(pl.List(FLOAT_MAP)).to_arrow()
    assert exported.values.offsets.to_pylist() == [0, 0, 1]
    assert exported.to_pylist() == [[None, [("c", 3.0)]]]

    # Nulling the outer `List` row must reach the Map rows it spans.
    lists = (
        df.group_by("keep", maintain_order=True)
        .agg("m")
        .with_columns(outer_keep=pl.Series([False, True]))
    )
    outer_nulled = lists.select(
        pl.when(pl.col("outer_keep")).then(pl.col("m")).otherwise(None)
    ).to_series()
    exported = outer_nulled.cast(pl.List(FLOAT_MAP)).to_arrow()
    assert exported.to_pylist() == [None, [[("c", 3.0)]]]
    assert exported.values.offsets.to_pylist() == [0, 0, 1]

    struct_nulled = df.select(
        pl.when(pl.col("keep")).then(pl.struct("m")).otherwise(None)
    ).to_series()
    exported = struct_nulled.cast(pl.Struct({"m": FLOAT_MAP})).to_arrow()
    assert exported.to_pylist() == [None, {"m": [("c", 3.0)]}]
    assert exported.field("m").offsets.to_pylist() == [0, 0, 1]

    # Two array rows, so the nulled row is not the whole column.
    wide = pl.Series(
        "m",
        [{"a": 1.0, "b": 2.0}, {"c": 3.0}, {"d": 4.0}, {"e": 5.0}],
        dtype=FLOAT_MAP,
    )
    arrays = (
        pl.DataFrame({"m": wide, "g": [1, 1, 2, 2]})
        .group_by("g", maintain_order=True)
        .agg("m")
        .with_columns(
            pl.col("m").cast(pl.Array(FLOAT_MAP, 2)), keep=pl.Series([False, True])
        )
    )
    array_nulled = arrays.select(
        pl.when(pl.col("keep")).then(pl.col("m")).otherwise(None)
    ).to_series()
    exported = array_nulled.cast(pl.Array(FLOAT_MAP, 2)).to_arrow()
    assert exported.to_pylist() == [None, [[("d", 4.0)], [("e", 5.0)]]]
    assert exported.values.offsets.to_pylist() == [0, 0, 0, 1, 2]


def _outer_nulled_container(inner: pl.Series) -> pl.Series:
    """Group `inner` into a two-row `List` and null the first row."""
    lists = (
        pl.DataFrame({"c": inner, "g": [1, 2]})
        .group_by("g", maintain_order=True)
        .agg("c")
        .with_columns(keep=pl.Series([False, True]))
    )
    return lists.select(
        pl.when(pl.col("keep")).then(pl.col("c")).otherwise(None)
    ).to_series()


def test_map_nested_container_is_compact_in_the_physical_tree() -> None:
    # Inspect physical descendants: reconstructing children could mask lost repairs.
    s = pl.Series("m", [{"a": 1.0, "b": 2.0}, {"c": 3.0}], dtype=FLOAT_MAP)
    # Rows `{}` and `{c: 3, d: 4}`: nulling row 0 leaves this second field alone.
    other = pl.Series("other", [{}, {"c": 3.0, "d": 4.0}], dtype=FLOAT_MAP)
    structs = pl.DataFrame({"m": s, "other": other}).select(pl.struct("m", "other"))[
        "m"
    ]

    exported = (
        _outer_nulled_container(structs)
        .cast(pl.List(pl.Struct({"m": FLOAT_MAP, "other": FLOAT_MAP})))
        .to_arrow()
    )
    assert exported.to_pylist() == [
        None,
        [{"m": [("c", 3.0)], "other": [("c", 3.0), ("d", 4.0)]}],
    ]
    assert exported.values.field("m").offsets.to_pylist() == [0, 0, 1]
    assert exported.values.field("other").offsets.to_pylist() == [0, 0, 2]


@pytest.mark.parametrize("container", ["list", "array", "struct"])
@pytest.mark.parametrize("cast_to_entries", [False, True], ids=["identity", "entries"])
def test_map_propagation_keeps_repaired_entry_children(
    container: str, cast_to_entries: bool
) -> None:
    pa = pytest.importorskip("pyarrow")
    inner = pa.MapArray.from_arrays([0, 2, 3], ["a", "b", "c"], [1, 2, 3])
    mask = pa.array([True, False])
    value_dtype: pl.DataType
    if container == "struct":
        values = pa.StructArray.from_arrays([inner], names=["m"], mask=mask)
        value_dtype = pl.Struct({"m": ENTRIES})
    elif container == "array":
        values = pa.FixedSizeListArray.from_arrays(inner, 1, mask=mask)
        value_dtype = pl.Array(ENTRIES, 1)
    else:
        values = pa.ListArray.from_arrays([0, 1, 2], inner, mask=mask)
        value_dtype = pl.List(ENTRIES)

    s = pl.from_arrow(pa.MapArray.from_arrays([0, 2], ["x", "y"], values))
    assert isinstance(s, pl.Series)
    target = (
        pl.List(pl.Struct({"key": pl.String, "value": value_dtype}))
        if cast_to_entries
        else s.dtype
    )
    result = s.cast(target)
    if not cast_to_entries:
        assert result.to_list() == s.to_list()

    # Inspect the physical tree without reconstructing children, which could repair
    # them and hide a discarded propagation result.
    exported_values = result.to_arrow().values.field("value")
    exported_inner = (
        exported_values.field("m") if container == "struct" else exported_values.values
    )
    assert exported_inner.is_null().to_pylist() == [True, False]
    assert exported_inner.offsets.to_pylist() == [0, 0, 1]
    assert exported_inner.values.to_pylist() == [{"key": "c", "value": 3}]


@pytest.mark.parametrize("wrap_in_struct", [False, True], ids=["map", "struct-of-map"])
def test_map_extension_container_null_rows_are_compacted_on_export(
    wrap_in_struct: bool,
) -> None:
    # `Extension` forwards propagation to its storage, so a Map under an extension
    # must be exported the same way.
    rows: list[Any] = [{"a": 1.0, "b": 2.0}, {"c": 3.0}]
    storage: pl.DataType = FLOAT_MAP
    live = [("c", 3.0)]
    if wrap_in_struct:
        rows = [{"m": row} for row in rows]
        storage = pl.Struct({"m": FLOAT_MAP})
        live = {"m": live}  # type: ignore[assignment]
    ext = pl.Extension(name=MAP_EXTENSION_NAME, storage=storage)

    exported = (
        _outer_nulled_container(pl.Series("c", rows, dtype=ext))
        .cast(pl.List(ext))
        .to_arrow()
    )
    assert exported.to_pylist() == [None, [live]]
    inner = exported.values.field("m") if wrap_in_struct else exported.values
    assert inner.offsets.to_pylist() == [0, 0, 1]


def test_map_sort_by_categorical_keys() -> None:
    # Ensure that categories are sorted by their values, not by their codes.
    cats = pl.Categories("test_map_sort_by_categorical_keys")
    dtype = pl.Map(pl.Categorical(cats), pl.Int64)
    s = pl.Series("m", [{"b": 1}, {"a": 1}, {"a": 1}], dtype=dtype)
    df = pl.DataFrame({"m": s, "x": [1, 2, 3], "y": [3, 4, 2]})

    assert df.sort("m", "y")["x"].to_list() == [3, 2, 1]
    assert df.select(pl.col("x").sort_by("m", "y"))["x"].to_list() == [3, 2, 1]

    grouped = df.with_columns(g=1).group_by("g").agg(pl.col("x").sort_by("m", "y"))
    assert grouped["x"].to_list() == [[3, 2, 1]]


@pytest.mark.parametrize(
    ("nest", "rescaled_dtype", "expected"),
    [
        pytest.param(
            lambda s: s.implode(),
            pl.List(RESCALED_MAP),
            [[NULL_ROW, LIVE_ROW]],
            id="list",
        ),
        pytest.param(
            lambda s: s.implode().cast(pl.Array(DEC_MAP, 2)),
            pl.Array(RESCALED_MAP, 2),
            [[NULL_ROW, LIVE_ROW]],
            id="array",
        ),
        pytest.param(
            lambda s: pl.select(pl.struct(pl.lit(s).alias("m"))).to_series(),
            pl.Struct({"m": RESCALED_MAP}),
            [{"m": NULL_ROW}, {"m": LIVE_ROW}],
            id="struct",
        ),
        pytest.param(
            lambda s: pl.select(pl.struct(pl.lit(s).alias("m"))).to_series().implode(),
            pl.List(pl.Struct({"m": RESCALED_MAP})),
            [[{"m": NULL_ROW}, {"m": LIVE_ROW}]],
            id="list-of-struct",
        ),
        pytest.param(
            lambda s: s.implode().implode(),
            pl.List(pl.List(RESCALED_MAP)),
            [[[NULL_ROW, LIVE_ROW]]],
            id="list-of-list",
        ),
    ],
)
def test_map_null_rows_survive_nesting(
    nest: Callable[[pl.Series], pl.Series],
    rescaled_dtype: pl.DataType,
    expected: list[Any],
) -> None:
    # The key rescaling below revalidates the entries, so it must see only the live
    # ones -- nested null propagation nulls the rest.
    nested = nest(_map_with_null_row_keeping_its_entries())
    assert nested.cast(rescaled_dtype).to_list() == expected


def test_map_null_entry_of_a_live_row_is_rejected() -> None:
    # Only null entries or keys that no live row owns may be dropped.
    entries = pl.Series("m", [[None, {"key": "a", "value": 1}]], dtype=ENTRIES)
    with pytest.raises(InvalidOperationError, match="Map entries cannot be null"):
        entries.cast(MAP)

    keys = pl.Series("m", [[{"key": None, "value": 1}]], dtype=ENTRIES)
    with pytest.raises(InvalidOperationError, match="Map keys cannot be null"):
        keys.cast(MAP)


def _list_of_entries_with_null_row(keys: list[str | None]) -> pl.Series:
    """A `List(Struct)` whose first, null row still spans an entry."""
    pa = pytest.importorskip("pyarrow")
    # `pa.MapArray.from_arrays` rejects null keys, so build the list by hand.
    entries = pa.StructArray.from_arrays(
        [pa.array(keys, type=pa.string()), pa.array([1, 2], type=pa.int64())],
        ["key", "value"],
    )
    arr = pa.ListArray.from_arrays(
        pa.array([0, 1, 2], type=pa.int32()),
        entries,
        mask=pa.array([True, False]),
    )
    return pl.from_arrow(pa.table({"m": arr}))["m"]  # type: ignore[index]


@pytest.mark.parametrize("keys", [[None, "b"], ["a", "b"]], ids=["invalid", "valid"])
def test_map_hidden_entry_is_compacted_on_export(keys: list[str | None]) -> None:
    # `List(Struct)` null propagation may null the entry under a null row. The Map keeps
    # it hidden; only the export must not carry a null entry or key into Arrow.
    s = _list_of_entries_with_null_row(keys).cast(MAP)
    exported = s.to_arrow()
    assert exported.offsets.to_pylist() == [0, 0, 1]
    assert exported.values.null_count == 0
    assert exported.keys.null_count == 0
    assert s.to_list() == [None, {"b": 2}]


def test_map_ipc_round_trip_keeps_duplicate_keys_with_valid_storage() -> None:
    pa = pytest.importorskip("pyarrow")
    tbl = pa.table(
        {"m": pa.array([[("a", 1), ("a", 2)]], type=pa.map_(pa.string(), pa.int64()))}
    )
    s = pl.from_arrow(tbl)["m"]  # type: ignore[index]
    out = pl.read_ipc(_ipc_buffer(s))["m"]
    # IPC import trusts key uniqueness while validating storage.
    assert out.cast(ENTRIES).to_list() == [
        [{"key": "a", "value": 1}, {"key": "a", "value": 2}]
    ]
    exported = out.to_arrow()
    assert exported.keys.null_count == 0
    assert exported.offsets.to_pylist() == [0, 2]


def test_map_decimal_rescale_collapsing_keys_deduplicates() -> None:
    # A key-changing cast re-establishes key uniqueness.
    s = pl.Series(
        "m",
        [{Decimal("1.50"): 1, Decimal("1.54"): 2}],
        pl.Map(pl.Decimal(10, 2), pl.Int64),
    )
    rescaled = s.cast(pl.Map(pl.Decimal(10, 1), pl.Int64))
    assert rescaled.to_list() == [{Decimal("1.5"): 2}]
    assert rescaled.to_arrow().offsets.to_pylist() == [0, 1]


SLICED_MAP_CASES = [
    pytest.param(
        MAP,
        pl.Map(pl.String, pl.Float64),
        [{"a": 1}, {"b": 2, "c": 3}, {"d": 4}],
        [{"b": 2.0, "c": 3.0}, {"d": 4.0}],
        id="flat",
    ),
    pytest.param(
        pl.Map(pl.String, pl.Map(pl.String, pl.Int64)),
        pl.Map(pl.String, pl.Map(pl.String, pl.Float64)),
        [{"a": {"x": 1}}, {"b": {"y": 2}, "c": {}}, {"d": {"z": 3}}],
        [{"b": {"y": 2.0}, "c": {}}, {"d": {"z": 3.0}}],
        id="map-in-map",
    ),
    pytest.param(
        pl.Map(pl.String, pl.List(pl.Int64)),
        pl.Map(pl.String, pl.List(pl.Float64)),
        [{"a": [1]}, {"b": [2, 3], "c": []}, {"d": [4]}],
        [{"b": [2.0, 3.0], "c": []}, {"d": [4.0]}],
        id="list-in-map",
    ),
]


@pytest.mark.parametrize(("dtype", "target", "rows", "expected"), SLICED_MAP_CASES)
def test_map_sliced_and_chunked_value_casts_use_the_offset_window(
    dtype: pl.Map,
    target: pl.Map,
    rows: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    # Slicing keeps the entries of the dropped row in the child, so replacing or casting
    # the values has to window the entries the same way the offsets do. Nested children
    # keep their own offsets, which only full normalization rebases.
    s = pl.Series("m", rows, dtype=dtype)
    sliced = s.slice(1, 2)
    chunked = pl.concat([s.slice(1, 1), s.slice(2, 1)])
    assert chunked.n_chunks() == 2

    for variant in (sliced, chunked):
        assert variant.to_list() == rows[1:]
        assert variant.map.entries().to_list() == [
            [{"key": key, "value": value} for key, value in row.items()]
            for row in rows[1:]
        ]

        cast = variant.cast(target)
        assert cast.to_list() == expected
        # Repacking the entries rebases the outer offsets onto them.
        assert cast.to_arrow().offsets.to_pylist()[0] == 0
        # Row encoding requires normalized offsets at every depth.
        assert variant.to_frame().group_by("m").len().height == 2
        assert variant.to_frame().sort("m").height == 2


@pytest.mark.parametrize(("dtype", "target", "rows", "expected"), SLICED_MAP_CASES)
def test_map_sliced_value_casts_survive_nesting(
    dtype: pl.Map,
    target: pl.Map,
    rows: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    sliced = pl.Series("m", rows, dtype=dtype).slice(1, 2)

    assert sliced.implode().cast(pl.List(target)).to_list() == [expected]
    assert pl.select(pl.struct(pl.lit(sliced).alias("m"))).to_series().cast(
        pl.Struct({"m": target})
    ).to_list() == [{"m": row} for row in expected]
