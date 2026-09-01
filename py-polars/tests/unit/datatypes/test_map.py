from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, SchemaError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import IO

MAP = pl.Map(pl.String, pl.Int64)
ENTRIES = pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))


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


@pytest.mark.parametrize(
    ("key_dtype", "key"),
    [
        (pl.Int64, 1),
        (pl.Float64, 1.5),
        (pl.Boolean, True),
        (pl.Date, date(2020, 1, 1)),
        (pl.String, "a"),
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
