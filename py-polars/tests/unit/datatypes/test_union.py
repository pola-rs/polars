from __future__ import annotations

import io
import pickle
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import PolarsDataType


def dense(
    types: list[int],
    offsets: list[int],
    children: list[pa.Array],
    names: list[str] | None = None,
    codes: list[int] | None = None,
) -> pa.UnionArray:
    names = names if names is not None else [f"v{i}" for i in range(len(children))]
    return pa.UnionArray.from_dense(
        pa.array(types, pa.int8()),
        pa.array(offsets, pa.int32()),
        children,
        names,
        codes,
    )


def to_pylist(s: pl.Series) -> list[Any]:
    """Export to Arrow, check that the result is valid and return its values."""
    arr = s.rechunk().to_arrow()
    arr.validate(full=True)
    values: list[Any] = arr.to_pylist()
    return values


def to_pa(s: pl.Series) -> pa.UnionArray:
    """Export to Arrow and check that the result is a valid dense union."""
    arr = s.rechunk().to_arrow()
    assert isinstance(arr, pa.UnionArray)
    arr.validate(full=True)
    return arr


def int_str() -> pa.UnionArray:
    return dense(
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 2, 2],
        [pa.array([1, 2, 3], pa.int64()), pa.array(["a", "b", "c"])],
        ["i", "s"],
    )


INT_STR = pl.Union({"i": pl.Int64, "s": pl.String})
INT_STR_VALUES = [1, "a", 2, "b", "c", 3]


def test_union_dtype() -> None:
    dtype = pl.Union({"i": pl.Int64, "s": pl.String})
    assert dtype == pl.Union([pl.Field("i", pl.Int64), pl.Field("s", pl.String)])
    assert dtype == pl.Union
    assert dtype != pl.Union({"s": pl.String, "i": pl.Int64})
    assert dtype != pl.Struct({"i": pl.Int64, "s": pl.String})
    assert hash(dtype) == hash(pl.Union({"i": pl.Int64, "s": pl.String}))
    assert repr(dtype) == "Union({'i': Int64, 's': String})"
    assert dtype.is_nested()
    assert pl.Series([], dtype=dtype).dtype == dtype

    with pytest.raises(TypeError, match="Union requires its variants"):
        pl.Series([], dtype=pl.Union)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"a": pl.Null},
        {"a": pl.Object},
        [pl.Field("a", pl.Int64), pl.Field("a", pl.String)],
    ],
)
def test_union_dtype_invalid(fields: Any) -> None:
    with pytest.raises((InvalidOperationError, pl.exceptions.DuplicateError)):
        pl.Series([], dtype=pl.Union(fields))


def test_union_import() -> None:
    s = pl.Series("u", int_str())
    assert s.dtype == INT_STR
    assert s.to_list() == INT_STR_VALUES
    assert s.null_count() == 0

    df = pl.from_arrow(pa.table({"u": int_str(), "x": list(range(6))}))
    assert isinstance(df, pl.DataFrame)
    assert df.schema == {"u": INT_STR, "x": pl.Int64}

    stream = pl.DataFrame(pa.table({"u": int_str()}))
    assert stream["u"].to_list() == INT_STR_VALUES

    chunked = pa.table({"u": pa.chunked_array([int_str(), int_str().slice(1, 2)])})
    df = pl.from_arrow(chunked, rechunk=False)
    assert isinstance(df, pl.DataFrame)
    assert df["u"].n_chunks() == 2
    assert df["u"].to_list() == [*INT_STR_VALUES, "a", 2]

    schema = pa.schema([("u", int_str().type)])
    assert pl.Schema(schema) == pl.Schema({"u": INT_STR})
    empty = pl.from_arrow(pa.table({"u": int_str()}).slice(0, 0))
    assert isinstance(empty, pl.DataFrame)
    assert empty.schema == {"u": INT_STR}


def test_union_round_trip() -> None:
    arr = int_str()
    out = to_pa(pl.Series(arr))
    # Polars exports strings as `large_string`, inside a union as outside.
    assert out.type.field(1).type == pa.large_string()
    assert out.to_pylist() == arr.to_pylist()
    assert out.type_codes.to_pylist() == arr.type_codes.to_pylist()
    assert out.offsets.to_pylist() == arr.offsets.to_pylist()


@pytest.mark.parametrize("with_nulls", [False, True])
def test_union_zero_copy(with_nulls: bool) -> None:
    ints = pa.array([1, None, 3] if with_nulls else [1, 2, 3], pa.int64())
    floats = pa.array([0.5, 1.5], pa.float64())
    arr = dense([0, 1, 0, 1, 0], [0, 0, 1, 1, 2], [ints, floats])

    out = to_pa(pl.Series(arr))
    assert out.buffers()[1].address == arr.buffers()[1].address
    assert out.field(0).buffers()[1].address == ints.buffers()[1].address
    assert out.field(1).buffers()[1].address == floats.buffers()[1].address
    assert out.offsets.to_pylist() == arr.offsets.to_pylist()
    assert out.equals(arr)


def test_union_nulls() -> None:
    arr = dense(
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [pa.array([1, None], pa.int64()), pa.array([None, "b"])],
        ["i", "s"],
    )
    s = pl.Series(arr)
    assert s.null_count() == 2
    assert s.is_null().to_list() == [False, True, True, False]
    assert s.to_list() == [1, None, None, "b"]

    out = to_pa(s)
    assert out.to_pylist() == [1, None, None, "b"]
    assert out.field(0).to_pylist() == [1, None]
    assert out.field(1).to_pylist() == [None, "b"]


def test_union_non_positional_ids() -> None:
    arr = dense(
        [7, 5, 7],
        [0, 0, 1],
        [pa.array([1], pa.int64()), pa.array(["a", "b"])],
        ["i", "s"],
        [5, 7],
    )
    s = pl.Series(arr)
    assert s.dtype == INT_STR
    assert s.to_list() == ["a", 1, "b"]
    out = to_pa(s)
    assert out.type_codes.to_pylist() == [1, 0, 1]
    assert out.to_pylist() == ["a", 1, "b"]


def test_union_null_variant_dropped() -> None:
    arr = dense(
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [pa.array([1, 2], pa.int64()), pa.nulls(2)],
        ["i", "n"],
    )
    s = pl.Series(arr)
    assert s.dtype == pl.Union({"i": pl.Int64})
    assert s.to_list() == [1, None, 2, None]
    assert to_pa(s).to_pylist() == [1, None, 2, None]

    schema = pl.from_arrow(pa.table({"u": arr}).slice(0, 0))
    assert isinstance(schema, pl.DataFrame)
    assert schema.schema == {"u": s.dtype}


def test_union_all_null_variants() -> None:
    arr = dense([0, 0], [0, 1], [pa.nulls(2)], ["n"])
    s = pl.Series(arr)
    assert s.dtype == pl.Null
    assert s.to_list() == [None, None]

    df = pl.from_arrow(pa.table({"u": arr}).slice(0, 0))
    assert isinstance(df, pl.DataFrame)
    assert df.schema == {"u": pl.Null}


def test_union_nested_all_null_variant() -> None:
    inner = dense([0], [0], [pa.nulls(1)], ["n"])
    arr = dense([0, 1], [0, 0], [inner, pa.array([5], pa.int64())], ["a", "b"])
    s = pl.Series(arr)
    assert s.dtype == pl.Union({"b": pl.Int64})
    assert s.to_list() == [None, 5]

    df = pl.from_arrow(pa.table({"u": arr}).slice(0, 0))
    assert isinstance(df, pl.DataFrame)
    assert df.schema == {"u": s.dtype}


def test_union_repeated_and_unreferenced_offsets() -> None:
    ints = pa.array([10, 20, 30, 40], pa.int64())
    arr = dense([0, 0, 0], [3, 3, 1], [ints], ["i"])
    s = pl.Series(arr)
    assert s.to_list() == [40, 40, 20]
    out = to_pa(s)
    assert out.to_pylist() == [40, 40, 20]
    assert out.offsets.to_pylist() == [0, 1, 2]


def test_union_sliced_input() -> None:
    arr = int_str().slice(2, 3)
    s = pl.Series(arr)
    assert s.to_list() == INT_STR_VALUES[2:5]
    assert to_pa(s).to_pylist() == INT_STR_VALUES[2:5]


def test_union_sliced_non_positional_c_interface() -> None:
    arr = dense(
        [7, 5, 7, 5, 7],
        [0, 0, 1, 1, 2],
        [pa.array([1, 2], pa.int64()), pa.array(["a", "b", "c"])],
        ["i", "s"],
        [5, 7],
    ).slice(1, 3)
    s = pl.Series(arr)
    assert s.to_list() == [1, "b", 2]

    # Export through the C stream interface and import again.
    back = pa.chunked_array(s).combine_chunks()
    back.validate(full=True)
    assert back.to_pylist() == [1, "b", 2]
    assert pl.Series(back).to_list() == [1, "b", 2]


def test_union_sparse_raises() -> None:
    arr = pa.UnionArray.from_sparse(
        pa.array([0, 1], pa.int8()),
        [pa.array([1, None], pa.int64()), pa.array([None, "a"])],
    )
    with pytest.raises(ComputeError, match="sparse"):
        pl.Series(arr)


def test_union_export_rebases_children() -> None:
    s = pl.concat([pl.Series(int_str()), pl.Series(int_str())], rechunk=True)
    middle = s.slice(4, 4)
    assert middle.to_list() == ["c", 3, 1, "a"]
    out = to_pa(middle)
    assert out.to_pylist() == ["c", 3, 1, "a"]
    assert out.offsets.to_pylist() == [0, 0, 1, 1]
    assert [len(out.field(i)) for i in range(2)] == [2, 2]


def big_union() -> pa.UnionArray:
    # A zero-field struct has no per-row buffers, so this child takes no memory.
    child = pa.StructArray.from_buffers(pa.struct([]), 2**30, [None])
    assert child.nbytes == 0
    return dense([0, 0], [0, 2**30 - 1], [child], ["e"])


def test_union_i32_boundary() -> None:
    one = pl.Series(big_union())

    two = pl.concat([one, one], rechunk=True)
    out = to_pa(two)
    assert out.offsets.to_pylist() == [0, 2**30 - 1, 2**30, 2**31 - 1]

    three = pl.concat([one, one, one], rechunk=True)
    with pytest.raises(InvalidOperationError, match="i32 offsets"):
        three.to_arrow()

    late = three.slice(5, 1)
    assert to_pa(late).offsets.to_pylist() == [0]


def test_union_gather_large_child() -> None:
    one = pl.Series(big_union())
    five = pl.concat([one] * 5, rechunk=True)
    assert len(five) == 10
    # The last row points past `u32::MAX` in a child of 5 * 2**30 values.
    for out in [
        five.gather([9]),
        five.filter(pl.Series([False] * 9 + [True])),
        five.shift(-9).head(1),
        five.gather([9, 0, 5, 8]),
    ]:
        arr = to_pa(out)
        assert arr.to_pylist()[0] == {}
        assert out.null_count() == 0


def tagged() -> pl.DataFrame:
    arr = dense(
        [0, 1],
        [0, 0],
        [pa.array([1], pa.int64()), pa.array([1], pa.int64())],
        ["a", "b"],
    )
    df = pl.from_arrow(pa.table({"u": arr}))
    assert isinstance(df, pl.DataFrame)
    return df


def test_union_tags_survive_scalars() -> None:
    df = tagged()
    stacked = df[0].vstack(df[1])
    assert to_pa(stacked["u"]).type_codes.to_pylist() == [0, 1]

    first = df.select(pl.col("u").last())
    assert to_pa(first["u"]).type_codes.to_pylist() == [1]

    broadcast = df.with_columns(f=pl.col("u").last())
    assert to_pa(broadcast["f"]).type_codes.to_pylist() == [1, 1]


def test_union_categorical_variant() -> None:
    cats = pa.DictionaryArray.from_arrays(
        pa.array([0, None, 1, None], pa.uint32()), pa.array(["x", "y"])
    )
    # Slot 1 references a null entry; entry 3 is null and unreferenced.
    arr = dense(
        [0, 0, 1, 0],
        [0, 1, 0, 2],
        [cats, pa.array([7], pa.int64())],
        ["c", "i"],
    )
    s = pl.Series(arr)
    assert s.to_list() == ["x", None, 7, "y"]
    shifted = s.shift(1)
    assert shifted.to_list() == [None, "x", None, 7]
    assert to_pa(s).to_pylist() == ["x", None, 7, "y"]
    assert to_pa(shifted).to_pylist() == [None, "x", None, 7]


def test_union_nesting() -> None:
    u = int_str()
    lst = pa.LargeListArray.from_arrays(pa.array([0, 2, 6]), u)
    fsl = pa.FixedSizeListArray.from_arrays(u, 2)
    st = pa.StructArray.from_arrays([u], ["u"])
    for arr, dtype in [
        (lst, pl.List(INT_STR)),
        (fsl, pl.Array(INT_STR, 2)),
        (st, pl.Struct({"u": INT_STR})),
    ]:
        s = pl.Series(arr)
        assert s.dtype == dtype
        out = s.to_arrow()
        out.validate(full=True)
        assert out.to_pylist() == arr.to_pylist()

    inner = dense([0, 1], [0, 0], [pa.array([1]), pa.array(["z"])], ["i", "s"])
    variants = dense(
        [0, 1, 2, 3],
        [0, 0, 0, 0],
        [
            pa.array([{"a": 1}]),
            pa.array([[1, 2]], pa.large_list(pa.int64())),
            pa.array([datetime(2020, 1, 1)], pa.timestamp("us")),
            inner,
        ],
        ["st", "l", "dt", "u"],
    )
    s = pl.Series(variants)
    expected: PolarsDataType = pl.Union(
        {
            "st": pl.Struct({"a": pl.Int64}),
            "l": pl.List(pl.Int64),
            "dt": pl.Datetime("us"),
            "u": pl.Union({"i": pl.Int64, "s": pl.String}),
        }
    )
    assert s.dtype == expected
    assert s.to_list() == [{"a": 1}, [1, 2], datetime(2020, 1, 1), 1]
    assert to_pa(s).to_pylist() == variants.to_pylist()


def test_union_structural_ops() -> None:
    s = pl.Series("u", int_str())
    assert s.slice(1, 3).to_list() == INT_STR_VALUES[1:4]
    assert s.reverse().to_list() == INT_STR_VALUES[::-1]
    assert s.gather([5, 0, 3]).to_list() == [3, 1, "b"]
    assert s.filter(s.is_not_null() & pl.Series([True, False] * 3)).to_list() == [
        1,
        2,
        "c",
    ]
    assert s.shift(2).to_list() == [None, None, *INT_STR_VALUES[:4]]
    assert s.shift(-2).to_list() == [*INT_STR_VALUES[2:], None, None]
    for out in [s.slice(1, 3), s.reverse(), s.gather([5, 0, 3]), s.shift(2)]:
        assert to_pa(out).to_pylist() == out.to_list()

    # Rechunking slices of one array does not duplicate the shared children.
    parts = pl.concat([s.slice(0, 2), s.slice(2, 2), s.slice(4, 2)], rechunk=True)
    out = to_pa(parts)
    assert out.to_pylist() == INT_STR_VALUES
    assert [len(out.field(i)) for i in range(2)] == [3, 3]


def test_union_frame_ops() -> None:
    df = pl.from_arrow(pa.table({"u": int_str(), "x": [5, 4, 3, 2, 1, 0]}))
    assert isinstance(df, pl.DataFrame)

    assert df.sort("x")["u"].to_list() == INT_STR_VALUES[::-1]
    other = pl.DataFrame({"x": [0, 5, 7], "y": ["p", "q", "r"]})
    joined = df.join(other, on="x").sort("x")
    assert joined["u"].to_list() == [3, 1]
    to_pa(joined["u"])

    diag = pl.concat([df, pl.DataFrame({"x": [9]})], how="diagonal")
    assert diag["u"].to_list() == [*INT_STR_VALUES, None]
    assert to_pa(diag["u"]).to_pylist() == [*INT_STR_VALUES, None]


def test_union_ipc(tmp_path: Path) -> None:
    df = pl.from_arrow(pa.table({"u": int_str(), "x": list(range(6))}))
    assert isinstance(df, pl.DataFrame)

    path = tmp_path / "u.arrow"
    df.write_ipc(path)
    out = pl.read_ipc(path)
    assert out.schema == df.schema
    assert out["u"].to_list() == INT_STR_VALUES
    assert pl.read_ipc(path, columns=["x"])["x"].to_list() == list(range(6))

    buf = io.BytesIO()
    df.write_ipc_stream(buf)
    buf.seek(0)
    assert pl.read_ipc_stream(buf)["u"].to_list() == INT_STR_VALUES

    roundtrip = pickle.loads(pickle.dumps(df))
    assert roundtrip["u"].to_list() == INT_STR_VALUES
    assert roundtrip.schema == df.schema


def nested_unions() -> list[pl.Series]:
    lists = pa.LargeListArray.from_arrays(pa.array([0, 2, 2, 6]), int_str())
    arrays = pa.FixedSizeListArray.from_arrays(int_str(), 2)
    return [pl.Series("l", lists), pl.Series("a", arrays)]


def test_union_nested_chunked_gather() -> None:
    for s in nested_unions():
        chunked = pl.concat([s.slice(1), s], rechunk=False)
        assert chunked.n_chunks() == 2
        values = s.to_arrow().to_pylist()
        combined = values[1:] + values
        idx = [1, len(s) - 1, len(combined) - 1, 0]
        assert to_pylist(chunked.gather(idx)) == [combined[i] for i in idx]

        broadcast = s.to_frame().with_columns(v=pl.col(s.name).first())["v"]
        assert to_pylist(broadcast) == [values[0]] * len(s)


def test_union_join_chunked_payload() -> None:
    lists = pa.LargeListArray.from_arrays(pa.array(range(7)), int_str())
    right = pl.DataFrame({"k": range(6), "u": pl.Series(int_str()), "l": lists})
    right = pl.concat([right, right.with_columns(pl.col("k") + 6)], rechunk=False)
    assert right.n_chunks() == 2
    left = pl.DataFrame({"k": [7, 0, 99, 11]})

    out = left.join(right, on="k", how="left")
    assert to_pa(out["u"]).to_pylist() == [
        INT_STR_VALUES[1],
        INT_STR_VALUES[0],
        None,
        INT_STR_VALUES[5],
    ]
    assert to_pylist(out["l"]) == [
        [INT_STR_VALUES[1]],
        [INT_STR_VALUES[0]],
        None,
        [INT_STR_VALUES[5]],
    ]


def test_union_duplicate_names_raise() -> None:
    arr = dense([0, 1], [0, 0], [pa.array([1]), pa.array(["a"])], ["x", "x"])
    with pytest.raises(pl.exceptions.DuplicateError, match="not unique"):
        pl.Series(arr)


def test_union_import_large_child() -> None:
    # A child longer than `u32::MAX`, with decreasing offsets and a dropped `Null`
    # variant, so the import rebuilds the children.
    child = pa.StructArray.from_buffers(pa.struct([]), 5 * 2**30, [None])
    arr = dense(
        [0, 0, 1, 0],
        [2**31 - 1, 0, 0, 5],
        [child, pa.nulls(1)],
        ["e", "n"],
    )
    s = pl.Series(arr)
    assert s.dtype == pl.Union({"e": pl.Struct({})})
    out = to_pa(s)
    assert out.to_pylist() == [{}, {}, None, {}]
    assert [len(out.field(0))] == [4]


def messy() -> pl.Series:
    """`[3, "b", 1, "a", None, "b", 1]`, with unreferenced values and a null slot."""
    arr = dense(
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 2, 2, 3, 3, 4],
        [pa.array([99, 3, 1, None, 1]), pa.array(["zz", "b", "a", "b"])],
        ["i", "s"],
    )
    return pl.Series("u", arr)


def messy_layouts() -> list[pl.Series]:
    s = messy()
    return [
        s,
        s.slice(1, 5),
        s.filter(pl.Series([True, False, True, True, True, False, True])),
        # A null slot with a placeholder in child 0.
        s.shift(1),
        pl.concat([s.slice(0, 3), s.slice(3)], rechunk=True),
        pl.concat([s.slice(0, 3), s.slice(3)], rechunk=False),
    ]


def union_key(v: Any) -> tuple[bool, Any]:
    """Unions order by variant, then by value."""
    return (isinstance(v, str), v)


def sorted_union(values: list[Any], *, descending: bool, nulls_last: bool) -> list[Any]:
    nulls = [v for v in values if v is None]
    rest = sorted(
        (v for v in values if v is not None), key=union_key, reverse=descending
    )
    return rest + nulls if nulls_last else nulls + rest


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_union_sort(descending: bool, nulls_last: bool) -> None:
    for s in messy_layouts():
        expected = sorted_union(
            s.to_list(), descending=descending, nulls_last=nulls_last
        )
        out = s.sort(descending=descending, nulls_last=nulls_last)
        assert to_pylist(out) == expected

        df = pl.DataFrame({"u": s, "x": range(len(s))})
        out_df = (
            df.lazy()
            .sort("u", "x", descending=descending, nulls_last=nulls_last)
            .collect()
        )
        assert to_pylist(out_df["u"]) == expected


def test_union_group_by_unique() -> None:
    for s in messy_layouts():
        values = s.to_list()
        firsts = list(dict.fromkeys(values))
        assert to_pylist(s.unique(maintain_order=True)) == firsts
        assert s.n_unique() == len(firsts)
        assert sorted(map(repr, to_pylist(s.unique()))) == sorted(map(repr, firsts))

        df = pl.DataFrame({"u": s, "x": range(len(s))})
        out = df.lazy().group_by("u").agg(pl.col("x").sum()).collect()
        expected: dict[Any, int] = {}
        for x, v in enumerate(values):
            expected[v] = expected.get(v, 0) + x
        assert dict(zip(to_pylist(out["u"]), out["x"], strict=True)) == expected


def test_union_group_by_tag() -> None:
    # Equal values of different variants are different keys.
    df = tagged().vstack(tagged())
    assert df["u"].n_unique() == 2
    out = df.lazy().group_by("u").agg(pl.len()).collect()
    assert sorted(out["len"].to_list()) == [2, 2]
    assert to_pa(out["u"].sort()).type_codes.to_pylist() == [0, 1]


def test_union_join_on_union() -> None:
    left = pl.DataFrame({"u": messy(), "l": range(7)})
    for right_u in messy_layouts():
        right = pl.DataFrame({"u": right_u, "r": range(len(right_u))})
        out = left.lazy().join(right.lazy(), on="u").collect()
        expected = [
            (lv, i, j)
            for i, lv in enumerate(messy().to_list())
            for j, rv in enumerate(right_u.to_list())
            if lv is not None and lv == rv and type(lv) is type(rv)
        ]
        got = list(zip(to_pylist(out["u"]), out["l"], out["r"], strict=True))
        assert sorted(got, key=repr) == sorted(expected, key=repr)

        matched = {i for _, i, _ in expected}
        semi = left.lazy().join(right.lazy(), on="u", how="semi").collect()
        assert semi["l"].to_list() == sorted(matched)
        anti = left.lazy().join(right.lazy(), on="u", how="anti").collect()
        assert anti["l"].to_list() == sorted(set(range(7)) - matched)
        lj = left.lazy().join(right.lazy(), on="u", how="left").collect()
        assert lj.height == len(expected) + 7 - len(matched)


def test_union_key_variants() -> None:
    cats = pa.DictionaryArray.from_arrays(
        pa.array([1, 0, 1, 2], pa.uint32()), pa.array(["x", "y", "unused"])
    )
    dec = pa.array([Decimal("1.50"), Decimal("-2.25")], pa.decimal128(10, 2))
    lists = pa.array([[2, 1], [1, 2, 3], []], pa.large_list(pa.int64()))
    arr = dense(
        [0, 1, 2, 0, 1, 2, 0, 2],
        [0, 0, 0, 1, 1, 1, 2, 2],
        [cats, dec, lists],
        ["c", "d", "l"],
    )
    s = pl.Series("u", arr)
    assert s.dtype == pl.Union(
        {"c": pl.Categorical(), "d": pl.Decimal(10, 2), "l": pl.List(pl.Int64)}
    )
    values = s.to_list()
    assert values == [
        "y",
        Decimal("1.50"),
        [2, 1],
        "x",
        Decimal("-2.25"),
        [1, 2, 3],
        "y",
        [],
    ]
    assert to_pylist(s.sort()) == [
        "x",
        "y",
        "y",
        Decimal("-2.25"),
        Decimal("1.50"),
        [],
        [1, 2, 3],
        [2, 1],
    ]
    assert to_pylist(s.unique(maintain_order=True)) == values[:6] + [[]]
    out = s.to_frame().lazy().group_by("u").agg(pl.len()).collect()
    assert sorted(out["len"].to_list()) == [1, 1, 1, 1, 1, 1, 2]


def test_union_nested_keys() -> None:
    u = messy()
    values = u.to_list()
    lists = pl.Series(
        "l", pa.LargeListArray.from_arrays(pa.array([0, 2, 4, 7]), u.to_arrow())
    )
    arrays = pl.Series("a", pa.FixedSizeListArray.from_arrays(u.slice(1).to_arrow(), 2))
    # Row 1 is a null struct over a valid union value.
    structs = pl.Series(
        "st",
        pa.StructArray.from_arrays(
            [u.to_arrow()], ["u"], mask=pa.array([False, True] * 3 + [False])
        ),
    )
    for s in [lists, arrays, structs]:
        assert to_pylist(s.unique(maintain_order=True)) == list(
            {repr(v): v for v in s.to_list()}.values()
        )
        assert to_pylist(s.sort())[::-1] == to_pylist(
            s.sort(descending=True, nulls_last=True)
        )
    assert to_pylist(lists.sort()) == [[None, "b", 1], [1, "a"], [3, "b"]]

    # The struct nulls hide different union values, so they are one group.
    st_values = structs.to_list()
    assert st_values[1] is None
    assert structs.n_unique() == len({repr(v) for v in st_values})
    out = structs.to_frame().lazy().group_by("st").agg(pl.len()).collect()
    assert dict(zip(map(repr, out["st"].to_list()), out["len"], strict=True))[
        "None"
    ] == st_values.count(None)
    assert values == [3, "b", 1, "a", None, "b", 1]


def test_union_large_child_keys() -> None:
    one = pl.Series(big_union())
    five = pl.concat([one] * 5, rechunk=True)
    for s in [one, five]:
        assert s.n_unique() == 1
        assert to_pylist(s.unique()) == [{}]
        assert to_pylist(s.sort()) == [{}] * len(s)
        out = s.to_frame().lazy().group_by(s.name).agg(pl.len()).collect()
        assert out["len"].to_list() == [len(s)]


@pytest.mark.parametrize(
    ("descending", "nulls_last"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_union_row_encode_roundtrip(descending: bool, nulls_last: bool) -> None:
    for s in messy_layouts():
        df = s.to_frame()
        encoded = df._row_encode(descending=[descending], nulls_last=[nulls_last])
        decoded = encoded._row_decode(
            ["u"], [s.dtype], descending=[descending], nulls_last=[nulls_last]
        ).struct.unnest()
        assert to_pylist(decoded["u"]) == s.to_list()
    unordered = messy().to_frame()._row_encode(unordered=True)
    assert unordered.n_unique() == 5


def test_union_implode_and_repeat() -> None:
    s = messy()
    df = pl.DataFrame({"k": [1, 1, 2, 2, 1, 2, 1], "u": s})
    imploded = df.lazy().select(pl.col("u").implode()).collect()["u"]
    assert to_pylist(imploded) == [s.to_list()]
    grouped = df.lazy().group_by("k").agg(pl.col("u")).collect().sort("k")
    assert to_pylist(grouped["u"]) == [[3, "b", None, 1], [1, "a", "b"]]

    lists = nested_unions()[0]
    repeated = lists.repeat_by(2)
    expected = lists.to_arrow().to_pylist()
    assert to_pylist(repeated) == [[v, v] for v in expected]


def py_eq(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def test_union_equality() -> None:
    layouts = messy_layouts()
    # The same values as `messy()`, with compact children and an explicit null slot.
    compact = pl.Series(
        "u",
        dense(
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 2, 2, 3],
            [pa.array([3, 1, None, 1]), pa.array(["b", "a", "b"])],
            ["i", "s"],
        ),
    )
    for lhs in layouts:
        for rhs in [*layouts, compact]:
            if len(lhs) != len(rhs):
                continue
            lv, rv = lhs.to_list(), rhs.to_list()
            eq = [
                None if a is None or b is None else py_eq(a, b)
                for a, b in zip(lv, rv, strict=True)
            ]
            eq_missing = [
                (a is None and b is None) or py_eq(a, b)
                for a, b in zip(lv, rv, strict=True)
            ]
            assert (lhs == rhs).to_list() == eq
            assert (lhs != rhs).to_list() == [None if v is None else not v for v in eq]
            assert lhs.eq_missing(rhs).to_list() == eq_missing
            assert lhs.ne_missing(rhs).to_list() == [not v for v in eq_missing]

    assert_series_equal(messy(), compact)
    assert_series_equal(layouts[4], layouts[0])
    with pytest.raises(AssertionError):
        assert_series_equal(messy(), messy().reverse())


def test_union_equality_broadcast() -> None:
    s = messy()
    values = s.to_list()
    df = s.to_frame()
    for i in [0, 1, 4]:
        out = df.select(
            eq=pl.col("u") == pl.col("u").gather(i),
            eq_missing=pl.col("u").eq_missing(pl.col("u").gather(i)),
            flipped=pl.col("u").gather(i) == pl.col("u"),
        )
        v = values[i]
        eq = [None if a is None or v is None else py_eq(a, v) for a in values]
        assert out["eq"].to_list() == eq
        assert out["flipped"].to_list() == eq
        assert out["eq_missing"].to_list() == [
            (a is None and v is None) or py_eq(a, v) for a in values
        ]


def test_union_equality_tags_and_large() -> None:
    df = tagged()
    assert (df["u"] == df["u"].reverse()).to_list() == [False, False]

    one = pl.Series(big_union())
    five = pl.concat([one] * 5, rechunk=True)
    assert (five == five.reverse()).all()
    assert (five.gather([9, 0]) == five.gather([0, 9])).all()


def test_union_zip() -> None:
    a = messy()
    b = pl.concat([a.slice(3), a.slice(0, 3)], rechunk=False)
    mask = [True, None, False, True, False, False, True]
    av, bv = a.to_list(), b.to_list()
    df = pl.DataFrame({"a": a, "b": b, "m": pl.Series(mask, dtype=pl.Boolean)})
    cases = {
        "both": (pl.col("a"), pl.col("b"), av, bv),
        "scalar_true": (pl.col("a").gather(1), pl.col("b"), [av[1]] * 7, bv),
        "scalar_false": (pl.col("a"), pl.col("b").gather(4), av, [bv[4]] * 7),
        "scalar_both": (
            pl.col("a").gather(0),
            pl.col("b").gather(2),
            [av[0]] * 7,
            [bv[2]] * 7,
        ),
    }
    for name, (t, f, tv, fv) in cases.items():
        out = (
            df.lazy()
            .select(pl.when("m").then(t).otherwise(f).alias(name))
            .collect()[name]
        )
        expected = [x if m else y for m, x, y in zip(mask, tv, fv, strict=True)]
        assert to_pylist(out) == expected
        # Every child value of the output belongs to one of its slots.
        arr = out.to_arrow()
        assert sum(len(arr.field(i)) for i in range(2)) == len(arr)

    assert to_pylist(a.zip_with(pl.Series(mask), b)) == [
        x if m else y for m, x, y in zip(mask, av, bv, strict=True)
    ]
    other = pl.Series([None] * 7, dtype=pl.Union({"x": pl.Int8}))
    with pytest.raises(pl.exceptions.SchemaError):
        a.zip_with(pl.Series(mask), other)


def test_union_zip_large_child() -> None:
    one = pl.Series(big_union())
    five = pl.concat([one] * 5, rechunk=True)
    mask = pl.Series([i % 3 == 0 for i in range(10)])
    out = five.zip_with(mask, five.reverse())
    assert to_pylist(out) == [{}] * 10


def test_union_cast_widens() -> None:
    for s in messy_layouts():
        values = s.to_list()
        # Reordering the variants only remaps the tags.
        flipped = s.cast(pl.Union({"s": pl.String, "i": pl.Int64}))
        assert to_pylist(flipped) == values
        assert to_pa(flipped).type_codes.to_pylist() == [
            1 - t for t in to_pa(s).type_codes.to_pylist()
        ]
        # A new variant stays empty.
        to = pl.Union({"b": pl.Boolean, "i": pl.Int64, "s": pl.String})
        for strict in [True, False]:
            wider = s.cast(to, strict=strict)
            assert wider.dtype == to
            assert to_pylist(wider) == values
        lazy = s.to_frame().lazy().select(pl.col("u").cast(to))
        assert lazy.collect_schema()["u"] == to
        assert to_pylist(lazy.collect()["u"]) == values


def test_union_cast_widens_nulls() -> None:
    s = pl.Series(
        dense(
            [0, 1, 0],
            [0, 0, 1],
            [
                pa.array([[None], []], pa.large_list(pa.null())),
                pa.array([{"a": None}], pa.struct([("a", pa.null())])),
            ],
            ["l", "st"],
        )
    )
    to = pl.Union(
        {"x": pl.Int8, "st": pl.Struct({"a": pl.Int64}), "l": pl.List(pl.Int64)}
    )
    out = s.cast(to)
    assert out.dtype == to
    assert to_pylist(out) == [[None], {"a": None}, []]

    inner = dense([0, 0], [0, 1], [pa.array([[None], []], pa.large_list(pa.null()))])
    nested = pl.Series(dense([0, 0], [0, 1], [inner], ["u"]))
    to = pl.Union({"u": pl.Union({"m": pl.Int8, "v0": pl.List(pl.String)})})
    out = nested.cast(to)
    assert out.dtype == to
    assert out.to_list() == [[None], []]

    for null in [pl.Series([None, None]), pl.Series([], dtype=pl.Null)]:
        out = null.cast(INT_STR)
        assert out.dtype == INT_STR
        assert out.to_list() == null.to_list()
    lazy = pl.LazyFrame({"x": [None]}).select(pl.col("x").cast(INT_STR))
    assert lazy.collect_schema()["x"] == INT_STR
    assert lazy.collect()["x"].to_list() == [None]


def test_union_cast_to_string() -> None:
    for s in messy_layouts():
        out = s.cast(pl.String)
        assert out.dtype == pl.String
        assert out.to_list() == [None if v is None else str(v) for v in s.to_list()]

    s = pl.Series(
        dense(
            [0, 1, 1],
            [0, 0, 1],
            [pa.array([{"a": 1}]), pa.array([1.5, None])],
            ["st", "f"],
        )
    )
    assert s.cast(pl.String).to_list() == ["{1}", "1.5", None]

    # The invalid value is not referenced by any slot.
    s = pl.Series(
        dense([0, 1], [0, 1], [pa.array([1]), pa.array([b"\xff", b"ok"])], ["i", "b"])
    )
    assert s.cast(pl.String).to_list() == ["1", "ok"]
    lazy = s.to_frame().lazy().select(pl.col("").cast(pl.String))
    assert lazy.collect_schema()[""] == pl.String
    assert lazy.collect()[""].to_list() == ["1", "ok"]


@pytest.mark.parametrize(
    "to",
    [
        pl.Union({"i": pl.Int64}),
        pl.Union({"i": pl.Float64, "s": pl.String}),
        pl.Union({"i": pl.Int64, "t": pl.String}),
        pl.Int64,
        pl.Binary,
        pl.Struct({"i": pl.Int64, "s": pl.String}),
        pl.List(INT_STR),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_union_cast_rejected(to: pl.DataType, strict: bool) -> None:
    # Only null slots hold `i`, so no value would be lost by dropping it.
    s = pl.Series([None, None], dtype=INT_STR)
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        s.cast(to, strict=strict)
    lazy = s.to_frame().lazy().select(pl.col("").cast(to, strict=strict))
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        lazy.collect_schema()
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        lazy.collect()


@pytest.mark.parametrize(
    "s",
    [
        pl.Series([1]),
        pl.Series([None], dtype=pl.Int64),
        pl.Series(["a"]),
        pl.Series([[1]]),
        pl.Series([{"i": 1}]),
    ],
)
def test_union_cast_into_rejected(s: pl.Series) -> None:
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        s.cast(INT_STR)
    lazy = s.to_frame().lazy().select(pl.col("").cast(INT_STR))
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        lazy.collect_schema()


U_I = pl.Union({"i": pl.Int64})
MAP_ENTRIES = pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))


@pytest.mark.parametrize(
    ("values", "dtype", "target"),
    [
        ([{"a": 1}], pl.Map(pl.String, pl.Int64), pl.Map(pl.String, U_I)),
        (
            [{"a": 1}],
            pl.Map(pl.String, pl.Int64),
            pl.List(pl.Struct({"key": pl.String, "value": U_I})),
        ),
        ([[{"key": "a", "value": 1}]], MAP_ENTRIES, pl.Map(pl.String, U_I)),
        ([{"x": 1}], pl.Struct({"x": pl.Int64}), pl.Struct({"x": U_I})),
        ([1], pl.Int64, pl.Struct({"x": U_I})),
        ([[1]], pl.List(pl.Int64), pl.Array(U_I, 1)),
        ([[1]], pl.List(pl.Int64), pl.List(pl.Struct({"x": U_I}))),
    ],
)
@pytest.mark.parametrize("null", [False, True])
def test_union_cast_nested_rejected(
    values: list[Any], dtype: pl.DataType, target: pl.DataType, null: bool
) -> None:
    # Rejected from the dtypes alone, also when no value reaches the union.
    s = pl.Series([None] if null else values, dtype=dtype)
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        s.cast(target)
    lazy = s.to_frame().lazy().select(pl.col("").cast(target))
    with pytest.raises(InvalidOperationError, match="cannot cast"):
        lazy.collect_schema()


def test_union_cast_struct_with_union_to_scalar_rejected() -> None:
    for values in ([None], [{"x": None}]):
        s = pl.Series(values, dtype=pl.Struct({"x": U_I}))
        with pytest.raises(InvalidOperationError, match="cannot cast"):
            s.cast(pl.Int64)
        lazy = s.to_frame().lazy().select(pl.col("").cast(pl.Int64))
        with pytest.raises(InvalidOperationError, match="cannot cast"):
            lazy.collect_schema()


@pytest.mark.parametrize(
    "target",
    [
        pl.Struct({"x": U_I}),
        pl.List(pl.Struct({"x": U_I})),
        pl.Array(U_I, 1),
        pl.Map(pl.String, U_I),
    ],
)
def test_union_cast_null_to_nested(target: pl.DataType) -> None:
    s = pl.Series([None, None])
    assert_series_equal(s.cast(target), pl.Series([None, None], dtype=target))
    lazy = s.to_frame().lazy().select(pl.col("").cast(target))
    assert lazy.collect_schema()[""] == target


def union_key_map() -> pl.Series:
    keys = pl.Series(dense([0, 0], [0, 1], [pa.array([1, 2])], ["i"]))
    entries = pl.struct(
        pl.lit(keys).alias("key"), pl.lit(pl.Series(["x", "y"])).alias("value")
    )
    return pl.select(pl.concat_list(entries)).to_series().cast(pl.Map(U_I, pl.String))


def test_union_cast_map_key_widens() -> None:
    m = union_key_map()
    target = pl.Map(INT_STR, pl.String)
    out = m.cast(target)
    assert out.dtype == target
    assert to_pylist(out) == [[(1, "x")], [(2, "y")]]
    lazy = m.to_frame().lazy().select(pl.col("key").cast(target))
    assert_series_equal(lazy.collect().to_series(), out)


def test_union_append_widens() -> None:
    small = pl.Series(dense([0], [0], [pa.array([1])], ["i"]))
    big = pl.Series(int_str())
    out = big.clone().append(small)
    assert out.dtype == INT_STR
    assert to_pylist(out) == INT_STR_VALUES + [1]
    with pytest.raises(pl.exceptions.SchemaError, match="does not widen"):
        small.clone().append(big)


def test_union_cast_rejects_placeholders_and_lists() -> None:
    s = pl.Series(dense([0], [0], [pa.array([[1]], pa.large_list(pa.int64()))], ["l"]))
    # `pl.List` is `List(Null)`, a list of nulls.
    with pytest.raises(InvalidOperationError, match="variant `l` has dtype"):
        s.cast(pl.Union({"l": pl.List}))
    with pytest.raises(InvalidOperationError, match="cannot be cast to"):
        s.cast(pl.String)
    with pytest.raises(InvalidOperationError, match="cannot be cast to"):
        s.to_frame().lazy().select(pl.col("").cast(pl.String)).collect_schema()


def test_union_supertypes() -> None:
    small = pl.Series([None], dtype=pl.Union({"i": pl.Int64}))
    big = pl.Series([None], dtype=pl.Union({"s": pl.String, "i": pl.Int64}))
    for a, b in [(small, big), (big, small)]:
        out = pl.select(pl.when(True).then(pl.lit(a)).otherwise(pl.lit(b))).to_series()
        assert out.dtype == big.dtype
        df = pl.concat([a.to_frame("u"), b.to_frame("u")], how="vertical_relaxed")
        assert df.schema["u"] == big.dtype

    out = pl.select(pl.when(True).then(pl.lit(small)).otherwise(None)).to_series()
    assert out.dtype == small.dtype
    other = pl.Series([None], dtype=pl.Union({"i": pl.Int32}))
    for o in [pl.lit("s"), pl.lit(1), pl.lit(other)]:
        with pytest.raises(pl.exceptions.SchemaError, match="supertype"):
            pl.select(pl.when(True).then(pl.lit(small)).otherwise(o))


def variant_of(v: Any) -> str | None:
    return None if v is None else "s" if isinstance(v, str) else "i"


@pytest.mark.parametrize("s", messy_layouts())
def test_union_tag_and_field(s: pl.Series) -> None:
    values = to_pylist(s)
    tag_dtype = pl.Enum(["i", "s"])
    tags = [variant_of(v) for v in values]
    assert_series_equal(s.union.tag(), pl.Series("u", tags, dtype=tag_dtype))
    for name, dtype in [("i", pl.Int64), ("s", pl.String)]:
        expected = [v if variant_of(v) == name else None for v in values]
        assert_series_equal(s.union.field(name), pl.Series("u", expected, dtype=dtype))

    lf = (
        s.to_frame()
        .lazy()
        .select(t=pl.col("u").union.tag(), i=pl.col("u").union.field("i"))
    )
    assert lf.collect_schema() == {"t": tag_dtype, "i": pl.Int64}
    for engine in ["in-memory", "streaming"]:
        out = lf.collect(engine=engine)  # type: ignore[call-overload]
        assert out["t"].to_list() == tags
        assert out["i"].to_list() == [
            v if variant_of(v) == "i" else None for v in values
        ]


def test_union_tag_and_field_errors() -> None:
    s = messy()
    with pytest.raises(pl.exceptions.StructFieldNotFoundError, match="no variant `z`"):
        s.union.field("z")
    with pytest.raises(InvalidOperationError, match="requires a Union dtype"):
        pl.Series([1]).union.tag()
    with pytest.raises(pl.exceptions.StructFieldNotFoundError, match="no variant `z`"):
        s.to_frame().lazy().select(pl.col("u").union.field("z")).collect_schema()


def test_union_value() -> None:
    df = pl.DataFrame({"x": [1, None, 3], "y": ["a", "b", None]})
    u = df.select(
        pl.when(pl.col("x").is_not_null())
        .then(pl.union_value(INT_STR, i=pl.col("x")))
        .otherwise(pl.union_value(INT_STR, s=pl.col("y")))
    ).to_series()
    assert u.dtype == INT_STR
    assert to_pylist(u) == [1, "b", 3]
    assert u.union.tag().to_list() == ["i", "s", "i"]

    # A null value gives a null row.
    out = df.select(pl.union_value(INT_STR, s=pl.col("y"))).to_series()
    assert out.null_count() == 1
    assert out.union.tag().to_list() == ["s", "s", None]

    # Multiple chunks, eager and streaming.
    x = pl.concat([pl.Series("x", [1, 2]), pl.Series("x", [None, 4])], rechunk=False)
    lf = x.to_frame().lazy().select(pl.union_value(INT_STR, i=pl.col("x")))
    for engine in ["in-memory", "streaming"]:
        out = lf.collect(engine=engine).to_series()  # type: ignore[call-overload]
        assert to_pylist(out) == [1, 2, None, 4]
        assert out.union.tag().to_list() == ["i", "i", None, "i"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (None, None), (pl.lit(1, pl.Int64), 1)],
)
def test_union_value_literals(value: Any, expected: Any) -> None:
    out = pl.select(pl.union_value(INT_STR, i=value)).to_series()
    assert out.dtype == INT_STR
    assert to_pylist(out) == [expected]
    out = pl.select(pl.union_value(INT_STR, {"s": "a"})).to_series()
    assert to_pylist(out) == ["a"]


@pytest.mark.parametrize(
    "variant",
    [
        {"i": pl.lit(1, pl.Int32)},
        {"i": 1.5},
        {"i": "a"},
        {"s": 1},
    ],
)
def test_union_value_dtype_mismatch(variant: dict[str, Any]) -> None:
    with pytest.raises(pl.exceptions.SchemaError, match="cast the value first"):
        pl.select(pl.union_value(INT_STR, variant))


def test_union_value_arguments() -> None:
    dtype = pl.Union({"a b": pl.Int64, "default": pl.String})
    out = pl.select(
        pl.union_value(dtype, {"a b": 1}), pl.union_value(dtype, default="x").alias("d")
    )
    assert out.dtypes == [dtype, dtype]
    with pytest.raises(TypeError, match="exactly one variant, got 0"):
        pl.union_value(INT_STR)
    with pytest.raises(TypeError, match="exactly one variant, got 2"):
        pl.union_value(INT_STR, i=1, s="a")
    with pytest.raises(ValueError, match="more than once"):
        pl.union_value(INT_STR, {"i": 1}, i=2)
    with pytest.raises(pl.exceptions.StructFieldNotFoundError, match="no variant `z`"):
        pl.select(pl.union_value(INT_STR, z=1))
    with pytest.raises(InvalidOperationError, match="requires a Union dtype"):
        pl.select(pl.union_value(pl.Int64, i=1))


def match_frame() -> pl.DataFrame:
    df = pl.DataFrame({"x": [1, None, 3, None], "y": ["a", "bc", "d", None]})
    return df.with_columns(
        k=pl.Series([5, 6, 7, 8]),
        u=pl.when(pl.col("x").is_not_null())
        .then(pl.union_value(INT_STR, i=pl.col("x")))
        .otherwise(pl.union_value(INT_STR, s=pl.col("y"))),
    )


@pytest.mark.parametrize(
    ("expr", "expected", "dtype"),
    [
        (
            pl.col("u").union.match(
                i=pl.element() * 10, s=pl.element().str.len_chars().cast(pl.Int64)
            ),
            [10, 2, 30, None],
            pl.Int64,
        ),
        (pl.col("u").union.match(i=pl.element(), default=0), [1, 0, 3, None], pl.Int64),
        (
            pl.col("u").union.match(i=pl.element(), default=None),
            [1, None, 3, None],
            pl.Int64,
        ),
        (
            pl.col("u").union.match({"s": pl.element()}, i=pl.lit("int")),
            ["int", "bc", "int", None],
            pl.String,
        ),
        # Arms can refer to other columns.
        (
            pl.col("u").union.match(i=pl.element() + pl.col("k"), s=pl.col("k")),
            [6, 6, 10, None],
            pl.Int64,
        ),
        # `element()` in a nested `eval` is the list element.
        (
            pl.col("u").union.match(
                i=pl.concat_list(pl.element()).list.eval(pl.element() * 2),
                default=None,
            ),
            [[2], None, [6], None],
            pl.List(pl.Int64),
        ),
    ],
)
def test_union_match(expr: pl.Expr, expected: list[Any], dtype: pl.DataType) -> None:
    df = match_frame()
    lf = df.lazy().select(expr)
    assert lf.collect_schema() == {"u": dtype}
    for engine in ["in-memory", "streaming"]:
        out = lf.collect(engine=engine).to_series()  # type: ignore[call-overload]
        assert out.dtype == dtype
        assert out.to_list() == expected


def test_union_match_converts_unions() -> None:
    to = pl.Union({"n": pl.Int64, "t": pl.String})
    out = match_frame().select(
        pl.col("u").union.match(
            i=pl.union_value(to, n=pl.element()), s=pl.union_value(to, t=pl.element())
        )
    )
    assert out.schema == {"u": to}
    assert to_pylist(out.to_series()) == [1, "bc", 3, None]
    assert out.to_series().union.tag().to_list() == ["n", "t", "n", None]


def test_union_match_nested() -> None:
    inner = pl.Union({"a": pl.Int64})
    df = match_frame()
    # The inner `element()` is the inner union's variant.
    out = df.select(
        pl.col("u").union.match(
            i=pl.union_value(inner, a=pl.element()).union.match(a=pl.element() + 1),
            default=None,
        )
    )
    assert out.to_series().to_list() == [2, None, 4, None]
    out = df.select(
        pl.concat_list("u").list.eval(
            pl.element().union.match(i=pl.element(), default=-1)
        )
    )
    assert out.to_series().to_list() == [[1], [-1], [3], [None]]


@pytest.mark.parametrize(
    ("expr", "error", "match"),
    [
        (
            pl.col("u").union.match(i=pl.element()),
            InvalidOperationError,
            r"no arm for the variants \[`s`\]",
        ),
        (
            pl.col("u").union.match(i=1, s=1, z=1),
            pl.exceptions.StructFieldNotFoundError,
            "no variant `z`",
        ),
        (
            pl.col("u").union.match(i=pl.element(), s=pl.element()),
            pl.exceptions.SchemaError,
            "same dtype",
        ),
        (
            pl.col("u").union.match(i=pl.element(), default=1.5),
            pl.exceptions.SchemaError,
            "same dtype",
        ),
        (
            pl.col("u").union.match(i=pl.element().sum(), default=0),
            InvalidOperationError,
            "must be elementwise",
        ),
        (
            pl.col("u").union.match(i=pl.all(), default=0),
            InvalidOperationError,
            "single expression",
        ),
        (
            pl.col("k").union.match(default=1),
            InvalidOperationError,
            "requires a Union dtype",
        ),
    ],
)
def test_union_match_errors(expr: pl.Expr, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        match_frame().lazy().select(expr).collect_schema()


def test_union_match_arguments() -> None:
    dtype = pl.Union({"a b": pl.Int64, "default": pl.Int64})
    df = pl.select(u=pl.union_value(dtype, {"a b": pl.Series([1, 2])}))
    out = df.select(pl.col("u").union.match({"a b": pl.element(), "default": 0}))
    assert out.to_series().to_list() == [1, 2]
    with pytest.raises(ValueError, match="more than once"):
        pl.col("u").union.match({"i": 1}, i=2)


def test_union_selector() -> None:
    import polars.selectors as cs

    df = match_frame()
    assert df.select(cs.union()).columns == ["u"]
    assert "u" not in df.select(~cs.union()).columns
    assert df.select(cs.nested()).columns == ["u"]
    assert repr(cs.union()) == "cs.union()"
    out = df.select(cs.union().union.match(i=pl.element(), default=0))
    assert out.to_series().to_list() == [1, 0, 3, None]


@pytest.mark.parametrize("s", messy_layouts())
def test_union_distinct_flags(s: pl.Series) -> None:
    values = to_pylist(s)
    keys = [(variant_of(v), v) for v in values]
    first = [keys.index(k) == i for i, k in enumerate(keys)]
    last = [len(keys) - 1 - keys[::-1].index(k) == i for i, k in enumerate(keys)]
    unique = [keys.count(k) == 1 for k in keys]
    assert s.is_first_distinct().to_list() == first
    assert s.is_last_distinct().to_list() == last
    assert s.is_unique().to_list() == unique
    assert s.is_duplicated().to_list() == [not u for u in unique]


@pytest.mark.parametrize("s", messy_layouts())
def test_union_write_json(s: pl.Series) -> None:
    import json

    df = s.to_frame()
    expected = [{"u": v} for v in to_pylist(s)]
    assert json.loads(df.write_json()) == expected
    assert [json.loads(line) for line in df.write_ndjson().splitlines()] == expected
    buf = io.BytesIO()
    df.lazy().sink_ndjson(buf)
    assert [json.loads(line) for line in buf.getvalue().splitlines()] == expected
    encoded = df.select(pl.struct("u").struct.json_encode()).to_series()
    assert [json.loads(v) for v in encoded] == expected


def test_union_write_parquet_raises(tmp_path: Path) -> None:
    df = messy().to_frame()
    with pytest.raises(ComputeError, match="not yet implemented"):
        df.write_parquet(tmp_path / "u.parquet")
    with pytest.raises(ComputeError, match="not yet implemented"):
        df.lazy().sink_parquet(tmp_path / "u.parquet")
