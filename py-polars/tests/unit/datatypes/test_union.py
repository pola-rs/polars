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
    return arr.to_pylist()


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


def test_union_cast_raises() -> None:
    s = pl.Series(int_str())
    assert s.cast(INT_STR).to_list() == INT_STR_VALUES
    with pytest.raises(InvalidOperationError):
        s.cast(pl.String)


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

    pl.testing.assert_series_equal(messy(), compact)
    pl.testing.assert_series_equal(layouts[4], layouts[0])
    with pytest.raises(AssertionError):
        pl.testing.assert_series_equal(messy(), messy().reverse())


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
