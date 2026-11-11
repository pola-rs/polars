from __future__ import annotations

import io
import pickle
from datetime import datetime
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


def test_union_import_large_child() -> None:
    # A child longer than `u32::MAX`, with decreasing offsets and a dropped `Null` variant,
    # so the import rebuilds the children.
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
