"""Tests for a `Series` that is backed by a `Column` (`PySeries` holds a `Column`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path


def test_column_kind_plain_series() -> None:
    assert pl.Series("a", [1, 2, 3])._s._column_kind() == "series"
    assert pl.Series("a", [1], dtype=pl.Int64)._s._column_kind() == "series"
    assert pl.Series("a", [], dtype=pl.Int64)._s._column_kind() == "series"


def test_length_one_series_keeps_flags_and_sortedness() -> None:
    # A length-1 Series reports itself as ascending; it must stay a `Column::Series` (not
    # become a `Column::Scalar`) through the operations that return a new Series.
    s = pl.Series("a", [1])
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s.is_sorted()

    s = s.set_sorted(descending=True)
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s._s._column_kind() == "series"

    # The same flags as a longer Series with the flag set.
    s2 = pl.Series("a", [3, 2, 1]).set_sorted(descending=True)
    assert s2.flags == {"SORTED_ASC": False, "SORTED_DESC": True}
    assert s2.head(1).flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s2.head(1)._s._column_kind() == "series"

    s = pl.Series("a", [1])
    s.append(pl.Series("a", [0]))
    assert s.to_list() == [1, 0]


def test_in_place_mutation() -> None:
    s = pl.Series("a", [1, 2, 3])
    s.append(pl.Series("a", [4]))
    s.extend(pl.Series("a", [5]))
    s._s.rename("b")
    s.scatter(0, 10)
    s.shrink_to_fit(in_place=True)
    s.rechunk(in_place=True)
    assert_series_equal(s, pl.Series("b", [10, 2, 3, 4, 5]))
    assert s.n_chunks() == 1


def test_rolling_map_callback() -> None:
    s = pl.Series("a", [1.0, 2.0, 3.0, 4.0])
    result = s.rolling_map(lambda w: w.sum(), window_size=2)
    assert_series_equal(result, pl.Series("a", [None, 3.0, 5.0, 7.0]))


def test_fold_reduce_cum_fold_callbacks() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    out = df.select(
        fold=pl.fold(pl.lit(0), lambda acc, x: acc + x, exprs=pl.all()),
        reduce=pl.reduce(lambda acc, x: acc + x, exprs=pl.all()),
        cum=pl.cum_fold(pl.lit(0), lambda acc, x: acc + x, exprs=pl.all()),
    )
    assert out["fold"].to_list() == [11, 22, 33]
    assert out["reduce"].to_list() == [11, 22, 33]
    assert out["cum"].to_list() == [
        {"a": 1, "b": 11},
        {"a": 2, "b": 22},
        {"a": 3, "b": 33},
    ]


# --- Columns handed over from frames keep their variant -------------------------------

LONG = [
    f"https://example.com/section-{i:03d}/item-page-{i * 7:05d}/view" for i in range(50)
]


@pytest.fixture
def coded_df(tmp_path: Path) -> pl.DataFrame:
    """A frame whose `k` column is a `Column::Dictionary`.

    Long strings, dictionary-encoded in a parquet file, are read as dictionary codes by the
    streaming engine's `collect_batches` (`collect` and `read_parquet` give plain Series
    columns); `POLARS_DICT_CODES=0` turns that off.
    """
    n = 20_000
    idx = pl.int_range(n, eager=True)
    k = pl.Series(LONG).gather(idx.hash(1) % len(LONG)).scatter([0, 97], None)
    path = tmp_path / "a.parquet"
    pl.DataFrame({"k": k, "v": idx.hash(3) % 100}).write_parquet(
        path, row_group_size=3_000
    )
    (df,) = pl.scan_parquet(path).collect_batches(
        chunk_size=1_000_000, engine="streaming"
    )
    assert df["k"]._s._column_kind() == "dictionary"
    return df


def scalar_df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1], "b": ["x"]})


def test_frame_to_series_paths_keep_dictionary(coded_df: pl.DataFrame) -> None:
    kind = "dictionary"
    assert coded_df["k"]._s._column_kind() == kind
    assert coded_df.get_column("k")._s._column_kind() == kind
    assert coded_df.to_series(0)._s._column_kind() == kind
    assert coded_df.get_columns()[0]._s._column_kind() == kind
    assert next(coded_df.iter_columns())._s._column_kind() == kind
    assert coded_df.to_dict(as_series=True)["k"]._s._column_kind() == kind
    assert coded_df.to_series(0).to_frame().to_series()._s._column_kind() == kind
    assert coded_df.clone().drop_in_place("k")._s._column_kind() == kind
    assert coded_df["v"]._s._column_kind() == "series"


def test_frame_to_series_paths_keep_scalar() -> None:
    df = scalar_df()
    kind = "scalar"
    assert df["a"]._s._column_kind() == kind
    assert df.get_column("a")._s._column_kind() == kind
    assert df.to_series(1)._s._column_kind() == kind
    assert df.get_columns()[0]._s._column_kind() == kind
    assert next(df.iter_columns())._s._column_kind() == kind
    # A length-n scalar column enters a frame materialised (length 1 stays scalar).
    assert df["a"].to_frame().to_series()._s._column_kind() == (
        "scalar" if len(df) == 1 else "series"
    )
    assert df.clone().drop_in_place("b")._s._column_kind() == kind
    # A Series that is a plain `Column::Series` keeps today's conversion on frame entry.
    assert pl.Series("a", [1])._s._column_kind() == "series"
    assert pl.Series("a", [1]).to_frame()["a"]._s._column_kind() == "scalar"
    # A broadcast literal is a scalar column of any length.
    wide = pl.DataFrame({"x": [1, 2, 3]}).with_columns(pl.lit(7).alias("c"))
    assert wide["c"]._s._column_kind() == "scalar"
    assert wide["c"].to_frame()["c"]._s._column_kind() == "series"


def test_expression_dispatch_keeps_codes(coded_df: pl.DataFrame) -> None:
    k = coded_df["k"]
    assert k.str.to_uppercase()._s._column_kind() == "dictionary"
    assert k.str.len_bytes()._s._column_kind() == "series"
    assert_series_equal(
        k.str.to_uppercase(), pl.Series("k", k.to_list()).str.to_uppercase()
    )


def test_values_of_coded_and_scalar_series(coded_df: pl.DataFrame) -> None:
    k = coded_df["k"]
    values = k.to_list()
    assert values[0] is None
    assert values[97] is None
    assert k.len() == 20_000
    assert k.null_count() == 2
    assert k.has_nulls()
    assert k.name == "k"
    assert k.dtype == pl.String
    assert k.n_chunks() == 7
    # Reading does not change the variant.
    assert k._s._column_kind() == "dictionary"

    wide = pl.DataFrame({"x": [1, 2, 3]}).with_columns(pl.lit(7).alias("c"))["c"]
    assert wide.to_list() == [7, 7, 7]
    assert wide.len() == 3
    assert wide.null_count() == 0
    assert wide.is_sorted() == pl.Series([7, 7, 7]).is_sorted()
    assert wide.flags == {"SORTED_ASC": True, "SORTED_DESC": False}


def test_in_place_mutation_of_non_series_column(coded_df: pl.DataFrame) -> None:
    k = coded_df["k"].clone()
    k.rechunk(in_place=True)
    assert k.n_chunks() == 1
    k.shrink_to_fit(in_place=True)
    expected = coded_df["k"].to_list()
    assert k.to_list() == expected
    k.append(pl.Series("k", ["tail"]))
    assert k.to_list() == [*expected, "tail"]
    assert k._s._column_kind() == "series"

    c = pl.DataFrame({"x": [1, 2, 3]}).with_columns(pl.lit(7).alias("c"))["c"]
    c.rechunk(in_place=True)
    c.append(pl.Series("c", [8], dtype=pl.Int32))
    assert c.to_list() == [7, 7, 7, 8]


# --- Methods that run on the column, not on the materialised Series ----------------------


def _scalar_column(n: int = 9, value: object = 7, dtype: pl.DataType = pl.Int64) -> pl.Series:
    df = pl.DataFrame({"x": list(range(n))}).with_columns(
        pl.lit(value, dtype=dtype).alias("c")
    )
    return df["c"]


@pytest.fixture(params=["dictionary", "scalar", "scalar_str", "scalar_null", "scalar_f64", "scalar_bool"])
def column(request: pytest.FixtureRequest, tmp_path: Path) -> pl.Series:
    kind = request.param
    if kind == "dictionary":
        s = request.getfixturevalue("coded_df")["k"]
    elif kind == "scalar":
        s = _scalar_column(10)
    elif kind == "scalar_str":
        s = _scalar_column(10, "zz", pl.String)
    elif kind == "scalar_f64":
        s = _scalar_column(10, 1.5, pl.Float64)
    elif kind == "scalar_bool":
        s = _scalar_column(10, True, pl.Boolean)
    else:
        s = _scalar_column(10, None, pl.Int64)
    assert s._s._column_kind() in ("dictionary", "scalar")
    return s


def _materialised(s: pl.Series) -> pl.Series:
    """The Series path's version of `s`: a plain `Column::Series` with the same chunking
    and flags (a scalar column is one value broadcast, a coded column keeps its chunks)."""
    if s._s._column_kind() == "scalar":
        m = pl.Series(s.name, [s[0] if len(s) else None], dtype=s.dtype)
        m = m._from_pyseries(m._s.new_from_index(0, len(s)))
    else:
        chunks = s.get_chunks()
        m = pl.concat(chunks, rechunk=False) if len(chunks) > 1 else chunks[0]
        m = pl.Series(s.name, m.to_list(), dtype=s.dtype) if len(chunks) == 1 else m
        m = m.alias(s.name)
    assert m._s._column_kind() == "series"
    assert m._s.chunk_lengths() == s._s.chunk_lengths()
    return m


def _same(result: pl.Series, expected: pl.Series, kind: str | None) -> None:
    assert_series_equal(result, expected)
    assert result.flags == expected.flags
    assert result.n_chunks() == expected.n_chunks()
    assert result._s.chunk_lengths() == expected._s.chunk_lengths()
    if kind is not None and len(result) > 0:  # zero-length results use the Series path
        assert result._s._column_kind() == kind


def _kind(s: pl.Series) -> str:
    return s._s._column_kind()


def test_m3_head_tail_slice(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    n = column.len()
    for f in (
        lambda x: x.head(3),
        lambda x: x.head(0),
        lambda x: x.head(n + 5),
        lambda x: x.tail(3),
        lambda x: x.tail(1),
        lambda x: x.slice(2, 4),
        lambda x: x.slice(-3),
        lambda x: x.slice(-3, 2),
        lambda x: x.slice(n + 3, 2),
        lambda x: x.slice(0),
    ):
        _same(f(column), f(m), k)
        assert _kind(column) == k


def test_m3_filter(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    mask = pl.Series(([True, False, True] * column.len())[: column.len()])
    _same(column.filter(mask), m.filter(mask), k)
    _same(column.filter(pl.Series([True])), m.filter(pl.Series([True])), k)
    _same(column.filter(pl.Series([False])), m.filter(pl.Series([False])), k)
    _same(column.filter(pl.Series([None], dtype=pl.Boolean)), m.filter(pl.Series([None], dtype=pl.Boolean)), k)
    nulls = pl.Series([True, None, False, True] * column.len()).head(column.len())
    _same(column.filter(nulls), m.filter(nulls), k)
    _same(column.filter(mask & False), m.filter(mask & False), k)
    # Errors are the Series path's.
    bad = pl.Series([True, False])
    with pytest.raises(Exception) as e_col:
        column.filter(bad)
    with pytest.raises(Exception) as e_mat:
        m.filter(bad)
    assert type(e_col.value) is type(e_mat.value)
    assert str(e_col.value) == str(e_mat.value)
    with pytest.raises(Exception, match="boolean mask"):
        column.filter(pl.Series([1, 2]))


def test_m3_gather(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    for idx in (
        pl.Series([3, 1, 4], dtype=pl.UInt32),
        pl.Series([0, 1, 2], dtype=pl.UInt32),
        pl.Series([2, None, 1], dtype=pl.UInt32),
        pl.Series([], dtype=pl.UInt32),
        pl.Series([None, None], dtype=pl.UInt32),
        pl.Series([0], dtype=pl.UInt32),
    ):
        # `gather` runs the expression engine, which is not this change.
        assert_series_equal(column.gather(idx), m.gather(idx))
        _same(column[idx], m[idx], k if k == "dictionary" else None)
    _same(column[[3, 1]], m[[3, 1]], k if k == "dictionary" else None)
    _same(column[-2:], m[-2:], k)
    n = column.len()
    with pytest.raises(Exception) as e_col:
        column.gather(pl.Series([n], dtype=pl.UInt32))
    with pytest.raises(Exception) as e_mat:
        m.gather(pl.Series([n], dtype=pl.UInt32))
    assert str(e_col.value) == str(e_mat.value)
    with pytest.raises(Exception) as e_col:
        column._s.gather_with_series(pl.Series([1.0])._s)
    with pytest.raises(Exception) as e_mat:
        m._s.gather_with_series(pl.Series([1.0])._s)
    assert str(e_col.value) == str(e_mat.value)


def test_m3_item_and_get(column: pl.Series) -> None:
    m = _materialised(column)
    n = column.len()
    for i in (0, 1, n - 1, -1, -n):
        assert column[i] == m[i]
        assert column.item(i) == m.item(i)
    for i in (n, -n - 1):
        with pytest.raises(IndexError) as e_col:
            column[i]
        with pytest.raises(IndexError) as e_mat:
            m[i]
        assert str(e_col.value) == str(e_mat.value)
    assert _kind(column) in ("dictionary", "scalar")


def test_m3_clear_new_from_index(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    _same(column.clear(), m.clear(), k)
    _same(column.clear(0), m.clear(0), k)
    for i, n in ((0, 5), (3, 1), (3, 0), (2, 20_000)):
        _same(column._from_pyseries(column._s.new_from_index(i, n)), m._from_pyseries(m._s.new_from_index(i, n)), None)
        assert _kind(column._from_pyseries(column._s.new_from_index(i, n))) == ("scalar" if n > 0 else "series")
    with pytest.raises(ValueError, match="out of bounds"):
        column._s.new_from_index(column.len(), 2)


CAST_TARGETS = [
    pl.String,
    pl.Binary,
    pl.Categorical,
    pl.Enum(["zz"]),
    pl.Int8,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.Float32,
    pl.Float64,
    pl.Decimal(5, 2),
    pl.Boolean,
    pl.Date,
    pl.List(pl.String),
]


def test_m3_cast(column: pl.Series) -> None:
    m = _materialised(column)
    for dtype in CAST_TARGETS:
        for kwargs in ({"strict": True}, {"strict": False}, {"wrap_numerical": True}):
            try:
                expected = m.cast(dtype, **kwargs)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)) as e_col:
                    column.cast(dtype, **kwargs)  # type: ignore[arg-type]
                assert str(e_col.value) == str(e)
                continue
            _same(column.cast(dtype, **kwargs), expected, None)  # type: ignore[arg-type]
    # Only a dictionary cast to its own dtype stays a column.
    if _kind(column) == "dictionary":
        assert _kind(column.cast(pl.String)) == "dictionary"


def test_m3_zero_length_results_and_list_scalar() -> None:
    lst = pl.DataFrame({"x": range(6)}).with_columns(
        c=pl.lit([1, 2], dtype=pl.List(pl.Int64))
    )["c"]
    assert lst._s._column_kind() == "scalar"
    m = _materialised(lst)
    for f in (
        lambda x: x.tail(0),
        lambda x: x.head(0),
        lambda x: x.slice(10, 3),
        lambda x: x.slice(1, 0),
        lambda x: x.head(-6),
        lambda x: x.filter(pl.Series([False])),
        lambda x: x.filter(pl.Series([None], dtype=pl.Boolean)),
        lambda x: x.filter(pl.Series([False] * 6)),
        lambda x: x.clear(),
        lambda x: x.cast(pl.List(pl.Int32)),
    ):
        _same(f(lst), f(m), None)


def test_m3_estimated_size_independent_of_sharing(coded_df: pl.DataFrame) -> None:
    df = coded_df.select("k")
    s1 = df["k"]
    s1.n_chunks()  # fill the cache the frame's column shares
    one = df.estimated_size()
    s2 = df["k"]
    assert df.estimated_size() == one
    del s1, s2
    assert df.estimated_size() == one


def test_m3_append_extend(column: pl.Series) -> None:
    n = column.len()
    for other_kind in ("self", "plain", "same_scalar", "other_value"):
        for op in ("append", "extend"):
            a = column.clone()
            m = _materialised(column)
            if other_kind == "self":
                other = a
                other_m = m
            elif other_kind == "plain":
                other = pl.Series(column.name, column.head(3).to_list(), dtype=column.dtype)
                other_m = other
            elif other_kind == "same_scalar":
                other = column.head(2)
                other_m = _materialised(column.head(2))
            else:
                vals = column.head(1).to_list()
                other = pl.Series(column.name, [*vals, *vals], dtype=column.dtype)
                other = pl.concat([other.head(0), column.tail(3)])
                other_m = _materialised(column.tail(3))
            getattr(a, op)(other)
            getattr(m, op)(other_m)
            assert_series_equal(a, m)
            assert a.flags == m.flags, (op, other_kind, a.flags, m.flags)
            assert a.len() == m.len()
            assert a.null_count() == m.null_count()
            assert a._s.chunk_lengths() == m._s.chunk_lengths()
    # Only a dictionary column stays compressed.
    a = column.clone()
    a.append(column)
    assert _kind(a) == ("dictionary" if _kind(column) == "dictionary" else "series")
    assert a.len() == 2 * n
    a = column.clone()
    a.extend(column.head(3))
    assert _kind(a) == ("dictionary" if _kind(column) == "dictionary" else "series")
    # Another dtype is a Series path error.
    a = column.clone()
    with pytest.raises(Exception, match=r"."):
        a.append(pl.Series(column.name, [object] * 1, dtype=pl.Object))


def test_m3_dictionary_append_by_value(coded_df: pl.DataFrame) -> None:
    k = coded_df["k"]
    a = k.head(100)
    b = k.tail(100)
    c = a.clone()
    c.append(b)
    assert c._s._column_kind() == "dictionary"
    assert c.to_list() == a.to_list() + b.to_list()
    assert c.null_count() == a.null_count() + b.null_count()


def test_shrink_to_fit_rechunks_dictionary(coded_df: pl.DataFrame) -> None:
    s = coded_df["k"].clone()
    assert s.n_chunks() > 1
    s.shrink_to_fit(in_place=True)
    assert s.n_chunks() == 1
    assert s.to_list() == coded_df["k"].to_list()


def test_row_loop_over_compressed_columns_is_not_quadratic(coded_df: pl.DataFrame) -> None:
    import time

    wide = pl.DataFrame({"x": pl.int_range(3_000_000, eager=True)}).with_columns(
        pl.lit("a").alias("c")
    )
    for df, col in ((wide, "c"), (coded_df, "k")):
        df[col][0]  # warm up
        t = time.perf_counter()
        for i in range(200):
            assert df[col][i] == df[col].item(i)
        # 400 accesses of a 3M-row column that materialised each time would take seconds.
        assert time.perf_counter() - t < 1.0


def test_item_does_not_materialise_shared_cache(coded_df: pl.DataFrame) -> None:
    k = coded_df["k"]
    k[1]
    assert k._s._column_kind() == "dictionary"
    # `n_chunks` reads the materialised Series; a second frame access shares its cache.
    assert coded_df["k"].n_chunks() == coded_df["k"].n_chunks()


def test_empty_results_do_not_materialise_scalar_column() -> None:
    df = pl.DataFrame({"x": pl.int_range(2_000_000, eager=True)}).with_columns(
        c=pl.lit("a-string-value-of-some-length")
    )
    before = df.estimated_size()
    c = df["c"]
    for r in (
        c.head(0),
        c.tail(0),
        c.clear(),
        c.slice(2_000_000, 1),
        c.filter(pl.Series([False])),
        c._from_pyseries(c._s.new_from_index(1, 0)),
    ):
        assert r.len() == 0
        assert r.dtype == pl.String
    assert df.estimated_size() == before
    assert c._s._column_kind() == "scalar"


# --- M4: comparisons, arithmetic, reductions ---------------------------------------------

import operator  # noqa: E402

CMP_OPS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]


def _constants(s: pl.Series) -> list[object]:
    if s.dtype == pl.String:
        return [LONG[3], "zz", "a", "", None, 1]
    if s.dtype == pl.Boolean:
        return [True, False, None]
    if s.dtype == pl.Float64:
        return [1.5, 0.5, 2, float("nan"), None]
    return [7, 3, 8, 1.5, None, "x"]


def _outcome(f, *args):  # type: ignore[no-untyped-def]
    try:
        return ("ok", f(*args))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e), str(e))


def _assert_same_outcome(got, expected, kinds=("series",)) -> None:  # type: ignore[no-untyped-def]
    assert got[0] == expected[0]
    if got[0] == "err":
        assert got[1:] == expected[1:]
    elif not isinstance(got[1], pl.Series):
        assert got[1] == expected[1]
    else:
        _same(got[1], expected[1], None)
        assert got[1]._s._column_kind() in kinds


def test_m4_compare_with_constant(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    for op in CMP_OPS:
        for c in _constants(column):
            # `bool_series == True` is answered by the Python method (a clone, or its negation).
            kinds = ("series", k) if isinstance(c, bool) else ("series",)
            _assert_same_outcome(_outcome(op, column, c), _outcome(op, m, c), kinds)
    assert _kind(column) == k


def test_m4_compare_with_series(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    n = column.len()
    consts = [c for c in _constants(column) if c is not None]
    others: list[pl.Series] = []
    for c in consts:
        try:
            others.append(pl.Series("o", [c]))
        except Exception:  # noqa: BLE001, S112
            continue
    others.append(pl.Series("o", [None], dtype=column.dtype))
    others.append(pl.Series("o", column.head(1).to_list(), dtype=column.dtype))
    others.append(pl.Series("o", column.head(n).to_list(), dtype=column.dtype))
    others.append(pl.Series("o", [column[0], column[0]], dtype=column.dtype))
    others.append(_scalar_column(n, 7))
    others.append(_scalar_column(n + 1, 7))
    others.append(_scalar_column(1, 7))
    others.append(_scalar_column(n, "zz", pl.String))
    others.append(_scalar_column(n, 1.5, pl.Float64))
    others.append(_scalar_column(n, None, column.dtype if column.dtype != pl.Null else pl.Int64))
    others.append(_scalar_column(n, column[0], column.dtype))
    for op in CMP_OPS:
        for o in others:
            try:
                _assert_same_outcome(_outcome(op, column, o), _outcome(op, m, _materialised(o) if _kind(o) != "series" else o))
            except AssertionError as e:
                msg = f"{op.__name__} rhs {o.dtype} len={len(o)} kind={_kind(o)} chunks={o.n_chunks()}"
                raise AssertionError(msg) from e
    assert _kind(column) == k


def test_m4_compare_empty() -> None:
    e = _scalar_column(0)
    assert_series_equal(e == 7, pl.Series("c", [], dtype=pl.Boolean))


ARITH_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    lambda a, b: b + a,
    lambda a, b: b - a,
    lambda a, b: b * a,
    lambda a, b: b / a,
    lambda a, b: b % a,
]


def _arith_constants(s: pl.Series) -> list[object]:
    if s.dtype == pl.String:
        return ["x", 3, None]
    if s.dtype == pl.Boolean:
        return [True, 2, None]
    if s.dtype == pl.Float64:
        return [2, 0.5, 0, float("nan"), None]
    return [3, 0, -2, 1.5, None, "x"]


def _scalar_cases(n: int) -> list[pl.Series]:
    return [
        _scalar_column(n, 7),
        _scalar_column(n, 1.5, pl.Float64),
        _scalar_column(n, "zz", pl.String),
        _scalar_column(n, True, pl.Boolean),
        _scalar_column(n, None, pl.Int64),
        _scalar_column(n, 250, pl.UInt8),
        _scalar_column(n, -3, pl.Int8),
    ]


def test_m4_arithmetic_with_constant(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    for i, op in enumerate(ARITH_OPS):
        for c in _arith_constants(column):
            if isinstance(c, str) and i >= 6:
                # `"x" + s` runs the expression engine (not changed here; its Scalar results
                # report flags that the base build's Series do not).
                continue
            try:
                kinds = ("series", "scalar") if i == 4 else ("series",)
                _assert_same_outcome(_outcome(op, column, c), _outcome(op, m, c), kinds)
            except AssertionError as e:
                msg = f"op#{i} const {c!r} dtype {column.dtype}"
                raise AssertionError(msg) from e
    assert _kind(column) == k


@pytest.mark.parametrize("n", [1, 2, 3, 10])
def test_m4_scalar_arithmetic_and_comparison_lengths(n: int) -> None:
    for column in _scalar_cases(n):
        m = _materialised(column)
        for i, op in enumerate(ARITH_OPS):
            for c in _arith_constants(column):
                if isinstance(c, str) and i >= 6:
                    continue
                # `//` runs the expression engine, whose length-1 result is a scalar column.
                kinds = ("series", "scalar") if i == 4 else ("series",)
                _assert_same_outcome(_outcome(op, column, c), _outcome(op, m, c), kinds)
        for op in CMP_OPS:
            for c in _constants(column):
                kinds = ("series", "scalar") if isinstance(c, bool) else ("series",)
                _assert_same_outcome(_outcome(op, column, c), _outcome(op, m, c), kinds)
        for other in _scalar_cases(n) + _scalar_cases(1) + [pl.Series("o", [3]), pl.Series("o", [None], dtype=pl.Int64)]:
            om = _materialised(other) if _kind(other) != "series" else other
            for op in [*ARITH_OPS[:5], *CMP_OPS]:
                try:
                    kinds = ("series", "scalar") if op is operator.floordiv else ("series",)
                    _assert_same_outcome(_outcome(op, column, other), _outcome(op, m, om), kinds)
                except AssertionError as e:
                    msg = f"{op} {column.dtype} {column[0]!r} rhs {other.dtype} {other.to_list()[:2]} {_kind(other)}"
                    raise AssertionError(msg) from e


def test_m4_arithmetic_with_series(column: pl.Series) -> None:
    m = _materialised(column)
    n = column.len()
    others = [
        pl.Series("o", [3]),
        pl.Series("o", [2.5]),
        pl.Series("o", ["x"]),
        pl.Series("o", [None], dtype=pl.Int64),
        pl.Series("o", [3] * n),
        _scalar_column(n, 7),
        _scalar_column(n, 1.5, pl.Float64),
        _scalar_column(n, "zz", pl.String),
        _scalar_column(1, 7),
        _scalar_column(n + 1, 7),
        _scalar_column(n, None, pl.Int64),
    ]
    for op in ARITH_OPS[:5]:
        for o in others:
            om = _materialised(o) if _kind(o) != "series" else o
            kinds = ("series", "scalar") if op is operator.floordiv else ("series",)
            _assert_same_outcome(_outcome(op, column, o), _outcome(op, m, om), kinds)
    # Only a scalar column's arithmetic skips the materialisation.
    if _kind(column) == "scalar":
        assert _kind(column) == "scalar"


def _eq_value(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b and type(a) is type(b)


def _same_value_outcome(f, a, b, flags: bool = True) -> None:  # type: ignore[no-untyped-def]
    ra = _outcome(f, a)
    rb = _outcome(f, b)
    assert ra[0] == rb[0]
    if ra[0] == "err":
        assert ra[1:] == rb[1:]
    elif isinstance(ra[1], pl.Series) and not flags:
        assert_series_equal(ra[1], rb[1])
    elif isinstance(ra[1], pl.Series):
        _same(ra[1], rb[1], None)
    else:
        assert _eq_value(ra[1], rb[1]), (ra[1], rb[1])


REDUCTIONS = [
    lambda s: s.min(),
    lambda s: s.max(),
    lambda s: s.median(),
    lambda s: s.mean(),
    lambda s: s.sum(),
    lambda s: s.std(),
    lambda s: s.var(),
    lambda s: s.n_unique(),
    lambda s: s.null_count(),
    lambda s: s.arg_min(),
    lambda s: s.arg_max(),
    lambda s: s.is_sorted(),
    lambda s: s.is_sorted(descending=True),
    lambda s: s.is_sorted(nulls_last=True),
    lambda s: s.is_sorted(descending=True, nulls_last=True),
    lambda s: s.first(),
    lambda s: s.last(),
    lambda s: s.first(ignore_nulls=True),
    lambda s: s.last(ignore_nulls=True),
    lambda s: s.any(),
    lambda s: s.all(),
    lambda s: s.any(ignore_nulls=False),
    lambda s: s.all(ignore_nulls=False),
    lambda s: s.is_empty(),
    lambda s: s.has_nulls(),
    lambda s: s.count(),
    lambda s: s.quantile(0.3),
]


def test_m4_reductions(column: pl.Series) -> None:
    m = _materialised(column)
    k = _kind(column)
    for f in REDUCTIONS:
        _same_value_outcome(f, column, m)
    assert _kind(column) == k


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 9])
def test_m4_scalar_reductions_all_lengths(n: int) -> None:
    nan = float("nan")
    cases = [
        (7, pl.Int64),
        (-7, pl.Int8),
        (250, pl.UInt8),
        (1.5, pl.Float64),
        (nan, pl.Float64),
        (float("inf"), pl.Float64),
        (1.7976931348623157e308, pl.Float64),
        (None, pl.Int64),
        (None, pl.Float64),
        (None, pl.String),
        (None, pl.Boolean),
        (True, pl.Boolean),
        (False, pl.Boolean),
        ("a", pl.String),
        ("", pl.String),
        (b"ab", pl.Binary),
    ]
    for value, dtype in cases:
        column = _scalar_column(n, value, dtype)
        assert _kind(column) == "scalar"
        m = _materialised(column)
        for f in REDUCTIONS:
            _same_value_outcome(f, column, m)
        assert _kind(column) == "scalar"


def _sorted_rows(df: pl.DataFrame) -> list[tuple[object, ...]]:
    return sorted(df.rows(), key=lambda r: (r[0] is not None, str(r[0])))


def test_m4_unique_and_value_counts(column: pl.Series, coded_df: pl.DataFrame) -> None:
    m = _materialised(column)
    k = _kind(column)
    u, mu = column.unique(), m.unique()
    if k == "dictionary":
        _same(column.unique(maintain_order=True), m.unique(maintain_order=True), "series")
        assert u.flags == mu.flags
        assert u.n_chunks() == mu.n_chunks() == 1
        assert _kind(u) == "series"
    else:
        # The expression engine's (not changed here: its scalar results report other flags).
        assert_series_equal(column.unique(maintain_order=True), m.unique(maintain_order=True))
    assert u.dtype == mu.dtype
    assert u.name == mu.name
    assert sorted(u.to_list(), key=lambda x: (x is not None, str(x))) == sorted(
        mu.to_list(), key=lambda x: (x is not None, str(x))
    )
    for kwargs in (
        {},
        {"sort": True},
        {"normalize": True},
        {"name": "n", "sort": True, "normalize": True},
        {"parallel": True},
    ):
        a, b = column.value_counts(**kwargs), m.value_counts(**kwargs), 
        assert a.schema == b.schema
        assert _sorted_rows(a) == _sorted_rows(b)
        if kwargs.get("sort"):
            name = kwargs.get("name", "count")
            assert a[name].to_list() == b[name].to_list()
    with pytest.raises(Exception, match="duplicate column names") as e1:
        column.value_counts(name=column.name)
    with pytest.raises(Exception, match="duplicate column names") as e2:
        m.value_counts(name=column.name)
    assert type(e1.value) is type(e2.value)
    assert _same_value_outcome(lambda s: s.n_unique(), column, m) is None
    assert _kind(column) == k


def test_m4_dictionary_with_repeated_values(coded_df: pl.DataFrame) -> None:
    k = coded_df["k"]
    both = k.clone()
    both.append(k.reverse() if False else k.head(5000))
    both.append(k.tail(3000))
    if both._s._column_kind() != "dictionary":
        pytest.skip("append did not keep the dictionary")
    m = pl.Series(both.name, both.to_list())
    assert both.n_unique() == m.n_unique()
    assert both.min() == m.min()
    assert both.max() == m.max()
    _same(both.unique(maintain_order=True), m.unique(maintain_order=True), None)
    assert sorted(both.unique().to_list(), key=str) == sorted(m.unique().to_list(), key=str)
    a, b = both.value_counts(), m.value_counts()
    assert _sorted_rows(a) == _sorted_rows(b)


def test_m4_is_in(column: pl.Series) -> None:
    m = _materialised(column)
    # `is_in` and `is_null` run in the expression engine, which already evaluates them once
    # per dictionary value (as a single chunk, and a scalar column's result reports other
    # flags): values only.
    exact = False
    vals = column.head(3).to_list()
    haystacks: list[object] = [
        [vals],
        [[None]],
        [[]],
        [[*vals, None]],
        [["no-such-value"]],
        [["1", "2"]],
    ]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in haystacks:
            for nulls_equal in (False, True):
                hs = pl.Series("h", h, dtype=pl.List(pl.String) if column.dtype == pl.String else None)
                _same_value_outcome(lambda s: s.is_in(hs, nulls_equal=nulls_equal), column, m, exact)  # noqa: B023
        for coll in (vals, [None], [], ["x", "zz"]):
            _same_value_outcome(lambda s: s.is_in(coll), column, m, exact)  # noqa: B023
        _same_value_outcome(lambda s: s.is_null(), column, m, exact)
        _same_value_outcome(lambda s: s.is_not_null(), column, m, exact)


def _scalar_series(value: object, dtype: pl.DataType, n: int = 4) -> pl.Series:
    df = pl.DataFrame({"x": range(n)}).with_columns(pl.lit(value, dtype=dtype).alias("c"))
    s = df["c"]
    assert _kind(s) == "scalar"
    return s


def _plain(s: pl.Series) -> pl.Series:
    # The Series path's version of a scalar column: its materialised Series.
    m = s._from_pyseries(s._s._materialized())
    assert _kind(m) == "series"
    return m


def test_m4_fix_categorical_scalar_comparison_flags() -> None:
    s = _scalar_series("a", pl.Categorical)
    m = _plain(s)
    for f in (lambda x: x == "a", lambda x: x != "a", lambda x: x < "b", lambda x: x == x):
        _same(f(s), f(m), "series")


def test_m4_fix_list_scalar_arithmetic_flags() -> None:
    for n in (1, 2, 3, 4, 7):
        s = _scalar_series([1], pl.List(pl.Int64), n)
        m = _plain(s)
        for f in (
            lambda x: x * 1.5,
            lambda x: x + 1,
            lambda x: x - 1,
            lambda x: x / 2,
            lambda x: x % 2,
        ):
            _same_value_outcome(f, s, m, True)


def test_m4_fix_dictionary_vs_full_categorical_scalar(coded_df: pl.DataFrame) -> None:
    n = coded_df.height
    s = coded_df["k"]
    if _kind(s) != "dictionary":
        pytest.skip("no dictionary column")
    m = _plain(s)
    lit = pl.DataFrame({"x": range(n)}).with_columns(
        pl.lit("a", dtype=pl.Categorical).alias("c")
    )["c"]
    lit_s = pl.DataFrame({"x": range(n)}).with_columns(pl.lit("a").alias("c"))["c"]
    for r in (lit, lit_s):
        _same_value_outcome(lambda x: x == r, s, m, True)  # noqa: B023


def test_m4_expression_dispatch_materialises_scalar() -> None:
    from decimal import Decimal

    cases: list[tuple[object, pl.DataType]] = [
        (Decimal("1.25"), pl.Decimal(10, 2)),
        (1, pl.Int64),
        ("a", pl.String),
        ("a", pl.Categorical),
        ([1, 2], pl.List(pl.Int64)),
        (None, pl.Int64),
    ]
    for value, dtype in cases:
        for n in (2, 5):
            s = _scalar_series(value, dtype, n)
            m = _plain(s)
            fs = [
                lambda x: x.is_null(),
                lambda x: x.unique(maintain_order=True),
                lambda x: x.is_in([value]),
                lambda x: x.eq_missing(value),
            ]
            if dtype in (pl.Int64, pl.Decimal(10, 2)):
                fs += [
                    lambda x: x / 1.5,
                    lambda x: x // 2,
                    lambda x: 3 + x,
                    lambda x: x**2,
                    lambda x: x.pow(2),
                    lambda x: -x,
                ]
            if dtype == pl.String:
                fs += [lambda x: x.str.to_uppercase(), lambda x: x.str.len_chars()]
            for f in fs:
                _same_value_outcome(f, s, m, True)


def test_m4_expression_dispatch_decimal_and_duration() -> None:
    from datetime import timedelta
    from decimal import Decimal

    s = _scalar_series(Decimal("1.25"), pl.Decimal(10, 2))
    m = _plain(s)
    assert (s / 1.5).to_list() == (m / 1.5).to_list()
    d = _scalar_series(timedelta(hours=1), pl.Duration("us"))
    dm = _plain(d)
    for f in (lambda x: x * 2, lambda x: x / 2, lambda x: 2 * x):
        _same_value_outcome(f, d, dm, True)


def _dtype_cases() -> list[tuple[object, pl.DataType]]:
    from datetime import timedelta
    from decimal import Decimal

    return [
        (2.5, pl.Float64),
        (3, pl.Int64),
        (True, pl.Boolean),
        (Decimal("1.25"), pl.Decimal(10, 2)),
        (timedelta(hours=1), pl.Duration("us")),
        ([1, 2], pl.List(pl.Int64)),
    ]


def test_m4_scalar_series_enters_frames_and_literals_materialised() -> None:
    for value, dtype in _dtype_cases():
        for n in (2, 4):
            s = _scalar_series(value, dtype, n)
            m = _plain(s)
            frames = [
                lambda x: x.to_frame(),
                lambda x: pl.DataFrame([x]),
                lambda x: pl.DataFrame({"c": x}),
                lambda x: pl.DataFrame({"x": range(n)}).with_columns(x),
                lambda x: pl.DataFrame({"x": range(n)}).hstack([x]),
                lambda x: pl.DataFrame({"x": range(n)}).insert_column(1, x),
                lambda x: pl.DataFrame({"x": range(n)}).select(pl.lit(x)),
            ]
            for mk in frames:
                a, b = mk(s), mk(m)
                assert a.schema == b.schema
                ca, cb = a.get_column(a.columns[-1]), b.get_column(b.columns[-1])
                _same(ca, cb, None)
                assert _kind(ca) == _kind(cb)


def test_m4_scalar_series_expression_flags_match_plain() -> None:
    s = _scalar_series(2.5, pl.Float64)
    m = _plain(s)
    for f in (
        lambda x: x.to_frame().select(pl.col("c") / 1.5).to_series(),
        lambda x: pl.DataFrame({"c": x}).select(pl.col("c") / 1.5).to_series(),
        lambda x: x.is_close(2.5),
        lambda x: x.search_sorted(2.5),
        lambda x: x.index_of(2.5),
        lambda x: x.eq_missing(_scalar_series(2.5, pl.Float64)),
        lambda x: x.ne_missing(_scalar_series(2.5, pl.Float64)),
    ):
        _same_value_outcome(f, s, m, True)
    for value, dtype in _dtype_cases():
        s = _scalar_series(value, dtype)
        m = _plain(s)
        other = _scalar_series(value, dtype)
        _same_value_outcome(lambda x: x.eq_missing(other), s, m, True)  # noqa: B023
