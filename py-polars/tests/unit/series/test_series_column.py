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
    assert df["a"].to_frame().to_series()._s._column_kind() == kind
    assert df.clone().drop_in_place("b")._s._column_kind() == kind
    # A Series that is a plain `Column::Series` keeps today's conversion on frame entry.
    assert pl.Series("a", [1])._s._column_kind() == "series"
    assert pl.Series("a", [1]).to_frame()["a"]._s._column_kind() == "scalar"
    # A broadcast literal is a scalar column of any length.
    wide = pl.DataFrame({"x": [1, 2, 3]}).with_columns(pl.lit(7).alias("c"))
    assert wide["c"]._s._column_kind() == "scalar"
    assert wide["c"].to_frame()["c"]._s._column_kind() == "scalar"


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
