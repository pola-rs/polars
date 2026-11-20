"""Scalar and Dictionary columns held by a `pl.Series` through literals, concat, UDFs, export."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

LONG = [
    f"https://example.com/section-{i:03d}/item-page-{i * 7:05d}/view" for i in range(50)
]
N = 20_000


def _kind(s: pl.Series) -> str:
    return s._s._column_kind()


def _coded_df(tmp_path: Path) -> pl.DataFrame:
    idx = pl.int_range(N, eager=True)
    k = pl.Series(LONG).gather(idx.hash(1) % len(LONG)).scatter([0, 97], None)
    path = tmp_path / "a.parquet"
    pl.DataFrame({"k": k, "v": idx.hash(3) % 100}).write_parquet(
        path, row_group_size=3_000
    )
    (df,) = pl.scan_parquet(path).collect_batches(
        chunk_size=1_000_000, engine="streaming"
    )
    assert _kind(df["k"]) == "dictionary"
    return df


def _scalar(n: int, value: object = 7, dtype: pl.DataType = pl.Int64) -> pl.Series:
    s = pl.DataFrame({"x": list(range(n))}).with_columns(
        pl.lit(value, dtype=dtype).alias("c")
    )["c"]
    assert _kind(s) == "scalar"
    return s


@pytest.fixture
def coded(tmp_path: Path) -> pl.Series:
    return _coded_df(tmp_path)["k"]


def _plain(s: pl.Series) -> pl.Series:
    m = pl.Series(s.name, s.to_list(), dtype=s.dtype)
    assert _kind(m) == "series"
    return m


@pytest.fixture(params=["dictionary", "scalar", "scalar_str"])
def column(request: pytest.FixtureRequest, tmp_path: Path) -> pl.Series:
    if request.param == "dictionary":
        return _coded_df(tmp_path)["k"]
    if request.param == "scalar":
        return _scalar(N)
    return _scalar(N, "zz", pl.String)


# --- lit(series) --------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_lit_series_keeps_variant(column: pl.Series, engine: str) -> None:
    base = pl.DataFrame({"i": pl.int_range(len(column), eager=True)})
    kind = _kind(column)
    expected = _plain(column)

    # The streaming engine does not keep Dictionary columns through its pipelines; a Scalar
    # column becomes a Series at the literal (the engine's scalar path differs from a plain
    # column's).
    keeps = kind == "dictionary" and engine == "in-memory"
    kind_in = "series" if kind == "scalar" else kind

    out = base.lazy().with_columns(column).collect(engine=engine)  # type: ignore[arg-type]
    assert_series_equal(out[column.name], expected)
    assert out.columns == ["i", column.name]
    assert not keeps or _kind(out[column.name]) == kind

    out = base.lazy().select(column).collect(engine=engine)  # type: ignore[arg-type]
    assert_series_equal(out[column.name], expected)
    assert not keeps or _kind(out[column.name]) == kind

    # Eager frame paths.
    assert _kind(base.with_columns(column)[column.name]) == kind_in
    assert _kind(base.select(column)[column.name]) == kind_in
    assert _kind(base.with_columns(pl.lit(column))[column.name]) == kind_in
    assert _kind(base.with_columns(r=column)["r"]) == kind_in


def test_lit_series_does_not_materialise_cache(column: pl.Series) -> None:
    if _kind(column) != "dictionary":
        pytest.skip("a Scalar column is materialised at the literal")
    base = pl.DataFrame({"i": pl.int_range(len(column), eager=True)})
    assert not column._s._is_materialised()
    for f in (
        lambda: base.with_columns(column),
        lambda: base.select(column),
        lambda: base.lazy().with_columns(column).collect(engine="in-memory"),
        lambda: base.lazy().select(column).collect(engine="in-memory"),
        lambda: base.lazy().with_columns(column).explain(),
        lambda: pl.lit(column),
        lambda: base.lazy().select(column, column.alias("b")).collect(engine="in-memory"),
    ):
        out = f()
        assert not column._s._is_materialised()
        if isinstance(out, pl.DataFrame):
            assert _kind(out[column.name]) == _kind(column)
            assert not out[column.name]._s._is_materialised()
    assert_series_equal(base.with_columns(column)[column.name], _plain(column))


def test_lit_series_equal_literals_recognised(column: pl.Series) -> None:
    n = len(column)
    df = pl.DataFrame({"i": pl.int_range(n, eager=True)})
    plain = _plain(column)
    got = df.select(a=column, b=column, c=pl.col("i") * 2)
    exp = df.select(a=plain, b=plain, c=pl.col("i") * 2)
    assert_frame_equal(got, exp)


def test_lit_series_other_consumers(column: pl.Series) -> None:
    n = len(column)
    df = pl.DataFrame({"i": pl.int_range(n, eager=True)})
    plain = _plain(column)
    got = df.select(pl.when(pl.col("i") % 2 == 0).then(column).otherwise(None).alias("w"))
    exp = df.select(pl.when(pl.col("i") % 2 == 0).then(plain).otherwise(None).alias("w"))
    assert_frame_equal(got, exp)

    needles = column.head(3)
    got = df.select(column.is_in(needles).alias("m"))
    exp = df.select(plain.is_in(needles.to_list()).alias("m"))
    assert_frame_equal(got, exp)

    got = df.with_columns(pl.col("i").is_in(pl.Series([1, 2, 3])).alias("m"))
    assert got["m"].sum() == 3


def test_lit_series_plain_unchanged() -> None:
    df = pl.DataFrame({"i": [1, 2, 3]})
    s = pl.Series("p", [4, 5, 6])
    assert _kind(df.with_columns(s)["p"]) == "series"
    assert df.select(pl.Series("p", [4])).height == 1


def test_lit_series_serialisation_unchanged(column: pl.Series) -> None:
    lf = pl.LazyFrame({"i": pl.int_range(len(column), eager=True)}).with_columns(column)
    plain = pl.LazyFrame({"i": pl.int_range(len(column), eager=True)}).with_columns(
        _plain(column)
    )
    # Byte equality with the base build is checked by the export probe.
    b = lf.serialize()
    assert_frame_equal(pl.LazyFrame.deserialize(io.BytesIO(b)).collect(), plain.collect())
    assert lf.explain() == plain.explain()


# --- pl.concat of Series ------------------------------------------------------------


def test_concat_series(column: pl.Series) -> None:
    kind = _kind(column)
    out = pl.concat([column, column.head(10)])
    assert_series_equal(out, pl.concat([_plain(column), _plain(column).head(10)]))
    # A Scalar column keeps the Series path.
    assert _kind(out) == ("dictionary" if kind == "dictionary" else "series")
    # The inputs are not modified.
    assert len(column) == N
    assert _kind(column) == kind

    out = pl.concat([column, column, column], rechunk=False)
    assert len(out) == 3 * N
    assert out.to_list() == column.to_list() * 3


def test_concat_series_mixed(coded: pl.Series) -> None:
    plain = _plain(coded)
    assert _kind(pl.concat([plain, coded])) == "series"
    assert_series_equal(pl.concat([plain, coded]), pl.concat([plain, plain]))
    out = pl.concat([coded, plain])
    assert_series_equal(out, pl.concat([plain, plain]))
    with pytest.raises(Exception, match="type"):
        pl.concat([coded, pl.Series("k", [1, 2])])


def test_concat_plain_series_unchanged() -> None:
    a, b = pl.Series("a", [1, 2]), pl.Series("a", [3])
    out = pl.concat([a, b])
    assert out.to_list() == [1, 2, 3]
    assert _kind(out) == "series"


# --- map_batches ----------------------------------------------------------------------


def test_map_batches_returns_column(coded: pl.Series) -> None:
    df = pl.DataFrame({"k": coded})
    assert _kind(df["k"]) == "dictionary"
    out = df.lazy().select(pl.col("k").map_batches(lambda s: s)).collect(engine="in-memory")
    assert _kind(out["k"]) == "dictionary"
    assert_series_equal(out["k"], _plain(coded))

    # A UDF that returns another dictionary-backed Series (here a slice), kept as one.
    out = (
        df.lazy()
        .select(pl.col("k").map_batches(lambda s: s.head(5)))
        .collect(engine="in-memory")
    )
    assert _kind(out["k"]) == "dictionary"
    assert out["k"].to_list() == coded.head(5).to_list()


def test_map_batches_scalar(column: pl.Series) -> None:
    if _kind(column) != "scalar":
        pytest.skip("scalar only")
    df = pl.DataFrame({"c": column})
    out = df.lazy().select(pl.col("c").map_batches(lambda s: s)).collect(engine="in-memory")
    assert _kind(out["c"]) == "series"
    assert_series_equal(out["c"], _plain(column))


# --- export ----------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["to_arrow", "to_arrow_old", "stream"])
def test_export_does_not_materialise_dictionary(tmp_path: Path, how: str) -> None:
    s = _coded_df(tmp_path)["k"]
    other = _coded_df(tmp_path)["k"]
    assert not s._s._is_materialised()
    if how == "to_arrow":
        arr = s.to_arrow(compat_level=pl.CompatLevel.newest())
        assert arr.type == pa.string_view()
    elif how == "to_arrow_old":
        arr = s.to_arrow(compat_level=pl.CompatLevel.oldest())
        assert arr.type == pa.large_string()
    else:
        arr = pa.chunked_array(s)
        assert [len(c) for c in arr.chunks] == [3000] * 6 + [2000]
    assert not s._s._is_materialised()
    assert not other._s._is_materialised()
    assert arr.to_pylist() == _plain(s).to_list()
    assert _kind(s) == "dictionary"


def test_concat_keeps_sorted_flag(coded: pl.Series) -> None:
    d = coded.slice(5, 1)
    out = pl.concat([d, d])
    assert out.flags == pl.concat([_plain(d), _plain(d)]).flags


def test_export_dictionary_shapes(tmp_path: Path) -> None:
    s = _coded_df(tmp_path)["k"]
    for t in (
        s,
        s.slice(5, 10_000),
        s.clear(),
        pl.concat([s, s.head(777)], rechunk=False),
    ):
        # An empty result takes the Series path (M3 rule on the task branch).
        assert _kind(t) == ("dictionary" if len(t) else "series")
        exp = _plain(t)
        n_chunks = t.n_chunks()
        arr = pa.chunked_array(t)
        assert arr.to_pylist() == exp.to_list()
        assert arr.num_chunks == n_chunks
        assert t.to_arrow(compat_level=pl.CompatLevel.newest()).to_pylist() == exp.to_list()
