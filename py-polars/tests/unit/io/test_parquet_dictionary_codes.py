"""String and Binary columns decoded from a parquet dictionary are held as codes.

Every result must equal the one computed on the same data built in memory, which has
no codes.
"""

from __future__ import annotations

import datetime as dt
import decimal
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

LONG = [
    f"https://example.com/section-{i:03d}/item-page-{i * 7:05d}/view" for i in range(50)
]


def _df(n: int = 20_000, with_nulls: bool = True) -> pl.DataFrame:
    idx = pl.int_range(n, eager=True)
    k = pl.Series(LONG).gather(idx.hash(1) % len(LONG))
    s = pl.Series([f"s{i}" for i in range(7)]).gather(idx.hash(2) % 7)
    if with_nulls:
        k = k.scatter(list(range(0, n, 97)), None)
    return pl.DataFrame({"k": k, "s": s, "v": idx.hash(3) % 100})


def _write(df: pl.DataFrame, path: Path, **kwargs: object) -> Path:
    kwargs.setdefault("row_group_size", 3_000)
    df.write_parquet(path, **kwargs)  # type: ignore[arg-type]
    return path


QUERIES: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    "group_by_long": lambda lf: lf.group_by("k").agg(pl.sum("v"), pl.len()),
    "group_by_two": lambda lf: lf.group_by("k", "s").agg(pl.sum("v")),
    "group_by_three": lambda lf: lf.group_by("s", "k", "v").len(),
    "unique": lambda lf: lf.select(pl.col("k").unique()),
    "n_unique": lambda lf: lf.select(pl.col("k").n_unique()),
    "extract": lambda lf: lf.select(pl.col("k").str.extract(r"item-page-(\d+)")),
    "upper_then_group": lambda lf: (
        lf.with_columns(pl.col("k").str.to_uppercase()).group_by("k").agg(pl.sum("v"))
    ),
    "contains": lambda lf: lf.select(pl.col("k").str.contains("section-01")),
    "len_bytes": lambda lf: lf.select(pl.col("k").str.len_bytes()),
    "is_null": lambda lf: lf.select(pl.col("k").is_null()),
    "fill_null": lambda lf: lf.select(pl.col("k").fill_null("x")),
    "replace": lambda lf: lf.select(pl.col("k").str.replace("view", "VIEW")),
    "cast_cat": lambda lf: lf.select(pl.col("k").cast(pl.Categorical)),
    "cast_cat_group": lambda lf: lf.group_by(pl.col("k").cast(pl.Categorical)).agg(
        pl.sum("v")
    ),
    "filter_eq": lambda lf: lf.filter(pl.col("k") == LONG[3]),
    "compare_constant": lambda lf: lf.select(
        (pl.col("k") == LONG[3]).alias("eq"),
        (pl.col("k") != LONG[3]).alias("ne"),
        (pl.col("k") == "absent").alias("eq_absent"),
        (pl.col("k") < LONG[10]).alias("lt"),
        (pl.col("k") >= LONG[10]).alias("ge"),
        (pl.lit(LONG[5]) > pl.col("k")).alias("lit_gt"),
        pl.col("k").eq_missing(LONG[3]).alias("eq_missing"),
        pl.col("k").ne_missing(LONG[3]).alias("ne_missing"),
        (pl.col("k") == pl.lit(None, dtype=pl.String)).alias("eq_null"),
        pl.col("k").ne_missing(pl.lit(None, dtype=pl.String)).alias("ne_missing_null"),
        (pl.col("s") == "s2").alias("s_eq"),
    ),
    "filter_is_in": lambda lf: lf.filter(pl.col("k").is_in(LONG[:5])),
    "filter_starts": lambda lf: lf.filter(pl.col("k").str.starts_with(LONG[4][:30])),
    "filter_other": lambda lf: lf.filter(pl.col("v") < 10),
    "slice": lambda lf: lf.slice(2_500, 1_000),
    "head": lambda lf: lf.head(5),
    "sort": lambda lf: lf.sort("k", "v", nulls_last=True, maintain_order=True),
    "concat_str": lambda lf: lf.select(pl.concat_str("k", "s")),
    "first_last": lambda lf: lf.select(
        pl.col("k").first(), pl.col("k").last().alias("l")
    ),
    "min_max": lambda lf: lf.select(pl.col("k").min(), pl.col("k").max().alias("mx")),
    "agg_in_group": lambda lf: lf.group_by("s").agg(pl.col("k").min()),
    "hash": lambda lf: lf.select(pl.col("k").hash(5)),
}

UNORDERED_QUERIES = {
    "group_by_long",
    "group_by_two",
    "group_by_three",
    "unique",
    "upper_then_group",
    "cast_cat_group",
    "agg_in_group",
}


@pytest.mark.parametrize("name", sorted(QUERIES))
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_queries(tmp_path: Path, name: str, with_nulls: bool) -> None:
    df = _df(with_nulls=with_nulls)
    path = _write(df, tmp_path / "f.parquet")
    q = QUERIES[name]
    expected = q(df.lazy()).collect()
    for engine in ["streaming", "in-memory"]:
        out = q(pl.scan_parquet(path)).collect(engine=engine)  # type: ignore[call-overload]
        ordered = name not in UNORDERED_QUERIES
        assert_frame_equal(
            out, expected, check_row_order=ordered, categorical_as_str=True
        )


@pytest.mark.parametrize(
    "f",
    [
        lambda: pl.lit("abc").str.replace(pl.col("s"), "X"),
        lambda: pl.lit("abc").replace_strict("x", "y", default=pl.col("s")),
        lambda: pl.lit("abc").str.contains(pl.col("s"), literal=True),
        lambda: pl.concat_str(pl.lit("x"), pl.col("s")),
    ],
)
def test_dictionary_codes_not_first_input(
    tmp_path: Path, f: Callable[[], pl.Expr]
) -> None:
    df = pl.DataFrame({"s": ["a"] * 256})
    path = _write(df, tmp_path / "f.parquet")
    try:
        expected = df.select(f())
    except pl.exceptions.PolarsError as e:
        with pytest.raises(type(e)):
            pl.scan_parquet(path).select(f()).collect()
    else:
        out = pl.scan_parquet(path).select(f()).collect()
        assert_frame_equal(out, expected)


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti", "full"])
def test_dictionary_codes_join(tmp_path: Path, how: str) -> None:
    df = _df()
    dim = pl.DataFrame(
        {"k": LONG[::2] + ["missing"], "attr": range(len(LONG[::2]) + 1)}
    )
    fact_path = _write(df, tmp_path / "fact.parquet")
    dim_path = _write(dim, tmp_path / "dim.parquet", row_group_size=10)
    expected = df.join(dim, on="k", how=how)  # type: ignore[arg-type]
    for left, right in [(fact_path, dim_path), (dim_path, fact_path)]:
        lf_l, lf_r = pl.scan_parquet(left), pl.scan_parquet(right)
        out = lf_l.join(lf_r, on="k", how=how).collect(engine="streaming")  # type: ignore[arg-type]
        exp = (
            expected if left == fact_path else dim.join(df, on="k", how=how)  # type: ignore[arg-type]
        )
        assert_frame_equal(out, exp, check_row_order=False)


def test_dictionary_codes_many_files_disjoint(tmp_path: Path) -> None:
    dfs = []
    for i in range(4):
        df = pl.DataFrame(
            {"k": [f"{LONG[j]}-{i}" for j in range(10)] * 300, "v": range(3000)}
        )
        _write(df, tmp_path / f"{i}.parquet", row_group_size=500)
        dfs.append(df)
    expected = pl.concat(dfs).group_by("k").agg(pl.sum("v"))
    out = pl.scan_parquet(tmp_path / "*.parquet").group_by("k").agg(pl.sum("v"))
    assert_frame_equal(out.collect(), expected, check_row_order=False)
    # Shared rows across files appear in one group.
    shared = pl.concat(
        [pl.DataFrame({"k": LONG[:5] * 100}) for _ in range(3)]
    ).with_row_index()
    for i in range(3):
        _write(
            shared.slice(i * 500, 500), tmp_path / f"s{i}.parquet", row_group_size=100
        )
    counts = pl.scan_parquet(tmp_path / "s*.parquet").group_by("k").len().collect()
    assert sorted(counts["len"].to_list()) == [300] * 5


def test_dictionary_codes_fallback_and_high_cardinality(tmp_path: Path) -> None:
    n = 30_000
    # Low cardinality, then unique values: the writer falls back to PLAIN part-way.
    vals = [LONG[i % 10] for i in range(n // 2)] + [f"u{i:08d}" for i in range(n // 2)]
    df = pl.DataFrame({"k": vals, "v": range(n)})
    table = df.to_arrow()
    for version in ["1.0", "2.0"]:
        path = tmp_path / f"fb{version}.parquet"
        pq.write_table(
            table,
            path,
            row_group_size=n,
            dictionary_pagesize_limit=200,
            data_page_version=version,
            data_page_size=1024,
        )
        out = pl.scan_parquet(path).group_by("k").agg(pl.sum("v")).collect()
        expected = df.group_by("k").agg(pl.sum("v"))
        assert_frame_equal(out, expected, check_row_order=False)

    # Row groups that are low cardinality, then high cardinality.
    path = _write(df, tmp_path / "drift.parquet", row_group_size=5_000)
    out = (
        pl.scan_parquet(path)
        .with_columns(pl.col("k").str.slice(0, 20))
        .group_by("k")
        .agg(pl.sum("v"))
        .collect()
    )
    expected = (
        df.with_columns(pl.col("k").str.slice(0, 20)).group_by("k").agg(pl.sum("v"))
    )
    assert_frame_equal(out, expected, check_row_order=False)


def test_dictionary_codes_function_error_only_on_used_values(tmp_path: Path) -> None:
    df = pl.DataFrame({"k": ["1", "2", "x"] * 1000, "v": range(3000)})
    path = _write(df, tmp_path / "f.parquet")
    # "x" is filtered out before the cast, which must not see it.
    out = (
        pl.scan_parquet(path)
        .filter(pl.col("k") != "x")
        .select(pl.col("k").cast(pl.Int64).sum())
        .collect()
    )
    assert out.item() == 3000
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.scan_parquet(path).select(pl.col("k").cast(pl.Int64)).collect()


def test_dictionary_codes_nested_untouched(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "l": [["a", "b"], ["a"]] * 500,
            "st": [{"f": "a"}, {"f": "b"}] * 500,
        }
    )
    path = _write(df, tmp_path / "f.parquet")
    assert_frame_equal(pl.read_parquet(path), df)


def test_dictionary_codes_round_trip_and_ipc(tmp_path: Path) -> None:
    df = _df()
    path = _write(df, tmp_path / "f.parquet")
    lf = pl.scan_parquet(path)
    assert_frame_equal(lf.collect(), df)
    lf.sink_parquet(tmp_path / "out.parquet")
    assert_frame_equal(pl.read_parquet(tmp_path / "out.parquet"), df)
    lf.sink_ipc(tmp_path / "out.ipc")
    assert_frame_equal(pl.read_ipc(tmp_path / "out.ipc"), df)
    arrow = lf.collect().to_arrow()
    assert isinstance(arrow, pa.Table)
    assert_frame_equal(pl.from_arrow(arrow), df)  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning")
def test_dictionary_codes_length_one_arguments(tmp_path: Path) -> None:
    df = pl.DataFrame({"s": ["a", "b", "c", None] * 500})
    path = _write(df, tmp_path / "f.parquet")
    for e in [
        pl.col("s").replace({"a": "x"}),
        pl.col("s").replace_strict({"a": "x", "b": "y", "c": "z"}, default="d"),
        pl.col("s").str.contains_any(["a", "c"]),
        pl.col("s").str.replace_many(["a", "b"], ["1", "2"]),
        pl.col("s").map_elements(lambda v: v + "!", return_dtype=pl.String),
    ]:
        assert_frame_equal(pl.scan_parquet(path).select(e).collect(), df.select(e))


@pytest.mark.parametrize("length", [0, 1, 2, 256, 257])
@pytest.mark.parametrize("dictionary_first", [False, True])
def test_dictionary_codes_apply_scalar_length(
    tmp_path: Path, length: int, dictionary_first: bool
) -> None:
    df = pl.DataFrame({"s": ["a", "b", None, "c"] * 64})
    path = _write(df, tmp_path / "f.parquet")
    # Collect one batch so the scan's morsel size cannot affect expression lengths.
    (coded,) = pl.scan_parquet(path).collect_batches(chunk_size=df.height)
    repeated = pl.repeat("x", length)
    args = [pl.col("s"), repeated] if dictionary_first else [repeated, pl.col("s")]
    expr = pl.concat_str(args, separator=":")
    if length in (1, 256):
        assert_frame_equal(coded.select(expr), df.select(expr))
    else:
        with pytest.raises(pl.exceptions.ShapeError) as expected:
            df.select(expr)
        with pytest.raises(pl.exceptions.ShapeError) as actual:
            coded.select(expr)
        assert str(actual.value) == str(expected.value)


def test_dictionary_codes_batches_with_chunked_columns(tmp_path: Path) -> None:
    df = pl.DataFrame({"k": [LONG[i % 7] for i in range(4000)], "v": range(4000)})
    path = _write(df, tmp_path / "f.parquet", row_group_size=4000)
    # Two batches of one row group share its dictionary; `v` gets two chunks.
    batches = list(pl.scan_parquet(path).collect_batches(chunk_size=1000))
    out = pl.concat(batches, rechunk=False)
    assert_frame_equal(out, df)
    assert_frame_equal(pl.from_arrow(out.to_arrow()), df)  # type: ignore[arg-type]
    out.write_ipc(tmp_path / "o.ipc")
    assert_frame_equal(pl.read_ipc(tmp_path / "o.ipc"), df)


@pytest.mark.parametrize("how", ["left", "right"])
@pytest.mark.parametrize("dictionary_on_left", [False, True])
@pytest.mark.parametrize("split_payload", [False, True])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_join_with_unaligned_chunks(
    tmp_path: Path,
    how: str,
    dictionary_on_left: bool,
    split_payload: bool,
    with_nulls: bool,
) -> None:
    values = ["a"] * 128 + ["b"] * 128
    payload = pl.Series("payload", values)
    if with_nulls:
        payload = payload.scatter([1, 129], None)
    plain = payload.to_frame()
    path = _write(plain, tmp_path / "f.parquet", row_group_size=256)
    coded = next(pl.scan_parquet(path).collect_batches(chunk_size=256)).rechunk()
    if split_payload:
        # Two payload chunks whose boundaries differ from the key's two chunks.
        coded = pl.concat([coded.head(64), coded.slice(64)], rechunk=False)

    key = pl.concat(
        [pl.Series("key", range(128, 256)), pl.Series("key", range(128))],
        rechunk=False,
    )
    coded.insert_column(0, key)
    plain.insert_column(0, key)
    # Duplicate matches force gathering the preserved side too; 300 has no match.
    other = pl.DataFrame({"key": [0, 0, 1, 128, 129, 300]})
    left, right = (coded, other) if dictionary_on_left else (other, coded)
    expected_left, expected_right = (
        (plain, other) if dictionary_on_left else (other, plain)
    )
    expected = expected_left.join(expected_right, on="key", how=how)  # type: ignore[arg-type]
    out = left.join(right, on="key", how=how)  # type: ignore[arg-type]
    assert_frame_equal(out, expected, check_row_order=False)


def test_dictionary_codes_group_by_mixed_sources_with_evictions(tmp_path: Path) -> None:
    # Small enough for CI's 4-row morsels; far more keys than a hot table holds.
    low = pl.DataFrame({"s": [LONG[i % 20] for i in range(20_000)], "v": 1})
    high = pl.DataFrame({"s": [f"h{i}" for i in range(20_000)], "v": 1})
    _write(low, tmp_path / "low.parquet", row_group_size=2_000)
    _write(high, tmp_path / "high.parquet", row_group_size=2_000)
    lf = pl.concat(
        [
            pl.scan_parquet(tmp_path / "low.parquet"),
            pl.scan_parquet(tmp_path / "high.parquet"),
        ]
        * 3
    )
    out = lf.group_by("s").agg(pl.sum("v")).collect()
    expected = pl.concat([low, high] * 3).group_by("s").agg(pl.sum("v"))
    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.parametrize("hot_table_size", ["2", "16", "64"])
def test_dictionary_codes_group_by_key_runs(
    tmp_path: Path, hot_table_size: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "1000")
    plmonkeypatch.setenv("POLARS_HOT_TABLE_SIZE", hot_table_size)
    plmonkeypatch.setenv("POLARS_MAX_HOT_TABLE_SIZE", hot_table_size)
    # Runs of equal keys with nulls inside and between them, a key coming back after
    # others, and more keys than a small hot table holds.
    keys = [LONG[(i // 5) % 30] if i % 11 != 4 else None for i in range(1200)]
    df = pl.DataFrame({"s": keys, "v": range(1200)})
    path = _write(df, tmp_path / "f.parquet", row_group_size=400)
    aggs = [pl.sum("v"), pl.col("v").min().alias("min"), pl.len()]
    out = pl.scan_parquet(path).group_by("s").agg(aggs).collect(engine="streaming")
    expected = df.group_by("s").agg(aggs)
    assert_frame_equal(out, expected, check_row_order=False)


def test_dictionary_codes_group_by_two_keys_partly_cached(tmp_path: Path) -> None:
    # ~6k combinations of two coded columns (few enough for the combined-code cache),
    # skewed so that batches of keys mix ones whose combined code was verified before
    # with new ones.
    n = 100_000
    idx = pl.int_range(n, eager=True)
    r = (idx.hash(1) % 1000).cast(pl.Float64) / 1000
    k = pl.Series([f"{s}-{i}" for i in range(4) for s in LONG]).gather(
        (r * r * 200).cast(pl.UInt32)
    )
    s = pl.Series([f"s{i:02d}" for i in range(30)]).gather(idx.hash(2) % 30)
    df = pl.DataFrame(
        {"k": k.scatter(list(range(0, n, 101)), None), "s": s, "v": idx.hash(3) % 100}
    )
    path = _write(df, tmp_path / "f.parquet", row_group_size=12_500)

    def q(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.group_by("k", "s").agg(pl.sum("v"), pl.len())

    out = q(pl.scan_parquet(path)).collect(engine="streaming")
    assert_frame_equal(out, q(df.lazy()).collect(), check_row_order=False)


@pytest.mark.parametrize(
    "op", ["eq", "ne", "lt", "le", "gt", "ge", "eq_missing", "ne_missing"]
)
def test_dictionary_codes_compare_with_constant(tmp_path: Path, op: str) -> None:
    df = _df()
    path = _write(df, tmp_path / "f.parquet")
    for e in [
        getattr(pl.col("k"), op)(LONG[7]),
        getattr(pl.lit(LONG[7]), op)(pl.col("k")),
        getattr(pl.col("k"), op)(pl.lit(None, pl.String)),
    ]:

        def q(lf: pl.LazyFrame, e: pl.Expr = e) -> pl.LazyFrame:
            return lf.select(e.alias("m"), pl.len(), pl.col("k").count())

        assert_frame_equal(q(pl.scan_parquet(path)).collect(), q(df.lazy()).collect())


def test_dictionary_codes_compare_fails_only_on_unused_values(tmp_path: Path) -> None:
    df = pl.DataFrame({"k": ["a", "b", "x"] * 1000, "v": range(3000)})
    path = _write(df, tmp_path / "f.parquet")
    # "x" is filtered out before the comparison, which fails on it.
    b = pl.lit("b", dtype=pl.Enum(["a", "b"]))

    def q(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("k") != "x").select((pl.col("k") < b).sum())

    assert q(pl.scan_parquet(path)).collect().item() == q(df.lazy()).collect().item()


@pytest.mark.parametrize("row_group_size", [1_000, 3_000, 50_000])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_sink_parquet_writes_used_values(
    tmp_path: Path, row_group_size: int, with_nulls: bool
) -> None:
    df = _df(with_nulls=with_nulls)
    path = _write(df, tmp_path / "f.parquet")

    def q(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("k").is_in(LONG[3:5]) | pl.col("k").is_null())

    out_path = tmp_path / "out.parquet"
    q(pl.scan_parquet(path)).sink_parquet(
        out_path, row_group_size=row_group_size, compression="uncompressed"
    )
    expected = q(df.lazy()).collect()
    out = pl.read_parquet(out_path)
    assert out.schema == expected.schema
    assert_frame_equal(out, expected)

    md = pq.ParquetFile(out_path).metadata
    k = md.schema.names.index("k")
    for rg in range(md.num_row_groups):
        col = md.row_group(rg).column(k)
        assert "RLE_DICTIONARY" in col.encodings
        # Only the used values are in the dictionary, not the 50 of the source's.
        assert col.data_page_offset - col.dictionary_page_offset < 400
        assert col.statistics.min >= LONG[3]
        assert col.statistics.max <= LONG[4]


def _write_duplicate_dictionary(path: Path, distinct: int) -> pa.Table:
    # pyarrow canonicalizes a dictionary with equal values, so write a unique one
    # uncompressed and overwrite its entries in place with equal-length values that
    # repeat every 8 codes. The writer holds no statistics or checksums to update.
    def value(i: int) -> str:
        return f"{LONG[0]}/{i:05d}"

    codes = pa.array(list(range(distinct)) * 8, type=pa.int32())
    unique = pa.array([value(i) for i in range(distinct)])
    table = pa.table({"k": pa.DictionaryArray.from_arrays(codes, unique)})
    pq.write_table(
        table,
        path,
        row_group_size=len(table),
        store_schema=False,
        compression=None,
        write_statistics=False,
        dictionary_pagesize_limit=1 << 24,
    )
    column = pq.ParquetFile(path).metadata.row_group(0).column(0)
    # The dictionary page directly precedes the first data page; its payload is PLAIN
    # byte arrays (a 4-byte length each) at the end of the page.
    size = distinct * (4 + len(value(0)))
    start = column.data_page_offset - size
    # Only the page header lies between the dictionary page's start and its payload.
    assert 0 < start - column.dictionary_page_offset < 64
    step = 4 + len(value(0))
    with path.open("r+b") as f:
        f.seek(start)
        payload = bytearray(f.read(size))
        for i in range(distinct):
            at = i * step
            assert payload[at + 4 : at + step] == value(i).encode()
            payload[at + 4 : at + step] = value(i % 8).encode()
        f.seek(start)
        f.write(payload)

    # The file's dictionary page really holds duplicate values (pyarrow's reader
    # would deduplicate them, so inspect the bytes).
    written = path.read_bytes()[start : start + size]
    entries = [written[i * step + 4 : (i + 1) * step] for i in range(distinct)]
    assert len(entries) == distinct
    assert len(set(entries)) == 8
    return pa.table({"k": pa.array([value(i % 8) for i in codes.to_pylist()])})


def test_dictionary_codes_sink_large_distinct_code_slices_unique(
    tmp_path: Path,
) -> None:
    # The source dictionary pays off for its whole row group, but each output row
    # group visits every code once.
    distinct = 32_768
    values = pa.array([f"{LONG[0]}/{i}" for i in range(distinct)])
    codes = pa.array(list(range(distinct)) * 8, type=pa.int32())
    table = pa.table({"k": pa.DictionaryArray.from_arrays(codes, values)})
    source = tmp_path / "source.parquet"
    pq.write_table(table, source, row_group_size=len(table), store_schema=False)
    target = tmp_path / "target.parquet"
    pl.scan_parquet(source).sink_parquet(target, row_group_size=distinct)
    expected = pl.DataFrame(table).with_columns(pl.col("k").cast(pl.String))
    assert_frame_equal(pl.read_parquet(target), expected)


def test_dictionary_codes_sink_large_distinct_code_slices_equal_values(
    tmp_path: Path,
) -> None:
    # Different codes of the source dictionary hold equal values, so the sink must
    # write the few values it has, not one per code.
    distinct = 32_768
    source = tmp_path / "source.parquet"
    table = _write_duplicate_dictionary(source, distinct)
    target = tmp_path / "target.parquet"
    pl.scan_parquet(source).sink_parquet(target, row_group_size=distinct)
    assert_frame_equal(pl.read_parquet(target), pl.DataFrame(table))
    metadata = pq.ParquetFile(target).metadata
    for i in range(metadata.num_row_groups):
        assert "RLE_DICTIONARY" in metadata.row_group(i).column(0).encodings


OVERSIZED_QUERIES: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    "group_by": lambda lf: lf.group_by("k").agg(pl.len(), pl.sum("v")),
    "group_by_two": lambda lf: lf.group_by("k", "s").agg(pl.sum("v")),
    "upper": lambda lf: lf.select(pl.col("k").str.to_uppercase()),
    "compare": lambda lf: lf.select(pl.col("k") == "https://example.com/p/00000007/i"),
    "join": lambda lf: lf.join(lf.select("k").unique().sort("k").head(50), on="k"),
    "sort": lambda lf: lf.sort("k", "v"),
    "unique": lambda lf: lf.select(pl.col("k").unique()),
    "concat": lambda lf: pl.concat([lf, lf.head(10)]),
}


@pytest.mark.parametrize("name", sorted(OVERSIZED_QUERIES))
@pytest.mark.parametrize("pushed_down", [False, True])
def test_dictionary_codes_dictionary_oversized_after_filter(
    tmp_path: Path, name: str, pushed_down: bool
) -> None:
    n, distinct = 50_000, 2_500
    idx = pl.int_range(n, eager=True)
    vals = pl.Series([f"https://example.com/p/{i:08d}/i" for i in range(distinct)])
    k = vals.gather(idx.hash(1) % distinct).scatter(list(range(0, n, 1001)), None)
    s = pl.Series([f"s{i}" for i in range(7)]).gather(idx.hash(2) % 7)
    df = pl.DataFrame({"k": k, "s": s, "v": idx})
    path = _write(df, tmp_path / "f.parquet", row_group_size=25_000)

    # A predicate on the rank isn't pushed into the scan, so the source keeps codes and
    # the operators get ~100 rows of 2.5k-value dictionaries; one on the value is pushed
    # down.
    pred = (pl.col("v") if pushed_down else pl.col("v").rank()) % 500 == 1
    q = OVERSIZED_QUERIES[name]
    expected = q(df.lazy().filter(pred)).collect()
    out = q(pl.scan_parquet(path).filter(pred)).collect(engine="streaming")
    assert_frame_equal(out, expected, check_row_order=name == "sort")


@pytest.mark.parametrize("parallel", ["auto", "prefiltered"])
def test_dictionary_codes_predicate_keeping_few_rows_does_not_keep_dictionary(
    tmp_path: Path, parallel: str
) -> None:
    # The dictionary pays off for the chunk's rows, but not for the one row the
    # predicate on the column keeps: that row is copied out of the dictionary.
    once = "only-once" + "x" * 100
    s = [once] + [f"{i % 1000:04}" + "x" * 100 for i in range(1, 100_000)]
    df = pl.DataFrame({"a": range(len(s)), "s": s})
    path = _write(df, tmp_path / "f.parquet", row_group_size=len(s))

    out = (
        pl.scan_parquet(path, parallel=parallel)  # type: ignore[arg-type]
        .filter(pl.col("s") == once)
        .collect()
    )
    assert_frame_equal(out, df.head(1))
    arr = out["s"].to_arrow(compat_level=pl.CompatLevel.newest())
    assert sum(b.size for b in arr.buffers()[2:] if b is not None) < 1000


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_dictionary_codes_set_sorted_is_kept(tmp_path: Path, engine: str) -> None:
    df = pl.DataFrame({"s": ["a"] * 1000 + ["b"] * 1000})
    path = _write(df, tmp_path / "f.parquet", row_group_size=2000)
    out = (
        pl.scan_parquet(path).select(pl.col("s").set_sorted()).collect(engine=engine)  # type: ignore[call-overload]
    )
    assert_frame_equal(out, df)
    assert out["s"].flags["SORTED_ASC"]


# Binary columns are decoded from a dictionary too, whether or not they are UTF-8.
BLONG = [
    LONG[i].encode() + (b"\xff" if i % 3 == 0 else b"") + bytes([i]) for i in range(50)
]


def _bdf(n: int = 20_000, with_nulls: bool = True) -> pl.DataFrame:
    idx = pl.int_range(n, eager=True)
    k = pl.Series(BLONG, dtype=pl.Binary).gather(idx.hash(1) % len(BLONG))
    s = pl.Series([bytes([i, 0xFE]) for i in range(7)], dtype=pl.Binary).gather(
        idx.hash(2) % 7
    )
    if with_nulls:
        k = k.scatter(list(range(0, n, 97)), None)
    return pl.DataFrame({"k": k, "s": s, "v": idx.hash(3) % 100})


def _write_binary(df: pl.DataFrame, path: Path, **kwargs: object) -> Path:
    # pyarrow writes `binary` without the UTF8 annotation, so it is read as Binary.
    kwargs.setdefault("row_group_size", 3_000)
    pq.write_table(df.to_arrow(), path, **kwargs)  # type: ignore[arg-type]
    return path


BINARY_QUERIES: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    "group_by": lambda lf: lf.group_by("k").agg(pl.sum("v"), pl.len()),
    "group_by_two": lambda lf: lf.group_by("k", "s").agg(pl.sum("v")),
    "group_by_three": lambda lf: lf.group_by("s", "k", "v").len(),
    "unique": lambda lf: lf.select(pl.col("k").unique()),
    "n_unique": lambda lf: lf.select(pl.col("k").n_unique()),
    "is_null": lambda lf: lf.select(pl.col("k").is_null()),
    "fill_null": lambda lf: lf.select(pl.col("k").fill_null(b"x")),
    "size_bytes": lambda lf: lf.select(pl.col("k").bin.size()),
    "contains": lambda lf: lf.select(pl.col("k").bin.contains(b"section-01")),
    "starts_with": lambda lf: lf.filter(
        pl.col("k").bin.starts_with(b"https://example")
    ),
    "encode_then_group": lambda lf: (
        lf.with_columns(pl.col("k").bin.encode("hex")).group_by("k").agg(pl.sum("v"))
    ),
    "filter_eq": lambda lf: lf.filter(pl.col("k") == BLONG[3]),
    "compare_constant": lambda lf: lf.select(
        (pl.col("k") == BLONG[3]).alias("eq"),
        (pl.col("k") != BLONG[3]).alias("ne"),
        (pl.col("k") == b"absent").alias("eq_absent"),
        (pl.col("k") < BLONG[10]).alias("lt"),
        (pl.col("k") >= BLONG[10]).alias("ge"),
        (pl.lit(BLONG[5]) > pl.col("k")).alias("lit_gt"),
        pl.col("k").eq_missing(BLONG[3]).alias("eq_missing"),
        pl.col("k").ne_missing(BLONG[3]).alias("ne_missing"),
        (pl.col("k") == pl.lit(None, dtype=pl.Binary)).alias("eq_null"),
        (pl.col("s") == bytes([2, 0xFE])).alias("s_eq"),
    ),
    "filter_is_in": lambda lf: lf.filter(pl.col("k").is_in(BLONG[:5])),
    "filter_other": lambda lf: lf.filter(pl.col("v") < 10),
    "slice": lambda lf: lf.slice(2_500, 1_000),
    "head": lambda lf: lf.head(5),
    "sort": lambda lf: lf.sort("k", "v", nulls_last=True, maintain_order=True),
    "first_last": lambda lf: lf.select(
        pl.col("k").first(), pl.col("k").last().alias("l")
    ),
    "min_max": lambda lf: lf.select(pl.col("k").min(), pl.col("k").max().alias("mx")),
    "agg_in_group": lambda lf: lf.group_by("s").agg(pl.col("k").min()),
    "hash": lambda lf: lf.select(pl.col("k").hash(5)),
    "cast_binary": lambda lf: lf.select(pl.col("k").cast(pl.Binary)),
}

UNORDERED_BINARY_QUERIES = {
    "group_by",
    "group_by_two",
    "group_by_three",
    "unique",
    "encode_then_group",
    "agg_in_group",
}


@pytest.mark.parametrize("name", sorted(BINARY_QUERIES))
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_binary_queries(
    tmp_path: Path, name: str, with_nulls: bool
) -> None:
    df = _bdf(with_nulls=with_nulls)
    path = _write_binary(df, tmp_path / "f.parquet")
    q = BINARY_QUERIES[name]
    expected = q(df.lazy()).collect()
    for engine in ["streaming", "in-memory"]:
        out = q(pl.scan_parquet(path)).collect(engine=engine)  # type: ignore[call-overload]
        ordered = name not in UNORDERED_BINARY_QUERIES
        assert_frame_equal(out, expected, check_row_order=ordered)


@pytest.mark.parametrize("dtype", [pl.String, pl.Binary])
def test_dictionary_codes_are_kept_for_string_and_binary(
    tmp_path: Path, dtype: pl.DataType
) -> None:
    # A coded column holds a code per row (4 bytes) and shares the dictionary; the
    # same values decoded one view per row would take at least 16 bytes per row.
    n = 50_000
    values = [LONG[i % 50] for i in range(n)]
    df = pl.DataFrame({"k": pl.Series(values).cast(dtype)})
    path = _write_binary(df, tmp_path / "f.parquet", row_group_size=n)
    # `collect` materializes its output; batches keep the codes.
    (out,) = pl.scan_parquet(path).collect_batches(chunk_size=n)
    assert out.schema == df.schema
    # Before comparing, which may cache a materialization that counts toward the size.
    assert out.estimated_size() < 8 * n
    assert_frame_equal(out, df)


@pytest.mark.parametrize("dtype", [pl.String, pl.Int64, pl.Datetime("us")])
def test_dictionary_codes_collect_reads_flat_batches_keep_codes(
    tmp_path: Path, dtype: pl.DataType
) -> None:
    # A scan that only feeds `collect` (through projections) has no use for codes and reads
    # flat; the result must equal the codes-off one. Batches keep codes (see
    # `test_dictionary_codes_are_kept_for_string_and_binary`).
    n = 40_000
    if dtype == pl.String:
        values = pl.Series([LONG[i % 50] for i in range(n)])
    else:
        values = (pl.int_range(n, eager=True) % 50 * 1_000_003).cast(dtype)
    df = pl.DataFrame({"c": values, "v": pl.int_range(n, eager=True)})
    path = _write_binary(df, tmp_path / "f.parquet", row_group_size=10_000)
    lf = pl.scan_parquet(path)
    for q in [
        lf,
        lf.select("c"),
        lf.select("v", "c"),
        lf.filter(pl.col("v") % 3 == 0).select("c"),
        lf.head(15_000),
    ]:
        expected = q.collect(engine="in-memory")
        assert_frame_equal(q.collect(engine="streaming"), expected)
        assert_frame_equal(_collect_as_batches(q), expected)
    out = lf.collect(engine="streaming")
    assert_frame_equal(out, df)
    assert out["c"].to_frame().estimated_size() >= 8 * n
    batches = list(lf.select("c").collect_batches(chunk_size=10_000))
    assert len(batches) == 4
    assert_frame_equal(pl.concat(batches), df.select("c"))


def test_dictionary_codes_binary_join(tmp_path: Path) -> None:
    df = _bdf()
    dim = pl.DataFrame(
        {"k": BLONG[::2] + [b"missing"], "attr": range(len(BLONG[::2]) + 1)},
        schema_overrides={"k": pl.Binary},
    )
    fact_path = _write_binary(df, tmp_path / "fact.parquet")
    dim_path = _write_binary(dim, tmp_path / "dim.parquet", row_group_size=10)
    for how in ["inner", "left", "semi", "anti", "full"]:
        for left, right, l_df, r_df in [
            (fact_path, dim_path, df, dim),
            (dim_path, fact_path, dim, df),
        ]:
            out = (
                pl.scan_parquet(left)
                .join(pl.scan_parquet(right), on="k", how=how)  # type: ignore[arg-type]
                .collect(engine="streaming")
            )
            assert_frame_equal(
                out,
                l_df.join(r_df, on="k", how=how),  # type: ignore[arg-type]
                check_row_order=False,
            )


def test_dictionary_codes_binary_concat_different_dictionaries(tmp_path: Path) -> None:
    dfs = []
    for i in range(4):
        df = pl.DataFrame(
            {
                "k": pl.Series(
                    [BLONG[j + 5 * i] for j in range(10)] * 300, dtype=pl.Binary
                ),
                "v": range(3000),
            }
        )
        _write_binary(df, tmp_path / f"{i}.parquet", row_group_size=500)
        dfs.append(df)
    expected = pl.concat(dfs)
    scans = [pl.scan_parquet(tmp_path / f"{i}.parquet") for i in range(4)]
    assert_frame_equal(pl.concat(scans).collect(), expected)
    assert_frame_equal(pl.scan_parquet(tmp_path / "*.parquet").collect(), expected)
    assert_frame_equal(
        pl.concat(
            [pl.read_parquet(tmp_path / f"{i}.parquet") for i in range(4)],
            rechunk=False,
        ),
        expected,
    )
    out = pl.concat(scans).group_by("k").agg(pl.sum("v")).collect()
    assert_frame_equal(
        out, expected.group_by("k").agg(pl.sum("v")), check_row_order=False
    )


def test_dictionary_codes_binary_dictionary_and_plain_row_groups(
    tmp_path: Path,
) -> None:
    n = 30_000
    # Low cardinality, then unique values: the writer falls back to PLAIN part-way,
    # and later row groups are PLAIN entirely.
    vals = [BLONG[i % 10] for i in range(n // 2)] + [
        b"\xff" + f"u{i:08d}".encode() for i in range(n // 2)
    ]
    df = pl.DataFrame({"k": pl.Series(vals, dtype=pl.Binary), "v": range(n)})
    for version in ["1.0", "2.0"]:
        path = tmp_path / f"fb{version}.parquet"
        pq.write_table(
            df.to_arrow(),
            path,
            row_group_size=n,
            dictionary_pagesize_limit=200,
            data_page_version=version,
            data_page_size=1024,
        )
        out = pl.scan_parquet(path).group_by("k").agg(pl.sum("v")).collect()
        assert_frame_equal(
            out, df.group_by("k").agg(pl.sum("v")), check_row_order=False
        )
    path = _write_binary(df, tmp_path / "drift.parquet", row_group_size=5_000)
    out = (
        pl.scan_parquet(path)
        .with_columns(pl.col("k").bin.slice(0, 20))
        .group_by("k")
        .agg(pl.sum("v"))
        .collect()
    )
    expected = (
        df.with_columns(pl.col("k").bin.slice(0, 20)).group_by("k").agg(pl.sum("v"))
    )
    assert_frame_equal(out, expected, check_row_order=False)
    assert_frame_equal(pl.scan_parquet(path).collect(), df)


def test_dictionary_codes_binary_cast_to_string(tmp_path: Path) -> None:
    valid = pl.DataFrame(
        {"k": pl.Series([LONG[i % 7].encode() for i in range(3000)], dtype=pl.Binary)}
    )
    path = _write_binary(valid, tmp_path / "valid.parquet")
    for strict in [True, False]:
        e = pl.col("k").cast(pl.String, strict=strict)
        assert_frame_equal(pl.scan_parquet(path).select(e).collect(), valid.select(e))
    assert pl.scan_parquet(path).select(e).collect().schema["k"] == pl.String

    # Invalid UTF-8 in some rows fails the cast, but a value no row uses is not cast.
    df = pl.DataFrame(
        {
            "k": pl.Series([b"a", b"\xff\xfe", b"b"] * 1000, dtype=pl.Binary),
            "v": range(3000),
        }
    )
    path = _write_binary(df, tmp_path / "invalid.parquet")
    with pytest.raises(pl.exceptions.ComputeError, match="invalid utf8"):
        df.select(pl.col("k").cast(pl.String))
    with pytest.raises(pl.exceptions.ComputeError, match="invalid utf8"):
        pl.scan_parquet(path).select(pl.col("k").cast(pl.String)).collect()
    # Binary to String fails on invalid UTF-8 even when not strict.
    e = pl.col("k").cast(pl.String, strict=False)
    with pytest.raises(pl.exceptions.ComputeError, match="invalid utf8"):
        df.select(e)
    with pytest.raises(pl.exceptions.ComputeError, match="invalid utf8"):
        pl.scan_parquet(path).select(e).collect()
    q = (
        pl.scan_parquet(path)
        .filter(pl.col("k") != b"\xff\xfe")
        .select(pl.col("k").cast(pl.String))
    )
    assert_frame_equal(
        q.collect(),
        df.filter(pl.col("k") != b"\xff\xfe").select(pl.col("k").cast(pl.String)),
    )


@pytest.mark.parametrize("row_group_size", [1_000, 3_000, 50_000])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_binary_sink_parquet(
    tmp_path: Path, row_group_size: int, with_nulls: bool
) -> None:
    df = _bdf(with_nulls=with_nulls)
    path = _write_binary(df, tmp_path / "f.parquet")

    def q(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("k").is_in(BLONG[3:5]) | pl.col("k").is_null())

    out_path = tmp_path / "out.parquet"
    q(pl.scan_parquet(path)).sink_parquet(
        out_path, row_group_size=row_group_size, compression="uncompressed"
    )
    expected = q(df.lazy()).collect()
    out = pl.read_parquet(out_path)
    assert out.schema == expected.schema
    assert_frame_equal(out, expected)

    md = pq.ParquetFile(out_path).metadata
    k = md.schema.names.index("k")
    for rg in range(md.num_row_groups):
        col = md.row_group(rg).column(k)
        assert "RLE_DICTIONARY" in col.encodings
        # Only the used values are in the dictionary, not the 50 of the source's.
        assert col.data_page_offset - col.dictionary_page_offset < 400
    lf = pl.scan_parquet(path)
    lf.sink_ipc(tmp_path / "out.ipc")
    assert_frame_equal(pl.read_ipc(tmp_path / "out.ipc"), df)
    assert_frame_equal(pl.from_arrow(lf.collect().to_arrow()), df)  # type: ignore[arg-type]


def test_dictionary_codes_binary_predicate_keeping_few_rows(tmp_path: Path) -> None:
    once = b"only-once\xff" + b"x" * 100
    s = [once] + [f"{i % 1000:04}".encode() + b"x" * 100 for i in range(1, 100_000)]
    df = pl.DataFrame({"a": range(len(s)), "s": pl.Series(s, dtype=pl.Binary)})
    path = _write_binary(df, tmp_path / "f.parquet", row_group_size=len(s))
    for parallel in ["auto", "prefiltered"]:
        out = (
            pl.scan_parquet(path, parallel=parallel)  # type: ignore[arg-type]
            .filter(pl.col("s") == once)
            .collect()
        )
        assert_frame_equal(out, df.head(1))
        arr = out["s"].to_arrow(compat_level=pl.CompatLevel.newest())
        assert sum(b.size for b in arr.buffers()[2:] if b is not None) < 1000


def test_dictionary_codes_binary_disabled_gives_equal_results(tmp_path: Path) -> None:
    df = _bdf()
    path = _write_binary(df, tmp_path / "f.parquet")
    script = (
        "import polars as pl, sys\n"
        "lf = pl.scan_parquet(sys.argv[1])\n"
        "out = lf.group_by('k').agg(pl.sum('v'), pl.len()).sort('k')\n"
        "out.collect().write_ipc(sys.argv[2])\n"
        "lf.filter(pl.col('k') == lf.head(1).collect()['k'][0]).collect().write_ipc(sys.argv[3])\n"
        "batches = lf.collect_batches(chunk_size=1_000_000)\n"
        "print(sum(b.estimated_size() for b in batches))\n"
    )
    results = {}
    for flag in ["1", "0"]:
        env = {**os.environ, "POLARS_DICT_CODES": flag}
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(tmp_path / f"g{flag}.ipc"),
                str(tmp_path / f"f{flag}.ipc"),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        results[flag] = int(out.stdout.strip())
    assert_frame_equal(
        pl.read_ipc(tmp_path / "g1.ipc"), pl.read_ipc(tmp_path / "g0.ipc")
    )
    assert_frame_equal(
        pl.read_ipc(tmp_path / "f1.ipc"), pl.read_ipc(tmp_path / "f0.ipc")
    )
    # Only with the flag on are the rows held as codes.
    assert results["1"] < results["0"]


def _same_outcome(
    coded: Callable[[], pl.DataFrame], plain: Callable[[], pl.DataFrame]
) -> None:
    try:
        expected = plain()
    except pl.exceptions.PolarsError as e:
        with pytest.raises(type(e)):
            coded()
    else:
        assert_frame_equal(coded(), expected)


@pytest.mark.parametrize("string_first", [False, True])
@pytest.mark.parametrize("how", ["vertical", "vertical_relaxed"])
def test_dictionary_codes_concat_string_with_binary(
    tmp_path: Path, string_first: bool, how: str
) -> None:
    s = pl.DataFrame({"k": [LONG[i % 7] for i in range(3000)]})
    b = pl.DataFrame(
        {"k": pl.Series([BLONG[i % 7] for i in range(3000)], dtype=pl.Binary)}
    )
    s_path = _write(s, tmp_path / "s.parquet")
    b_path = _write_binary(b, tmp_path / "b.parquet")
    plain = [s.lazy(), b.lazy()]
    scans = [pl.scan_parquet(s_path), pl.scan_parquet(b_path)]
    if not string_first:
        plain.reverse()
        scans.reverse()
    _same_outcome(
        lambda: pl.concat(scans, how=how).collect(),  # type: ignore[arg-type]
        lambda: pl.concat(plain, how=how).collect(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("name", ["eq", "lt", "is_in"])
def test_dictionary_codes_binary_compared_with_string(
    tmp_path: Path, name: str
) -> None:
    df = _bdf()
    path = _write_binary(df, tmp_path / "f.parquet")
    exprs = {
        "eq": pl.col("k") == "abc",
        "lt": pl.col("k") < "abc",
        "is_in": pl.col("k").is_in(["abc"]),
    }
    e = exprs[name]
    _same_outcome(
        lambda: pl.scan_parquet(path).select(e).collect(),
        lambda: df.lazy().select(e).collect(),
    )


def test_dictionary_codes_string_cast_to_binary_sink_round_trip(tmp_path: Path) -> None:
    df = _df()
    path = _write(df, tmp_path / "f.parquet")
    q = pl.scan_parquet(path).with_columns(pl.col("k").cast(pl.Binary))
    out_path = tmp_path / "out.parquet"
    q.sink_parquet(out_path, compression="uncompressed")
    expected = df.with_columns(pl.col("k").cast(pl.Binary))
    out = pl.read_parquet(out_path)
    assert out.schema == expected.schema
    assert_frame_equal(out, expected)
    md = pq.ParquetFile(out_path).metadata
    k = md.schema.names.index("k")
    for rg in range(md.num_row_groups):
        assert "RLE_DICTIONARY" in md.row_group(rg).column(k).encodings


@pytest.mark.parametrize("with_nulls", [False, True])
@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_dictionary_codes_random_function_is_not_evaluated_per_distinct_value(
    tmp_path: Path, with_nulls: bool, engine: str
) -> None:
    n = 20_000
    k = pl.Series(["a/b/c/d/e/f"] * n)
    if with_nulls:
        k = k.scatter(list(range(0, n, 97)), None)
    path = _write(pl.DataFrame({"k": k}), tmp_path / "f.parquet")
    df = pl.DataFrame({"k": k})
    split = pl.col("k").str.split("/")
    # Equal rows are sampled one by one: neighbouring rows are rarely sampled alike (1 in 6
    # for one element of 6; a value evaluated once would give all of them the same).
    for e in [split.list.sample(n=1), split.list.sample(fraction=1.0, shuffle=True)]:
        out = pl.scan_parquet(path).select(r=e).collect(engine=engine)  # type: ignore[call-overload]
        r = out["r"].list.join(",").drop_nulls()
        assert (r == r.shift(1)).drop_nulls().mean() < 0.5
    # A seeded sample is the one of the flat column.
    for e in [split.list.sample(n=2, seed=0), split.list.sample(n=2, seed=3).list.len()]:
        out = pl.scan_parquet(path).select(e).collect(engine=engine)  # type: ignore[call-overload]
        assert_frame_equal(out, df.select(e))


def test_dictionary_codes_random_function_equals_dictionary_codes_off(
    tmp_path: Path,
) -> None:
    df = _df()
    path = _write(df, tmp_path / "f.parquet")
    script = (
        "import polars as pl, sys\n"
        "q = pl.scan_parquet(sys.argv[1]).select("
        "pl.col('k').str.split('/').list.sample(n=2, seed=0))\n"
        "q.collect().write_ipc(sys.argv[2])\n"
    )
    for flag in ["1", "0"]:
        subprocess.run(
            [sys.executable, "-c", script, str(path), str(tmp_path / f"r{flag}.ipc")],
            env={**os.environ, "POLARS_DICT_CODES": flag},
            check=True,
        )
    assert_frame_equal(
        pl.read_ipc(tmp_path / "r1.ipc"), pl.read_ipc(tmp_path / "r0.ipc")
    )


def test_dictionary_codes_streaming_on_a_non_byte_dictionary(tmp_path: Path) -> None:
    # `len_chars` of a coded String column is a coded UInt32 column.
    n = 20_000
    idx = pl.int_range(n, eager=True)
    df = pl.DataFrame(
        {
            "s": pl.Series([f"v{'x' * (i % 9)}" for i in range(40)]).gather(
                idx.hash(5) % 40
            ),
            "v": idx.hash(6) % 100,
        }
    )
    path = _write(df, tmp_path / "f.parquet", row_group_size=2_000)
    dim = pl.DataFrame(
        {"n": pl.Series(range(0, 12, 2), dtype=pl.UInt32), "attr": range(6)}
    )
    n_ = pl.col("s").str.len_chars().alias("n")
    queries: dict[str, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
        "group_by": lambda lf: lf.select(n_, "v").group_by("n").agg(pl.sum("v")),
        "join": lambda lf: lf.select(n_, "v").join(dim.lazy(), on="n", how="left"),
        "unique": lambda lf: lf.select(n_).unique(),
    }
    for name, q in queries.items():
        expected = q(df.lazy()).collect()
        out = q(pl.scan_parquet(path)).collect(engine="streaming")
        assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.parametrize("dim_on_left", [False, True])
def test_dictionary_codes_join_runtime_filter(tmp_path: Path, dim_on_left: bool) -> None:
    # A forced hash join publishes its build keys as a dynamic predicate on the probe scan,
    # which is evaluated on a dictionary-coded key.
    df = _df()
    fact = pl.scan_parquet(_write(df, tmp_path / "fact.parquet"))
    dim = pl.LazyFrame({"k": LONG[:5], "d": range(5)}).filter(pl.col("d") >= 0)
    q = dim.join(fact, on="k") if dim_on_left else fact.join(dim, on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    expected = (
        dim.join(df.lazy(), on="k") if dim_on_left else df.lazy().join(dim, on="k")
    ).collect()
    out = q.collect(engine="streaming")
    assert out.height > 0
    assert_frame_equal(out, expected, check_row_order=False)


# ---------------------------------------------------------------------------------------
# Top-level columns of every type that parquet dictionary-encodes.

TOPLEVEL_ROWS = 40_000
TOPLEVEL_CARD = 50


def _ints(dtype: type, scale: int = 1, offset: int = 0) -> Callable[[], pa.Array]:
    return lambda: pa.array(
        np.arange(TOPLEVEL_CARD).astype(dtype) * dtype(scale) + dtype(offset)
    )


def _base_values() -> dict[str, Callable[[], pa.Array]]:
    k = np.arange(TOPLEVEL_CARD)
    floats = k * 1.5
    floats[0], floats[1], floats[2], floats[3] = -0.0, np.nan, np.inf, -np.inf
    decimals = [decimal.Decimal(int(i) * 7919 + 1) / 100 for i in k]
    return {
        "i8": lambda: pa.array((k - 25).astype(np.int8)),
        "i16": lambda: pa.array((k * 600 - 15_000).astype(np.int16)),
        "i32": lambda: pa.array((k * 40_000_001 - 10**9).astype(np.int32)),
        "i64": lambda: pa.array(k.astype(np.int64) * 10**12 - 10**13),
        "u8": lambda: pa.array((k * 5).astype(np.uint8)),
        "u16": lambda: pa.array((k * 1_300).astype(np.uint16)),
        "u32": lambda: pa.array((k.astype(np.uint32) * np.uint32(87_000_001))),
        "u64": lambda: pa.array(k.astype(np.uint64) * np.uint64(2**58)),
        "date": lambda: pa.array((k * 1_000 - 5_000).astype(np.int32), pa.date32()),
        "time_us": lambda: pa.array(k.astype(np.int64) * 1_700_001, pa.time64("us")),
        "time_ns": lambda: pa.array(k.astype(np.int64) * 1_234_567_891, pa.time64("ns")),
        "time_ms": lambda: pa.array((k * 1_000_007).astype(np.int32), pa.time32("ms")),
        "dt_us_tz": lambda: pa.array(
            k.astype(np.int64) * 10**11 + 1_700_000_000_000_000,
            pa.timestamp("us", "Europe/Amsterdam"),
        ),
        "dt_us": lambda: pa.array(
            k.astype(np.int64) * 10**11 + 1_700_000_000_000_000, pa.timestamp("us")
        ),
        "dt_ms": lambda: pa.array(
            k.astype(np.int64) * 10**8 + 1_700_000_000_000, pa.timestamp("ms")
        ),
        "dt_ns_tz": lambda: pa.array(
            k.astype(np.int64) * 10**14 + 1_700_000_000_000_000_000,
            pa.timestamp("ns", "Asia/Tokyo"),
        ),
        "dur_us": lambda: pa.array(k.astype(np.int64) * 10**9, pa.duration("us")),
        "dur_ms": lambda: pa.array(k.astype(np.int64) * 10**6, pa.duration("ms")),
        "f32": lambda: pa.array(floats.astype(np.float32)),
        "f64": lambda: pa.array(floats),
        "f16": lambda: pa.array(floats.astype(np.float16)),
        "dec9": lambda: pa.array(decimals, pa.decimal128(9, 2)),
        "dec18": lambda: pa.array(decimals, pa.decimal128(18, 2)),
        "dec38": lambda: pa.array(
            [d * 10**15 for d in decimals], pa.decimal128(38, 2)
        ),
        "fsb16": lambda: pa.array([bytes([i + 1]) * 16 for i in k], pa.binary(16)),
        "fsb5": lambda: pa.array([bytes([i + 1]) * 5 for i in k], pa.binary(5)),
        "large_binary": lambda: pa.array(
            [bytes([i + 1]) * (i % 9) + b"x" for i in k], pa.large_binary()
        ),
    }


TOPLEVEL_TYPES = sorted(_base_values())
# Values narrower than a 4-byte code are not held as codes.
NARROW_TYPES = {"i8", "i16", "u8", "u16", "f16"}
# Decimals are stored as INT32 and INT64 when asked to.
INT_DECIMALS = {"dec9", "dec18"}


def _toplevel_table(name: str, with_nulls: bool) -> pa.Table:
    n = TOPLEVEL_ROWS
    dictionary = _base_values()[name]()
    idx = pa.array((np.arange(n) * 7919 + np.arange(n) // 13) % TOPLEVEL_CARD)
    col = dictionary.take(idx)
    if with_nulls:
        mask = np.zeros(n, dtype=bool)
        mask[::97] = True
        col = pc.if_else(pa.array(mask), pa.scalar(None, col.type), col)
    v = pa.array(np.arange(n, dtype=np.int64) * 31 % 100)
    return pa.table({"c": col, "v": v})


def _write_toplevel(table: pa.Table, path: Path, name: str, **kwargs: object) -> Path:
    kwargs.setdefault("row_group_size", 10_000)
    kwargs.setdefault("use_dictionary", True)
    if name in INT_DECIMALS:
        kwargs["store_decimal_as_integer"] = True
    pq.write_table(table, path, **kwargs)  # type: ignore[arg-type]
    return path


@pytest.fixture(scope="module")
def toplevel_files(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, bool], tuple[Path, pl.DataFrame]]:
    """A file per type, and its content as read with dictionary codes off."""
    root = tmp_path_factory.mktemp("toplevel")
    paths = {}
    for name in TOPLEVEL_TYPES:
        for with_nulls in [False, True]:
            path = root / f"{name}-{int(with_nulls)}.parquet"
            _write_toplevel(_toplevel_table(name, with_nulls), path, name)
            paths[name, with_nulls] = path
    script = (
        "import polars as pl, sys\n"
        "for i, p in enumerate(sys.argv[2:]):\n"
        "    pl.read_parquet(p).write_ipc(f'{sys.argv[1]}/{i}.ipc')\n"
    )
    env = {**os.environ, "POLARS_DICT_CODES": "0"}
    subprocess.run(
        [sys.executable, "-c", script, str(root), *map(str, paths.values())],
        env=env,
        check=True,
    )
    return {
        key: (path, pl.read_ipc(root / f"{i}.ipc"))
        for i, (key, path) in enumerate(paths.items())
    }


def test_dictionary_codes_toplevel_types_are_kept(tmp_path: Path) -> None:
    # Held as codes, a column takes a different size than read flat, with the same dtype. One row
    # group, as batches of different dictionaries of such values are materialized together.
    names = TOPLEVEL_TYPES
    paths = [
        str(
            _write_toplevel(
                _toplevel_table(n, True),
                tmp_path / f"{n}.parquet",
                n,
                row_group_size=TOPLEVEL_ROWS,
            )
        )
        for n in names
    ]
    script = (
        "import polars as pl, sys\n"
        "for p in sys.argv[1:]:\n"
        "    (b,) = pl.scan_parquet(p).collect_batches(chunk_size=40_000)\n"
        "    print(p, b.schema['c'], b.select('c').estimated_size(), b.height, sep='\\t')\n"
    )
    sizes = {}
    for flag in ["1", "0"]:
        out = subprocess.run(
            [sys.executable, "-c", script, *paths],
            env={**os.environ, "POLARS_DICT_CODES": flag},
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out.stderr
        sizes[flag] = [line.split("\t") for line in out.stdout.splitlines()]
    for name, on, off in zip(names, sizes["1"], sizes["0"], strict=True):
        assert on[0] == off[0]
        assert on[1] == off[1], f"dtype of {name}"
        if name in NARROW_TYPES:
            assert on[2] == off[2], f"{name} was kept as codes"
        else:
            assert on[2] != off[2], f"{name} was not kept as codes"
        assert on[3] == off[3] == "40000"


TOPLEVEL_QUERIES: dict[str, Callable[[pl.LazyFrame, object], pl.LazyFrame]] = {
    "scan": lambda lf, v: lf,
    "filter_eq": lambda lf, v: lf.filter(pl.col("c") == pl.lit(v)),
    "filter_ne": lambda lf, v: lf.filter(pl.col("c") != pl.lit(v)),
    "filter_gt": lambda lf, v: lf.filter(pl.col("c") > pl.lit(v)),
    "filter_is_null": lambda lf, v: lf.filter(pl.col("c").is_null()),
    "filter_is_in": lambda lf, v: lf.filter(pl.col("c").is_in([v])),
    "compare": lambda lf, v: lf.select(
        (pl.col("c") == pl.lit(v)).alias("eq"), (pl.col("c") < pl.lit(v)).alias("lt")
    ),
    "group_by": lambda lf, v: lf.group_by("c").agg(pl.sum("v"), pl.len()),
    "group_by_two": lambda lf, v: lf.group_by("c", "v").len(),
    "unique": lambda lf, v: lf.select(pl.col("c").unique()),
    "n_unique": lambda lf, v: lf.select(pl.col("c").n_unique()),
    "sort": lambda lf, v: lf.sort("c", "v", nulls_last=True, maintain_order=True),
    "min_max": lambda lf, v: lf.select(pl.col("c").min(), pl.col("c").max().alias("x")),
    "first_last": lambda lf, v: lf.select(pl.col("c").first(), pl.col("c").last().alias("l")),
    "is_null": lambda lf, v: lf.select(pl.col("c").is_null()),
    "fill_null": lambda lf, v: lf.select(pl.col("c").fill_null(pl.lit(v))),
    "slice": lambda lf, v: lf.slice(5_000, 12_345),
    "hash": lambda lf, v: lf.select(pl.col("c").hash(3)),
    "cast_string": lambda lf, v: lf.select(pl.col("c").cast(pl.String)),
    "cast_string_group": lambda lf, v: lf.group_by(pl.col("c").cast(pl.String)).agg(
        pl.sum("v")
    ),
}
UNORDERED_TOPLEVEL = {"group_by", "group_by_two", "unique", "cast_string_group"}


@pytest.mark.parametrize("name", TOPLEVEL_QUERIES)
@pytest.mark.parametrize("with_nulls", [False, True])
@pytest.mark.parametrize("col", TOPLEVEL_TYPES)
def test_dictionary_codes_toplevel_queries(
    toplevel_files: dict[tuple[str, bool], tuple[Path, pl.DataFrame]],
    col: str,
    with_nulls: bool,
    name: str,
) -> None:
    path, expected_df = toplevel_files[col, with_nulls]
    v = expected_df["c"].drop_nulls()[7]
    q = TOPLEVEL_QUERIES[name]
    same = _same_outcome_unordered if name in UNORDERED_TOPLEVEL else _same_outcome
    # `collect` of a scan feeding the sink reads flat; batches keep codes, so predicates and
    # slices pushed into the scan also run the coded decode.
    collectors: dict[str, Callable[[pl.LazyFrame], pl.DataFrame]] = {
        "streaming": lambda lf: lf.collect(engine="streaming"),
        "in-memory": lambda lf: lf.collect(engine="in-memory"),
        "batches": _collect_as_batches,
    }
    for collect in collectors.values():
        same(
            lambda: collect(q(pl.scan_parquet(path), v)),  # type: ignore[arg-type]
            lambda: q(expected_df.lazy(), v).collect(),
        )


def _collect_as_batches(lf: pl.LazyFrame) -> pl.DataFrame:
    batches = list(lf.collect_batches(chunk_size=7_000))
    return pl.concat(batches) if batches else lf.head(0).collect()


def _same_outcome_unordered(
    coded: Callable[[], pl.DataFrame], plain: Callable[[], pl.DataFrame]
) -> None:
    try:
        expected = plain()
    except pl.exceptions.PolarsError as e:
        with pytest.raises(type(e)):
            coded()
    else:
        assert_frame_equal(coded(), expected, check_row_order=False)


@pytest.mark.parametrize("how", ["inner", "left", "semi", "anti"])
@pytest.mark.parametrize("col", ["i64", "dt_us_tz", "dec38", "f64", "dec9", "fsb16"])
def test_dictionary_codes_toplevel_join(
    toplevel_files: dict[tuple[str, bool], tuple[Path, pl.DataFrame]],
    col: str,
    how: str,
) -> None:
    path, df = toplevel_files[col, True]
    dim = (
        df.select(pl.col("c").unique(maintain_order=True))
        .drop_nulls()
        .head(25)
        .with_row_index("attr")
    )
    for engine in ["streaming", "in-memory"]:
        out = (
            pl.scan_parquet(path)
            .join(dim.lazy(), on="c", how=how)  # type: ignore[arg-type]
            .collect(engine=engine)  # type: ignore[call-overload]
        )
        assert_frame_equal(
            out, df.join(dim, on="c", how=how), check_row_order=False  # type: ignore[arg-type]
        )
    # Joining on a String built from the column against a String table.
    keys = pl.DataFrame({"s": df["c"].drop_nulls().unique().cast(pl.String)[:20]})
    keys = keys.with_row_index("attr")
    q = lambda lf: lf.with_columns(s=pl.col("c").cast(pl.String)).join(  # noqa: E731
        keys.lazy(), on="s", how="left"
    )
    assert_frame_equal(
        q(pl.scan_parquet(path)).collect(engine="streaming"),
        q(df.lazy()).collect(),
        check_row_order=False,
    )


ELEMENTWISE: dict[str, dict[str, Callable[[], pl.Expr]]] = {
    "dt_us_tz": {
        "strftime": lambda: pl.col("c").dt.strftime("%Y-%m-%d %H:%M"),
        "convert_time_zone": lambda: pl.col("c").dt.convert_time_zone("Asia/Tokyo"),
        "convert_then_hour": lambda: (
            pl.col("c").dt.convert_time_zone("America/New_York").dt.hour()
        ),
        "replace_time_zone": lambda: pl.col("c").dt.replace_time_zone(None),
        "date": lambda: pl.col("c").dt.date(),
        "cast_date": lambda: pl.col("c").cast(pl.Date),
        "add_duration": lambda: pl.col("c") + pl.duration(hours=1),
    },
    "dt_ns_tz": {"strftime": lambda: pl.col("c").dt.strftime("%Y-%m")},
    "dt_us": {
        "truncate": lambda: pl.col("c").dt.truncate("1d"),
        "cast_ms": lambda: pl.col("c").cast(pl.Datetime("ms")),
    },
    "date": {"strftime": lambda: pl.col("c").dt.strftime("%d/%m/%Y")},
    "i64": {
        "cast_string": lambda: pl.col("c").cast(pl.String),
        "add_one": lambda: pl.col("c") + 1,
        "mul": lambda: pl.col("c") * pl.lit(3, pl.Int64),
        "cast_f64": lambda: pl.col("c").cast(pl.Float64),
        "cast_i32_strict_false": lambda: pl.col("c").cast(pl.Int32, strict=False),
    },
    "u64": {
        "cast_string": lambda: pl.col("c").cast(pl.String),
        "shift_right": lambda: pl.col("c") // 2,
    },
    "i8": {"neg": lambda: -pl.col("c"), "cast_string": lambda: pl.col("c").cast(pl.String)},
    "dec38": {
        "mul": lambda: pl.col("c") * 2,
        "cast_string": lambda: pl.col("c").cast(pl.String),
        "cast_f64": lambda: pl.col("c").cast(pl.Float64),
        "cast_dec": lambda: pl.col("c").cast(pl.Decimal(38, 4)),
    },
    "dec9": {
        "mul": lambda: pl.col("c") * 2,
        "cast_string": lambda: pl.col("c").cast(pl.String),
    },
    "dec18": {"cast_string": lambda: pl.col("c").cast(pl.String)},
    "f64": {
        "cast_string": lambda: pl.col("c").cast(pl.String),
        "mul": lambda: pl.col("c") * 2,
        "round": lambda: pl.col("c").round(1),
        "is_nan": lambda: pl.col("c").is_nan(),
    },
    "f32": {"cast_string": lambda: pl.col("c").cast(pl.String)},
    "time_us": {"strftime": lambda: pl.col("c").dt.strftime("%H:%M:%S")},
    "dur_us": {"total_seconds": lambda: pl.col("c").dt.total_seconds()},
    "large_binary": {"len": lambda: pl.col("c").bin.size()},
    "fsb16": {"is_null": lambda: pl.col("c").is_null()},
}


@pytest.mark.parametrize(
    ("col", "name"), [(c, n) for c, d in ELEMENTWISE.items() for n in d]
)
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_toplevel_elementwise_chains(
    toplevel_files: dict[tuple[str, bool], tuple[Path, pl.DataFrame]],
    col: str,
    name: str,
    with_nulls: bool,
) -> None:
    path, df = toplevel_files[col, with_nulls]
    e = ELEMENTWISE[col][name]()
    for tail in [
        lambda lf: lf.select(e),
        lambda lf: lf.select(e.alias("k"), "v").group_by("k").agg(pl.sum("v")),
        lambda lf: lf.select(e.alias("k")).unique(),
        lambda lf: lf.filter(pl.col("v") < 10).select(e),
    ]:
        for engine in ["streaming", "in-memory"]:
            _same_outcome_unordered(
                lambda: tail(pl.scan_parquet(path)).collect(engine=engine),  # type: ignore[arg-type]
                lambda: tail(df.lazy()).collect(),
            )


@pytest.mark.parametrize("col", ["i64", "dt_us_tz", "dec38", "f64", "large_binary"])
def test_dictionary_codes_toplevel_plain_fallback(tmp_path: Path, col: str) -> None:
    # Low cardinality, then unique values: the writer falls back to PLAIN part-way, so the
    # chunk is read flat.
    n = 30_000
    table = _toplevel_table(col, True)
    base = _base_values()[col]()
    uniq = {
        "i64": pa.array(np.arange(n // 2, dtype=np.int64) * 7 + 10**15),
        "dt_us_tz": pa.array(
            np.arange(n // 2, dtype=np.int64) * 1_000_003 + 10**15,
            pa.timestamp("us", "Europe/Amsterdam"),
        ),
        "dec38": pa.array(
            [decimal.Decimal(i) / 100 + 10**20 for i in range(n // 2)],
            pa.decimal128(38, 2),
        ),
        "f64": pa.array(np.arange(n // 2) * 0.37 + 1e9),
        "large_binary": pa.array(
            [f"u{i:08d}".encode() for i in range(n // 2)], pa.large_binary()
        ),
    }[col]
    first = base.take(pa.array(np.arange(n // 2) % TOPLEVEL_CARD))
    full = pa.chunked_array([first, uniq]).combine_chunks()
    table = pa.table({"c": full, "v": pa.array(np.arange(n, dtype=np.int64))})
    expected = pl.from_arrow(table)
    assert isinstance(expected, pl.DataFrame)
    for version in ["1.0", "2.0"]:
        path = _write_toplevel(
            table,
            tmp_path / f"{version}.parquet",
            col,
            row_group_size=n,
            dictionary_pagesize_limit=200,
            data_page_version=version,
            data_page_size=1024,
        )
        out = pl.scan_parquet(path).group_by("c").agg(pl.sum("v")).collect()
        assert_frame_equal(
            out, expected.group_by("c").agg(pl.sum("v")), check_row_order=False
        )
        assert_frame_equal(pl.read_parquet(path), expected)
    # Row groups that are low cardinality, then high cardinality.
    path = _write_toplevel(table, tmp_path / "drift.parquet", col, row_group_size=5_000)
    assert_frame_equal(pl.read_parquet(path), expected)
    out = pl.scan_parquet(path).filter(pl.col("v") % 2 == 0).collect()
    assert_frame_equal(out, expected.filter(pl.col("v") % 2 == 0))


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_toplevel_int96(
    tmp_path: Path, unit: str, with_nulls: bool
) -> None:
    n = 20_000
    k = np.arange(30, dtype=np.int64)
    vals = pa.array(k * 10**11 + 1_700_000_000_000_000, pa.timestamp("us")).cast(
        pa.timestamp("ns")
    )
    col = vals.take(pa.array(np.arange(n) % 30))
    if with_nulls:
        mask = np.zeros(n, dtype=bool)
        mask[::53] = True
        col = pc.if_else(pa.array(mask), pa.scalar(None, col.type), col)
    table = pa.table({"c": col, "v": pa.array(np.arange(n) % 100)})
    path = tmp_path / "f.parquet"
    pq.write_table(
        table,
        path,
        use_dictionary=True,
        use_deprecated_int96_timestamps=True,
        row_group_size=5_000,
    )
    expected = pl.from_arrow(table)
    assert isinstance(expected, pl.DataFrame)
    lf = pl.scan_parquet(path)
    assert_frame_equal(lf.collect(), expected)
    assert_frame_equal(
        lf.group_by("c").agg(pl.sum("v")).collect(),
        expected.group_by("c").agg(pl.sum("v")),
        check_row_order=False,
    )
    e = pl.col("c").dt.strftime("%Y-%m-%d %H:%M:%S")
    assert_frame_equal(lf.select(e).collect(), expected.select(e))
    assert_frame_equal(
        lf.filter(pl.col("c") > expected["c"][9]).collect(),
        expected.filter(pl.col("c") > expected["c"][9]),
    )


def test_dictionary_codes_toplevel_int96_far_future(
    tmp_path: Path,
) -> None:
    # INT96 values far in the future (Polars reads INT96 as microseconds, so none overflow).
    n = 10_000
    days = np.arange(40)
    vals = pa.array(
        (days * 86_400 * 10**6 + 1_600_000_000_000_000).astype(np.int64), pa.timestamp("us")
    ).cast(pa.timestamp("us"))
    far = pa.array(
        [dt.datetime(3000 + i, 1, 1) for i in range(10)], pa.timestamp("us")
    )
    dictionary = pa.chunked_array([vals, far]).combine_chunks()
    col = dictionary.take(pa.array(np.arange(n) % 50))
    table = pa.table({"c": col, "v": pa.array(np.arange(n) % 100)})
    path = tmp_path / "f.parquet"
    pq.write_table(
        table, path, use_dictionary=True, use_deprecated_int96_timestamps=True
    )
    script = (
        "import polars as pl, sys\n"
        "lf = pl.scan_parquet(sys.argv[1])\n"
        "lf.group_by('c').agg(pl.sum('v'), pl.len()).sort('c').collect().write_ipc(sys.argv[2])\n"
        "lf.select(pl.col('c').n_unique()).collect().write_ipc(sys.argv[3])\n"
        "lf.select(pl.col('c').unique().sort()).collect().write_ipc(sys.argv[4])\n"
    )
    out = {}
    for flag in ["1", "0"]:
        files = [str(tmp_path / f"{flag}{i}.ipc") for i in range(3)]
        subprocess.run(
            [sys.executable, "-c", script, str(path), *files],
            env={**os.environ, "POLARS_DICT_CODES": flag},
            check=True,
        )
        out[flag] = [pl.read_ipc(f) for f in files]
    for a, b in zip(out["1"], out["0"], strict=True):
        assert_frame_equal(a, b)


@pytest.mark.parametrize("col", TOPLEVEL_TYPES)
def test_dictionary_codes_toplevel_sink_round_trip(
    toplevel_files: dict[tuple[str, bool], tuple[Path, pl.DataFrame]],
    tmp_path: Path,
    col: str,
) -> None:
    path, df = toplevel_files[col, True]
    out = tmp_path / "out.parquet"
    pl.scan_parquet(path).sink_parquet(out)
    assert_frame_equal(pl.read_parquet(out), df)
    assert_frame_equal(pl.scan_parquet(out).collect(engine="in-memory"), df)  # type: ignore[call-overload]
    back = pq.read_table(out)
    assert back.num_rows == df.height
    # The same through a function that keeps the column coded.
    out2 = tmp_path / "out2.parquet"
    pl.scan_parquet(path).filter(pl.col("v") < 50).sink_parquet(out2)
    assert_frame_equal(pl.read_parquet(out2), df.filter(pl.col("v") < 50))


def test_dictionary_codes_toplevel_float_bits_round_trip(tmp_path: Path) -> None:
    n = 20_000
    nan_payload = np.array([0x7FF8_0000_0000_0001, 0xFFF8_0000_0000_0002], dtype=np.uint64)
    special = np.concatenate(
        [
            np.array([0.0, -0.0, np.inf, -np.inf, 1.5]).view(np.uint64),
            nan_payload,
        ]
    )
    bits = special[np.arange(n) % len(special)]
    table = pa.table({"c": pa.array(bits.view(np.float64))})
    path = tmp_path / "f.parquet"
    pq.write_table(table, path, use_dictionary=True, row_group_size=5_000)
    out = tmp_path / "o.parquet"
    pl.scan_parquet(path).sink_parquet(out)
    for p in [path, out]:
        got = pl.read_parquet(p)["c"].to_numpy().view(np.uint64)
        np.testing.assert_array_equal(got, bits)
        got = pq.read_table(p)["c"].to_numpy().view(np.uint64)
        np.testing.assert_array_equal(got, bits)


def test_dictionary_codes_toplevel_unaffected_types(tmp_path: Path) -> None:
    n = 5_000
    table = pa.table(
        {
            "b": pa.array([i % 3 == 0 for i in range(n)]),
            "cat": pa.array([f"c{i % 5}" for i in range(n)]),
            "l": pa.array([[i % 4, 1] for i in range(n)]),
            "s": pa.array([{"a": i % 4} for i in range(n)]),
        }
    )
    path = tmp_path / "f.parquet"
    pq.write_table(table, path, use_dictionary=True)
    expected = pl.from_arrow(table)
    assert isinstance(expected, pl.DataFrame)
    assert_frame_equal(pl.scan_parquet(path).collect(), expected)


# ---------------------------------------------------------------------------------------
# Writing coded columns of every type as parquet dictionaries.

WRITABLE_DICTIONARIES = [
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "date",
    "time_us",
    "time_ns",
    "dt_us",
    "dt_us_tz",
    "dt_ns_tz",
    "dt_ms",
    "dur_us",
    "dur_ms",
    "f32",
    "f64",
    "dec9",
    "dec18",
    "dec38",
    "large_binary",
]


def _encodings(path: Path, column: int = 0) -> set[str]:
    md = pq.read_metadata(path)
    return {e for rg in range(md.num_row_groups) for e in md.row_group(rg).column(column).encodings}


def _single_group_file(tmp_path: Path, name: str, with_nulls: bool) -> Path:
    return _write_toplevel(
        _toplevel_table(name, with_nulls),
        tmp_path / "in.parquet",
        name,
        row_group_size=TOPLEVEL_ROWS,
    )


@pytest.mark.parametrize("compression", ["zstd", "uncompressed"])
@pytest.mark.parametrize("with_nulls", [False, True])
@pytest.mark.parametrize("name", WRITABLE_DICTIONARIES)
def test_dictionary_codes_sink_writes_non_byte_dictionaries(
    tmp_path: Path, name: str, with_nulls: bool, compression: str
) -> None:
    path = _single_group_file(tmp_path, name, with_nulls)
    expected = pl.from_arrow(pq.read_table(path))
    assert isinstance(expected, pl.DataFrame)
    out = tmp_path / "out.parquet"
    pl.scan_parquet(path).sink_parquet(out, compression=compression)  # type: ignore[arg-type]
    assert "RLE_DICTIONARY" in _encodings(out), name
    assert_frame_equal(pl.read_parquet(out), expected)
    assert_frame_equal(pl.scan_parquet(out).collect(engine="in-memory"), expected)  # type: ignore[call-overload]
    # Other readers.
    back = pq.read_table(out)
    assert back.num_rows == TOPLEVEL_ROWS
    assert_frame_equal(pl.from_arrow(back), expected)  # type: ignore[arg-type]
    # The source's rows after a filter keep only the values they use.
    out2 = tmp_path / "out2.parquet"
    pl.scan_parquet(path).filter(pl.col("v") < 3).sink_parquet(out2, compression=compression)  # type: ignore[arg-type]
    assert_frame_equal(pl.read_parquet(out2), expected.filter(pl.col("v") < 3))


@pytest.mark.parametrize("name", ["i64", "dt_us_tz", "f64", "dec38", "large_binary"])
def test_dictionary_codes_sink_distinct_values_are_written_flat(
    tmp_path: Path, name: str
) -> None:
    # Many distinct values behind a large dictionary: no dictionary page.
    n = 30_000
    uniq = {
        "i64": pa.array(np.arange(n, dtype=np.int64) * 7 + 10**15),
        "dt_us_tz": pa.array(
            np.arange(n, dtype=np.int64) * 1_000_003 + 10**15,
            pa.timestamp("us", "Europe/Amsterdam"),
        ),
        "f64": pa.array(np.arange(n) * 0.37 + 1e9),
        "dec38": pa.array([decimal.Decimal(i) / 100 for i in range(n)], pa.decimal128(38, 2)),
        "large_binary": pa.array([f"u{i:08d}".encode() for i in range(n)], pa.large_binary()),
    }[name]
    table = pa.table({"c": uniq})
    path = _write_toplevel(table, tmp_path / "in.parquet", name, row_group_size=n)
    out = tmp_path / "out.parquet"
    pl.scan_parquet(path).sink_parquet(out)
    assert "RLE_DICTIONARY" not in _encodings(out)
    assert_frame_equal(pl.read_parquet(out), pl.from_arrow(table))  # type: ignore[arg-type]


def test_dictionary_codes_sink_writes_each_value_once(tmp_path: Path) -> None:
    out = tmp_path / "out.parquet"
    # Elementwise results with fewer distinct values than inputs, as `dt.date` of timestamps.
    n = 40_000
    stamps = pa.array(
        np.arange(500, dtype=np.int64) * 3_600 * 10**6 + 1_700_000_000_000_000,
        pa.timestamp("us"),
    )
    path = tmp_path / "in2.parquet"
    pq.write_table(
        pa.table({"c": stamps.take(pa.array(np.arange(n) % 500))}),
        path,
        use_dictionary=True,
        row_group_size=n,
    )
    lf = pl.scan_parquet(path).select(pl.col("c").dt.date())
    lf.sink_parquet(out)
    result = pl.read_parquet(out)
    assert_frame_equal(result, lf.collect())
    md = pq.read_metadata(out).row_group(0).column(0)
    assert "RLE_DICTIONARY" in md.encodings
    # About 21 distinct days (5-bit keys), not 500 values (9-bit keys, 45 kB).
    assert md.total_uncompressed_size < 30_000


@pytest.mark.parametrize("name", ["dec9", "dec18", "dec38"])
def test_dictionary_codes_sink_decimal_statistics(tmp_path: Path, name: str) -> None:
    n = 20_000
    values = [decimal.Decimal(v) / 100 for v in (-12345, 7, -3, 900, 123, -99999)]
    ty = {"dec9": 9, "dec18": 18, "dec38": 38}[name]
    col = pa.array(values, pa.decimal128(ty, 2)).take(pa.array(np.arange(n) % len(values)))
    path = tmp_path / "in.parquet"
    _write_toplevel(pa.table({"c": col}), path, name, row_group_size=n)
    out = tmp_path / "out.parquet"
    pl.scan_parquet(path).sink_parquet(out, statistics={"min": True, "max": True, "null_count": True})
    assert "RLE_DICTIONARY" in _encodings(out)
    stats = pq.read_metadata(out).row_group(0).column(0).statistics
    assert (stats.min, stats.max) == (min(values), max(values))
    # Row group pruning by the statistics.
    lf = pl.scan_parquet(out)
    assert lf.filter(pl.col("c") > decimal.Decimal(10_000)).collect().height == 0
    assert lf.filter(pl.col("c") < decimal.Decimal(-1_000)).collect().height == 0
    assert lf.filter(pl.col("c") == decimal.Decimal("-123.45")).collect().height > 0


def test_dictionary_codes_sink_float_dictionary_is_bit_exact(tmp_path: Path) -> None:
    n = 20_000
    special = np.array(
        [0x0, 0x8000_0000_0000_0000, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000, 0x3FF8_0000_0000_0000,
         0x7FF8_0000_0000_0001, 0xFFF8_0000_0000_0002],
        dtype=np.uint64,
    )
    bits = special[np.arange(n) % len(special)]
    path = tmp_path / "in.parquet"
    pq.write_table(
        pa.table({"c": pa.array(bits.view(np.float64))}), path, use_dictionary=True, row_group_size=n
    )
    out = tmp_path / "out.parquet"
    pl.scan_parquet(path).sink_parquet(out)
    assert "RLE_DICTIONARY" in _encodings(out)
    for p in [out, path]:
        np.testing.assert_array_equal(pl.read_parquet(p)["c"].to_numpy().view(np.uint64), bits)
        np.testing.assert_array_equal(pq.read_table(p)["c"].to_numpy().view(np.uint64), bits)


# ---------------------------------------------------------------------------------------
# Review leads: sorted flags, extension types, 128-bit values, strict casts.


def _coded_batches(path: Path, chunk_size: int) -> pl.DataFrame:
    """The file read as batches, which keep dictionary codes (`collect` reads flat)."""
    return pl.concat(
        pl.scan_parquet(path).collect_batches(chunk_size=chunk_size), rechunk=True
    )


def test_dictionary_codes_keep_the_sorted_flag(tmp_path: Path) -> None:
    n = 100_000
    table = pa.table(
        {
            "c": pa.array(np.arange(n, dtype=np.int64) // 1_000 * 10**9),
            "d": pa.array((np.arange(n, dtype=np.int64) // 1_000)[::-1] * 7),
        }
    )
    path = tmp_path / "sorted.parquet"
    pq.write_table(
        table,
        path,
        row_group_size=n,
        use_dictionary=True,
        sorting_columns=[pq.SortingColumn(0)],
    )
    script = (
        "import polars as pl, sys\n"
        "lf = pl.scan_parquet(sys.argv[1])\n"
        "(b,) = lf.collect_batches(chunk_size=100_000)\n"
        "print(b['c'].flags['SORTED_ASC'], b['d'].flags['SORTED_ASC'], b.height)\n"
        "o = lf.collect()\n"
        "print(o['c'].flags['SORTED_ASC'], o['d'].flags['SORTED_ASC'])\n"
        "p = lf.filter(pl.col('d') >= 0).collect_batches(chunk_size=100_000)\n"
        "print(pl.concat(p)['c'].flags['SORTED_ASC'])\n"
    )
    out = {}
    for flag in ["1", "0"]:
        r = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            env={**os.environ, "POLARS_DICT_CODES": flag},
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        out[flag] = r.stdout
    assert out["1"] == out["0"]
    assert out["0"].splitlines()[0] == "True False 100000"
    assert out["0"].splitlines()[1] == "True False"


def test_dictionary_codes_extension_type_reads_as_storage(tmp_path: Path) -> None:
    n = 20_000
    values = [bytes([i + 1]) * 16 for i in range(100)]
    storage = pa.array([values[i % 100] for i in range(n)], pa.binary(16))
    table = pa.table({"c": pa.ExtensionArray.from_storage(pa.uuid(), storage)})
    path = tmp_path / "uuid.parquet"
    pq.write_table(table, path, row_group_size=n, use_dictionary=True)
    flat = pl.scan_parquet(path).collect()
    coded = _coded_batches(path, n)
    assert coded.schema == flat.schema
    assert_frame_equal(coded, flat)
    assert coded["c"].to_list()[:3] == [values[0], values[1], values[2]]


def test_dictionary_codes_128_bit_integers_round_trip(tmp_path: Path) -> None:
    n = 20_000
    k = np.arange(n) % 100
    df = pl.DataFrame(
        {
            "i": pl.Series([(int(i) - 50) * 10**30 for i in k], dtype=pl.Int128),
            "u": pl.Series([int(i) * 10**30 for i in k], dtype=pl.UInt128),
        }
    )
    path = tmp_path / "wide.parquet"
    df.write_parquet(path, row_group_size=n)
    flat = pl.scan_parquet(path).collect()
    assert_frame_equal(flat, df)
    assert_frame_equal(_coded_batches(path, n), df)
    out = tmp_path / "wide-out.parquet"
    pl.scan_parquet(path).sink_parquet(out)
    assert_frame_equal(pl.read_parquet(out), df)


def test_dictionary_codes_negative_fixed_length_decimals(tmp_path: Path) -> None:
    n = 30_000
    vals = [
        decimal.Decimal(i - 50) * decimal.Decimal(10**20) + decimal.Decimal("0.07")
        for i in range(100)
    ]
    col = [None if i % 11 == 0 else vals[i % 100] for i in range(n)]
    table = pa.table({"c": pa.array(col, pa.decimal128(38, 2))})
    path = tmp_path / "dec.parquet"
    pq.write_table(table, path, row_group_size=n, use_dictionary=True)
    expected = pl.from_arrow(table)
    assert_frame_equal(pl.scan_parquet(path).collect(), expected)
    coded = _coded_batches(path, n)
    assert_frame_equal(coded, expected)
    assert coded.select(
        lo=pl.col("c").min(), hi=pl.col("c").max()
    ).row(0) == (
        vals[0],
        vals[99],
    )
    out = tmp_path / "dec-out.parquet"
    pl.scan_parquet(path).sink_parquet(out)
    assert_frame_equal(pl.read_parquet(out), expected)


def test_dictionary_codes_strict_cast_only_on_used_values(tmp_path: Path) -> None:
    n = 30_000
    values = np.array([1, 2, 3, 1_000_000_000_000], dtype=np.int64)
    df = pl.DataFrame(
        {"c": values[np.arange(n) % 4], "v": np.arange(n, dtype=np.int64)}
    )
    path = tmp_path / "ints.parquet"
    df.write_parquet(path, row_group_size=n)
    lf = pl.scan_parquet(path)
    # The large value is filtered out before the cast, which must not see it.
    for engine in ["streaming", "in-memory"]:
        out = (
            lf.filter(pl.col("c") < 10)
            .select(pl.col("c").cast(pl.Int8, strict=True).sum())
            .collect(engine=engine)  # type: ignore[call-overload]
        )
        assert out.item() == int(df.filter(pl.col("c") < 10)["c"].sum())
        with pytest.raises(pl.exceptions.InvalidOperationError):
            lf.select(pl.col("c").cast(pl.Int8, strict=True)).collect(engine=engine)  # type: ignore[call-overload]
    # The same on coded batches, as elementwise maps of coded morsels see the dictionary.
    batches = lf.filter(pl.col("c") < 10).select(pl.col("c").cast(pl.Int8, strict=True))
    assert pl.concat(batches.collect_batches(chunk_size=n)).height == 3 * n // 4
