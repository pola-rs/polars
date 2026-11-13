"""String columns decoded from a parquet dictionary are held as dictionary codes.

Every result must equal the one computed on the same data built in memory, which has
no codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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


@pytest.mark.parametrize("name", sorted(QUERIES))
@pytest.mark.parametrize("with_nulls", [False, True])
def test_dictionary_codes_queries(tmp_path: Path, name: str, with_nulls: bool) -> None:
    df = _df(with_nulls=with_nulls)
    path = _write(df, tmp_path / "f.parquet")
    q = QUERIES[name]
    expected = q(df.lazy()).collect()
    for engine in ["streaming", "in-memory"]:
        out = q(pl.scan_parquet(path)).collect(engine=engine)  # type: ignore[call-overload]
        ordered = name in {"slice", "head", "sort", "filter_eq", "filter_is_in"}
        assert_frame_equal(
            out, expected, check_row_order=ordered, categorical_as_str=True
        )


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


def test_dictionary_codes_nested_and_binary_untouched(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "l": [["a", "b"], ["a"]] * 500,
            "b": [b"x", b"y"] * 500,
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
