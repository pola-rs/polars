"""Struct-field projection pushdown: read and compute only the fields used."""

from __future__ import annotations

import sys
import time
import types
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import ComputeError, PolarsError, SchemaError, ShapeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

NO_PUSHDOWN = pl.QueryOptFlags(projection_pushdown=False)
ENGINES = ("in-memory", "streaming")


def assert_same_as_without_pushdown(
    lf: pl.LazyFrame, *, check_row_order: bool = True
) -> pl.DataFrame:
    expected = lf.collect(optimizations=NO_PUSHDOWN)
    assert_engines_equal(lf, expected, check_row_order=check_row_order)
    return expected


def assert_engines_equal(lf: pl.LazyFrame, expected: pl.DataFrame, **kw: Any) -> None:
    for engine in ENGINES:
        assert_frame_equal(lf.collect(engine=engine), expected, **kw)  # type: ignore[call-overload]


def assert_raises_with_and_without_pushdown(
    lf: pl.LazyFrame, exc: type[Exception] = PolarsError
) -> None:
    with pytest.raises(exc):
        lf.collect(optimizations=NO_PUSHDOWN)
    for engine in ENGINES:
        with pytest.raises(exc):
            lf.collect(engine=engine)  # type: ignore[call-overload]


def project_fields(lf: pl.LazyFrame, **kw: Any) -> str | None:
    """The field selection of the (first) scan, or None if it reads whole columns."""
    for line in lf.explain(**kw).splitlines():
        if "PROJECT FIELDS:" in line:
            return line.split("PROJECT FIELDS:")[1].strip()
    return None


def scan(
    df: pl.DataFrame, fmt: str, tmp_path: Path, name: str = "data"
) -> pl.LazyFrame:
    """`df` written as `fmt` and scanned, or `df.lazy()` for "memory"."""
    path = tmp_path / f"{name}.{fmt}"
    if fmt == "parquet":
        df.write_parquet(path)
        return pl.scan_parquet(path)
    if fmt.startswith("ipc"):
        df.write_ipc(path, compression="zstd" if fmt == "ipc-zstd" else "uncompressed")
        return pl.scan_ipc(path)
    if fmt == "ndjson":
        df.write_ndjson(path)
        return pl.scan_ndjson(path, schema=df.schema)
    return df.lazy()


def corrupt_chunk(path: Path, leaf: str, row_groups: list[int] | None = None) -> None:
    """Overwrite the column chunk of `leaf` (a dotted path) with garbage."""
    md = pq.ParquetFile(path).metadata
    data = bytearray(path.read_bytes())
    found = False
    for rg in row_groups if row_groups is not None else range(md.num_row_groups):
        chunks = [md.row_group(rg).column(i) for i in range(md.num_columns)]
        starts = sorted(
            min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset)
            for c in chunks
        )
        for col in (c for c in chunks if c.path_in_schema == leaf):
            found = True
            start = min(
                col.data_page_offset, col.dictionary_page_offset or col.data_page_offset
            )
            end = min(
                [start + col.total_compressed_size, *(s for s in starts if s > start)]
            )
            data[start:end] = b"\xff" * (end - start)
    assert found, leaf
    path.write_bytes(bytes(data))


STRUCT_DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "s": [
            {"a": 1, "b": "x", "c": {"x": 1.5, "y": [1, 2]}},
            None,
            {"a": 3, "b": None, "c": None},
            {"a": None, "b": "z", "c": {"x": None, "y": None}},
        ],
    },
    schema={
        "id": pl.Int64,
        "s": pl.Struct(
            {
                "a": pl.Int64,
                "b": pl.String,
                "c": pl.Struct({"x": pl.Float64, "y": pl.List(pl.Int64)}),
            }
        ),
    },
)
FORMATS = ["parquet", "ipc", "ipc-zstd", "ndjson"]

s = pl.col("s")
a, b, c = (s.struct.field(f) for f in "abc")

QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(a),
    lambda lf: lf.select(c.struct.field("x")),
    lambda lf: lf.select(c.struct.field("y")),
    lambda lf: lf.select(s.struct.field("a", "b")),
    lambda lf: lf.select(s.is_null()),
    lambda lf: lf.select(
        s.is_not_null().sum().alias("n"),
        s.count().alias("count"),
        s.null_count().alias("null_count"),
    ),
    lambda lf: lf.select(c.is_null()),
    lambda lf: lf.select(s, a.alias("x")),
    lambda lf: lf.select(a.alias("x"), s),
    lambda lf: lf.select(a.alias("x"), c),
    lambda lf: lf.with_columns(t=b).select("t", "id"),
    lambda lf: lf.select(a + pl.col("id")),
    lambda lf: lf.filter(a > 1).select(b),
    lambda lf: lf.filter(a.cast(pl.Float64) > 1.5).select(b),
    lambda lf: lf.filter(a > 0).select(s),
    lambda lf: lf.select(s.struct.rename_fields(["p", "q", "r"]).struct.field("q")),
    lambda lf: lf.select(
        s.struct.rename_fields(["p", "q", "r"]).struct.field("r").struct.field("x"),
    ),
    lambda lf: lf.select(s.struct.with_fields(d=a * 2).struct.field("d", "b")),
    lambda lf: lf.select(s.struct.with_fields(pl.field("a") * 2).struct.field("a")),
    lambda lf: lf.select(s.struct.with_fields(pl.field("a") * 2)),
    lambda lf: lf.select(s.name.prefix_fields("p_").struct.field("p_b")),
    lambda lf: lf.select(s.name.suffix_fields("_s").struct.field("c_s")),
    lambda lf: lf.select(s.struct.drop(["a"]).struct.field("b")),
    lambda lf: lf.select(s.struct.drop(["a"])),
    lambda lf: lf.select(s.struct.json_encode()),
    lambda lf: lf.select(s.struct.unnest()),
    lambda lf: lf.unnest("s").select("b"),
    lambda lf: lf.unnest("s", separator="::").select("s::c"),
    lambda lf: lf.group_by(b, maintain_order=True).agg(a.sum()),
    lambda lf: lf.group_by("id", maintain_order=True).agg(s.first().struct.field("a")),
    lambda lf: lf.sort(a, nulls_last=True).select(b),
    lambda lf: lf.head(2).select(a),
    lambda lf: lf.select(s.first().struct.field("a"), s.last().struct.field("b")),
    lambda lf: lf.select(s.filter(pl.col("id") > 1).struct.field("a")),
    lambda lf: lf.select(
        pl.when(pl.col("id") > 2).then(s).otherwise(s).struct.field("a")
    ),
    lambda lf: lf.with_columns(t=s).select(pl.col("t").struct.field("a")),
    lambda lf: lf.rename({"s": "t"}).select(pl.col("t").struct.field("a")),
    lambda lf: lf.join(lf, on="id", suffix="_r", maintain_order="left").select(
        pl.col("s_r").struct.field("a"), b
    ),
    lambda lf: lf.cache().select(a),
]


@pytest.mark.parametrize("fmt", [*FORMATS, "memory"])
@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    assert_same_as_without_pushdown(query(scan(STRUCT_DF, fmt, tmp_path)))


@pytest.mark.parametrize("fmt", FORMATS)
def test_struct_field_pruning_explain(fmt: str, tmp_path: Path) -> None:
    lf = scan(STRUCT_DF, fmt, tmp_path)

    assert project_fields(lf.select(a)) == "s{a}"
    assert project_fields(lf.select(c.struct.field("x"))) == "s{c{x}}"
    assert project_fields(lf.select(s.is_null())) == "s{}"
    # A field read by the predicate is read as well.
    assert project_fields(lf.filter(b == "x").select(a)) in ("s{a, b}", "s{b, a}")
    # Used whole, in either order; nothing pruned without projection pushdown.
    for q in [
        lf.select(s),
        lf.select(s, a.alias("x")),
        lf.select(a.alias("x"), s),
        lf.select(a.alias("x"), s.is_null().alias("n"), s),
    ]:
        assert project_fields(q) is None
    assert project_fields(lf.select(a), optimizations=NO_PUSHDOWN) is None


def test_struct_field_pruning_skips_unselected_leaves(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "s": [
                {"a": i, "b": f"long string {i}" * 10, "c": {"x": i, "y": f"{i}" * 50}}
                for i in range(1000)
            ]
        }
    )
    path = tmp_path / "corrupt.parquet"
    pq.write_table(df.to_arrow(), path, use_dictionary=False, write_statistics=False)
    corrupt_chunk(path, "s.b")
    corrupt_chunk(path, "s.c.y")

    # Reading the corrupted leaves fails, so these only pass if they are not read.
    q = pl.scan_parquet(path).select(a, c.struct.field("x"))
    with pytest.raises(ComputeError):
        q.collect(optimizations=NO_PUSHDOWN)
    assert_engines_equal(q, df.select(a, c.struct.field("x")))
    assert_engines_equal(
        pl.scan_parquet(path).select(s.is_null().sum()),
        pl.DataFrame({"s": [0]}, schema={"s": pl.get_index_type()}),
    )


def test_struct_field_pruning_ipc_skips_unselected_fields(tmp_path: Path) -> None:
    marker = "QZQZQZQZ"
    n = 1000
    table = pa.table(
        {"s": [{"a": i, "b": marker * 4, "c": {"x": i, "y": marker}} for i in range(n)]}
    )
    path = tmp_path / "corrupt.arrow"
    with pa.ipc.new_file(path, table.schema) as writer:
        writer.write_table(table, max_chunksize=300)
    # Make the string values of `b` and `c.y` invalid UTF-8.
    data = path.read_bytes()
    assert marker.encode() in data
    path.write_bytes(data.replace(marker.encode(), b"\xff" * len(marker)))

    q = pl.scan_ipc(path).select(a, c.struct.field("x"))
    with pytest.raises(PolarsError):
        q.collect(optimizations=NO_PUSHDOWN)
    assert_engines_equal(q, pl.DataFrame({"a": range(n), "x": range(n)}))
    for engine in ENGINES:
        q = pl.scan_ipc(path).select(s.is_null().sum())
        assert q.collect(engine=engine).item() == 0  # type: ignore[call-overload]


def test_struct_field_pruning_ndjson_skips_unselected_fields(tmp_path: Path) -> None:
    path = tmp_path / "s.ndjson"
    path.write_text(
        '{"s": {"a": 1, "b": {"not": "an int"}}}\n'
        '{"s": null}\n'
        '{"s": {"a": 3, "b": "nor this"}}\n'
    )
    lf = pl.scan_ndjson(path, schema={"s": pl.Struct({"a": pl.Int64, "b": pl.Int64})})

    # Parsing `b` fails, so these only pass if it is not parsed.
    with pytest.raises(PolarsError):
        lf.select(a).collect(optimizations=NO_PUSHDOWN)
    assert_engines_equal(
        lf.select(a, s.is_null().alias("n")),
        pl.DataFrame({"a": [1, None, 3], "n": [False, True, False]}),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_struct_field_pruning_validity_only(fmt: str, tmp_path: Path) -> None:
    df = pl.DataFrame({"s": [{"a": 1}, None, {"a": None}, None, {"a": 5}] * 1000})
    lf = scan(df, fmt, tmp_path)

    out = assert_same_as_without_pushdown(
        lf.select(s.is_null().alias("n"), s.is_not_null().sum().alias("v"))
    )
    assert out["n"].to_list() == [False, True, False, True, False] * 1000
    assert out["v"].to_list() == [3000] * 5000
    assert_same_as_without_pushdown(lf.select(pl.len()))
    assert_same_as_without_pushdown(lf.select(s.null_count()))


def test_struct_field_pruning_ndjson_empty_object_vs_null(tmp_path: Path) -> None:
    # `{}` is a valid struct with null fields, `null` a null struct: validity-only reads
    # must tell them apart, also for list elements.
    path = tmp_path / "x.ndjson"
    path.write_text(
        '{"s": {}, "l": [{}, null, {"a": 1}]}\n'
        '{"s": null, "l": null}\n'
        '{"s": {"a": 2}, "l": []}\n'
    )
    el = pl.Struct({"a": pl.Int64, "b": pl.String})
    lf = pl.scan_ndjson(path, schema={"s": el, "l": pl.List(el)})
    for q in [s.is_null(), l.explode().is_null(), l.list.first().is_null()]:
        assert_same_as_without_pushdown(lf.select(q))


def test_struct_field_pruning_ndjson_schema(tmp_path: Path) -> None:
    path = tmp_path / "s.ndjson"
    path.write_text(
        '{"s": {"a": 1, "b": "x", "extra": [1]}, "t": 1}\n'
        '{"s": {"b": "y"}, "t": 2}\n'
        '{"s": null, "t": 3}\n'
        '{"t": 4}\n'
    )
    for schema in [
        None,
        pl.Schema({"s": pl.Struct({"a": pl.Int64, "b": pl.String}), "t": pl.Int64()}),
        pl.Schema({"s": pl.Struct({"b": pl.String, "a": pl.Int32, "z": pl.Float64})}),
    ]:
        for ignore_errors in [False, True]:
            lf = pl.scan_ndjson(path, schema=schema, ignore_errors=ignore_errors)
            for q in [lf.select(b), lf.select(a, s.is_null()), lf.select(s.is_null())]:
                assert_same_as_without_pushdown(q)


def test_struct_field_pruning_dotted_names(tmp_path: Path) -> None:
    df = pl.DataFrame({"s": [{"a.b": 1, "a": {"b": 2, "c": 3}}, None], "s.a": [4, 5]})
    lf = scan(df, "parquet", tmp_path)
    for q in [
        lf.select(s.struct.field("a.b")),
        lf.select(a.struct.field("b")),
        lf.select(s.struct.field("a.b"), pl.col("s.a")),
        lf.select(a.struct.field("c"), s.struct.field("a.b")),
    ]:
        assert_same_as_without_pushdown(q)


@pytest.mark.parametrize("policy", ["extra", "missing"])
def test_struct_field_pruning_cast_policies(tmp_path: Path, policy: str) -> None:
    pl.DataFrame(
        {"s": [{"a": 1, "b": "x"}, None, {"a": 3, "b": None}]},
    ).write_parquet(tmp_path / "1.parquet")
    # With `missing`, the second file lacks `b`, the only field read by some queries
    # below: the struct validity must still come from the file.
    other = (
        [{"z": 0, "b": "y", "a": 4}]
        if policy == "extra"
        else [None, {"a": 5}, {"a": None}]
    )
    pl.DataFrame({"s": other}).write_parquet(tmp_path / "2.parquet")
    paths = [tmp_path / "1.parquet", tmp_path / "2.parquet"]

    def queries(lf: pl.LazyFrame) -> list[pl.LazyFrame]:
        return [
            lf.select(s),
            lf.select(a),
            lf.select(b),
            lf.select(s.is_null()),
            lf.select(b, s.is_null().alias("is_null")),
            # With a predicate, the projection is resolved before the reader starts.
            lf.filter(a > 0).select(a),
            lf.filter(a > 1).select(s),
        ]

    # Under the default `ScanCastOptions` (raise on extra / missing struct fields), a
    # scan whose files disagree on the fields of `s` raises with or without pruning.
    for q in queries(pl.scan_parquet(paths)):
        assert_raises_with_and_without_pushdown(q, SchemaError)

    # Allowed by the policy, the fields are still pruned.
    cast_options = (
        pl.ScanCastOptions(extra_struct_fields="ignore")
        if policy == "extra"
        else pl.ScanCastOptions(missing_struct_fields="insert")
    )
    lf = pl.scan_parquet(paths, cast_options=cast_options)
    assert project_fields(lf.select(a)) == "s{a}"
    for q in queries(lf):
        assert_same_as_without_pushdown(q)
    if policy == "missing":
        out = lf.select(b, s.is_null()).collect()
        assert out["s"].to_list() == [False, True, False, True, False, False]


@pytest.mark.parametrize("fmt", ["ipc", "ipc-zstd"])
def test_struct_field_pruning_ipc_schema_evolution(fmt: str, tmp_path: Path) -> None:
    # Struct fields missing from, or extra in, the second file: whatever the cast
    # policy makes of them without pushdown must not change with it.
    for name, rows in [
        ("1", [{"a": 1, "b": "x"}, None, {"a": 3, "b": None}]),
        ("2", [None, {"a": 5}, {"a": None}]),
        ("3", [{"z": 0, "b": "y", "a": 3}]),
    ]:
        scan(pl.DataFrame({"s": rows}), fmt, tmp_path, name)

    for other in ["2", "3"]:
        lf = pl.scan_ipc([tmp_path / f"1.{fmt}", tmp_path / f"{other}.{fmt}"])
        for q in [
            lf.select(b, s.is_null().alias("is_null")),
            lf.select(a),
            lf.select(s.is_null()),
            lf.select(s),
        ]:
            try:
                q.collect(optimizations=NO_PUSHDOWN)
            except PolarsError as e:
                assert_raises_with_and_without_pushdown(q, type(e))
            else:
                assert_same_as_without_pushdown(q)


@pytest.mark.parametrize("fmt", ["ipc", "ipc-zstd"])
def test_struct_field_pruning_ipc_nested_and_dictionaries(
    fmt: str, tmp_path: Path
) -> None:
    row = {
        "cat": "u",
        "enum": "p",
        "inner": {"x": 1, "cat": "v", "l": [{"q": 1}]},
        "arr": [1, 2],
    }
    inner = pl.Struct(
        {
            "x": pl.Int32,
            "cat": pl.Categorical(),
            "l": pl.List(pl.Struct({"q": pl.Int8})),
        }
    )
    df = pl.DataFrame(
        {
            "s": [row, None, {"cat": None, "enum": "q", "inner": None, "arr": None}]
            * 100
        },
        schema={
            "s": pl.Struct(
                {
                    "cat": pl.Categorical(),
                    "enum": pl.Enum(["p", "q"]),
                    "inner": inner,
                    "arr": pl.Array(pl.Int64, 2),
                }
            )
        },
    )
    lf = scan(df, fmt, tmp_path)
    inner_ = s.struct.field("inner")
    for q in [
        s.struct.field("enum"),
        s.struct.field("cat"),
        s.struct.field("arr"),
        inner_.struct.field("cat"),
        inner_.struct.field("l"),
        inner_.is_null(),
    ]:
        assert_same_as_without_pushdown(lf.select(q))
    assert_same_as_without_pushdown(
        lf.select(inner_.struct.field("x"), s.struct.field("arr"))
    )
    assert_same_as_without_pushdown(lf.select(inner_.struct.field("x"), s.is_null()))


@pytest.mark.parametrize("fmt", FORMATS)
def test_struct_field_pruning_multiple_files(fmt: str, tmp_path: Path) -> None:
    for i in range(3):
        scan(
            pl.DataFrame(
                {
                    "s": [{"a": i, "b": str(i), "c": {"x": i, "y": "y"}}, None],
                    "i": [i, i],
                }
            ),
            fmt,
            tmp_path,
            str(i),
        )
    paths = [tmp_path / f"{i}.{fmt}" for i in range(3)]
    scan_fn = {"parquet": pl.scan_parquet, "ndjson": pl.scan_ndjson}
    lf: pl.LazyFrame = scan_fn.get(fmt, pl.scan_ipc)(paths)  # type: ignore[operator]
    for q in [
        lf.select(b, "i"),
        lf.filter(pl.col("i") > 0).select(c.struct.field("y")),
        lf.head(3).select(a),
        lf.tail(3).select(a),
        lf.with_row_index().select("index", a),
        lf.select(a, c),
    ]:
        assert_same_as_without_pushdown(q)


def test_struct_field_pruning_hive(tmp_path: Path) -> None:
    for part in [1, 2]:
        (tmp_path / f"p={part}").mkdir()
        pl.DataFrame(
            {"s": [{"a": part, "b": str(part)}, None]},
        ).write_parquet(tmp_path / f"p={part}" / "data.parquet")

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True)
    assert_same_as_without_pushdown(lf.select(a, pl.col("p")))
    assert_same_as_without_pushdown(lf.filter(pl.col("p") == 2).select(b))


def test_struct_field_pruning_concat(tmp_path: Path) -> None:
    lf = scan(STRUCT_DF, "parquet", tmp_path)

    # The branches read different fields, but must have equal dtypes to be
    # concatenated.
    q = pl.concat(
        [
            lf.filter(b == "x"),
            lf,
            lf.select(pl.col("id"), s.struct.with_fields(pl.lit(1).alias("e"))),
        ],
        how="diagonal_relaxed",
    ).select(a)
    assert_same_as_without_pushdown(q)

    q = pl.concat([lf.filter(c.struct.field("x") > 1.0), lf.head(1)])
    assert_same_as_without_pushdown(q.select(a, c.struct.field("y")))


def test_struct_field_pruning_issue_23503() -> None:
    def raising_udf(_: Any) -> Any:
        msg = "should not run"
        raise ValueError(msg)

    lf = pl.LazyFrame({"a": [1, 2, 3]}).with_columns(
        b=pl.struct(
            pl.col("a")
            .map_batches(raising_udf, return_dtype=pl.Int64)
            .alias("field_a"),
            pl.lit(1).alias("field_b"),
        )
    )

    q = lf.select(pl.col("b").struct.field("field_b"))
    expected = pl.DataFrame({"field_b": [1, 1, 1]}, schema={"field_b": pl.Int32})
    assert_engines_equal(q, expected)
    with pytest.raises((ComputeError, ValueError), match="should not run"):
        q.collect(optimizations=NO_PUSHDOWN)


@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(pl.col("t").struct.field("y")),
        lambda lf: lf.select(pl.col("t").is_null()),
        lambda lf: lf.select(pl.col("t").struct.field("y"), pl.col("t").is_null()),
        lambda lf: lf.select(pl.col("t")),
        lambda lf: lf.select(pl.col("t").struct.field("x").sum()),
        lambda lf: lf.select(pl.col("t").struct.field("x"), pl.col("a")),
    ],
)
@pytest.mark.parametrize(
    "build",
    [
        pl.struct(pl.col("a").alias("x"), (pl.col("b") * 2).alias("y")),
        pl.struct(pl.col("a").sum().alias("x"), pl.col("b").alias("y")),
        pl.struct(pl.col("a").alias("x"), pl.lit(1).alias("y")),
        pl.struct(pl.col("a").sum().alias("x"), pl.lit(1).alias("y")),
        pl.struct(pl.col("a").alias("p"), pl.col("b").alias("q")).struct.rename_fields(
            ["x", "y"]
        ),
        pl.struct(pl.col("a").alias("x")).struct.with_fields(y=pl.field("x") + 1),
        pl.struct(pl.col("a").alias("x"), pl.col("b").alias("z")).struct.with_fields(
            y=pl.col("b") + 1
        ),
    ],
)
def test_struct_field_pruning_mid_plan(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], build: pl.Expr
) -> None:
    base = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    assert_same_as_without_pushdown(query(base.with_columns(t=build)))
    assert_same_as_without_pushdown(query(base.select(build.alias("t"), "a")))


def test_struct_field_pruning_mid_plan_drops_computation() -> None:
    lf = pl.LazyFrame({"a": [1, 2]}).with_columns(
        t=pl.struct(
            pl.col("a").alias("x"),
            pl.col("a")
            .map_batches(lambda _: 1 / 0, return_dtype=pl.Float64)  # type: ignore[arg-type]
            .alias("y"),
        ).struct.rename_fields(["p", "q"])
    )
    assert_frame_equal(
        lf.select(pl.col("t").struct.field("p")).collect(), pl.DataFrame({"p": [1, 2]})
    )


def test_struct_field_pruning_keeps_inputs_of_unknown_height() -> None:
    # `pl.struct` broadcasts a length-1 input to the height of the others. Dropping the
    # unselected input `y` is only valid if the kept inputs have the struct's height;
    # with an input of unknown height (`filter`) it does not.
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    def query(min_a: int) -> pl.LazyFrame:
        st = pl.struct(
            pl.col("a").filter(pl.col("a") > min_a).alias("x"), pl.col("b").alias("y")
        )
        return lf.select(st.struct.field("x"))

    # `x` has 1 row, broadcast.
    assert assert_same_as_without_pushdown(query(2)).height == 3
    # `x` has 2 rows: dropping `y` would also hide a length mismatch.
    assert_raises_with_and_without_pushdown(query(1), ShapeError)


def test_struct_field_pruning_not_for_gpu_engine(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The GPU engine ignores the field selection of scans, so it gets no struct pruning.
    # `collect(engine="gpu")` optimizes through the post-opt callback path. A stand-in
    # `cudf_polars` records the scan schema it is given and falls back to the CPU.
    seen: list[Any] = []

    def scan_schema(nt: Any) -> Any:
        while type(nt.view_current_node()).__name__ != "Scan":
            nt.set_node(nt.get_inputs()[0])
        return nt.get_schema()

    fake = types.ModuleType("cudf_polars")
    fake.execute_with_cudf = lambda nt, *_, **__: seen.append(scan_schema(nt))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cudf_polars", fake)

    lf = scan(pl.DataFrame({"s": [{"a": 1, "b": "x"}, None]}), "parquet", tmp_path)
    q = lf.select(a)
    assert_frame_equal(q.collect(engine="gpu"), pl.DataFrame({"a": [1, None]}))
    assert len(seen) == 1
    assert seen[0]["s"] == lf.collect_schema()["s"]
    # The node visitor serves cudf-polars as well.
    assert scan_schema(q._ldf.visit())["s"] == lf.collect_schema()["s"]


def test_struct_field_pruning_validity_leaf_deep_nesting(tmp_path: Path) -> None:
    # The choice of the leaf to read a struct validity from was exponential in the
    # depth of nesting. Each file of a scan pays it at reader init.
    dtype: pl.DataType = pl.Int64()
    value: Any = 1
    for i in range(26):
        dtype = pl.Struct({"p": dtype, "q": pl.Int64})
        value = {"p": value, "q": i}
    lf = scan(
        pl.DataFrame({"s": [value, None]}, schema={"s": dtype}), "parquet", tmp_path
    )

    q = lf.select(s.is_null())
    expected = q.collect(optimizations=NO_PUSHDOWN)
    t0 = time.perf_counter()
    q.collect(optimizations=NO_PUSHDOWN)
    t1 = time.perf_counter()
    out = q.collect()
    t2 = time.perf_counter()

    assert_frame_equal(out, expected)
    assert t2 - t1 < 0.25 + 10 * (t1 - t0), (t2 - t1, t1 - t0)


DF_D = pl.DataFrame(
    {
        "id": [0, 1, 1, 2],
        "s": [{"a": 1, "b": 2}, None, {"a": None, "b": 3}, {"a": 4, "b": None}],
        "l": [[{"a": 1, "b": 2}], None, [], [None]],
    }
)
l = pl.col("l")  # noqa: E741
# A struct read for validity only is a `Struct({})`, whose length the streaming
# first/last reductions lost; and a `with_fields` whose new fields nobody reads
# became an empty `with_fields`, which the streaming engine rejected.
FIRST_LAST_QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    # Validity only, then first / last.
    lambda lf: lf.select(s.first().is_null()),
    lambda lf: lf.select(s.last().count()),
    lambda lf: lf.head(1).select(s.item().is_null()),
    lambda lf: lf.select(s.drop_nulls().first().is_null()),
    lambda lf: lf.sort("id", descending=True).select(s.first().is_null()),
    lambda lf: pl.concat([lf, lf]).select(s.last().is_null()),
    lambda lf: lf.join(lf.select("id", t=s), on="id", maintain_order="left").select(
        pl.col("t").first().is_null()
    ),
    lambda lf: lf.group_by("id", maintain_order=True).agg(s.first().is_null()),
    lambda lf: lf.unique(subset=["id"], maintain_order=True).select(s.count()),
    lambda lf: lf.unique(subset=["id"], keep="last", maintain_order=True).select(
        s.null_count()
    ),
    lambda lf: lf.explode("l").select(l.last().is_null()),
    lambda lf: lf.select(l.explode().last().is_null()),
    lambda lf: lf.select(l.list.first().first().is_null()),
    lambda lf: (
        lf.explode("l").unique(subset=["id"], maintain_order=True).select(l.count())
    ),
    # `with_fields` whose new fields nobody reads, on a non-elementwise input.
    lambda lf: lf.select(
        s.filter(pl.col("id") > 0)
        .struct.with_fields(pl.field("a").alias("zz"))
        .is_null()
    ),
    lambda lf: lf.select(
        s.filter(pl.col("id") > 0).struct.with_fields(pl.field("a").alias("zz")).count()
    ),
    lambda lf: lf.select(
        s.sort_by("id").struct.with_fields(pl.field("a").alias("zz")).is_null()
    ),
]


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson", "memory"])
@pytest.mark.parametrize("query", FIRST_LAST_QUERIES)
def test_struct_field_pruning_first_last_and_with_fields(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    assert_same_as_without_pushdown(query(scan(DF_D, fmt, tmp_path)))


def test_first_last_of_struct_without_fields_streaming() -> None:
    lf = pl.LazyFrame(
        {"id": [0, 1, 1, 2], "s": pl.Series([{}, None, {}, {}], dtype=pl.Struct({}))}
    )
    for q in [
        lf.select(s.first().is_null()),
        lf.select(s.last().is_null()),
        lf.select(s.drop_nulls().first().is_null()),
        lf.select(s.drop_nulls().last().is_null()),
        lf.group_by("id", maintain_order=True).agg(
            s.first().is_null(), s.last().is_null().alias("t")
        ),
        lf.unique(subset=["id"], maintain_order=True).select(s.count()),
    ]:
        assert_frame_equal(
            q.collect(engine="streaming", optimizations=NO_PUSHDOWN),
            q.collect(engine="in-memory", optimizations=NO_PUSHDOWN),
        )
