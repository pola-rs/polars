"""Struct-field projection pushdown: read and compute only the fields used."""

from __future__ import annotations

import sys
import time
import types
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import ComputeError, SchemaError, ShapeError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

NO_PUSHDOWN = pl.QueryOptFlags(projection_pushdown=False)


def assert_same_as_without_pushdown(
    lf: pl.LazyFrame, *, check_row_order: bool = True
) -> pl.DataFrame:
    expected = lf.collect(optimizations=NO_PUSHDOWN)
    for engine in ("in-memory", "streaming"):
        assert_frame_equal(
            lf.collect(engine=engine),  # type: ignore[call-overload]
            expected,
            check_row_order=check_row_order,
        )
    return expected


@pytest.fixture
def struct_df() -> pl.DataFrame:
    return pl.DataFrame(
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


@pytest.fixture
def struct_path(tmp_path: Path, struct_df: pl.DataFrame) -> Path:
    path = tmp_path / "s.parquet"
    struct_df.write_parquet(path)
    return path


s = pl.col("s")

QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(s.struct.field("a")),
    lambda lf: lf.select(s.struct.field("c").struct.field("x")),
    lambda lf: lf.select(s.struct.field("c").struct.field("y")),
    lambda lf: lf.select(s.struct.field("a", "b")),
    lambda lf: lf.select(s.is_null()),
    lambda lf: lf.select(
        s.is_not_null().sum().alias("n"),
        s.count().alias("count"),
        s.null_count().alias("null_count"),
    ),
    lambda lf: lf.select(s.struct.field("c").is_null()),
    lambda lf: lf.select(s, s.struct.field("a").alias("x")),
    lambda lf: lf.select(s.struct.field("a").alias("x"), s),
    lambda lf: lf.select(s.struct.field("a").alias("x"), s.struct.field("c")),
    lambda lf: lf.with_columns(t=s.struct.field("b")).select("t", "id"),
    lambda lf: lf.select(s.struct.field("a") + pl.col("id")),
    lambda lf: lf.filter(s.struct.field("a") > 1).select(s.struct.field("b")),
    lambda lf: lf.filter(s.struct.field("a").cast(pl.Float64) > 1.5).select(
        s.struct.field("b")
    ),
    lambda lf: lf.filter(s.struct.field("a") > 0).select(s),
    lambda lf: lf.select(
        s.struct.rename_fields(["p", "q", "r"]).struct.field("q"),
    ),
    lambda lf: lf.select(
        s.struct.rename_fields(["p", "q", "r"]).struct.field("r").struct.field("x"),
    ),
    lambda lf: lf.select(
        s.struct.with_fields(d=s.struct.field("a") * 2).struct.field("d", "b"),
    ),
    lambda lf: lf.select(
        s.struct.with_fields(pl.field("a") * 2).struct.field("a"),
    ),
    lambda lf: lf.select(s.struct.with_fields(pl.field("a") * 2)),
    lambda lf: lf.select(s.name.prefix_fields("p_").struct.field("p_b")),
    lambda lf: lf.select(s.name.suffix_fields("_s").struct.field("c_s")),
    lambda lf: lf.select(s.struct.drop(["a"]).struct.field("b")),
    lambda lf: lf.select(s.struct.drop(["a"])),
    lambda lf: lf.select(s.struct.json_encode()),
    lambda lf: lf.select(s.struct.unnest()),
    lambda lf: lf.unnest("s").select("b"),
    lambda lf: lf.unnest("s", separator="::").select("s::c"),
    lambda lf: lf.group_by(s.struct.field("b"), maintain_order=True).agg(
        s.struct.field("a").sum()
    ),
    lambda lf: lf.group_by("id", maintain_order=True).agg(s.first().struct.field("a")),
    lambda lf: lf.sort(s.struct.field("a"), nulls_last=True).select(
        s.struct.field("b")
    ),
    lambda lf: lf.head(2).select(s.struct.field("a")),
    lambda lf: lf.select(s.first().struct.field("a"), s.last().struct.field("b")),
    lambda lf: lf.select(s.filter(pl.col("id") > 1).struct.field("a")),
    lambda lf: lf.select(
        pl.when(pl.col("id") > 2).then(s).otherwise(s).struct.field("a")
    ),
    lambda lf: lf.with_columns(t=s).select(pl.col("t").struct.field("a")),
    lambda lf: lf.rename({"s": "t"}).select(pl.col("t").struct.field("a")),
    lambda lf: lf.join(lf, on="id", suffix="_r", maintain_order="left").select(
        pl.col("s_r").struct.field("a"), s.struct.field("b")
    ),
    lambda lf: lf.cache().select(s.struct.field("a")),
]


@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning_parquet(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], struct_path: Path
) -> None:
    assert_same_as_without_pushdown(query(pl.scan_parquet(struct_path)))


@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning_in_memory(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], struct_df: pl.DataFrame
) -> None:
    assert_same_as_without_pushdown(query(struct_df.lazy()))


def test_struct_field_pruning_explain(struct_path: Path) -> None:
    lf = pl.scan_parquet(struct_path)

    assert "PROJECT FIELDS: s{a}" in lf.select(s.struct.field("a")).explain()
    assert (
        "PROJECT FIELDS: s{c{x}}"
        in lf.select(s.struct.field("c").struct.field("x")).explain()
    )
    assert "PROJECT FIELDS: s{}" in lf.select(s.is_null()).explain()
    # A field read by the predicate is read as well.
    plan = lf.filter(s.struct.field("b") == "x").select(s.struct.field("a")).explain()
    assert "PROJECT FIELDS: s{a, b}" in plan or "PROJECT FIELDS: s{b, a}" in plan

    # Used whole, in either order.
    for q in [
        lf.select(s, s.struct.field("a").alias("x")),
        lf.select(s.struct.field("a").alias("x"), s),
        lf.select(s.struct.field("a").alias("x"), s.is_null().alias("n"), s),
    ]:
        assert "PROJECT FIELDS" not in q.explain()

    # Nothing pruned without projection pushdown.
    assert "PROJECT FIELDS" not in lf.select(s.struct.field("a")).explain(
        optimizations=NO_PUSHDOWN
    )


def _corrupt_leaf(path: Path, leaf: str) -> None:
    """Overwrite the column chunk of `leaf` (a dotted path) with garbage."""
    md = pq.ParquetFile(path).metadata
    data = bytearray(path.read_bytes())
    for rg in range(md.num_row_groups):
        chunks = [md.row_group(rg).column(i) for i in range(md.num_columns)]
        starts = sorted(
            min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset)
            for c in chunks
        )
        for col in chunks:
            if col.path_in_schema != leaf:
                continue
            start = min(
                col.data_page_offset, col.dictionary_page_offset or col.data_page_offset
            )
            end = min(
                [start + col.total_compressed_size] + [s for s in starts if s > start]
            )
            data[start:end] = b"\xff" * (end - start)
    path.write_bytes(bytes(data))


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
    _corrupt_leaf(path, "s.b")
    _corrupt_leaf(path, "s.c.y")

    lf = pl.scan_parquet(path)

    # Reading the corrupted leaves fails, so these only pass if they are not read.
    with pytest.raises(ComputeError):
        lf.select(s.struct.field("a")).collect(optimizations=NO_PUSHDOWN)

    for engine in ("in-memory", "streaming"):
        assert_frame_equal(
            lf.select(
                s.struct.field("a"), s.struct.field("c").struct.field("x")
            ).collect(engine=engine),  # type: ignore[call-overload]
            df.select(s.struct.field("a"), s.struct.field("c").struct.field("x")),
        )
        assert_frame_equal(
            lf.select(s.is_null().sum()).collect(engine=engine),  # type: ignore[call-overload]
            pl.DataFrame({"s": [0]}, schema={"s": pl.get_index_type()}),
        )


def test_struct_field_pruning_dotted_names(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "s": [{"a.b": 1, "a": {"b": 2, "c": 3}}, None],
            "s.a": [4, 5],
        }
    )
    path = tmp_path / "dots.parquet"
    df.write_parquet(path)
    lf = pl.scan_parquet(path)

    for q in [
        lf.select(s.struct.field("a.b")),
        lf.select(s.struct.field("a").struct.field("b")),
        lf.select(s.struct.field("a.b"), pl.col("s.a")),
        lf.select(s.struct.field("a").struct.field("c"), s.struct.field("a.b")),
    ]:
        assert_same_as_without_pushdown(q)


def test_struct_field_pruning_schema_evolution(tmp_path: Path) -> None:
    pl.DataFrame(
        {"s": [{"a": 1, "b": "x"}, None, {"a": 3, "b": None}]},
    ).write_parquet(tmp_path / "1.parquet")
    # Lacks `b`, the only field read below: the struct validity must still come
    # from the file.
    pl.DataFrame(
        {"s": [None, {"a": 5}, {"a": None}]},
    ).write_parquet(tmp_path / "2.parquet")

    lf = pl.scan_parquet(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        schema={"s": pl.Struct({"a": pl.Int64, "b": pl.String})},
        cast_options=pl.ScanCastOptions(missing_struct_fields="insert"),
    )

    out = assert_same_as_without_pushdown(
        lf.select(s.struct.field("b"), s.is_null().alias("is_null"))
    )
    assert out["is_null"].to_list() == [False, True, False, True, False, False]

    assert_same_as_without_pushdown(lf.select(s.struct.field("a")))
    assert_same_as_without_pushdown(lf.select(s.is_null()))
    assert_same_as_without_pushdown(lf.filter(s.struct.field("a") > 1).select(s))


def test_struct_field_pruning_extra_fields(tmp_path: Path) -> None:
    pl.DataFrame({"s": [{"a": 1, "b": 2}]}).write_parquet(tmp_path / "1.parquet")
    pl.DataFrame({"s": [{"z": 0, "b": 4, "a": 3}]}).write_parquet(
        tmp_path / "2.parquet"
    )

    lf = pl.scan_parquet(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        cast_options=pl.ScanCastOptions(extra_struct_fields="ignore"),
    )

    assert_same_as_without_pushdown(lf.select(s.struct.field("b")))
    assert_same_as_without_pushdown(lf.select(s))


def test_struct_field_pruning_multiple_files_and_hive(tmp_path: Path) -> None:
    for part in [1, 2]:
        (tmp_path / f"p={part}").mkdir()
        pl.DataFrame(
            {"s": [{"a": part, "b": str(part)}, None]},
        ).write_parquet(tmp_path / f"p={part}" / "data.parquet")

    lf = pl.scan_parquet(tmp_path, hive_partitioning=True)
    assert_same_as_without_pushdown(lf.select(s.struct.field("a"), pl.col("p")))
    assert_same_as_without_pushdown(
        lf.filter(pl.col("p") == 2).select(s.struct.field("b"))
    )


def test_struct_field_pruning_concat(struct_path: Path) -> None:
    lf = pl.scan_parquet(struct_path)

    # The branches read different fields, but must have equal dtypes to be
    # concatenated.
    q = pl.concat(
        [
            lf.filter(s.struct.field("b") == "x"),
            lf,
            lf.select(pl.col("id"), s.struct.with_fields(pl.lit(1).alias("e"))),
        ],
        how="diagonal_relaxed",
    ).select(s.struct.field("a"))
    assert_same_as_without_pushdown(q)

    q = pl.concat(
        [lf.filter(s.struct.field("c").struct.field("x") > 1.0), lf.head(1)]
    ).select(s.struct.field("a"), s.struct.field("c").struct.field("y"))
    assert_same_as_without_pushdown(q)


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
    assert_frame_equal(q.collect(), expected)
    assert_frame_equal(q.collect(engine="streaming"), expected)

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
        lf.select(pl.col("t").struct.field("p")).collect(),
        pl.DataFrame({"p": [1, 2]}),
    )


def test_struct_field_pruning_keeps_inputs_of_unknown_height() -> None:
    # `pl.struct` broadcasts a length-1 input to the height of the others. Dropping the
    # unselected input `y` is only valid if the kept inputs have the struct's height;
    # with an input of unknown height (`filter`) it does not.
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    st = pl.struct(
        pl.col("a").filter(pl.col("a") > 2).alias("x"),  # 1 row, broadcast
        pl.col("b").alias("y"),  # 3 rows
    )
    expected = assert_same_as_without_pushdown(lf.select(st.struct.field("x")))
    assert expected.height == 3

    # Dropping `y` would also hide a length mismatch.
    st = pl.struct(
        pl.col("a").filter(pl.col("a") > 1).alias("x"),  # 2 rows
        pl.col("b").alias("y"),  # 3 rows
    )
    q = lf.select(st.struct.field("x"))
    with pytest.raises(ShapeError):
        q.collect(optimizations=NO_PUSHDOWN)
    for engine in ("in-memory", "streaming"):
        with pytest.raises(ShapeError):
            q.collect(engine=engine)  # type: ignore[call-overload]


def test_struct_field_pruning_not_for_gpu_engine(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The GPU engine ignores the field selection of scans, so it gets no struct pruning.
    # `collect(engine="gpu")` optimizes through the post-opt callback path. A stand-in
    # `cudf_polars` records the scan schema it is given and falls back to the CPU.
    seen: list[Any] = []

    def execute_with_cudf(nt: Any, *args: Any, **kwargs: Any) -> None:
        while type(nt.view_current_node()).__name__ != "Scan":
            nt.set_node(nt.get_inputs()[0])
        seen.append(nt.get_schema())

    fake = types.ModuleType("cudf_polars")
    fake.execute_with_cudf = execute_with_cudf  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cudf_polars", fake)

    pl.DataFrame({"s": [{"a": 1, "b": "x"}, None]}).write_parquet(
        tmp_path / "s.parquet"
    )
    lf = pl.scan_parquet(tmp_path / "s.parquet")
    q = lf.select(s.struct.field("a"))
    assert_frame_equal(q.collect(engine="gpu"), pl.DataFrame({"a": [1, None]}))

    assert len(seen) == 1
    assert seen[0]["s"] == lf.collect_schema()["s"]

    # The node visitor serves cudf-polars as well.
    nt = q._ldf.visit()
    while type(nt.view_current_node()).__name__ != "Scan":
        nt.set_node(nt.get_inputs()[0])
    assert nt.get_schema()["s"] == lf.collect_schema()["s"]


def test_struct_field_pruning_validity_leaf_deep_nesting(tmp_path: Path) -> None:
    # The choice of the leaf to read a struct validity from was exponential in the
    # depth of nesting. Each file of a scan pays it at reader init.
    depth = 26
    dtype: pl.DataType = pl.Int64()
    value: Any = 1
    for i in range(depth):
        dtype = pl.Struct({"p": dtype, "q": pl.Int64})
        value = {"p": value, "q": i}
    path = tmp_path / "deep.parquet"
    pl.DataFrame({"s": [value, None]}, schema={"s": dtype}).write_parquet(path)

    q = pl.scan_parquet(path).select(s.is_null())
    expected = q.collect(optimizations=NO_PUSHDOWN)

    t0 = time.perf_counter()
    q.collect(optimizations=NO_PUSHDOWN)
    t1 = time.perf_counter()
    out = q.collect()
    t2 = time.perf_counter()

    assert_frame_equal(out, expected)
    assert t2 - t1 < 0.25 + 10 * (t1 - t0), (t2 - t1, t1 - t0)


@pytest.mark.parametrize("policy", ["extra", "missing"])
def test_struct_field_pruning_cast_policy_errors(tmp_path: Path, policy: str) -> None:
    # Under the default `ScanCastOptions` (raise on extra / missing struct fields), a
    # scan whose files disagree on the fields of `s` raises with or without pruning.
    pl.DataFrame({"s": [{"a": 1, "b": 2}]}).write_parquet(tmp_path / "1.parquet")
    other = {"a": 3, "b": 4, "z": 0} if policy == "extra" else {"a": 3}
    pl.DataFrame({"s": [other]}).write_parquet(tmp_path / "2.parquet")
    paths = [tmp_path / "1.parquet", tmp_path / "2.parquet"]
    lf = pl.scan_parquet(paths)

    for q in [
        lf.select(s),
        lf.select(s.struct.field("a")),
        lf.select(s.is_null()),
        # With a predicate, the projection is resolved before the reader starts.
        lf.filter(s.struct.field("a") > 0).select(s.struct.field("a")),
    ]:
        with pytest.raises(SchemaError):
            q.collect(optimizations=NO_PUSHDOWN)
        for engine in ("in-memory", "streaming"):
            with pytest.raises(SchemaError):
                q.collect(engine=engine)  # type: ignore[call-overload]

    # Allowed by the policy, the fields are still pruned.
    cast_options = (
        pl.ScanCastOptions(extra_struct_fields="ignore")
        if policy == "extra"
        else pl.ScanCastOptions(missing_struct_fields="insert")
    )
    lf = pl.scan_parquet(paths, cast_options=cast_options)
    q = lf.select(s.struct.field("a"))
    assert "PROJECT FIELDS: s{a}" in q.explain()
    assert_frame_equal(assert_same_as_without_pushdown(q), pl.DataFrame({"a": [1, 3]}))
    assert_same_as_without_pushdown(
        lf.filter(s.struct.field("a") > 0).select(s.struct.field("a"))
    )
