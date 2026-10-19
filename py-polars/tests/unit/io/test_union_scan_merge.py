from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.conftest import normalize_path_separator_pl

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

SCAN_AND_WRITE_FUNCS = [
    (pl.scan_ipc, pl.DataFrame.write_ipc),
    (pl.scan_parquet, pl.DataFrame.write_parquet),
    (pl.scan_csv, pl.DataFrame.write_csv),
    (pl.scan_ndjson, pl.DataFrame.write_ndjson),
]


def n_scans(lf: pl.LazyFrame) -> int:
    return lf.explain().count("SCAN [")


def write_parts(
    tmp_path: Path, write: Any, n: int = 3
) -> tuple[list[Path], pl.DataFrame]:
    dfs = [
        pl.DataFrame({"a": [10 * i + j for j in range(4)], "b": ["x", "y", "z", "w"]})
        for i in range(n)
    ]
    paths = [tmp_path / f"{i}.data" for i in range(n)]
    for df, path in zip(dfs, paths, strict=True):
        write(df, path)
    return paths, pl.concat(dfs)


@pytest.mark.write_disk
@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
@pytest.mark.parametrize("ordered", [True, False])
def test_union_scans_merged(
    tmp_path: Path, scan: Any, write: Any, ordered: bool
) -> None:
    paths, df = write_parts(tmp_path, write)
    lfs = [scan(p).filter(pl.col("a") % 2 == 0).select("a") for p in paths]
    lf = pl.concat(lfs) if ordered else pl.union(lfs)

    assert n_scans(lf) == 1
    assert "UNION" not in lf.explain()

    out = lf.collect()
    if not ordered:
        out = out.sort("a")
    assert_frame_equal(out, df.filter(pl.col("a") % 2 == 0).select("a"))
    assert lf.select(pl.len()).collect().item() == 6


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("read", "write"),
    [
        ("read_csv", pl.DataFrame.write_csv),
        ("read_parquet", pl.DataFrame.write_parquet),
    ],
)
def test_union_scans_merged_sql(tmp_path: Path, read: str, write: Any) -> None:
    paths, df = write_parts(tmp_path, write)
    query = " UNION ALL ".join(
        f"SELECT a FROM {read}('{p}') WHERE b <> 'x'" for p in paths
    )
    lf = pl.SQLContext().execute(query)

    assert n_scans(lf) == 1
    assert_frame_equal(
        lf.collect().sort("a"), df.filter(pl.col("b") != "x").select("a")
    )


@pytest.mark.write_disk
def test_union_scans_merged_ordered_runs(tmp_path: Path) -> None:
    paths, df = write_parts(tmp_path, pl.DataFrame.write_csv, n=4)
    mid = pl.DataFrame({"a": [-1], "b": ["m"]})
    lfs = [pl.scan_csv(p) for p in paths]

    lf = pl.concat([lfs[0], lfs[1], mid.lazy(), lfs[2], lfs[3]])
    assert n_scans(lf) == 2
    assert_frame_equal(lf.collect(), pl.concat([df[:8], mid, df[8:]]))

    lf = pl.union([lfs[0], mid.lazy(), lfs[1], lfs[2], lfs[3]])
    assert n_scans(lf) == 1
    assert_frame_equal(lf.collect().sort("a"), pl.concat([mid, df]))

    # Nested unions are flattened before merging.
    lf = pl.concat([pl.concat([lfs[0], lfs[1]]), pl.concat([lfs[2], lfs[3]])])
    assert n_scans(lf) == 1
    assert_frame_equal(lf.collect(), df)


@pytest.mark.write_disk
def test_union_scans_merged_row_wise(tmp_path: Path) -> None:
    paths, df = write_parts(tmp_path, pl.DataFrame.write_csv)

    def q(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.with_columns(c=pl.col("a") * 2)
            .filter(pl.col("c") > 5)
            .select("a", d=pl.col("c") + 1, e=pl.lit(1))
        )

    lf = pl.concat([q(pl.scan_csv(p)) for p in paths])
    assert n_scans(lf) == 1
    assert_frame_equal(lf.collect(), q(df.lazy()).collect())

    # A sorted flag only holds for each file.
    lf = pl.concat(
        [pl.scan_csv(p).select(pl.col("a").set_sorted()) for p in paths[::-1]]
    )
    assert n_scans(lf) == 3
    assert lf.select(pl.col("a").min()).collect(engine="in-memory").item() == 0

    # Different expressions on top of the scans.
    lf = pl.concat(
        [
            pl.scan_csv(paths[0]).select(x=pl.col("a")),
            pl.scan_csv(paths[1]).select(x=pl.col("a") + 1),
        ]
    )
    assert n_scans(lf) == 2


# Not row-wise: these give a different result per file.
@pytest.mark.write_disk
@pytest.mark.parametrize(
    "f",
    [
        lambda lf: lf.select(pl.col("a").sum()),
        lambda lf: lf.select(pl.col("a").head(1)),
        lambda lf: lf.with_columns(pl.col("a").cum_sum()),
        lambda lf: lf.filter(pl.col("a") > pl.col("a").mean()),
        lambda lf: lf.select(pl.lit([7, 8]).explode()),
        lambda lf: lf.select(pl.lit(7).filter(pl.lit(True))),
        lambda lf: lf.select(pl.lit(1).is_in(pl.Series([[1], [2], [3], [4]]))),
        lambda lf: lf.select(
            pl.col("a").filter(pl.col("a") % 2 == 0)
            + pl.col("a").filter(pl.col("a") % 2 == 1)
        ),
    ],
    ids=[
        "sum",
        "head",
        "cum_sum",
        "filter_by_mean",
        "explode_literal",
        "filter_literal",
        "is_in_per_row_sets",
        "filtered_operands",
    ],
)
def test_union_scans_not_row_wise_not_merged(
    tmp_path: Path, f: Callable[[pl.LazyFrame], pl.LazyFrame]
) -> None:
    paths, _ = write_parts(tmp_path, pl.DataFrame.write_csv)
    lf = pl.concat([f(pl.scan_csv(p)) for p in paths])
    assert n_scans(lf) == 3
    assert_frame_equal(
        lf.collect(), pl.concat([f(pl.scan_csv(p)).collect() for p in paths])
    )


@pytest.mark.write_disk
def test_union_scans_merged_after_simplify(tmp_path: Path) -> None:
    paths, df = write_parts(tmp_path, pl.DataFrame.write_csv, n=2)
    lf = pl.concat(
        [
            pl.scan_csv(paths[0]).filter(pl.col("a") > 1),
            pl.scan_csv(paths[1]).filter((pl.col("a") > 1) & pl.lit(True)),
        ]
    )
    assert n_scans(lf) == 1
    assert_frame_equal(lf.collect(), df.filter(pl.col("a") > 1))


@pytest.mark.write_disk
def test_union_scans_not_merged(tmp_path: Path) -> None:
    paths, df = write_parts(tmp_path, pl.DataFrame.write_csv, n=2)
    a, b = paths

    cases = [
        (pl.scan_csv(a).filter(pl.col("a") > 1), pl.scan_csv(b)),
        (pl.scan_csv(a), pl.scan_csv(b, quote_char="'")),
        (
            pl.scan_csv(a),
            pl.scan_csv(b, schema_overrides={"a": pl.Int32}).cast({"a": pl.Int64}),
        ),
        (pl.scan_csv(a, row_index_name="i"), pl.scan_csv(b, row_index_name="i")),
        (pl.scan_csv(a, n_rows=2), pl.scan_csv(b, n_rows=2)),
    ]
    for lfs in cases:
        assert n_scans(pl.concat(lfs)) == 2

    # The row index restarts for every file.
    out = pl.concat(
        [pl.scan_csv(a, row_index_name="i"), pl.scan_csv(b, row_index_name="i")]
    ).collect()
    assert out["i"].to_list() == [0, 1, 2, 3, 0, 1, 2, 3]

    out = pl.concat([pl.scan_csv(a, n_rows=2), pl.scan_csv(b, n_rows=2)]).collect()
    assert_frame_equal(out, pl.concat([df[:2], df[4:6]]))


@pytest.mark.write_disk
def test_union_scans_merged_hive_not_merged(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2], "p": [1, 2]})
    df.write_parquet(tmp_path, partition_by="p")
    lfs = [
        pl.scan_parquet(tmp_path / f"p={i}" / "*.parquet", hive_partitioning=True)
        for i in (1, 2)
    ]
    lf = pl.concat(lfs)
    assert n_scans(lf) == 2
    assert_frame_equal(lf.collect(), df)


@pytest.mark.write_disk
@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_union_scans_merged_include_file_paths(
    tmp_path: Path, scan: Any, write: Any
) -> None:
    paths, df = write_parts(tmp_path, write, n=2)
    lf = pl.concat([scan(p, include_file_paths="f") for p in paths])
    assert n_scans(lf) == 1
    assert_frame_equal(
        lf.collect(),
        df.with_columns(
            f=pl.Series([str(paths[0])] * 4 + [str(paths[1])] * 4)
        ).with_columns(normalize_path_separator_pl(pl.col("f"))),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("ordered", [True, False])
def test_union_scans_merged_slice(tmp_path: Path, ordered: bool) -> None:
    paths, df = write_parts(tmp_path, pl.DataFrame.write_parquet)
    lfs = [pl.scan_parquet(p).filter(pl.col("a") % 2 == 1) for p in paths]
    lf = pl.concat(lfs) if ordered else pl.union(lfs)

    out = lf.head(4).collect()
    assert out.height == 4
    if ordered:
        assert_frame_equal(out, df.filter(pl.col("a") % 2 == 1).head(4))
    assert lf.slice(1, 3).collect().height == 3


def test_union_scans_iceberg_default_values_not_merged() -> None:
    # Default values are looked up by source index.
    field_a = pa.field("a", pa.int64(), metadata={b"PARQUET:field_id": b"1"})
    field_p = pa.field("p", pa.int64(), metadata={b"PARQUET:field_id": b"2"})
    sources = []
    for values in [[1, 2], [3, 4]]:
        buf = io.BytesIO()
        pq.write_table(pa.table({"a": values}, schema=pa.schema([field_a])), buf)
        sources.append(buf.getvalue())

    initial: dict[int, pl.Series | str] = {2: pl.Series("p", [7])}
    lfs = [
        pl.scan_parquet(
            s,
            schema={"a": pl.Int64, "p": pl.Int64},
            missing_columns="insert",
            _column_mapping=("iceberg-column-mapping", pa.schema([field_a, field_p])),
            _default_values=("iceberg", (initial, {})),
        )
        for s in sources
    ]
    lf = pl.concat(lfs)
    assert n_scans(lf) == 2
    assert_frame_equal(lf.collect(), pl.DataFrame({"a": [1, 2, 3, 4], "p": [7] * 4}))
