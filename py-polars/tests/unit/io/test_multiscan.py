from __future__ import annotations

import io
import re
import sys
from functools import partial
from typing import IO, TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st

import polars as pl
from polars.meta.index_type import get_index_type
from polars.testing import assert_frame_equal
from tests.unit.io.conftest import normalize_path_separator_pl

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

SCAN_AND_WRITE_FUNCS = [
    (pl.scan_ipc, pl.DataFrame.write_ipc),
    (pl.scan_parquet, pl.DataFrame.write_parquet),
    (pl.scan_csv, pl.DataFrame.write_csv),
    (pl.scan_ndjson, pl.DataFrame.write_ndjson),
]


@pytest.mark.write_disk
@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_include_file_paths(tmp_path: Path, scan: Any, write: Any) -> None:
    a_path = tmp_path / "a"
    b_path = tmp_path / "b"

    write(pl.DataFrame({"a": [5, 10]}), a_path)
    write(pl.DataFrame({"a": [1996]}), b_path)

    out = scan([a_path, b_path], include_file_paths="f")

    assert_frame_equal(
        out.collect(),
        pl.DataFrame(
            {
                "a": [5, 10, 1996],
                "f": [str(a_path), str(a_path), str(b_path)],
            }
        ).with_columns(normalize_path_separator_pl(pl.col("f"))),
    )


@pytest.mark.parametrize(
    ("scan", "write", "ext", "supports_missing_columns", "supports_hive_partitioning"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc", False, True),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet", True, True),
        (pl.scan_csv, pl.DataFrame.write_csv, "csv", False, False),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson, "jsonl", False, False),
    ],
)
@pytest.mark.parametrize("missing_column", [False, True])
@pytest.mark.parametrize("row_index", [False, True])
@pytest.mark.parametrize("include_file_paths", [False, True])
@pytest.mark.parametrize("hive", [False, True])
@pytest.mark.parametrize("col", [False, True])
@pytest.mark.write_disk
def test_multiscan_projection(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
    ext: str,
    supports_missing_columns: bool,
    supports_hive_partitioning: bool,
    missing_column: bool,
    row_index: bool,
    include_file_paths: bool,
    hive: bool,
    col: bool,
) -> None:
    a = pl.DataFrame({"col": [5, 10, 1996]})
    b = pl.DataFrame({"col": [13, 37]})

    if missing_column and supports_missing_columns:
        a = a.with_columns(missing=pl.Series([420, 2000, 9]))

    a_path: Path
    b_path: Path
    multiscan_path: Path

    if hive and supports_hive_partitioning:
        (tmp_path / "hive_col=0").mkdir()
        a_path = tmp_path / "hive_col=0" / f"a.{ext}"
        (tmp_path / "hive_col=1").mkdir()
        b_path = tmp_path / "hive_col=1" / f"b.{ext}"

        multiscan_path = tmp_path

    else:
        a_path = tmp_path / f"a.{ext}"
        b_path = tmp_path / f"b.{ext}"

        multiscan_path = tmp_path / f"*.{ext}"

    write(a, a_path)
    write(b, b_path)

    base_projection = []
    if missing_column and supports_missing_columns:
        base_projection += ["missing"]
    if row_index:
        base_projection += ["row_index"]
    if include_file_paths:
        base_projection += ["file_path"]
    if hive and supports_hive_partitioning:
        base_projection += ["hive_col"]
    if col:
        base_projection += ["col"]

    ifp = "file_path" if include_file_paths else None
    ri = "row_index" if row_index else None

    args = {
        "missing_columns": "insert" if missing_column else "raise",
        "include_file_paths": ifp,
        "row_index_name": ri,
        "hive_partitioning": hive,
    }

    if not supports_missing_columns:
        del args["missing_columns"]
    if not supports_hive_partitioning:
        del args["hive_partitioning"]

    for projection in [
        base_projection,
        base_projection[::-1],
    ]:
        assert_frame_equal(
            scan(multiscan_path, **args).collect(engine="streaming").select(projection),
            scan(multiscan_path, **args).select(projection).collect(engine="streaming"),
        )

    for remove in range(len(base_projection)):
        new_projection = base_projection.copy()
        new_projection.pop(remove)

        for projection in [
            new_projection,
            new_projection[::-1],
        ]:
            assert_frame_equal(
                scan(multiscan_path, **args)
                .collect(engine="streaming")
                .select(projection),
                scan(multiscan_path, **args)
                .select(projection)
                .collect(engine="streaming"),
            )


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
    ],
)
@pytest.mark.write_disk
def test_multiscan_hive_predicate(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
    ext: str,
) -> None:
    a = pl.DataFrame({"col": [5, 10, 1996]})
    b = pl.DataFrame({"col": [13, 37]})
    c = pl.DataFrame({"col": [3, 5, 2024]})

    (tmp_path / "hive_col=0").mkdir()
    a_path = tmp_path / "hive_col=0" / f"0.{ext}"
    (tmp_path / "hive_col=1").mkdir()
    b_path = tmp_path / "hive_col=1" / f"0.{ext}"
    (tmp_path / "hive_col=2").mkdir()
    c_path = tmp_path / "hive_col=2" / f"0.{ext}"

    multiscan_path = tmp_path

    write(a, a_path)
    write(b, b_path)
    write(c, c_path)

    full = scan(multiscan_path).collect(engine="streaming")
    full_ri = full.with_row_index("ri", 42)

    last_pred = None
    try:
        for pred in [
            pl.col.hive_col == 0,
            pl.col.hive_col == 1,
            pl.col.hive_col == 2,
            pl.col.hive_col < 2,
            pl.col.hive_col > 0,
            pl.col.hive_col != 1,
            pl.col.hive_col != 3,
            pl.col.col == 13,
            pl.col.col != 13,
            (pl.col.col != 13) & (pl.col.hive_col == 1),
            (pl.col.col != 13) & (pl.col.hive_col != 1),
        ]:
            last_pred = pred
            assert_frame_equal(
                full.filter(pred),
                scan(multiscan_path).filter(pred).collect(engine="streaming"),
            )

            assert_frame_equal(
                full_ri.filter(pred),
                scan(multiscan_path)
                .with_row_index("ri", 42)
                .filter(pred)
                .collect(engine="streaming"),
            )
    except Exception as _:
        print(last_pred)
        raise


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
@pytest.mark.write_disk
def test_multiscan_row_index(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
) -> None:
    a = pl.DataFrame({"col": [5, 10, 1996]})
    b = pl.DataFrame({"col": [42]})
    c = pl.DataFrame({"col": [13, 37]})

    write(a, tmp_path / "a")
    write(b, tmp_path / "b")
    write(c, tmp_path / "c")

    col = pl.concat([a, b, c]).to_series()
    g = tmp_path / "*"

    assert_frame_equal(
        scan(g, row_index_name="ri").collect(),
        pl.DataFrame(
            [
                pl.Series("ri", range(6), get_index_type()),
                col,
            ]
        ),
    )

    start = 42
    assert_frame_equal(
        scan(g, row_index_name="ri", row_index_offset=start).collect(),
        pl.DataFrame(
            [
                pl.Series("ri", range(start, start + 6), get_index_type()),
                col,
            ]
        ),
    )

    start = 42
    assert_frame_equal(
        scan(g, row_index_name="ri", row_index_offset=start).slice(3, 3).collect(),
        pl.DataFrame(
            [
                pl.Series("ri", range(start + 3, start + 6), get_index_type()),
                col.slice(3, 3),
            ]
        ),
    )

    start = 42
    assert_frame_equal(
        scan(g, row_index_name="ri", row_index_offset=start)
        .filter(pl.col("col") < 15)
        .collect(),
        pl.DataFrame(
            [
                pl.Series("ri", [start + 0, start + 1, start + 4], get_index_type()),
                pl.Series("col", [5, 10, 13]),
            ]
        ),
    )

    with pytest.raises(
        pl.exceptions.DuplicateError, match="duplicate column name index"
    ):
        scan(g).with_row_index().with_row_index().collect()

    assert_frame_equal(
        scan(g)
        .with_row_index()
        .with_row_index("index_1", offset=1)
        .with_row_index("index_2", offset=2)
        .collect(),
        pl.DataFrame(
            [
                pl.Series("index_2", [2, 3, 4, 5, 6, 7], get_index_type()),
                pl.Series("index_1", [1, 2, 3, 4, 5, 6], get_index_type()),
                pl.Series("index", [0, 1, 2, 3, 4, 5], get_index_type()),
                col,
            ]
        ),
    )


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
        pytest.param(
            pl.scan_csv,
            pl.DataFrame.write_csv,
            "csv",
            marks=pytest.mark.xfail(
                reason="See https://github.com/pola-rs/polars/issues/21211"
            ),
        ),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson, "jsonl"),
    ],
)
@pytest.mark.write_disk
def test_schema_mismatch_type_mismatch(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
    ext: str,
) -> None:
    a = pl.DataFrame({"xyz_col": [5, 10, 1996]})
    b = pl.DataFrame({"xyz_col": ["a", "b", "c"]})

    a_path = tmp_path / f"a.{ext}"
    b_path = tmp_path / f"b.{ext}"

    multiscan_path = tmp_path / f"*.{ext}"

    write(a, a_path)
    write(b, b_path)

    q = scan(multiscan_path)

    # NDJSON will just parse according to `projected_schema`
    cx = (
        pytest.raises(
            pl.exceptions.ComputeError,
            match=re.escape("cannot parse 'a' (string) as Int64"),
        )
        if scan is pl.scan_ndjson
        else pytest.raises(
            pl.exceptions.SchemaError,  # type: ignore[arg-type]
            match=(
                "data type mismatch for column xyz_col: "
                "incoming: String != target: Int64"
            ),
        )
    )

    with cx:
        q.collect(engine="streaming")


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        # (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"), # TODO: _
        # (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"), # TODO: _
        pytest.param(
            pl.scan_csv,
            pl.DataFrame.write_csv,
            "csv",
            marks=pytest.mark.xfail(
                reason="See https://github.com/pola-rs/polars/issues/21211"
            ),
        ),
        # (pl.scan_ndjson, pl.DataFrame.write_ndjson, "jsonl"), # TODO: _
    ],
)
@pytest.mark.write_disk
def test_schema_mismatch_order_mismatch(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
    ext: str,
) -> None:
    a = pl.DataFrame({"x": [5, 10, 1996], "y": ["a", "b", "c"]})
    b = pl.DataFrame({"y": ["x", "y"], "x": [1, 2]})

    a_path = tmp_path / f"a.{ext}"
    b_path = tmp_path / f"b.{ext}"

    multiscan_path = tmp_path / f"*.{ext}"

    write(a, a_path)
    write(b, b_path)

    q = scan(multiscan_path)

    with pytest.raises(pl.exceptions.SchemaError):
        q.collect(engine="streaming")


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_multiscan_head(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
) -> None:
    a = io.BytesIO()
    b = io.BytesIO()
    for f in [a, b]:
        write(pl.Series("c1", range(10)).to_frame(), f)
        f.seek(0)

    assert_frame_equal(
        scan([a, b]).head(5).collect(engine="streaming"),
        pl.Series("c1", range(5)).to_frame(),
    )


@pytest.mark.parametrize(
    ("scan", "write"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson),
        (
            pl.scan_csv,
            pl.DataFrame.write_csv,
        ),
    ],
)
def test_multiscan_tail(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
) -> None:
    a = io.BytesIO()
    b = io.BytesIO()
    for f in [a, b]:
        write(pl.Series("c1", range(10)).to_frame(), f)
        f.seek(0)

    assert_frame_equal(
        scan([a, b]).tail(5).collect(engine="streaming"),
        pl.Series("c1", range(5, 10)).to_frame(),
    )


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_multiscan_slice_middle(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
) -> None:
    fs = [io.BytesIO() for _ in range(13)]
    for f in fs:
        write(pl.Series("c1", range(7)).to_frame(), f)
        f.seek(0)

    offset = 5 * 7 - 5
    expected = (
        list(range(2, 7))  # fs[4]
        + list(range(7))  # fs[5]
        + list(range(5))  # fs[6]
    )
    expected_series = [pl.Series("c1", expected)]
    ri_expected_series = [
        pl.Series("ri", range(offset, offset + 17), get_index_type())
    ] + expected_series

    assert_frame_equal(
        scan(fs).slice(offset, 17).collect(engine="streaming"),
        pl.DataFrame(expected_series),
    )
    assert_frame_equal(
        scan(fs, row_index_name="ri").slice(offset, 17).collect(engine="streaming"),
        pl.DataFrame(ri_expected_series),
    )

    # Negative slices
    offset = -(13 * 7 - offset)
    assert_frame_equal(
        scan(fs).slice(offset, 17).collect(engine="streaming"),
        pl.DataFrame(expected_series),
    )
    assert_frame_equal(
        scan(fs, row_index_name="ri").slice(offset, 17).collect(engine="streaming"),
        pl.DataFrame(ri_expected_series),
    )


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
@given(offset=st.integers(-100, 100), length=st.integers(0, 101))
def test_multiscan_slice_parametric(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
    offset: int,
    length: int,
) -> None:
    ref = io.BytesIO()
    write(pl.Series("c1", [i % 7 for i in range(13 * 7)]).to_frame(), ref)
    ref.seek(0)

    fs = [io.BytesIO() for _ in range(13)]
    for f in fs:
        write(pl.Series("c1", range(7)).to_frame(), f)
        f.seek(0)

    assert_frame_equal(
        scan(ref).slice(offset, length).collect(),
        scan(fs).slice(offset, length).collect(engine="streaming"),
    )

    ref.seek(0)
    for f in fs:
        f.seek(0)

    assert_frame_equal(
        scan(ref, row_index_name="ri", row_index_offset=42)
        .slice(offset, length)
        .collect(),
        scan(fs, row_index_name="ri", row_index_offset=42)
        .slice(offset, length)
        .collect(engine="streaming"),
    )

    assert_frame_equal(
        scan(ref, row_index_name="ri", row_index_offset=42)
        .slice(offset, length)
        .select("ri")
        .collect(),
        scan(fs, row_index_name="ri", row_index_offset=42)
        .slice(offset, length)
        .select("ri")
        .collect(engine="streaming"),
    )


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_many_files(scan: Any, write: Any) -> None:
    f = io.BytesIO()
    write(pl.DataFrame({"a": [5, 10, 1996]}), f)
    bs = f.getvalue()

    out = scan([bs] * 1023)

    assert_frame_equal(
        out.collect(),
        pl.DataFrame(
            {
                "a": [5, 10, 1996] * 1023,
            }
        ),
    )


def test_deadlock_stop_requested(plmonkeypatch: PlMonkeyPatch) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    f = io.BytesIO()
    df.write_parquet(f, row_group_size=1)

    plmonkeypatch.setenv("POLARS_MAX_THREADS", "2")
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "1")

    left_fs = [io.BytesIO(f.getbuffer()) for _ in range(10)]
    right_fs = [io.BytesIO(f.getbuffer()) for _ in range(10)]

    left = pl.scan_parquet(left_fs)  # type: ignore[arg-type]
    right = pl.scan_parquet(right_fs)  # type: ignore[arg-type]

    left.join(right, pl.col.a == pl.col.a).collect(engine="streaming").height


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_deadlock_linearize(scan: Any, write: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    f = io.BytesIO()
    write(df, f)
    fs = [io.BytesIO(f.getbuffer()) for _ in range(10)]
    lf = scan(fs).head(100)

    assert_frame_equal(
        lf.collect(
            engine="streaming", optimizations=pl.QueryOptFlags(slice_pushdown=False)
        ),
        pl.concat([df] * 10),
    )


@pytest.mark.parametrize(
    ("scan", "write"),
    SCAN_AND_WRITE_FUNCS,
)
def test_row_index_filter_22612(scan: Any, write: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    f = io.BytesIO()

    if write is pl.DataFrame.write_parquet:
        df.write_parquet(f, row_group_size=5)
        assert pq.read_metadata(f).num_row_groups == 2
    else:
        write(df, f)

    for end in range(2, 10):
        assert_frame_equal(
            scan(f)
            .with_row_index()
            .filter(pl.col("index") >= end - 2, pl.col("index") <= end)
            .collect(),
            df.with_row_index().slice(end - 2, 3),
        )

        assert_frame_equal(
            scan(f)
            .with_row_index()
            .filter(pl.col("index").is_between(end - 2, end))
            .collect(),
            df.with_row_index().slice(end - 2, 3),
        )


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_row_index_name_in_file(scan: Any, write: Any) -> None:
    f = io.BytesIO()
    write(pl.DataFrame({"index": 1}), f)

    with pytest.raises(
        pl.exceptions.DuplicateError,
        match="cannot add row_index with name 'index': column already exists in file",
    ):
        scan(f).with_row_index().collect()


def test_extra_columns_not_ignored_22218() -> None:
    dfs = [pl.DataFrame({"a": 1, "b": 1}), pl.DataFrame({"a": 2, "c": 2})]

    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]

    dfs[0].write_parquet(files[0])
    dfs[1].write_parquet(files[1])

    with pytest.raises(
        pl.exceptions.SchemaError,
        match=r"extra column in file outside of expected schema: c, hint: specify .*or pass",
    ):
        pl.scan_parquet(files, missing_columns="insert").select(pl.all()).collect()

    assert_frame_equal(
        pl.scan_parquet(
            files,
            missing_columns="insert",
            extra_columns="ignore",
        )
        .select(pl.all())
        .collect(),
        pl.DataFrame({"a": [1, 2], "b": [1, None]}),
    )


@pytest.mark.parametrize(("scan", "write"), SCAN_AND_WRITE_FUNCS)
def test_scan_null_upcast(scan: Any, write: Any) -> None:
    dfs = [
        pl.DataFrame({"a": [1, 2, 3]}),
        pl.select(a=pl.lit(None, dtype=pl.Null)),
    ]

    files = [io.BytesIO(), io.BytesIO()]

    write(dfs[0], files[0])
    write(dfs[1], files[1])

    # Prevent CSV schema inference from loading as string (it looks at multiple
    # files).
    if scan is pl.scan_csv:
        scan = partial(scan, schema=dfs[0].schema)

    assert_frame_equal(
        scan(files).collect(),
        pl.DataFrame({"a": [1, 2, 3, None]}),
    )


@pytest.mark.parametrize(
    ("scan", "write"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson),
    ],
)
def test_scan_null_upcast_to_nested(scan: Any, write: Any) -> None:
    schema = {"a": pl.List(pl.Struct({"field": pl.Int64}))}

    dfs = [
        pl.DataFrame(
            {"a": [[{"field": 1}], [{"field": 2}], []]},
            schema=schema,
        ),
        pl.select(a=pl.lit(None, dtype=pl.Null)),
    ]

    files = [io.BytesIO(), io.BytesIO()]

    write(dfs[0], files[0])
    write(dfs[1], files[1])

    # Prevent CSV schema inference from loading as string (it looks at multiple
    # files).
    if scan is pl.scan_csv:
        scan = partial(scan, schema=schema)

    assert_frame_equal(
        scan(files).collect(),
        pl.DataFrame(
            {"a": [[{"field": 1}], [{"field": 2}], [], None]},
            schema=schema,
        ),
    )


@pytest.mark.parametrize(
    ("scan", "write"),
    [
        (pl.scan_parquet, pl.DataFrame.write_parquet),
    ],
)
@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "file:" if sys.platform != "win32" else "file:/",
        "file://" if sys.platform != "win32" else "file:///",
    ],
)
@pytest.mark.parametrize("use_glob", [True, False])
def test_scan_ignore_hidden_files_21762(
    tmp_path: Path, scan: Any, write: Any, use_glob: bool, prefix: str
) -> None:
    file_names: list[str] = ["a.ext", "_a.ext", ".a.ext", "a_.ext"]

    for file_name in file_names:
        write(pl.DataFrame({"rel_path": file_name}), tmp_path / file_name)

    (tmp_path / "folder").mkdir()

    for file_name in file_names:
        write(
            pl.DataFrame({"rel_path": f"folder/{file_name}"}),
            tmp_path / "folder" / file_name,
        )

    (tmp_path / "_folder").mkdir()

    for file_name in file_names:
        write(
            pl.DataFrame({"rel_path": f"_folder/{file_name}"}),
            tmp_path / "_folder" / file_name,
        )

    suffix = "/**/*.ext" if use_glob else "/" if prefix.startswith("file:") else ""
    root = f"{prefix}{tmp_path}{suffix}"

    assert_frame_equal(
        scan(root).sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    ".a.ext",
                    "_a.ext",
                    "_folder/.a.ext",
                    "_folder/_a.ext",
                    "_folder/a.ext",
                    "_folder/a_.ext",
                    "a.ext",
                    "a_.ext",
                    "folder/.a.ext",
                    "folder/_a.ext",
                    "folder/a.ext",
                    "folder/a_.ext",
                ]
            }
        ),
    )

    assert_frame_equal(
        scan(root, hidden_file_prefix=".").sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    "_a.ext",
                    "_folder/_a.ext",
                    "_folder/a.ext",
                    "_folder/a_.ext",
                    "a.ext",
                    "a_.ext",
                    "folder/_a.ext",
                    "folder/a.ext",
                    "folder/a_.ext",
                ]
            }
        ),
    )

    assert_frame_equal(
        scan(root, hidden_file_prefix=[".", "_"]).sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    "_folder/a.ext",
                    "_folder/a_.ext",
                    "a.ext",
                    "a_.ext",
                    "folder/a.ext",
                    "folder/a_.ext",
                ]
            }
        ),
    )

    assert_frame_equal(
        scan(root, hidden_file_prefix=(".", "_")).sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    "_folder/a.ext",
                    "_folder/a_.ext",
                    "a.ext",
                    "a_.ext",
                    "folder/a.ext",
                    "folder/a_.ext",
                ]
            }
        ),
    )

    # Top-level glob only
    root = f"{tmp_path}/*.ext"

    assert_frame_equal(
        scan(root).sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    ".a.ext",
                    "_a.ext",
                    "a.ext",
                    "a_.ext",
                ]
            }
        ),
    )

    assert_frame_equal(
        scan(root, hidden_file_prefix=".").sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    "_a.ext",
                    "a.ext",
                    "a_.ext",
                ]
            }
        ),
    )

    assert_frame_equal(
        scan(root, hidden_file_prefix=[".", "_"]).sort("*"),
        pl.LazyFrame(
            {
                "rel_path": [
                    "a.ext",
                    "a_.ext",
                ]
            }
        ),
    )

    # Direct file passed
    with pytest.raises(pl.exceptions.ComputeError, match="expanded paths were empty"):
        scan(tmp_path / "_a.ext", hidden_file_prefix="_").collect()


def test_row_count_estimate_multifile(io_files_path: Path) -> None:
    src = io_files_path / "foods*.parquet"
    # test that it doesn't check only the first file
    assert "ESTIMATED ROWS: 54" in pl.scan_parquet(src).explain()


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
    ],
)
@pytest.mark.parametrize(
    ("predicate", "expected_indices"),
    [
        ((pl.col.x == 1) & True, [0]),
        (True & (pl.col.x == 1), [0]),
    ],
)
@pytest.mark.write_disk
def test_hive_predicate_filtering_edge_case_25630(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
    ext: str,
    predicate: pl.Expr,
    expected_indices: list[int],
) -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]}).with_row_index()

    (tmp_path / "y=0").mkdir()
    (tmp_path / "y=1").mkdir()

    # previously we could panic if hive columns were all filtered out of the projection
    write(df.filter(pl.col.y == 0).drop("y"), tmp_path / "y=0" / f"data.{ext}")
    write(df.filter(pl.col.y == 1).drop("y"), tmp_path / "y=1" / f"data.{ext}")

    res = scan(tmp_path).filter(predicate).select("index").collect(engine="streaming")
    expected = pl.DataFrame(
        data={"index": expected_indices},
        schema={"index": pl.get_index_type()},
    )
    assert_frame_equal(res, expected)
