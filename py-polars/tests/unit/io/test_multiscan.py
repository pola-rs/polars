from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any, Callable

import pytest

import polars as pl
from polars.meta.index_type import get_index_type
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("scan", "write"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_csv, pl.DataFrame.write_csv),
        # (pl.scan_ndjson, pl.DataFrame.write_ndjson), not yet implemented for streaming
    ],
)
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
        ),
    )


@pytest.mark.parametrize(
    ("scan", "write", "ext", "supports_missing_columns", "supports_hive_partitioning"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc", False, True),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet", True, True),
        (pl.scan_csv, pl.DataFrame.write_csv, "csv", False, False),
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
        "allow_missing_columns": missing_column,
        "include_file_paths": ifp,
        "row_index_name": ri,
        "hive_partitioning": hive,
    }

    if not supports_missing_columns:
        del args["allow_missing_columns"]
    if not supports_hive_partitioning:
        del args["hive_partitioning"]

    for projection in [
        base_projection,
        base_projection[::-1],
    ]:
        assert_frame_equal(
            scan(multiscan_path, **args)
            .collect(new_streaming=True)  # type: ignore[call-overload]
            .select(projection),
            scan(multiscan_path, **args).select(projection).collect(new_streaming=True),  # type: ignore[call-overload]
        )

    for remove in range(len(base_projection)):
        new_projection = base_projection.copy()
        new_projection.pop(remove)

        for projection in [
            new_projection,
            new_projection[::-1],
        ]:
            print(projection)
            assert_frame_equal(
                scan(multiscan_path, **args)
                .collect(new_streaming=True)  # type: ignore[call-overload]
                .select(projection),
                scan(multiscan_path, **args)
                .select(projection)
                .collect(new_streaming=True),  # type: ignore[call-overload]
            )


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
        (pl.scan_csv, pl.DataFrame.write_csv, "csv"),
    ],
)
@pytest.mark.write_disk
def test_multiscan_row_index(
    tmp_path: Path,
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, Path], Any],
    ext: str,
) -> None:
    a = pl.DataFrame({"col": [5, 10, 1996]})
    b = pl.DataFrame({"col": [42]})
    c = pl.DataFrame({"col": [13, 37]})

    write(a, tmp_path / f"a.{ext}")
    write(b, tmp_path / f"b.{ext}")
    write(c, tmp_path / f"c.{ext}")

    col = pl.concat([a, b, c]).to_series()
    g = tmp_path / f"*.{ext}"

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
        pytest.param(
            pl.scan_ndjson,
            pl.DataFrame.write_ndjson,
            "ndjson",
            marks=pytest.mark.xfail(reason="NYI"),
        ),
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

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="data type mismatch for column xyz_col: expected: i64, found: str",
    ):
        q.collect(new_streaming=True)  # type: ignore[call-overload]


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
        pytest.param(
            pl.scan_ndjson,
            pl.DataFrame.write_ndjson,
            "ndjson",
            marks=pytest.mark.xfail(reason="NYI"),
        ),
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
        q.collect(new_streaming=True)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
        (
            pl.scan_csv,
            pl.DataFrame.write_csv,
            "csv",
        ),
        pytest.param(
            pl.scan_ndjson,
            pl.DataFrame.write_ndjson,
            "ndjson",
            marks=pytest.mark.xfail(reason="NYI"),
        ),
    ],
)
def test_multiscan_head(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
    ext: str,
) -> None:
    if ext == "ndjson" and os.environ["POLARS_AUTO_NEW_STREAMING"] == "1":
        msg = "NYI"
        raise Exception(msg)  # noqa: TRY002

    a = io.BytesIO()
    b = io.BytesIO()
    for f in [a, b]:
        write(pl.Series("c1", range(10)).to_frame(), f)
        f.seek(0)

    assert_frame_equal(
        scan([a, b]).head(5).collect(new_streaming=True),  # type: ignore[call-overload]
        pl.Series("c1", range(5)).to_frame(),
    )


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
        (
            pl.scan_csv,
            pl.DataFrame.write_csv,
            "csv",
        ),
        pytest.param(
            pl.scan_ndjson,
            pl.DataFrame.write_ndjson,
            "ndjson",
            marks=pytest.mark.xfail(reason="NYI"),
        ),
    ],
)
def test_multiscan_tail(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
    ext: str,
) -> None:
    if ext == "ndjson" and os.environ["POLARS_AUTO_NEW_STREAMING"] == "1":
        msg = "NYI"
        raise Exception(msg)  # noqa: TRY002

    a = io.BytesIO()
    b = io.BytesIO()
    for f in [a, b]:
        write(pl.Series("c1", range(10)).to_frame(), f)
        f.seek(0)

    assert_frame_equal(
        scan([a, b]).tail(5).collect(new_streaming=True),  # type: ignore[call-overload]
        pl.Series("c1", range(5, 10)).to_frame(),
    )


@pytest.mark.parametrize(
    ("scan", "write", "ext"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc, "ipc"),
        (pl.scan_parquet, pl.DataFrame.write_parquet, "parquet"),
        (
            pl.scan_csv,
            pl.DataFrame.write_csv,
            "csv",
        ),
        pytest.param(
            pl.scan_ndjson,
            pl.DataFrame.write_ndjson,
            "ndjson",
            marks=pytest.mark.xfail(reason="NYI"),
        ),
    ],
)
def test_multiscan_slice_middle(
    scan: Callable[..., pl.LazyFrame],
    write: Callable[[pl.DataFrame, io.BytesIO | Path], Any],
    ext: str,
) -> None:
    if ext == "ndjson" and os.environ["POLARS_AUTO_NEW_STREAMING"] == "1":
        msg = "NYI"
        raise Exception(msg)  # noqa: TRY002

    fs = [io.BytesIO() for _ in range(13)]
    for f in fs:
        write(pl.Series("c1", range(7)).to_frame(), f)
        f.seek(0)

    expected = (
        list(range(2, 7))  # fs[4]
        + list(range(7))  # fs[5]
        + list(range(5))  # fs[6]
    )

    assert_frame_equal(
        scan(fs).slice(5 * 7 - 5, 17).collect(new_streaming=True),  # type: ignore[call-overload]
        pl.Series("c1", expected).to_frame(),
    )
