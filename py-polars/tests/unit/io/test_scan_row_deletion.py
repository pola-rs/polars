from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def data_files_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates: 5 parquet files, 5 rows each."""
    tmp_path = tmp_path_factory.mktemp("data-files")
    _create_data_files(tmp_path)

    return tmp_path


@pytest.fixture
def write_position_deletes(tmp_path: Path) -> WritePositionDeletes:
    return WritePositionDeletes(tmp_path=tmp_path)


class WritePositionDeletes:  # noqa: D101
    def __init__(self, *, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.i = 0

    def __call__(self, positions: pl.Series) -> str:
        path = self.tmp_path / f"{self.i}"

        (
            positions.alias("pos")
            .to_frame()
            .select(pl.lit("").alias("file_path"), "pos")
            .write_parquet(path)
        )

        self.i += 1

        return str(path)


# 5 files x 5 rows each. Contains `physical_index` [0, 1, .., 24].
def _create_data_files(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True, parents=True)
    df = pl.select(physical_index=pl.int_range(25, dtype=pl.UInt32))

    parts = []

    for i in [0, 5, 10, 15, 20]:
        O = "0" if i < 10 else ""  # noqa: E741
        path = tmp_path / f"offset={O}{i}/data.parquet"
        path.parent.mkdir(exist_ok=True, parents=True)

        part_df = df.slice(i, 5)
        part_df.write_parquet(path)

        parts.append(part_df.with_columns(offset=pl.lit(i, dtype=pl.Int64)))

    assert_frame_equal(pl.scan_parquet(tmp_path).collect(), pl.concat(parts))


@pytest.mark.parametrize("row_index_offset", [0, 27, 38, 73])
def test_scan_row_deletions(
    data_files_path: Path,
    write_position_deletes: WritePositionDeletes,
    row_index_offset: int,
) -> None:
    deletion_files = (
        "iceberg-position-delete",
        {
            0: [
                write_position_deletes(pl.Series([1, 2])),
            ],
            1: [
                write_position_deletes(pl.Series([0, 1, 2])),
            ],
            4: [
                write_position_deletes(pl.Series([2, 3])),
            ],
        },
    )

    def apply_row_index_offset(values: list[int]) -> list[int]:
        return [x + row_index_offset for x in values]

    q = pl.scan_parquet(
        data_files_path,
        _deletion_files=deletion_files,  # type: ignore[arg-type]
        hive_partitioning=False,
    ).with_row_index(offset=row_index_offset)

    assert q.select(pl.len()).collect().item() == 18

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([
                    0, 1, 2,
                    3, 4,
                    5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14,
                    15, 16, 17
                ]),
                "physical_index": [
                    0, 3, 4,
                    8, 9,
                    10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19,
                    20, 21, 24
                ],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        )
    )  # fmt: skip

    # head()

    assert_frame_equal(
        q.head(3).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([0, 1, 2]),
                "physical_index": [0, 3, 4],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    assert_frame_equal(
        q.head(10).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "physical_index": [0, 3, 4, 8, 9, 10, 11, 12, 13, 14],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    # tail()

    assert_frame_equal(
        q.tail(3).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([15, 16, 17]),
                "physical_index": [20, 21, 24],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    assert_frame_equal(
        q.tail(10).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset(
                    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                ),
                "physical_index": [
                    13, 14, 15, 16, 17, 18, 19, 20, 21,
                    24
                ],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )  # fmt: skip

    # slice(positive_offset)

    assert_frame_equal(
        q.slice(2, 10).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                "physical_index": [4, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    assert_frame_equal(
        q.slice(5, 10).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                "physical_index": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    assert_frame_equal(
        q.slice(10, 10).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([10, 11, 12, 13, 14, 15, 16, 17]),
                "physical_index": [15, 16, 17, 18, 19, 20, 21, 24],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    # slice(negative_offset)
    assert_frame_equal(
        q.slice(-3, 2).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([15, 16]),
                "physical_index": [20, 21],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    assert_frame_equal(
        q.slice(-23, 10).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([0, 1, 2, 3, 4]),
                "physical_index": [0, 3, 4, 8, 9],
            },
            schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
        ),
    )

    # filter: skip_files
    q = pl.scan_parquet(
        data_files_path,
        _deletion_files=deletion_files,  # type: ignore[arg-type]
    ).with_row_index(offset=row_index_offset)

    assert_frame_equal(
        q.filter(pl.col("offset").is_in([10, 20])).collect(),
        pl.DataFrame(
            {
                "index": apply_row_index_offset([5, 6, 7, 8, 9, 15, 16, 17]),
                "physical_index": [10, 11, 12, 13, 14, 20, 21, 24],
                "offset": [10, 10, 10, 10, 10, 20, 20, 20],
            },
            schema={
                "index": pl.get_index_type(),
                "physical_index": pl.UInt32,
                "offset": pl.Int64,
            },
        ),
    )


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.parametrize("ideal_morsel_size", [999, 50, 33])
@pytest.mark.parametrize("force_empty_capabilities", [True, False])
def test_scan_row_deletion_single_large(
    tmp_path: Path,
    write_position_deletes: WritePositionDeletes,
    ideal_morsel_size: int,
    force_empty_capabilities: bool,
) -> None:
    path = tmp_path / "data.parquet"
    pl.DataFrame({"physical_index": range(100)}).write_parquet(path)

    positions = pl.Series([
        0,   1,  2,  3,  4,  5,  6,  7,  8, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 64, 65, 66, 67, 68, 69, 70, 71, 79,
        80, 81, 82, 83, 84, 90, 91, 92, 93, 97,
        98
    ])  # fmt: skip

    deletion_positions_path = write_position_deletes(positions)

    script_args: list[str] = [
        str(ideal_morsel_size),
        "1" if force_empty_capabilities else "0",
        str(path),
        deletion_positions_path,
    ]

    # Use a process to ensure ideal morsel size is set correctly.
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import os
import sys

(
    _,
    ideal_morsel_size,
    force_empty_capabilities,
    data_file_path,
    deletion_positions_path,
) = sys.argv

os.environ["POLARS_VERBOSE"] = "0"
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["POLARS_IDEAL_MORSEL_SIZE"] = ideal_morsel_size
os.environ["POLARS_FORCE_EMPTY_READER_CAPABILITIES"] = force_empty_capabilities

import polars as pl
from polars.testing import assert_frame_equal

full_expected_physical = [
     9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 72, 73, 74, 75, 76, 77, 78,
    85, 86, 87, 88, 89, 94, 95, 96, 99
]  # fmt: skip

deletion_files = (
    "iceberg-position-delete",
    {0: [deletion_positions_path]},
)

q = pl.scan_parquet(data_file_path, _deletion_files=deletion_files).with_row_index()

assert_frame_equal(
    q.collect(),
    pl.DataFrame({"physical_index": full_expected_physical}).with_row_index(),
)

assert_frame_equal(
    q.tail(999).collect(),
    pl.DataFrame({"physical_index": full_expected_physical}).with_row_index(),
)

# Note: The negative slice is important here. Otherwise row_index does not get
# lowered into the post-apply pipeline.
for negative_offset in range(1, 49):
    assert_frame_equal(
        q.tail(negative_offset).collect(),
        pl.DataFrame(
            {"physical_index": full_expected_physical[-negative_offset:]}
        ).with_row_index(offset=49 - negative_offset),
    )

assert_frame_equal(
    q.slice(20).collect(),
    pl.DataFrame({"physical_index": full_expected_physical[20:]}).with_row_index(
        offset=20
    ),
)

print("OK", end="")
    """,
            *script_args,
        ],
        stderr=subprocess.STDOUT,
    )

    assert out == b"OK"


@pytest.mark.write_disk
def test_scan_row_deletion_skips_file_with_all_rows_deleted(
    tmp_path: Path,
    write_position_deletes: WritePositionDeletes,
) -> None:
    # Create our own copy because we mutate one of the data files
    data_files_path = tmp_path / "data-files"
    _create_data_files(data_files_path)

    # Corrupt a parquet file
    def remove_data(path: Path) -> None:
        v = path.read_bytes()
        metadata_and_footer_len = 8 + int.from_bytes(v[-8:][:4], "little")
        path.write_bytes(b"\x00" * (len(v) - metadata_and_footer_len))
        path.write_bytes(v[-metadata_and_footer_len:])

    remove_data(data_files_path / "offset=05/data.parquet")

    q = pl.scan_parquet(
        data_files_path / "offset=05/data.parquet", hive_partitioning=False
    )

    # Baseline: The metadata is readable but the row groups are not

    assert q.collect_schema() == {"physical_index": pl.UInt32}
    assert q.select(pl.len()).collect().item() == 5

    with pytest.raises(pl.exceptions.ComputeError, match="Invalid thrift"):
        q.collect()

    q = pl.scan_parquet(data_files_path, hive_partitioning=False)

    with pytest.raises(pl.exceptions.ComputeError, match="Invalid thrift"):
        q.collect()

    q = pl.scan_parquet(
        data_files_path,
        _deletion_files=(
            "iceberg-position-delete",
            {
                1: [
                    write_position_deletes(pl.Series([0, 1, 2])),
                    write_position_deletes(pl.Series([3, 4])),
                ]
            },
        ),
        hive_partitioning=False,
    )

    expect = pl.DataFrame(
        {
            "index": [
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 19,
            ],
            "physical_index": [
                0, 1, 2, 3, 4,
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 19,
                20, 21, 22, 23, 24,
            ],
        },
        schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
    )  # fmt: skip

    assert_frame_equal(q.collect(), expect.drop("index"))
    assert_frame_equal(q.with_row_index().collect(), expect)

    expect = pl.DataFrame(
        {
            "index": [
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 19,
            ],
            "physical_index": [
                15, 16, 17, 18, 19,
                20, 21, 22, 23, 24,
            ],
        },
        schema={"index": pl.get_index_type(), "physical_index": pl.UInt32},
    )  # fmt: skip

    assert_frame_equal(q.slice(10).collect(), expect.drop("index"))
    assert_frame_equal(q.with_row_index().slice(10).collect(), expect)
