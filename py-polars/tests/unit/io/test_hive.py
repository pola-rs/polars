import warnings
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.write_disk()
def test_hive_partitioned_predicate_pushdown(
    io_files_path: Path, tmp_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    df = pl.read_ipc(io_files_path / "*.ipc")

    root = tmp_path / "partitioned_data"

    # Ignore the pyarrow legacy warning until we can write properly with new settings.
    warnings.filterwarnings("ignore")
    pq.write_to_dataset(
        df.to_arrow(),
        root_path=root,
        partition_cols=["category", "fats_g"],
        use_legacy_dataset=True,
    )
    q = pl.scan_parquet(root / "**/*.parquet", hive_partitioning=False)
    # checks schema
    assert q.columns == ["calories", "sugars_g"]
    # checks materialization
    assert q.collect().columns == ["calories", "sugars_g"]

    q = pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True)
    assert q.columns == ["calories", "sugars_g", "category", "fats_g"]

    # Partitioning changes the order
    sort_by = ["fats_g", "category", "calories", "sugars_g"]

    # The hive partitioned columns are appended,
    # so we must ensure we assert in the proper order.
    df = df.select(["calories", "sugars_g", "category", "fats_g"])
    for streaming in [True, False]:
        for pred in [
            pl.col("category") == "vegetables",
            pl.col("category") != "vegetables",
            pl.col("fats_g") > 0.5,
            (pl.col("fats_g") == 0.5) & (pl.col("category") == "vegetables"),
        ]:
            assert_frame_equal(
                q.filter(pred).sort(sort_by).collect(streaming=streaming),
                df.filter(pred).sort(sort_by),
            )
            err = capfd.readouterr().err
            assert "hive partitioning" in err

    # tests: 11536
    assert q.filter(pl.col("sugars_g") == 25).collect().shape == (1, 4)

    # tests: 12570
    assert q.filter(pl.col("fats_g") == 1225.0).select("category").collect().shape == (
        0,
        1,
    )


@pytest.mark.write_disk()
def test_hive_partitioned_predicate_pushdown_skips_correct_number_of_files(
    io_files_path: Path, tmp_path: Path, monkeypatch: Any, capfd: Any
) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")
    df = pl.DataFrame({"d": pl.arange(0, 5, eager=True)}).with_columns(
        a=pl.col("d") % 5
    )
    root = tmp_path / "test_int_partitions"
    df.write_parquet(
        root,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": ["a"]},
    )

    q = pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True)
    assert q.filter(pl.col("a").is_in([1, 4])).collect().shape == (2, 2)
    assert "hive partitioning: skipped 3 files" in capfd.readouterr().err


@pytest.mark.write_disk()
def test_hive_partitioned_slice_pushdown(io_files_path: Path, tmp_path: Path) -> None:
    df = pl.read_ipc(io_files_path / "*.ipc")

    root = tmp_path / "partitioned_data"

    # Ignore the pyarrow legacy warning until we can write properly with new settings.
    warnings.filterwarnings("ignore")
    pq.write_to_dataset(
        df.to_arrow(),
        root_path=root,
        partition_cols=["category", "fats_g"],
        use_legacy_dataset=True,
    )

    q = pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True)

    # tests: 11682
    for streaming in [True, False]:
        assert (
            q.head(1)
            .collect(streaming=streaming)
            .select(pl.all_horizontal(pl.all().count() == 1))
            .item()
        )
        assert q.head(0).collect(streaming=streaming).columns == [
            "calories",
            "sugars_g",
            "category",
            "fats_g",
        ]


@pytest.mark.write_disk()
def test_hive_partitioned_projection_pushdown(
    io_files_path: Path, tmp_path: Path
) -> None:
    df = pl.read_ipc(io_files_path / "*.ipc")

    root = tmp_path / "partitioned_data"

    # Ignore the pyarrow legacy warning until we can write properly with new settings.
    warnings.filterwarnings("ignore")
    pq.write_to_dataset(
        df.to_arrow(),
        root_path=root,
        partition_cols=["category", "fats_g"],
        use_legacy_dataset=True,
    )

    q = pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True)
    columns = ["sugars_g", "category"]
    for streaming in [True, False]:
        assert q.select(columns).collect(streaming=streaming).columns == columns

    # test that hive partition columns are projected with the correct height when
    # the projection contains only hive partition columns (11796)
    for parallel in ("row_groups", "columns"):
        q = pl.scan_parquet(
            root / "**/*.parquet",
            hive_partitioning=True,
            parallel=parallel,  # type: ignore[arg-type]
        )

        expected = q.collect().select("category")
        result = q.select("category").collect()

        assert_frame_equal(result, expected)


@pytest.mark.write_disk()
def test_hive_partitioned_err(io_files_path: Path, tmp_path: Path) -> None:
    df = pl.read_ipc(io_files_path / "*.ipc")
    root = tmp_path / "sugars_g=10"
    root.mkdir()
    df.write_parquet(root / "file.parquet")

    with pytest.raises(pl.ComputeError, match="invalid hive partitions"):
        pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True)
