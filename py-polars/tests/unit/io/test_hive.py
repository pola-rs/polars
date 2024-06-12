import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.skip(
    reason="Broken by pyarrow 15 release: https://github.com/pola-rs/polars/issues/13892"
)
@pytest.mark.xdist_group("streaming")
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

    # Ensure the CSE can work with hive partitions.
    q = q.filter(pl.col("a").gt(2))
    result = q.join(q, on="a", how="left").collect(comm_subplan_elim=True)
    expected = {
        "a": [3, 4],
        "d": [3, 4],
        "a_right": [3, 4],
        "d_right": [3, 4],
    }
    assert result.to_dict(as_series=False) == expected


@pytest.mark.skip(
    reason="Broken by pyarrow 15 release: https://github.com/pola-rs/polars/issues/13892"
)
@pytest.mark.xdist_group("streaming")
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


@pytest.mark.skip(
    reason="Broken by pyarrow 15 release: https://github.com/pola-rs/polars/issues/13892"
)
@pytest.mark.xdist_group("streaming")
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

    with pytest.raises(pl.DuplicateError, match="invalid Hive partition schema"):
        pl.scan_parquet(root / "**/*.parquet", hive_partitioning=True).collect()


@pytest.mark.write_disk()
def test_hive_partitioned_projection_skip_files(
    io_files_path: Path, tmp_path: Path
) -> None:
    # ensure that it makes hive columns even when . in dir value
    # and that it doesn't make hive columns from filename with =
    df = pl.DataFrame(
        {"sqlver": [10012.0, 10013.0], "namespace": ["eos", "fda"], "a": [1, 2]}
    )
    root = tmp_path / "partitioned_data"
    for dir_tuple, sub_df in df.partition_by(
        ["sqlver", "namespace"], include_key=False, as_dict=True
    ).items():
        new_path = root / f"sqlver={dir_tuple[0]}" / f"namespace={dir_tuple[1]}"
        new_path.mkdir(parents=True, exist_ok=True)
        sub_df.write_parquet(new_path / "file=8484.parquet")
    test_df = (
        pl.scan_parquet(str(root) + "/**/**/*.parquet")
        # don't care about column order
        .select("sqlver", "namespace", "a", pl.exclude("sqlver", "namespace", "a"))
        .collect()
    )
    assert_frame_equal(df, test_df)


@pytest.fixture()
def dataset_path(tmp_path: Path) -> Path:
    tmp_path.mkdir(exist_ok=True)

    # Set up Hive partitioned Parquet file
    root = tmp_path / "dataset"
    part1 = root / "c=1"
    part2 = root / "c=2"
    root.mkdir()
    part1.mkdir()
    part2.mkdir()
    df1 = pl.DataFrame({"a": [1, 2], "b": [11.0, 12.0]})
    df2 = pl.DataFrame({"a": [3, 4], "b": [13.0, 14.0]})
    df3 = pl.DataFrame({"a": [5, 6], "b": [15.0, 16.0]})
    df1.write_parquet(part1 / "one.parquet")
    df2.write_parquet(part1 / "two.parquet")
    df3.write_parquet(part2 / "three.parquet")

    return root


@pytest.mark.write_disk()
def test_scan_parquet_hive_schema(dataset_path: Path) -> None:
    result = pl.scan_parquet(dataset_path / "**/*.parquet", hive_partitioning=True)
    assert result.schema == OrderedDict({"a": pl.Int64, "b": pl.Float64, "c": pl.Int64})

    result = pl.scan_parquet(
        dataset_path / "**/*.parquet",
        hive_partitioning=True,
        hive_schema={"c": pl.Int32},
    )

    expected_schema = OrderedDict({"a": pl.Int64, "b": pl.Float64, "c": pl.Int32})
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema


@pytest.mark.write_disk()
def test_read_parquet_invalid_hive_schema(dataset_path: Path) -> None:
    with pytest.raises(
        pl.SchemaFieldNotFoundError,
        match='path contains column not present in the given Hive schema: "c"',
    ):
        pl.read_parquet(
            dataset_path / "**/*.parquet",
            hive_partitioning=True,
            hive_schema={"nonexistent": pl.Int32},
        )


def test_read_parquet_hive_schema_with_pyarrow() -> None:
    with pytest.raises(
        TypeError,
        match="cannot use `hive_partitions` with `use_pyarrow=True`",
    ):
        pl.read_parquet("test.parquet", hive_schema={"c": pl.Int32}, use_pyarrow=True)
