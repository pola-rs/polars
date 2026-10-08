"""
Tests for Iceberg scan planning with the `polars_iceberg` plugin.

These cover the plumbing (plugin ID handshake and refusal, host storage, errors) and
behaviour specific to the plugin planner. Iceberg semantics are covered by re-running
the main Iceberg test suite with the plugin planner (`test_iceberg_plugin_suite.py`).
"""

from __future__ import annotations

import importlib.metadata
import io
import pickle
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
import polars._plr as plr
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch

pytest.importorskip("polars_iceberg")
pytest.importorskip("pyiceberg")

import polars_iceberg
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.types import LongType, NestedField, StringType

from tests.unit.io.test_iceberg import (
    new_iceberg_table,
    sqlcatalog_uses_null_pool_connections,  # noqa: F401
)

pytestmark = pytest.mark.write_disk

# Pip requirements as they appear in error messages.
POLARS = f"polars=={importlib.metadata.version('polars')}"
PLUGIN = f"polars_iceberg=={importlib.metadata.version('polars_iceberg')}"

ID_V1 = "polars.io_plugin.iceberg.v1"


@pytest.fixture(autouse=True)
def _plugin_planner(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", "plugin")


TEST_DF = pl.DataFrame(
    {
        "a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
        "b": pl.Series(["x", "y", "z", None, "w"], dtype=pl.String),
    }
)


def _new_table(tmp_path: Path, name: str = "table") -> Any:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "b", StringType()),
        ),
        name=name,
    )
    tbl.append(TEST_DF.to_arrow())
    return tbl


@pytest.fixture
def table(tmp_path: Path) -> Any:
    return _new_table(tmp_path)


@pytest.fixture
def metadata_path(table: Any) -> str:
    """Metadata location of a table holding `TEST_DF`."""
    return str(table.metadata_location)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_iceberg_plugin_collect(metadata_path: str, engine: str) -> None:
    lf = pl.scan_iceberg(metadata_path)
    assert_frame_equal(lf.collect(engine=engine), TEST_DF)  # type: ignore[call-overload]


def test_iceberg_plugin_collect_schema(metadata_path: str) -> None:
    lf = pl.scan_iceberg(metadata_path)
    assert lf.collect_schema() == pl.Schema({"a": pl.Int64, "b": pl.String})


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_iceberg_plugin_operations_on_top(metadata_path: str, engine: str) -> None:
    lf = pl.scan_iceberg(metadata_path)

    def check(q: pl.LazyFrame, expected: pl.DataFrame) -> None:
        assert_frame_equal(q.collect(engine=engine), expected)  # type: ignore[call-overload]

    check(lf.select("b"), TEST_DF.select("b"))
    check(lf.filter(pl.col("a") > 2), TEST_DF.filter(pl.col("a") > 2))
    check(
        lf.filter(pl.col("b").is_null()).select("a"),
        TEST_DF.filter(pl.col("b").is_null()).select("a"),
    )
    check(lf.head(2), TEST_DF.head(2))
    check(lf.slice(1, 3), TEST_DF.slice(1, 3))
    check(lf.select(pl.len()), TEST_DF.select(pl.len()))
    check(
        lf.with_columns(c=pl.col("a") * 2).sort("a", descending=True),
        TEST_DF.with_columns(c=pl.col("a") * 2).sort("a", descending=True),
    )


def test_iceberg_plugin_multiple_scans(tmp_path: Path) -> None:
    lf1 = pl.scan_iceberg(_new_table(tmp_path, "t1").metadata_location)
    lf2 = pl.scan_iceberg(_new_table(tmp_path, "t2"))

    assert_frame_equal(
        pl.concat([lf1, lf2]).collect(),
        pl.concat([TEST_DF, TEST_DF]),
    )

    assert_frame_equal(
        lf1.join(lf2.select("a", c="b"), on="a", maintain_order="left").collect(),
        TEST_DF.join(TEST_DF.select("a", c="b"), on="a", maintain_order="left"),
    )


def test_iceberg_plugin_serialize_roundtrip(metadata_path: str) -> None:
    lf = pl.scan_iceberg(metadata_path).filter(pl.col("a") > 1)
    expected = TEST_DF.filter(pl.col("a") > 1)

    serialized = lf.serialize()
    assert_frame_equal(
        pl.LazyFrame.deserialize(io.BytesIO(serialized)).collect(), expected
    )

    assert_frame_equal(pickle.loads(pickle.dumps(lf)).collect(), expected)


def test_iceberg_plugin_table_object_pickles_as_metadata_location(table: Any) -> None:
    lf = pl.scan_iceberg(table)
    assert_frame_equal(pickle.loads(pickle.dumps(lf)).collect(), TEST_DF)


def test_iceberg_plugin_snapshot_id(table: Any) -> None:
    first = table.current_snapshot().snapshot_id
    table.append(TEST_DF.head(1).to_arrow())

    assert pl.scan_iceberg(table).collect().height == TEST_DF.height + 1
    assert_frame_equal(pl.scan_iceberg(table, snapshot_id=first).collect(), TEST_DF)
    assert_frame_equal(
        pl.scan_iceberg(table, from_snapshot_id_exclusive=first).collect(),
        TEST_DF.head(1),
    )


def test_iceberg_plugin_snapshot_id_not_found(metadata_path: str) -> None:
    with pytest.raises(ValueError, match="snapshot ID not found: 1234"):
        pl.scan_iceberg(metadata_path, snapshot_id=1234).collect()


def test_iceberg_plugin_ids() -> None:
    assert ID_V1 in plr._IO_PLUGIN_IDS
    assert ID_V1 in polars_iceberg._polars_io_plugin_ids


def test_iceberg_plugin_is_used(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)

    stderr = capfd.readouterr().err
    assert "plan with the polars_iceberg plugin" in stderr
    assert "polars-iceberg: resolve():" in stderr
    assert "begin path expansion" not in stderr


def test_iceberg_plugin_not_used_with_reader_override_pyiceberg(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    lf = pl.scan_iceberg(metadata_path, reader_override="pyiceberg")
    assert_frame_equal(lf.collect(), TEST_DF)
    assert "polars_iceberg plugin" not in capfd.readouterr().err


def test_iceberg_plugin_planner_env_var_invalid(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", "other")
    with pytest.raises(ValueError, match="unknown value for POLARS_ICEBERG_PLANNER"):
        pl.scan_iceberg(metadata_path).collect()


def test_iceberg_plugin_repeat_collect_unchanged(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    lf = pl.scan_iceberg(metadata_path)

    assert_frame_equal(lf.collect(), TEST_DF)
    capfd.readouterr()

    assert_frame_equal(lf.collect(), TEST_DF)
    stderr = capfd.readouterr().err
    assert "to_dataset_scan(): early return" in stderr
    assert "polars-iceberg: resolve():" not in stderr


def test_iceberg_plugin_io_error(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_ICEBERG_PLUGIN_TESTING_FAIL", "io")
    with pytest.raises(
        pl.exceptions.ComputeError, match=r"polars-iceberg-testing-missing"
    ):
        pl.scan_iceberg(metadata_path).collect()


def test_iceberg_plugin_panic_is_error(
    metadata_path: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_ICEBERG_PLUGIN_TESTING_FAIL", "panic")
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"polars_iceberg panicked: testing panic requested",
    ):
        pl.scan_iceberg(metadata_path).collect()

    capfd.readouterr()


def test_iceberg_plugin_selects_newest_shared_id(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    newer = "polars.io_plugin.iceberg.v2"
    requested = []
    capsule = polars_iceberg._capsule

    def _capsule(id: str) -> Any:
        requested.append(id)
        return capsule(id)

    # The plugin offers a newer ID too; this Polars only knows v1.
    plmonkeypatch.setattr(polars_iceberg, "_polars_io_plugin_ids", (ID_V1, newer))
    plmonkeypatch.setattr(polars_iceberg, "_capsule", _capsule)
    assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)
    assert requested == [ID_V1]


@pytest.mark.parametrize(
    ("plugin_ids", "hint"),
    [
        (("polars.io_plugin.iceberg.v99",), "upgrade polars, or install an older"),
        (
            ("polars.io_plugin.iceberg.v0",),
            "upgrade polars_iceberg with `pip install --upgrade 'polars_iceberg>=0.1.0'`",
        ),
        ((), "upgrade polars_iceberg with"),
    ],
)
def test_iceberg_plugin_no_shared_id(
    metadata_path: str,
    plmonkeypatch: PlMonkeyPatch,
    plugin_ids: tuple[str, ...],
    hint: str,
) -> None:
    plmonkeypatch.setattr(polars_iceberg, "_polars_io_plugin_ids", plugin_ids)
    with pytest.raises(pl.exceptions.ComputeError) as exc:
        pl.scan_iceberg(metadata_path).collect()

    msg = str(exc.value)
    assert msg.startswith(f"{PLUGIN} is incompatible with {POLARS}: "), msg
    assert f"provides plugin IDs {list(plugin_ids)}" in msg
    assert f"Polars supports {plr._IO_PLUGIN_IDS}" in msg
    assert hint in msg


def test_iceberg_plugin_unknown_capsule_name_refused(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Polars dispatches by the capsule name only, whatever the plugin claims.
    other = "polars.io_plugin.iceberg.v2"
    plmonkeypatch.setattr(
        polars_iceberg, "_capsule", lambda _: polars_iceberg._capsule_for_testing(other)
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=rf"unsupported polars_iceberg plugin ID \"{re.escape(other)}\"",
    ):
        pl.scan_iceberg(metadata_path).collect()


def test_iceberg_plugin_non_capsule_refused(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setattr(polars_iceberg, "_capsule", lambda _: object())
    with pytest.raises(TypeError, match="expected a PyCapsule, got object"):
        pl.scan_iceberg(metadata_path).collect()


def test_iceberg_plugin_not_installed(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setitem(sys.modules, "polars_iceberg", None)
    with pytest.raises(
        ModuleNotFoundError,
        match=re.escape(
            "Iceberg scan planning with the polars_iceberg plugin requires the "
            "polars_iceberg package; install it with "
            "`pip install --upgrade 'polars_iceberg>=0.1.0'`"
        ),
    ):
        pl.scan_iceberg(metadata_path).collect()


def test_iceberg_plugin_default_planner_uses_plugin(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)
    assert "plan with the polars_iceberg plugin" in capfd.readouterr().err


def test_iceberg_plugin_not_installed_default_planner_warns(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setitem(sys.modules, "polars_iceberg", None)
    with pytest.warns(
        pl.exceptions.PerformanceWarning,
        match=re.escape(
            "install it with `pip install --upgrade 'polars_iceberg>=0.1.0'`. "
            "Planning the Iceberg scan with PyIceberg instead"
        ),
    ):
        assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)
    assert "polars_iceberg plugin" not in capfd.readouterr().err


def test_iceberg_plugin_incompatible_default_planner_warns(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")
    plmonkeypatch.setattr(
        polars_iceberg, "_polars_io_plugin_ids", ("polars.io_plugin.iceberg.v999",)
    )
    with pytest.warns(pl.exceptions.PerformanceWarning, match="is incompatible with"):
        assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)


def test_iceberg_plugin_not_installed_pyiceberg_planner_no_warning(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", "pyiceberg")
    plmonkeypatch.setitem(sys.modules, "polars_iceberg", None)
    # Warnings are errors in the test suite.
    assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)


def _plugin_scan_unsupported(*args: Any, **kwargs: Any) -> Any:
    msg = "polars_iceberg: iceberg: unsupported: equality delete files (x.parquet)"
    raise NotImplementedError(msg)


def test_iceberg_plugin_unsupported_default_planner_falls_back(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setattr(plr, "_iceberg_plugin_scan", _plugin_scan_unsupported)
    # Warnings are errors in the test suite.
    assert_frame_equal(pl.scan_iceberg(metadata_path).collect(), TEST_DF)
    assert (
        "plugin planner unsupported, plan with PyIceberg: polars_iceberg: iceberg: "
        "unsupported: equality delete files" in capfd.readouterr().err
    )


def test_iceberg_plugin_unsupported_plugin_planner_raises(
    metadata_path: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setattr(plr, "_iceberg_plugin_scan", _plugin_scan_unsupported)
    with pytest.raises(NotImplementedError, match="equality delete files"):
        pl.scan_iceberg(metadata_path).collect()


def test_iceberg_plugin_row_index(metadata_path: str) -> None:
    lf = pl.scan_iceberg(metadata_path).with_row_index()
    expected = TEST_DF.with_row_index()
    assert_frame_equal(lf.collect(), expected)
    assert_frame_equal(
        lf.filter(pl.col("a") > 2).collect(), expected.filter(pl.col("a") > 2)
    )
    assert lf.select(pl.len()).collect().item() == TEST_DF.height


def _add_position_deletes(
    tbl: Any,
    deletes: dict[str, list[int]],
    *,
    single_delete_file: bool = False,
    partition: Any = None,
    path_bounds: bool = True,
) -> Any:
    """
    Commit a snapshot adding one position delete file per data file.

    With `single_delete_file`, one delete file holds the deletes of all data files (it
    is then scoped to the partition rather than to a data file). `partition` is the
    partition tuple of the delete files (default: unpartitioned). Without
    `path_bounds`, the delete files have no `file_path` bounds, so they are scoped to
    the partition even when they reference one data file.

    PyIceberg cannot write merge-on-read deletes, so the delete files, the delete
    manifest and the manifest list are written with its low-level writers.
    """
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.manifest import (
        DataFile,
        DataFileContent,
        FileFormat,
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestWriterV2,
        write_manifest_list,
    )
    from pyiceberg.table.refs import SnapshotRefType
    from pyiceberg.table.snapshots import Operation, Snapshot, Summary
    from pyiceberg.table.update import AddSnapshotUpdate, SetSnapshotRefUpdate
    from pyiceberg.typedef import Record

    class DeleteManifestWriter(ManifestWriterV2):  # type: ignore[misc]
        def content(self) -> ManifestContent:
            return ManifestContent.DELETES

        @property
        def _meta(self) -> dict[str, str]:
            return {**super()._meta, "content": "deletes"}

    path_field_id, pos_field_id = 2147483546, 2147483545
    delete_schema = pa.schema(
        [
            pa.field(
                "file_path",
                pa.string(),
                nullable=False,
                metadata={"PARQUET:field_id": str(path_field_id)},
            ),
            pa.field(
                "pos",
                pa.int64(),
                nullable=False,
                metadata={"PARQUET:field_id": str(pos_field_id)},
            ),
        ]
    )

    parent = tbl.current_snapshot()
    snapshot_id = uuid.uuid4().int >> 65
    sequence_number = tbl.metadata.last_sequence_number + 1

    groups = (
        [deletes]
        if single_delete_file
        else [{data_path: positions} for data_path, positions in deletes.items()]
    )
    delete_files = []
    for group in groups:
        path = f"{tbl.location()}/data/delete-{uuid.uuid4()}.parquet"
        local_path = path.removeprefix("file://")
        rows = [(p, pos) for p in sorted(group) for pos in group[p]]
        pq.write_table(
            pa.table(
                {"file_path": [p for p, _ in rows], "pos": [pos for _, pos in rows]},
                schema=delete_schema,
            ),
            local_path,
        )
        data_file = DataFile.from_args(
            content=DataFileContent.POSITION_DELETES,
            file_path=path,
            file_format=FileFormat.PARQUET,
            partition=Record() if partition is None else partition,
            record_count=len(rows),
            file_size_in_bytes=Path(local_path).stat().st_size,
            lower_bounds={path_field_id: min(group).encode()} if path_bounds else {},
            upper_bounds={path_field_id: max(group).encode()} if path_bounds else {},
        )
        data_file.spec_id = tbl.spec().spec_id
        delete_files.append(data_file)

    manifest_path = f"{tbl.location()}/metadata/{uuid.uuid4()}-m0.avro"
    with DeleteManifestWriter(
        spec=tbl.spec(),
        schema=tbl.schema(),
        output_file=tbl.io.new_output(manifest_path),
        snapshot_id=snapshot_id,
        avro_compression="deflate",
    ) as writer:
        for data_file in delete_files:
            writer.add(
                ManifestEntry.from_args(
                    status=ManifestEntryStatus.ADDED,
                    snapshot_id=snapshot_id,
                    data_file=data_file,
                )
            )
    delete_manifest = writer.to_manifest_file()

    list_path = f"{tbl.location()}/metadata/snap-{snapshot_id}-{uuid.uuid4()}.avro"
    with write_manifest_list(
        format_version=2,
        output_file=tbl.io.new_output(list_path),
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent.snapshot_id,
        sequence_number=sequence_number,
        avro_compression="deflate",
    ) as list_writer:
        list_writer.add_manifests([delete_manifest, *parent.manifests(tbl.io)])

    snapshot = Snapshot.model_validate(
        {
            "snapshot-id": snapshot_id,
            "parent-snapshot-id": parent.snapshot_id,
            "sequence-number": sequence_number,
            "manifest-list": list_path,
            "summary": Summary(Operation.DELETE),
            "schema-id": tbl.schema().schema_id,
        }
    )
    set_ref = SetSnapshotRefUpdate.model_validate(
        {"ref-name": "main", "type": SnapshotRefType.BRANCH, "snapshot-id": snapshot_id}
    )
    tbl.catalog.commit_table(
        tbl,
        requirements=(),
        updates=(AddSnapshotUpdate(snapshot=snapshot), set_ref),
    )
    return tbl.catalog.load_table(tbl.name())


def _data_file_paths(tbl: Any) -> list[str]:
    return [task.file.file_path for task in tbl.scan().plan_files()]


@pytest.mark.parametrize("fast_deletion_count", [False, True])
def test_iceberg_plugin_position_deletes(
    tmp_path: Path, fast_deletion_count: bool
) -> None:
    tbl = _new_table(tmp_path)
    tbl.append(TEST_DF.to_arrow())
    paths = _data_file_paths(tbl)
    assert len(paths) == 2

    tbl = _add_position_deletes(tbl, {paths[0]: [0, 2], paths[1]: [4]})

    expected = pl.DataFrame(tbl.scan().to_arrow())
    assert expected.height == 2 * TEST_DF.height - 3

    lf = pl.scan_iceberg(tbl, fast_deletion_count=fast_deletion_count)
    assert_frame_equal(lf.collect(), expected, check_row_order=False)
    assert lf.select(pl.len()).collect().item() == expected.height
    assert_frame_equal(
        lf.filter(pl.col("a") > 2).collect(),
        expected.filter(pl.col("a") > 2),
        check_row_order=False,
    )

    # Deletes from a later snapshot do not apply to an earlier one.
    first = tbl.snapshots()[0].snapshot_id
    assert_frame_equal(pl.scan_iceberg(tbl, snapshot_id=first).collect(), TEST_DF)


@pytest.mark.parametrize("planner", [None, "pyiceberg"])
@pytest.mark.parametrize("remove_data_file", [False, True])
def test_iceberg_plugin_partition_scoped_position_deletes(
    tmp_path: Path,
    planner: str | None,
    remove_data_file: bool,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # Delete files referencing several data files are read by PyIceberg (the plugin
    # raises `NotImplementedError`, the PyIceberg planner falls back).
    if planner is None:
        plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")
    else:
        plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", planner)

    tbl = _new_table(tmp_path)
    tbl.append(TEST_DF.with_columns(pl.col("a") + 10).to_arrow())
    paths = _data_file_paths(tbl)
    assert len(paths) == 2

    # One delete file applying to both data files.
    tbl = _add_position_deletes(
        tbl, {paths[0]: [0, 2], paths[1]: [4]}, single_delete_file=True
    )
    if remove_data_file:
        # The delete file stays live, with deletes of a data file that is not.
        tbl.delete("a < 10")
        assert len(_data_file_paths(tbl)) == 1

    expected = pl.DataFrame(tbl.scan().to_arrow())
    if remove_data_file:
        assert expected.height < TEST_DF.height
    else:
        assert expected.height == 2 * TEST_DF.height - 3

    for fast_deletion_count in [False, True]:
        lf = pl.scan_iceberg(tbl, fast_deletion_count=fast_deletion_count)
        assert_frame_equal(lf.collect(), expected, check_row_order=False)
        assert lf.select(pl.len()).collect().item() == expected.height


def test_iceberg_plugin_partition_scoped_position_deletes_plugin_planner_raises(
    tmp_path: Path,
) -> None:
    tbl = _new_table(tmp_path)
    tbl.append(TEST_DF.to_arrow())
    paths = _data_file_paths(tbl)
    tbl = _add_position_deletes(tbl, {paths[0]: [0]}, path_bounds=False)

    with pytest.raises(
        NotImplementedError,
        match="position delete file not limited to one data file",
    ):
        pl.scan_iceberg(tbl).collect()


@pytest.mark.parametrize("planner", ["plugin", "pyiceberg"])
def test_iceberg_plugin_decimal_partition_deletes_after_precision_widening(
    tmp_path: Path, planner: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    from decimal import Decimal

    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.typedef import Record
    from pyiceberg.types import DecimalType

    plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", planner)

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "d", DecimalType(9, 2)),
        ),
        partition_spec=PartitionSpec(PartitionField(2, 1000, IdentityTransform(), "d")),
    )
    df = pl.DataFrame(
        {"a": [1, 2, 3], "d": pl.Series([Decimal("1.00")] * 3, dtype=pl.Decimal(9, 2))}
    )
    tbl.append(df.to_arrow())
    [path] = _data_file_paths(tbl)

    # The partition value of the data file is 4 bytes, the one of the delete file 5.
    with tbl.update_schema() as update:
        update.update_column("d", DecimalType(10, 2))
    tbl = tbl.catalog.load_table(tbl.name())

    # Matched to the data file by partition only.
    tbl = _add_position_deletes(
        tbl, {path: [1]}, partition=Record(Decimal("1.00")), path_bounds=False
    )

    expected = pl.DataFrame(tbl.scan().to_arrow())
    assert expected.height == 2

    if planner == "plugin":
        # The delete file is matched (so the plugin reports it as unsupported, instead
        # of ignoring it).
        with pytest.raises(NotImplementedError, match="not limited to one data file"):
            pl.scan_iceberg(tbl).collect()
    else:
        assert_frame_equal(
            pl.scan_iceberg(tbl).collect(), expected, check_row_order=False
        )

    plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")
    lf = pl.scan_iceberg(tbl)
    assert_frame_equal(lf.collect(), expected, check_row_order=False)
    assert lf.select(pl.len()).collect().item() == 2


def test_iceberg_plugin_gzip_metadata(metadata_path: str) -> None:
    import gzip

    # As written with `write.metadata.compression-codec=gzip`.
    local_path = Path(metadata_path.removeprefix("file://"))
    gz_path = local_path.with_name(
        local_path.name.removesuffix(".metadata.json") + ".gz.metadata.json"
    )
    gz_path.write_bytes(gzip.compress(local_path.read_bytes()))

    assert_frame_equal(pl.scan_iceberg(str(gz_path)).collect(), TEST_DF)


@pytest.mark.parametrize("planner", [None, "plugin"])
def test_iceberg_plugin_unsupported_avro_codec(
    tmp_path: Path, planner: str | None, plmonkeypatch: PlMonkeyPatch
) -> None:
    if planner is None:
        plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "a", LongType())),
        properties={"write.avro.compression-codec": "bzip2"},
    )
    tbl.append(TEST_DF.select("a").to_arrow())

    if planner == "plugin":
        with pytest.raises(NotImplementedError, match="Avro codec 'bzip2'"):
            pl.scan_iceberg(tbl).collect()
    else:
        # Planned with PyIceberg.
        assert_frame_equal(pl.scan_iceberg(tbl).collect(), TEST_DF.select("a"))


@pytest.mark.parametrize("planner", [None, "plugin"])
def test_iceberg_plugin_custom_file_io(
    tmp_path: Path, planner: str | None, plmonkeypatch: PlMonkeyPatch
) -> None:
    from pyiceberg.io.pyarrow import PyArrowFileIO
    from pyiceberg.table import Table

    if planner is None:
        plmonkeypatch.delenv("POLARS_ICEBERG_PLANNER")

    class CustomFileIO(PyArrowFileIO):  # type: ignore[misc]
        pass

    tbl = _new_table(tmp_path)
    tbl = Table(
        identifier=tbl.name(),
        metadata=tbl.metadata,
        metadata_location=tbl.metadata_location,
        io=CustomFileIO(tbl.io.properties),
        catalog=tbl.catalog,
    )

    if planner == "plugin":
        with pytest.raises(NotImplementedError, match="custom PyIceberg FileIO"):
            pl.scan_iceberg(tbl).collect()
    else:
        # Planned with PyIceberg, through the table's FileIO.
        assert_frame_equal(pl.scan_iceberg(tbl).collect(), TEST_DF)


def test_iceberg_plugin_decimal_initial_default_exponent(tmp_path: Path) -> None:
    from decimal import Decimal

    from pyiceberg.types import DecimalType

    tbl = _new_table(tmp_path)
    with tbl.update_schema() as update:
        update.add_column("x", DecimalType(8, 7), default_value=Decimal("1E-7"))
        update.add_column("z", DecimalType(8, 7), default_value=Decimal("0E-7"))
    tbl = tbl.catalog.load_table(tbl.name())

    expected = pl.DataFrame(tbl.scan().to_arrow())
    assert expected["x"].to_list() == [Decimal("0.0000001")] * TEST_DF.height
    assert_frame_equal(pl.scan_iceberg(tbl).collect(), expected)


def test_iceberg_plugin_prunes_buckets(tmp_path: Path) -> None:
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import BucketTransform

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()), NestedField(2, "b", StringType())
        ),
        partition_spec=PartitionSpec(
            PartitionField(1, 1000, BucketTransform(16), "a_bucket"),
            PartitionField(2, 1001, BucketTransform(4), "b_bucket"),
        ),
    )
    df = pl.DataFrame({"a": range(100), "b": [str(i % 7) for i in range(100)]})
    tbl.append(df.to_arrow())
    n_files = len(_data_file_paths(tbl))

    def check(predicate: pl.Expr, iceberg_filter: str) -> None:
        lf = pl.scan_iceberg(tbl).filter(predicate)
        assert_frame_equal(lf.collect(), df.filter(predicate), check_row_order=False)
        # Same candidate files as PyIceberg.
        expected = len(list(tbl.scan(row_filter=iceberg_filter).plan_files()))
        assert _num_sources(lf) == expected < n_files

    check(pl.col("a") == 50, "a = 50")
    check(pl.col("a").is_in([1, 2, 50]), "a in (1, 2, 50)")
    check(pl.col("b") == "3", "b = '3'")
    check((pl.col("a") == 50) & (pl.col("b") == "1"), "a = 50 and b = '1'")


def _num_sources(lf: pl.LazyFrame) -> int:
    plan = lf.explain()
    if m := re.search(r"\.\.\. (\d+) other sources", plan):
        return 1 + int(m.group(1))
    return plan.count(".parquet")


def test_iceberg_plugin_time_travel_filter_uses_snapshot_schema(tmp_path: Path) -> None:
    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "x", LongType()),
            NestedField(2, "y", LongType()),
        ),
    )
    df = pl.DataFrame({"x": [1, 2, 3], "y": [100, 200, 300]})
    tbl.append(df.to_arrow())
    first = tbl.current_snapshot().snapshot_id

    # Swap the names: in the current schema, `x` is field 2.
    with tbl.update_schema() as update:
        update.rename_column("x", "tmp")
    with tbl.update_schema() as update:
        update.rename_column("y", "x")
    with tbl.update_schema() as update:
        update.rename_column("tmp", "y")
    tbl.append(pl.DataFrame({"y": [4], "x": [400]}).to_arrow())

    # `x` refers to field 1 in the snapshot's schema; binding it to the current schema
    # (field 2, with bounds [100, 300]) would prune the only file.
    predicate = pl.col("x") == 2
    assert_frame_equal(
        pl.scan_iceberg(tbl, snapshot_id=first).filter(predicate).collect(),
        df.filter(predicate),
    )


def test_iceberg_plugin_prunes_partitions(tmp_path: Path) -> None:
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import DayTransform, IdentityTransform
    from pyiceberg.types import TimestampType

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "a", LongType()),
            NestedField(2, "b", StringType()),
            NestedField(3, "ts", TimestampType()),
        ),
        partition_spec=PartitionSpec(
            PartitionField(2, 1000, IdentityTransform(), "b"),
            PartitionField(3, 1001, DayTransform(), "ts_day"),
        ),
    )
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["x", "y", "x", "z"],
            "ts": pl.Series(
                ["2025-01-01 10:00", "2025-01-02 10:00", "2025-01-03 10:00", None]
            ).str.to_datetime(time_unit="us"),
        }
    )
    tbl.append(df.to_arrow())
    assert len(_data_file_paths(tbl)) == 4

    def check(predicate: pl.Expr, n_files: int) -> None:
        lf = pl.scan_iceberg(tbl).filter(predicate)
        assert_frame_equal(lf.collect(), df.filter(predicate), check_row_order=False)
        plan = lf.explain()
        assert plan.count(".parquet") == n_files, plan

    check(pl.col("b") == "x", 2)
    check(pl.col("b").is_in(["y", "z"]), 2)
    check(pl.col("b") != "x", 2)
    check(pl.col("ts") >= pl.datetime(2025, 1, 2, 12), 1)
    check(pl.col("ts").is_null(), 1)
    check((pl.col("b") == "x") & (pl.col("ts") < pl.datetime(2025, 1, 2)), 1)


def test_iceberg_plugin_nan_is_not_pruned(tmp_path: Path) -> None:
    from pyiceberg.types import DoubleType

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(NestedField(1, "v", DoubleType())),
    )
    # One file with only NaN, one with regular values.
    tbl.append(pl.DataFrame({"v": [float("nan")]}).to_arrow())
    tbl.append(pl.DataFrame({"v": [0.0, 1.0]}).to_arrow())

    df = pl.DataFrame({"v": [float("nan"), 0.0, 1.0]})
    for predicate in [
        pl.col("v") > 5.0,
        pl.col("v") >= 1.0,
        pl.col("v") < 0.5,
        pl.col("v").is_nan(),
        pl.col("v").is_not_nan(),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


@pytest.mark.parametrize("planner", ["plugin", "pyiceberg"])
def test_iceberg_plugin_float_literals_are_exact(
    tmp_path: Path, planner: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    from decimal import Decimal

    from pyiceberg.types import DecimalType, DoubleType, FloatType

    plmonkeypatch.setenv("POLARS_ICEBERG_PLANNER", planner)

    tbl, _ = new_iceberg_table(
        tmp_path,
        schema=IcebergSchema(
            NestedField(1, "d", DoubleType()),
            NestedField(2, "f", FloatType()),
            NestedField(3, "dec", DecimalType(20, 6)),
        ),
    )
    df = pl.DataFrame(
        {
            # Values that lose precision when rounded to a few significant digits,
            # or when parsed slightly inexactly.
            "d": [0.123456789012345, 123456789.98765433, 1.0],
            "f": pl.Series([0.1, 3.4e38, float("inf")], dtype=pl.Float32),
            "dec": pl.Series(
                [Decimal("12345678901234.123456"), Decimal("1"), Decimal("2")],
                dtype=pl.Decimal(20, 6),
            ),
        }
    )
    # One file per row, so that each row is subject to pruning on its own.
    for i in range(df.height):
        tbl.append(df.slice(i, 1).to_arrow().cast(tbl.schema().as_arrow()))

    for predicate in [
        pl.col("d").is_in([0.123456789012345]),
        pl.col("d").is_in([123456789.98765433]),
        pl.col("d") == 123456789.98765433,
        pl.col("f").is_in(pl.Series([0.1], dtype=pl.Float32).implode()),
        pl.col("f").is_in(pl.Series([3.4e38], dtype=pl.Float32).implode()),
        pl.col("f") == float("inf"),
        pl.col("f").is_in(pl.Series([float("inf")], dtype=pl.Float32).implode()),
        pl.col("dec").is_in(
            pl.Series(
                [Decimal("12345678901234.123456")], dtype=pl.Decimal(20, 6)
            ).implode()
        ),
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


_CACHE_STATS = re.compile(
    r"metadata file cache: hits: (\d+), misses: (\d+), cached bytes: (\d+)"
)


@pytest.fixture
def _fresh_metadata_file_cache() -> Any:
    from polars.io.iceberg._cache import reset_metadata_file_cache

    reset_metadata_file_cache()
    yield
    reset_metadata_file_cache()


def _scan_cache_stats(
    capfd: pytest.CaptureFixture[str], lf: pl.LazyFrame, expected: pl.DataFrame
) -> tuple[int, int, int] | None:
    """Collect `lf` and return its metadata file cache (hits, misses, cached bytes)."""
    capfd.readouterr()
    assert_frame_equal(lf.collect(), expected, check_row_order=False)
    m = _CACHE_STATS.findall(capfd.readouterr().err)
    assert len(m) <= 1
    return tuple(map(int, m[0])) if m else None  # type: ignore[return-value]


@pytest.mark.usefixtures("_fresh_metadata_file_cache")
def test_iceberg_plugin_metadata_file_cache(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    from polars.io.iceberg._cache import get_metadata_file_cache

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    for i in range(3):
        tbl.append(pl.DataFrame({"a": [i]}).to_arrow())

    expected = pl.DataFrame({"a": [0, 1, 2]})

    # Table metadata, manifest list and 3 manifests.
    hits, misses, cached_bytes = _scan_cache_stats(
        capfd, pl.scan_iceberg(tbl), expected
    )  # type: ignore[misc]
    assert (hits, misses) == (0, 5)
    assert cached_bytes > 0
    assert cached_bytes == get_metadata_file_cache().plugin_cache().total_bytes
    assert len(get_metadata_file_cache().plugin_cache()) == 5

    stats = _scan_cache_stats(capfd, pl.scan_iceberg(tbl), expected)
    assert stats == (5, 0, cached_bytes)

    # After a commit, only the new files are fetched.
    tbl.append(pl.DataFrame({"a": [3]}).to_arrow())
    stats = _scan_cache_stats(
        capfd, pl.scan_iceberg(tbl), pl.DataFrame({"a": [0, 1, 2, 3]})
    )
    assert stats is not None
    assert stats[:2] == (3, 3)

    # Dropped with the PyIceberg planner's cache.
    from polars.io.iceberg._cache import reset_metadata_file_cache

    reset_metadata_file_cache()
    stats = _scan_cache_stats(
        capfd, pl.scan_iceberg(tbl), pl.DataFrame({"a": [0, 1, 2, 3]})
    )
    assert stats is not None
    assert stats[:2] == (0, 6)


@pytest.mark.usefixtures("_fresh_metadata_file_cache")
def test_iceberg_plugin_metadata_file_cache_incremental(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    tbl, _ = new_iceberg_table(
        tmp_path, schema=IcebergSchema(NestedField(1, "a", LongType()))
    )
    for i in range(3):
        tbl.append(pl.DataFrame({"a": [i]}).to_arrow())

    snapshots = tbl.snapshots()

    def scan() -> pl.LazyFrame:
        return pl.scan_iceberg(
            tbl,
            from_snapshot_id_exclusive=snapshots[0].snapshot_id,
            to_snapshot_id_inclusive=snapshots[-1].snapshot_id,
        )

    expected = pl.DataFrame({"a": [1, 2]})
    first = _scan_cache_stats(capfd, scan(), expected)
    assert first is not None
    assert first[1] > 0

    second = _scan_cache_stats(capfd, scan(), expected)
    assert second is not None
    assert second[:2] == (first[0] + first[1], 0)


@pytest.mark.usefixtures("_fresh_metadata_file_cache")
def test_iceberg_plugin_metadata_file_cache_disabled(
    table: Any, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    from polars.io.iceberg._cache import reset_metadata_file_cache

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_ICEBERG_METADATA_CACHE_MB", "0")
    reset_metadata_file_cache()

    for _ in range(2):
        assert _scan_cache_stats(capfd, pl.scan_iceberg(table), TEST_DF) is None


@pytest.mark.usefixtures("_fresh_metadata_file_cache")
def test_iceberg_plugin_metadata_file_cache_scoped_to_storage_options(
    table: Any, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    metadata_path = table.metadata_location

    stats = _scan_cache_stats(capfd, pl.scan_iceberg(metadata_path), TEST_DF)
    assert stats is not None
    assert stats[:2] == (0, 3)

    # Other storage options: not shared.
    storage_options = {"max_retries": 3}
    stats = _scan_cache_stats(
        capfd,
        pl.scan_iceberg(metadata_path, storage_options=storage_options),
        TEST_DF,
    )
    assert stats is not None
    assert stats[:2] == (0, 3)

    stats = _scan_cache_stats(
        capfd,
        pl.scan_iceberg(metadata_path, storage_options=storage_options),
        TEST_DF,
    )
    assert stats is not None
    assert stats[:2] == (3, 0)


def test_iceberg_plugin_storage_scope() -> None:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    from polars.io.iceberg._cache import plugin_storage_scope

    io = PyArrowFileIO({"s3.region": "us-east-1"})
    scope = plugin_storage_scope(io, None)
    assert scope is not None
    assert plugin_storage_scope(io, {}) == scope
    assert (
        plugin_storage_scope(PyArrowFileIO({"s3.region": "us-east-1"}), None) == scope
    )
    assert (
        plugin_storage_scope(PyArrowFileIO({"s3.region": "eu-west-1"}), None) != scope
    )

    with_options = plugin_storage_scope(io, {"s3.region": "eu-west-1"})
    assert with_options is not None
    assert with_options != scope

    # Options that cannot be fingerprinted are not cached.
    assert plugin_storage_scope(io, {"key": object()}) is None
    assert plugin_storage_scope(PyArrowFileIO({"key": object()}), None) is None  # type: ignore[dict-item]
