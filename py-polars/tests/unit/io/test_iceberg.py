# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
import io
import itertools
import os
import pickle
import sys
import warnings
import zoneinfo
from datetime import date, datetime
from decimal import Decimal as D
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import pyiceberg
import pytest
from pyiceberg.expressions import literal
from pyiceberg.partitioning import (
    BucketTransform,
    IdentityTransform,
    PartitionField,
    PartitionSpec,
)
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    FixedType,
    FloatType,
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
    TimestamptzType,
    TimeType,
    UUIDType,
)

import polars as pl
from polars._utils.various import parse_version
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import IcebergDataset, _NativeIcebergScanData
from polars.io.iceberg._utils import (
    _convert_predicate,
    _normalize_windows_iceberg_file_uri,
    _to_ast,
    try_convert_pyarrow_predicate,
)
from polars.testing import assert_frame_equal
from tests.unit.io.conftest import normalize_path_separator_pl

if TYPE_CHECKING:
    from pyiceberg.table import Table

    from tests.conftest import PlMonkeyPatch

    # Mypy does not understand the constructors and we can't construct the inputs
    # explicitly since they are abstract base classes.
    And = Any
    EqualTo = Any
    GreaterThan = Any
    GreaterThanOrEqual = Any
    In = Any
    IsNull = Any
    LessThan = Any
    LessThanOrEqual = Any
    Not = Any
    Or = Any
    Reference = Any
else:
    from pyiceberg.expressions import (
        And,
        EqualTo,
        GreaterThan,
        GreaterThanOrEqual,
        In,
        IsNull,
        LessThan,
        LessThanOrEqual,
        Not,
        Or,
        Reference,
    )


with warnings.catch_warnings():
    # Upstream issue at https://github.com/apache/iceberg-python/issues/2648.
    warnings.simplefilter("ignore", pydantic.warnings.PydanticDeprecatedSince212)
    # Upstream issue at https://github.com/apache/iceberg-python/issues/2849.
    warnings.simplefilter("ignore", DeprecationWarning)
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.io.pyarrow import schema_to_pyarrow


def new_pl_iceberg_dataset(source: str | Table) -> IcebergDataset:
    from pyiceberg.table import Table

    return IcebergDataset(
        table_=NoPickleOption(source if isinstance(source, Table) else None),
        metadata_path_=source if not isinstance(source, Table) else None,
        snapshot_id=None,
        iceberg_storage_properties=None,
        reader_override=None,
        use_metadata_statistics=True,
        fast_deletion_count=False,
        use_pyiceberg_filter=True,
    )


# PyIceberg on Windows uses `file://C:/` rather than `file:///C:/`.
def format_file_uri_iceberg(absolute_local_path: str | Path) -> str:
    absolute_local_path = str(absolute_local_path)

    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha()
        assert absolute_local_path[1] == ":"
        p = absolute_local_path.replace("\\", "/")
        return f"file://{p}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"


@pytest.fixture
def iceberg_path(io_files_path: Path) -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path(__file__).parent.resolve()

    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")  # noqa: PTH211

    iceberg_path = io_files_path / "iceberg-table" / "metadata" / "v2.metadata.json"
    return format_file_uri_iceberg(f"{iceberg_path.resolve()}")


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
@pytest.mark.ci_only
class TestIcebergScanIO:
    """Test coverage for `iceberg` scan ops."""

    def test_scan_iceberg_plain(self, iceberg_path: str) -> None:
        q = pl.scan_iceberg(iceberg_path)
        assert len(q.collect()) == 3
        assert q.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_snapshot_id(self, iceberg_path: str) -> None:
        q = pl.scan_iceberg(iceberg_path, snapshot_id=7051579356916758811)
        assert len(q.collect()) == 3
        assert q.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_snapshot_id_not_found(self, iceberg_path: str) -> None:
        with pytest.raises(ValueError, match="snapshot ID not found"):
            pl.scan_iceberg(iceberg_path, snapshot_id=1234567890).collect()

    def test_scan_iceberg_filter_on_partition(self, iceberg_path: str) -> None:
        ts1 = datetime(2023, 3, 1, 18, 15)
        ts2 = datetime(2023, 3, 1, 19, 25)
        ts3 = datetime(2023, 3, 2, 22, 0)

        lf = pl.scan_iceberg(iceberg_path)

        res = lf.filter(pl.col("ts") >= ts2)
        assert len(res.collect()) == 2

        res = lf.filter(pl.col("ts") > ts2).select(pl.col("id"))
        assert res.collect().rows() == [(3,)]

        res = lf.filter(pl.col("ts") <= ts2).select("id", "ts")
        assert res.collect().rows(named=True) == [
            {"id": 1, "ts": ts1},
            {"id": 2, "ts": ts2},
        ]

        res = lf.filter(pl.col("ts") > ts3)
        assert len(res.collect()) == 0

        for constraint in (
            (pl.col("ts") == ts1) | (pl.col("ts") == ts3),
            pl.col("ts").is_in([ts1, ts3]),
        ):
            res = lf.filter(constraint).select("id")
            assert res.collect().rows() == [(1,), (3,)]

    def test_scan_iceberg_filter_on_column(self, iceberg_path: str) -> None:
        lf = pl.scan_iceberg(iceberg_path)
        res = lf.filter(pl.col("id") < 2)
        assert res.collect().rows() == [(1, "1", datetime(2023, 3, 1, 18, 15))]

        res = lf.filter(pl.col("id") == 2)
        assert res.collect().rows() == [(2, "2", datetime(2023, 3, 1, 19, 25))]

        res = lf.filter(pl.col("id").is_in([1, 3]))
        assert res.collect().rows() == [
            (1, "1", datetime(2023, 3, 1, 18, 15)),
            (3, "3", datetime(2023, 3, 2, 22, 0)),
        ]


@pytest.mark.ci_only
class TestIcebergExpressions:
    """Test coverage for `iceberg` expressions comprehension."""

    def test_is_null_expression(self) -> None:
        expr = _to_ast("(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == IsNull("id")

    def test_is_not_null_expression(self) -> None:
        expr = _to_ast("~(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == Not(IsNull("id"))

    def test_isin_expression(self) -> None:
        expr = _to_ast("(pa.compute.field('id')).isin([1,2,3])")
        assert _convert_predicate(expr) == In(
            "id", {literal(1), literal(2), literal(3)}
        )

    def test_parse_combined_expression(self) -> None:
        expr = _to_ast(
            "(((pa.compute.field('str') == '2') & (pa.compute.field('id') > 10)) | (pa.compute.field('id')).isin([1,2,3]))"
        )
        assert _convert_predicate(expr) == Or(
            left=And(
                left=EqualTo(term=Reference(name="str"), literal=literal("2")),
                right=GreaterThan(term="id", literal=literal(10)),
            ),
            right=In("id", {literal(1), literal(2), literal(3)}),
        )

    def test_parse_gt(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")

    def test_parse_gteq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")

    def test_parse_eq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
        assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")

    def test_parse_lt(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
        assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")

    def test_parse_lteq(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
        assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")

    def test_compare_boolean(self) -> None:
        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(True))")
        assert _convert_predicate(expr) == EqualTo("ts", True)

        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(False))")
        assert _convert_predicate(expr) == EqualTo("ts", False)

    def test_bare_boolean_field(self) -> None:
        expr = try_convert_pyarrow_predicate("pa.compute.field('is_active')")
        assert expr == EqualTo("is_active", True)

    def test_bare_boolean_field_negated(self) -> None:
        expr = try_convert_pyarrow_predicate("~pa.compute.field('is_active')")
        assert expr == Not(EqualTo("is_active", True))


@pytest.mark.write_disk
def test_iceberg_dataset_does_not_pickle_table_object(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "row_index", IntegerType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    df = pl.DataFrame(
        {"row_index": [0, 1, 2, 3, 4]},
        schema={"row_index": pl.Int32},
    )

    df.write_iceberg(tbl, mode="append")

    dataset = new_pl_iceberg_dataset(tbl)

    assert dataset.table_.get() is not None
    dataset = pickle.loads(pickle.dumps(dataset))
    assert dataset.table_.get() is None

    assert_frame_equal(dataset.to_dataset_scan()[0].collect(), df)  # type: ignore[index]


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings("ignore:Delete operation did not match any records")
@pytest.mark.filterwarnings(
    "ignore:Iceberg does not have a dictionary type. <class 'pyarrow.lib.DictionaryType'> will be inferred as large_string on read."
)
def test_write_iceberg(df: pl.DataFrame, tmp_path: Path) -> None:
    # time64[ns] type is currently not supported in pyiceberg.
    # https://github.com/apache/iceberg-python/issues/1169
    df = df.drop("time", "cat", "enum")

    # in-memory catalog
    catalog = SqlCatalog(
        "default", uri="sqlite:///:memory:", warehouse=format_file_uri_iceberg(tmp_path)
    )
    catalog.create_namespace("foo")
    table = catalog.create_table(
        "foo.bar",
        schema=df.to_arrow().schema,
    )

    df.write_iceberg(table, mode="overwrite")
    actual = pl.scan_iceberg(table).collect()

    assert_frame_equal(df, actual)

    # append on top of already written data, expecting twice the data
    df.write_iceberg(table, mode="append")
    # double the `df` by vertically stacking the dataframe on top of itself
    expected = df.vstack(df)
    actual = pl.scan_iceberg(table).collect()
    assert_frame_equal(expected, actual, check_dtypes=False)


@pytest.mark.write_disk
def test_scan_iceberg_row_index_renamed(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "row_index", IntegerType()),
            NestedField(2, "file_path", StringType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"row_index": [0, 1, 2, 3, 4], "file_path": None},
        schema={"row_index": pl.Int32, "file_path": pl.String},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.rename_column("row_index", "row_index_in_file")
        sch.rename_column("file_path", "file_path_in_file")

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]
    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={
            "row_index_in_file": pl.Int32,
            "file_path_in_file": pl.String,
        },
        _column_mapping=(
            "iceberg-column-mapping",
            new_pl_iceberg_dataset(tbl).arrow_schema(),
        ),
        include_file_paths="file_path",
        row_index_name="row_index",
        row_index_offset=3,
    )

    assert_frame_equal(
        q.collect().with_columns(normalize_path_separator_pl(pl.col("file_path"))),
        pl.DataFrame(
            {
                "row_index": [3, 4, 5, 6, 7],
                "row_index_in_file": [0, 1, 2, 3, 4],
                "file_path_in_file": None,
                "file_path": file_paths[0],
            },
            schema={
                "row_index": pl.get_index_type(),
                "row_index_in_file": pl.Int32,
                "file_path_in_file": pl.String,
                "file_path": pl.String,
            },
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_polars_storage_options_keys(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "row_index", IntegerType()),
            NestedField(2, "file_path", StringType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"row_index": [0, 1, 2, 3, 4], "file_path": None},
        schema={"row_index": pl.Int32, "file_path": pl.String},
    ).write_iceberg(tbl, mode="append")

    capfd.readouterr()

    pl.scan_iceberg(
        tbl,
        storage_options={
            "file_cache_ttl": 7,
            "max_retries": 3,
            "retry_timeout_ms": 9873,
            "retry_init_backoff_ms": 9874,
            "retry_max_backoff_ms": 9875,
            "retry_base_multiplier": 3.14159,
        },
    ).collect()

    capture = capfd.readouterr().err

    assert "file_cache_ttl: 7" in capture

    assert (
        """\
max_retries: Some(3), \
retry_timeout: Some(9.873s), \
retry_init_backoff: Some(9.874s), \
retry_max_backoff: Some(9.875s), \
retry_base_multiplier: Some(TotalOrdWrap(3.14159)) }"""
        in capture
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("reader_override", ["pyiceberg", "native"])
def test_scan_iceberg_collect_without_version_scans_latest(
    tmp_path: Path,
    reader_override: str,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )

    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", LongType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    q = pl.scan_iceberg(tbl, reader_override=reader_override)  # type: ignore[arg-type]

    assert_frame_equal(q.collect(), pl.DataFrame(schema={"a": pl.Int64}))

    pl.DataFrame({"a": 1}).write_iceberg(tbl, mode="append")

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1}))

    snapshot = tbl.current_snapshot()
    assert snapshot is not None
    snapshot_id = snapshot.snapshot_id

    q_with_id = pl.scan_iceberg(
        tbl,
        reader_override=reader_override,  # type: ignore[arg-type]
        snapshot_id=snapshot_id,
    )

    assert_frame_equal(q_with_id.collect(), pl.DataFrame({"a": 1}))

    pl.DataFrame({"a": 2}).write_iceberg(tbl, mode="append")

    assert_frame_equal(q.collect(), pl.DataFrame({"a": [2, 1]}))

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    assert_frame_equal(q_with_id.collect(), pl.DataFrame({"a": 1}))

    assert (
        "IcebergDataset: to_dataset_scan(): early return (snapshot_id_key = "
        in capfd.readouterr().err
    )


@pytest.mark.write_disk
def test_scan_iceberg_extra_columns(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", IntegerType()),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"a": [0, 1, 2, 3, 4]},
        schema={"a": pl.Int32},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.delete_column("a")
        sch.add_column("a", IntegerType())

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]

    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Int32},
        _column_mapping=(
            "iceberg-column-mapping",
            new_pl_iceberg_dataset(tbl).arrow_schema(),
        ),
    )

    # The original column is considered an extra column despite having the same
    # name as the physical ID does not match.

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="extra column in file outside of expected schema: a",
    ):
        q.collect()

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Int32},
        _column_mapping=(
            "iceberg-column-mapping",
            new_pl_iceberg_dataset(tbl).arrow_schema(),
        ),
        extra_columns="ignore",
        missing_columns="insert",
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": [None, None, None, None, None],
            },
            schema={"a": pl.Int32},
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_extra_struct_fields(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", StructType(NestedField(1, "a", IntegerType()))),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame(
        {"a": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]},
        schema={"a": pl.Struct({"a": pl.Int32})},
    ).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.delete_column(("a", "a"))
        sch.add_column(("a", "a"), IntegerType())

    file_paths = [
        _normalize_windows_iceberg_file_uri(x.file.file_path)
        for x in tbl.scan().plan_files()
    ]

    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Struct({"a": pl.Int32})},
        _column_mapping=(
            "iceberg-column-mapping",
            new_pl_iceberg_dataset(tbl).arrow_schema(),
        ),
    )

    # The original column is considered an extra column despite having the same
    # name as the physical ID does not match.

    with pytest.raises(
        pl.exceptions.SchemaError,
        match="encountered extra struct field: a",
    ):
        q.collect()

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Struct({"a": pl.Int32})},
        _column_mapping=(
            "iceberg-column-mapping",
            new_pl_iceberg_dataset(tbl).arrow_schema(),
        ),
        cast_options=pl.ScanCastOptions(
            extra_struct_fields="ignore", missing_struct_fields="insert"
        ),
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            {
                "a": [
                    {"a": None},
                    {"a": None},
                    {"a": None},
                    {"a": None},
                    {"a": None},
                ],
            },
            schema={"a": pl.Struct({"a": pl.Int32})},
        ),
    )


@pytest.mark.write_disk
def test_scan_iceberg_column_deletion(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(1, "a", StructType(NestedField(0, "inner", StringType())))
        ),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame({"a": [{"inner": "A"}]}).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.delete_column("a").add_column(
            "a", StructType(NestedField(0, "inner", StringType()))
        )

    pl.DataFrame({"a": [{"inner": "A"}]}).write_iceberg(tbl, mode="append")

    expect = pl.DataFrame({"a": [{"inner": "A"}, None]})

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )


@pytest.mark.write_disk
def test_scan_iceberg_nested_column_cast_deletion_rename(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=ListType(
                    element_id=next_field_id(),
                    element=StructType(
                        NestedField(
                            field_id=next_field_id(),
                            name="field_1",
                            field_type=MapType(
                                key_id=next_field_id(),
                                key_type=ListType(
                                    element_id=next_field_id(), element=TimestampType(), element_required=False
                                ),
                                value_id=next_field_id(),
                                value_type=ListType(
                                    element_id=next_field_id(), element=IntegerType(), element_required=False
                                ),
                                value_required=False,
                            ),
                            required=False,
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_3", field_type=StringType(), required=False
                        ),
                    ),
                    element_required=False,
                ),
                required=False,
            ),
            NestedField(field_id=next_field_id(), name="column_2", field_type=StringType(), required=False),
            NestedField(
                field_id=next_field_id(),
                name="column_3",
                field_type=MapType(
                    key_id=next_field_id(),
                    key_type=StructType(
                        NestedField(
                            field_id=next_field_id(), name="field_1", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_3", field_type=IntegerType(), required=False
                        ),
                    ),
                    value_id=next_field_id(),
                    value_type=StructType(
                        NestedField(
                            field_id=next_field_id(), name="field_1", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False
                        ),
                        NestedField(
                            field_id=next_field_id(), name="field_3", field_type=IntegerType(), required=False
                        ),
                    ),
                    value_required=False,
                ),
                required=False,
            ),
        ),
    )  # fmt: skip

    tbl = catalog.load_table("namespace.table")

    df_dict = {
        "column_1": [
            [
                {
                    "field_1": [
                        {"key": [datetime(2025, 1, 1), None], "value": [1, 2, None]},
                        {"key": [datetime(2025, 1, 1), None], "value": None},
                    ],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [
                {
                    "field_1": [{"key": [datetime(2025, 1, 1), None], "value": None}],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [{"field_1": [], "field_2": None, "field_3": None}],
            [None],
            [],
        ],
        "column_2": ["1", "2", "3", "4", None],
        "column_3": [
            [
                {
                    "key": {"field_1": 1, "field_2": 2, "field_3": 3},
                    "value": {"field_1": 7, "field_2": 8, "field_3": 9},
                }
            ],
            [
                {
                    "key": {"field_1": 1, "field_2": 2, "field_3": 3},
                    "value": {"field_1": 7, "field_2": 8, "field_3": 9},
                }
            ],
            [
                {
                    "key": {"field_1": None, "field_2": None, "field_3": None},
                    "value": {"field_1": None, "field_2": None, "field_3": None},
                }
            ],
            [
                {
                    "key": {"field_1": None, "field_2": None, "field_3": None},
                    "value": None,
                }
            ],
            [],
        ],
    }

    df = pl.DataFrame(
        df_dict,
        schema={
            "column_1": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.List(
                            pl.Struct({"key": pl.List(pl.Datetime("us")), "value": pl.List(pl.Int32)})
                        ),
                        "field_2": pl.Int32,
                        "field_3": pl.String,
                    }
                )
            ),
            "column_2": pl.String,
            "column_3": pl.List(
                pl.Struct(
                    {
                        "key": pl.Struct({"field_1": pl.Int32, "field_2": pl.Int32, "field_3": pl.Int32}),
                        "value": pl.Struct({"field_1": pl.Int32, "field_2": pl.Int32, "field_3": pl.Int32}),
                    }
                )
            ),
        },
    )  # fmt: skip

    # The Iceberg table schema has a `Map` type, whereas the polars DataFrame
    # stores `list[struct{..}]` - directly using `write_iceberg()` causes the
    # following error:
    # * ValueError: PyArrow table contains more columns:
    #   column_1.element.field_1.element
    # We workaround this by constructing a pyarrow table an arrow schema.
    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.large_list(
                        pa.struct(
                            [
                                (
                                    "field_1",
                                    pa.map_(pa.large_list(pa.timestamp("us")), pa.large_list(pa.int32())),
                                ),
                                ("field_2", pa.int32()),
                                ("field_3", pa.string()),
                            ]
                        )
                    ),
                ),
                ("column_2", pa.string()),
                (
                    "column_3",
                    pa.map_(
                        pa.struct([("field_1", pa.int32()), ("field_2", pa.int32()), ("field_3", pa.int32())]),
                        pa.struct([("field_1", pa.int32()), ("field_2", pa.int32()), ("field_3", pa.int32())]),
                    ),
                ),
            ]
        ),
    )  # fmt: skip

    assert_frame_equal(pl.DataFrame(arrow_tbl), df)

    tbl.append(arrow_tbl)

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), df)
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), df)

    # Change schema
    # Note: Iceberg doesn't allow modifying the "key" part of the Map type.

    # Promote types
    with tbl.update_schema() as sch:
        sch.update_column(("column_1", "field_2"), LongType())
        sch.update_column(("column_3", "value", "field_1"), LongType())
        sch.update_column(("column_3", "value", "field_2"), LongType())
        sch.update_column(("column_3", "value", "field_3"), LongType())

    # Delete/Rename:
    # * Delete `*_2` fields
    # * Rename:
    #   * `{x}_1` -> `{x}_2`
    #   * `{x}_3` -> `{x}_1`
    #     * And move the field position to 1st

    # Delete `*_2` fields/columns.
    with tbl.update_schema() as sch:
        sch.delete_column("column_2")
        sch.delete_column(("column_3", "value", "field_2"))
        sch.delete_column(("column_1", "field_2"))

    # Shift nested fields in `column_1`
    with tbl.update_schema() as sch:
        sch.rename_column(("column_1", "field_1"), "field_2")

    with tbl.update_schema() as sch:
        sch.rename_column(("column_1", "field_3"), "field_1")

    with tbl.update_schema() as sch:
        sch.move_first(("column_1", "field_1"))

    # Shift nested fields in `column_2`
    with tbl.update_schema() as sch:
        sch.rename_column(("column_3", "value", "field_1"), "field_2")

    with tbl.update_schema() as sch:
        sch.rename_column(("column_3", "value", "field_3"), "field_1")

    with tbl.update_schema() as sch:
        sch.move_first(("column_3", "value", "field_1"))

    # Shift top-level columns
    with tbl.update_schema() as sch:
        sch.rename_column("column_1", "column_2")

    with tbl.update_schema() as sch:
        sch.rename_column("column_3", "column_1")

    with tbl.update_schema() as sch:
        sch.move_first("column_1")

    expect = pl.DataFrame(
        {
            "column_2": [
                [
                    {
                        "field_2": [
                            {"key": [datetime(2025, 1, 1, 0, 0), None], "value": [1, 2, None]},
                            {"key": [datetime(2025, 1, 1), None], "value": None},
                        ],
                        "field_1": "F3",
                    }
                ],
                [{"field_2": [{"key": [datetime(2025, 1, 1, 0, 0), None], "value": None}], "field_1": "F3"}],
                [{"field_2": [], "field_1": None}],
                [None],
                [],
            ],
            "column_1": [
                [{"key": {"field_1": 1, "field_2": 2, "field_3": 3}, "value": {"field_2": 7, "field_1": 9}}],
                [{"key": {"field_1": 1, "field_2": 2, "field_3": 3}, "value": {"field_2": 7, "field_1": 9}}],
                [
                    {
                        "key": {"field_1": None, "field_2": None, "field_3": None},
                        "value": {"field_2": None, "field_1": None},
                    }
                ],
                [{"key": {"field_1": None, "field_2": None, "field_3": None}, "value": None}],
                [],
            ],
        },
        schema={
            "column_1": pl.List(
                pl.Struct(
                    {
                        "key": pl.Struct({"field_1": pl.Int32, "field_2": pl.Int32, "field_3": pl.Int32}),
                        "value": pl.Struct({"field_1": pl.Int64, "field_2": pl.Int64}),
                    }
                )
            ),
            "column_2": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.String,
                        "field_2": pl.List(
                            pl.Struct(
                                {
                                    "key": pl.List(pl.Datetime(time_unit="us", time_zone=None)),
                                    "value": pl.List(pl.Int32),
                                }
                            )
                        ),
                    }
                )
            ),
        },
    )  # fmt: skip

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), expect
    )
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), expect)


@pytest.mark.write_disk
@pytest.mark.xfail(
    reason="""\
[Upstream Issue]
PyIceberg writes NULL as empty lists into the Parquet file.
* Issue on Polars repo - https://github.com/pola-rs/polars/issues/23715
* Issue on PyIceberg repo - https://github.com/apache/iceberg-python/issues/2246
"""
)
def test_scan_iceberg_nulls_multiple_nesting(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=ListType(
                    element_id=next_field_id(),
                    element=StructType(
                        NestedField(
                            field_id=next_field_id(),
                            name="field_1",
                            field_type=ListType(
                                element_id=next_field_id(),
                                element=StructType(
                                    NestedField(field_id=next_field_id(), name="key", field_type=ListType(
                                        element_id=next_field_id(),
                                        element=TimestampType(),
                                        element_required=False,
                                    ), required=True),
                                    NestedField(field_id=next_field_id(), name="value", field_type=ListType(
                                        element_id=next_field_id(),
                                        element=IntegerType(),
                                        element_required=False,
                                    ), required=False),
                                ),
                                element_required=False
                            ),
                            required=False,
                        ),
                        NestedField(field_id=next_field_id(), name="field_2", field_type=IntegerType(), required=False),
                        NestedField(field_id=next_field_id(), name="field_3", field_type=StringType(), required=False),
                    ),
                    element_required=False,
                ),
                required=False,
            ),
        ),
    )  # fmt: skip

    tbl = catalog.load_table("namespace.table")

    df_dict = {
        "column_1": [
            [
                {
                    "field_1": [
                        {"key": [datetime(2025, 1, 1), None], "value": [1, 2, None]}
                    ],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [
                {
                    "field_1": [{"key": [datetime(2025, 1, 1), None], "value": None}],
                    "field_2": 7,
                    "field_3": "F3",
                }
            ],
            [{"field_1": None, "field_2": None, "field_3": None}],
            [None],
            None,
        ],
    }

    df = pl.DataFrame(
        df_dict,
        schema={
            "column_1": pl.List(
                pl.Struct(
                    {
                        "field_1": pl.List(
                            pl.Struct(
                                {
                                    "key": pl.List(pl.Datetime("us")),
                                    "value": pl.List(pl.Int32),
                                }
                            )
                        ),
                        "field_2": pl.Int32,
                        "field_3": pl.String,
                    }
                )
            ),
        },
    )

    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.large_list(
                        pa.struct(
                            [
                                (
                                    "field_1",
                                    pa.large_list(
                                        pa.struct(
                                            [
                                                pa.field(
                                                    "key",
                                                    pa.large_list(pa.timestamp("us")),
                                                    nullable=False,
                                                ),
                                                ("value", pa.large_list(pa.int32())),
                                            ]
                                        )
                                    ),
                                ),
                                ("field_2", pa.int32()),
                                ("field_3", pa.string()),
                            ]
                        )
                    ),
                )
            ]
        ),
    )

    assert_frame_equal(pl.DataFrame(arrow_tbl), df)

    tbl.append(arrow_tbl)

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), df)
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), df)


@pytest.mark.write_disk
def test_scan_iceberg_nulls_nested(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=ListType(
                    element_id=next_field_id(),
                    element=IntegerType(),
                    element_required=False,
                ),
                required=False,
            ),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    df = pl.DataFrame(
        {
            "column_1": [
                [1, 2],
                [None],
                None,
            ],
        },
        schema={
            "column_1": pl.List(pl.Int32),
        },
    )

    df_dict = df.to_dict(as_series=False)

    assert_frame_equal(pl.DataFrame(df_dict, schema=df.schema), df)

    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=pa.schema(
            [
                (
                    "column_1",
                    pa.large_list(pa.int32()),
                )
            ]
        ),
    )

    assert_frame_equal(pl.DataFrame(arrow_tbl), df)

    tbl.append(arrow_tbl)

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(), df)
    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), df)


@pytest.mark.write_disk
def test_scan_iceberg_parquet_prefilter_with_column_mapping(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count())

    catalog.create_table(
        "namespace.table",
        IcebergSchema(
            NestedField(
                field_id=next_field_id(),
                name="column_1",
                field_type=StringType(),
                required=False,
            ),
            NestedField(
                field_id=next_field_id(),
                name="column_2",
                field_type=IntegerType(),
                required=False,
            ),
            NestedField(
                field_id=next_field_id(),
                name="column_3",
                field_type=StringType(),
                required=False,
            ),
        ),
    )

    tbl = catalog.load_table("namespace.table")

    df = pl.DataFrame(
        {
            "column_1": ["A", "B", "C", "D", "E", "F"],
            "column_2": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int32),
            "column_3": ["P", "Q", "R", "S", "T", "U"],
        }
    )

    df.slice(0, 3).write_iceberg(tbl, mode="append")
    df.slice(3).write_iceberg(tbl, mode="append")

    with tbl.update_schema() as sch:
        sch.update_column("column_2", LongType())

    with tbl.update_schema() as sch:
        sch.delete_column("column_1")

    with tbl.update_schema() as sch:
        sch.rename_column("column_3", "column_1")

    with tbl.update_schema() as sch:
        sch.rename_column("column_2", "column_3")

    with tbl.update_schema() as sch:
        sch.move_first("column_1")

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect().sort("column_3"),
        pl.DataFrame(
            {
                "column_1": ["P", "Q", "R", "S", "T", "U"],
                "column_3": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int64),
            }
        ),
    )

    # Upstream issue - PyIceberg filter does not handle schema evolution
    with pytest.raises(Exception, match="unpack requires a buffer of 8 bytes"):
        pl.scan_iceberg(
            tbl, reader_override="native", use_pyiceberg_filter=True
        ).filter(pl.col("column_3") == 5).collect()

    q = pl.scan_iceberg(
        tbl, reader_override="native", use_pyiceberg_filter=False
    ).filter(pl.col("column_3") == 5)

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        cx.setenv("POLARS_FORCE_EMPTY_READER_CAPABILITIES", "0")
        capfd.readouterr()
        out = q.collect()
        capture = capfd.readouterr().err

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "column_1": ["T"],
                "column_3": pl.Series([5], dtype=pl.Int64),
            }
        ),
    )

    # First file
    assert "Source filter mask initialization via table statistics" in capture
    assert "Predicate pushdown allows skipping 1 / 2 files" in capture
    # Second file
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 1 / 1 row groups" in capture
    )
    assert (
        "[ParquetFileReader]: Pre-filtered decode enabled (1 live, 1 non-live)"
        in capture
    )


# Note: This test also generally covers primitive type round-tripping.
@pytest.mark.parametrize("test_uuid", [True, False])
@pytest.mark.write_disk
def test_fill_missing_fields_with_identity_partition_values(
    test_uuid: bool, tmp_path: Path
) -> None:
    from datetime import time

    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    min_version = parse_version(pyiceberg.__version__) >= (0, 10, 0)

    test_decimal_and_fixed = min_version
    test_uuid = test_uuid and min_version

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(next_field_id(), "height_provider", IntegerType()),
        NestedField(next_field_id(), "BooleanType", BooleanType()),
        NestedField(next_field_id(), "IntegerType", IntegerType()),
        NestedField(next_field_id(), "LongType", LongType()),
        NestedField(next_field_id(), "FloatType", FloatType()),
        NestedField(next_field_id(), "DoubleType", DoubleType()),
        NestedField(next_field_id(), "DateType", DateType()),
        NestedField(next_field_id(), "TimeType", TimeType()),
        NestedField(next_field_id(), "TimestampType", TimestampType()),
        NestedField(next_field_id(), "TimestamptzType", TimestamptzType()),
        NestedField(next_field_id(), "StringType", StringType()),
        NestedField(next_field_id(), "BinaryType", BinaryType()),
        *(
            [
                NestedField(next_field_id(), "DecimalType", DecimalType(18, 2)),
                NestedField(next_field_id(), "FixedType", FixedType(1)),
            ]
            if test_decimal_and_fixed
            else []
        ),
        *([NestedField(next_field_id(), "UUIDType", UUIDType())] if test_uuid else []),
    )

    arrow_tbl = pa.Table.from_pydict(
        {
            "height_provider": [1],
            "BooleanType": [True],
            "IntegerType": [1],
            "LongType": [1],
            "FloatType": [1.0],
            "DoubleType": [1.0],
            "DateType": [date(2025, 1, 1)],
            "TimeType": [time(11, 30)],
            "TimestampType": [datetime(2025, 1, 1)],
            "TimestamptzType": [datetime(2025, 1, 1)],
            "StringType": ["A"],
            "BinaryType": [b"A"],
            **(
                {"DecimalType": [D("1.0")], "FixedType": [b"A"]}
                if test_decimal_and_fixed
                else {}
            ),
            **({"UUIDType": [b"0000111100001111"]} if test_uuid else {}),
        },
        schema=schema_to_pyarrow(iceberg_schema, include_field_ids=False),
    )

    tbl = catalog.create_table(
        "namespace.table",
        iceberg_schema,
        partition_spec=PartitionSpec(
            # We have this to offset the indices
            PartitionField(
                iceberg_schema.fields[0].field_id, 0, BucketTransform(32), "bucket"
            ),
            *(
                PartitionField(field.field_id, 0, IdentityTransform(), field.name)
                for field in iceberg_schema.fields[1:]
            ),
        ),
    )

    if test_uuid:
        # Note: If this starts working one day we can include it in tests.
        with pytest.raises(
            pa.ArrowNotImplementedError,
            match=r"Keys of type extension<arrow\.uuid>",
        ):
            tbl.append(arrow_tbl)

        return

    tbl.append(arrow_tbl)

    expect = pl.DataFrame(
        [
            pl.Series('height_provider', [1], dtype=pl.Int32),
            pl.Series('BooleanType', [True], dtype=pl.Boolean),
            pl.Series('IntegerType', [1], dtype=pl.Int32),
            pl.Series('LongType', [1], dtype=pl.Int64),
            pl.Series('FloatType', [1.0], dtype=pl.Float32),
            pl.Series('DoubleType', [1.0], dtype=pl.Float64),
            pl.Series('DateType', [date(2025, 1, 1)], dtype=pl.Date),
            pl.Series('TimeType', [time(11, 30)], dtype=pl.Time),
            pl.Series('TimestampType', [datetime(2025, 1, 1, 0, 0)], dtype=pl.Datetime(time_unit='us', time_zone=None)),
            pl.Series('TimestamptzType', [datetime(2025, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))], dtype=pl.Datetime(time_unit='us', time_zone='UTC')),
            pl.Series('StringType', ['A'], dtype=pl.String),
            pl.Series('BinaryType', [b'A'], dtype=pl.Binary),
            *(
                [
                    pl.Series('DecimalType', [D('1.00')], dtype=pl.Decimal(precision=18, scale=2)),
                    pl.Series('FixedType', [b'A'], dtype=pl.Binary),
                ]
                if test_decimal_and_fixed
                else []
            ),
        ]
    )  # fmt: skip

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )

    dfiles = [*tbl.scan().plan_files()]

    assert len(dfiles) == 1

    p = dfiles[0].file.file_path.removeprefix("file://")

    # Drop every column except 'height_provider'
    pq.write_table(
        pa.Table.from_pydict(
            {"height_provider": [1]},
            schema=schema_to_pyarrow(iceberg_schema.select("height_provider")),
        ),
        p,
    )

    out = pl.DataFrame(tbl.scan().to_arrow())

    assert_frame_equal(
        out.select(pl.col(c).cast(dt) for c, dt in expect.schema.items()),
        expect,
    )

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), expect)


@pytest.mark.write_disk
def test_fill_missing_fields_with_identity_partition_values_nested(
    tmp_path: Path,
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(next_field_id(), "height_provider", IntegerType()),
        NestedField(
            next_field_id(),
            "struct_1",
            StructType(
                NestedField(
                    next_field_id(),
                    "struct_2",
                    StructType(NestedField(2001, "field_1", LongType())),
                )
            ),
        ),
    )

    tbl = catalog.create_table(
        "namespace.table",
        iceberg_schema,
        partition_spec=PartitionSpec(
            PartitionField(2001, 0, IdentityTransform(), "field_1")
        ),
    )

    pl.DataFrame(
        {"height_provider": [0], "struct_1": [{"struct_2": {"field_1": 300}}]},
        schema=pl.Schema(iceberg_schema.as_arrow()),
    ).write_iceberg(tbl, mode="append")

    expect = pl.DataFrame(
        [
            pl.Series("height_provider", [0], dtype=pl.Int32),
            pl.Series(
                "struct_1",
                [{"struct_2": {"field_1": 300}}],
                dtype=pl.Struct({"struct_2": pl.Struct({"field_1": pl.Int64})}),
            ),
        ]
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )

    # Note: We will still match even if the partition field is renamed, since it still
    # has the same source field ID.
    with tbl.update_spec() as spu:
        spu.rename_field("field_1", "AAA")

    pl.DataFrame(
        {"height_provider": [None], "struct_1": [{"struct_2": {"field_1": 301}}]},
        schema=pl.Schema(iceberg_schema.as_arrow()),
    ).write_iceberg(tbl, mode="append")

    with tbl.update_spec() as spu:
        spu.remove_field("AAA")

    pl.DataFrame(
        {"height_provider": [None], "struct_1": [{"struct_2": {"field_1": 302}}]},
        schema=pl.Schema(iceberg_schema.as_arrow()),
    ).write_iceberg(tbl, mode="append")

    for i, data_file in enumerate(tbl.scan().plan_files()):
        p = data_file.file.file_path.removeprefix("file://")

        pq.write_table(
            pa.Table.from_pydict(
                {"height_provider": [i]},
                schema=schema_to_pyarrow(iceberg_schema.select("height_provider")),
            ),
            p,
        )

    # Deleting partitions only takes effect for newly added files.
    expect = pl.DataFrame(
        [
            pl.Series("height_provider", [0, 1, 2], dtype=pl.Int32),
            pl.Series(
                "struct_1",
                [
                    None,
                    {"struct_2": {"field_1": 301}},
                    {"struct_2": {"field_1": 300}},
                ],
                dtype=pl.Struct({"struct_2": pl.Struct({"field_1": pl.Int64})}),
            ),
        ]
    )

    assert_frame_equal(pl.scan_iceberg(tbl, reader_override="native").collect(), expect)
    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").select("struct_1").collect(),
        expect.select("struct_1"),
    )


@pytest.mark.write_disk
def test_scan_iceberg_min_max_statistics_filter(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import datetime

    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    test_decimal_and_fixed = parse_version(pyiceberg.__version__) >= (0, 10, 0)

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(next_field_id(), "height_provider", IntegerType()),
        NestedField(next_field_id(), "BooleanType", BooleanType()),
        NestedField(next_field_id(), "IntegerType", IntegerType()),
        NestedField(next_field_id(), "LongType", LongType()),
        NestedField(next_field_id(), "FloatType", FloatType()),
        NestedField(next_field_id(), "DoubleType", DoubleType()),
        NestedField(next_field_id(), "DateType", DateType()),
        NestedField(next_field_id(), "TimeType", TimeType()),
        NestedField(next_field_id(), "TimestampType", TimestampType()),
        NestedField(next_field_id(), "TimestamptzType", TimestamptzType()),
        NestedField(next_field_id(), "StringType", StringType()),
        NestedField(next_field_id(), "BinaryType", BinaryType()),
        *(
            [
                NestedField(next_field_id(), "DecimalType", DecimalType(18, 2)),
                NestedField(
                    next_field_id(), "DecimalTypeLargeValue", DecimalType(38, 0)
                ),
                NestedField(
                    next_field_id(), "DecimalTypeLargeNegativeValue", DecimalType(38, 0)
                ),
                NestedField(next_field_id(), "FixedType", FixedType(1)),
            ]
            if test_decimal_and_fixed
            else []
        ),
    )

    pl_schema = pl.Schema(
        {
            "height_provider": pl.Int32(),
            "BooleanType": pl.Boolean(),
            "IntegerType": pl.Int32(),
            "LongType": pl.Int64(),
            "FloatType": pl.Float32(),
            "DoubleType": pl.Float64(),
            "DateType": pl.Date(),
            "TimeType": pl.Time(),
            "TimestampType": pl.Datetime(time_unit="us", time_zone=None),
            "TimestamptzType": pl.Datetime(time_unit="us", time_zone="UTC"),
            "StringType": pl.String(),
            "BinaryType": pl.Binary(),
            "DecimalType": pl.Decimal(precision=18, scale=2),
            "DecimalTypeLargeValue": pl.Decimal(precision=38, scale=0),
            "DecimalTypeLargeNegativeValue": pl.Decimal(precision=38, scale=0),
            "FixedType": pl.Binary(),
        }
    )

    df_dict = {
        "height_provider": [1],
        "BooleanType": [True],
        "IntegerType": [1],
        "LongType": [1],
        "FloatType": [1.0],
        "DoubleType": [1.0],
        "DateType": [datetime.date(2025, 1, 1)],
        "TimeType": [datetime.time(11, 30)],
        "TimestampType": [datetime.datetime(2025, 1, 1)],
        "TimestamptzType": [datetime.datetime(2025, 1, 1)],
        "StringType": ["A"],
        "BinaryType": [b"A"],
        **(
            {
                "DecimalType": [D("1.00")],
                # This helps ensure loads are done with the correct endianness.
                "DecimalTypeLargeValue": [D("73377733337777733333377777773333333377")],
                "DecimalTypeLargeNegativeValue": [
                    D("-73377733337777733333377777773333333377")
                ],
                "FixedType": [b"A"],
            }
            if test_decimal_and_fixed
            else {}
        ),
    }

    arrow_tbl = pa.Table.from_pydict(
        df_dict,
        schema=schema_to_pyarrow(iceberg_schema, include_field_ids=False),
    )

    tbl = catalog.create_table(
        "namespace.table",
        iceberg_schema,
        partition_spec=PartitionSpec(
            # We have this to offset the indices
            PartitionField(
                iceberg_schema.fields[0].field_id, 0, BucketTransform(32), "bucket"
            ),
            *(
                PartitionField(field.field_id, 0, IdentityTransform(), field.name)
                for field in iceberg_schema.fields[1:]
            ),
        ),
    )

    tbl.append(arrow_tbl)

    expect = pl.DataFrame(df_dict, schema=pl_schema)

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="pyiceberg").collect(),
        expect,
    )

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )

    # Begin inspecting statistics

    scan_data = new_pl_iceberg_dataset(tbl)._to_dataset_scan_impl()

    assert isinstance(scan_data, _NativeIcebergScanData)
    assert scan_data.statistics_loader is None
    assert scan_data.min_max_statistics is None

    scan_data = new_pl_iceberg_dataset(tbl)._to_dataset_scan_impl(
        filter_columns=["height_provider"]
    )

    assert isinstance(scan_data, _NativeIcebergScanData)
    assert scan_data.min_max_statistics is not None

    min_max_values = scan_data.min_max_statistics.with_columns(
        pl.all().cast(pl.String)
    ).transpose(include_header=True)

    assert_frame_equal(
        min_max_values,
        pl.DataFrame(
            [
                ("len", "1"),
                ("height_provider_nc", "0"),
                ("height_provider_min", "1"),
                ("height_provider_max", "1"),
            ],
            orient="row",
            schema=min_max_values.schema,
        ),
    )

    scan_data = new_pl_iceberg_dataset(tbl)._to_dataset_scan_impl(
        filter_columns=pl_schema.names()
    )

    assert isinstance(scan_data, _NativeIcebergScanData)
    assert scan_data.statistics_loader is not None

    non_coalesced_min_max_values = (
        scan_data.statistics_loader.finish(len(scan_data.sources), {})
        .with_columns(pl.all().cast(pl.String))
        .transpose(include_header=True)
    )

    assert_frame_equal(
        non_coalesced_min_max_values,
        pl.DataFrame(
            [
                ("len", "1"),
                ("height_provider_nc", "0"),
                ("height_provider_min", "1"),
                ("height_provider_max", "1"),
                ("BooleanType_nc", "0", "1"),
                ("BooleanType_min", "true"),
                ("BooleanType_max", "true"),
                ("IntegerType_nc", "0", "1"),
                ("IntegerType_min", "1"),
                ("IntegerType_max", "1"),
                ("LongType_nc", "0", "1"),
                ("LongType_min", "1"),
                ("LongType_max", "1"),
                ("FloatType_nc", "0", "1"),
                ("FloatType_min", None),
                ("FloatType_max", None),
                ("DoubleType_nc", "0", "1"),
                ("DoubleType_min", None),
                ("DoubleType_max", None),
                ("DateType_nc", "0", "1"),
                ("DateType_min", "2025-01-01"),
                ("DateType_max", "2025-01-01"),
                ("TimeType_nc", "0", "1"),
                ("TimeType_min", "11:30:00"),
                ("TimeType_max", "11:30:00"),
                ("TimestampType_nc", "0", "1"),
                ("TimestampType_min", "2025-01-01 00:00:00.000000"),
                ("TimestampType_max", "2025-01-01 00:00:00.000000"),
                ("TimestamptzType_nc", "0", "1"),
                ("TimestamptzType_min", "2025-01-01 00:00:00.000000+00:00"),
                ("TimestamptzType_max", "2025-01-01 00:00:00.000000+00:00"),
                ("StringType_nc", "0"),
                ("StringType_min", "A"),
                ("StringType_max", "A"),
                ("BinaryType_nc", "0"),
                ("BinaryType_min", "A"),
                ("BinaryType_max", "A"),
                ("DecimalType_nc", "0"),
                ("DecimalType_min", "1.00"),
                ("DecimalType_max", "1.00"),
                ("DecimalTypeLargeValue_nc", "0", "1"),
                ("DecimalTypeLargeValue_min", "73377733337777733333377777773333333377"),
                ("DecimalTypeLargeValue_max", "73377733337777733333377777773333333377"),
                ("DecimalTypeLargeNegativeValue_nc", "0"),
                (
                    "DecimalTypeLargeNegativeValue_min",
                    "-73377733337777733333377777773333333377",
                ),
                (
                    "DecimalTypeLargeNegativeValue_max",
                    "-73377733337777733333377777773333333377",
                ),
                ("FixedType_nc", "0"),
                ("FixedType_min", "A"),
                ("FixedType_max", "A"),
            ],
            orient="row",
            schema=non_coalesced_min_max_values.schema,
        ),
    )

    assert scan_data.min_max_statistics is not None

    coalesced_min_max_values = scan_data.min_max_statistics.with_columns(
        pl.all().cast(pl.String)
    ).transpose(include_header=True)

    coalesced_ne_non_coalesced = pl.concat(
        [
            non_coalesced_min_max_values.select(
                pl.struct(pl.all()).alias("non_coalesced")
            ),
            coalesced_min_max_values.select(pl.struct(pl.all()).alias("coalesced")),
        ],
        how="horizontal",
    ).filter(pl.first() != pl.last())

    # Float statistics are available after coalescing from an identity partition field.
    assert_frame_equal(
        coalesced_ne_non_coalesced,
        pl.DataFrame(
            [
                (
                    {"column": "FloatType_min", "column_0": None},
                    {"column": "FloatType_min", "column_0": "1.0"},
                ),
                (
                    {"column": "FloatType_max", "column_0": None},
                    {"column": "FloatType_max", "column_0": "1.0"},
                ),
                (
                    {"column": "DoubleType_min", "column_0": None},
                    {"column": "DoubleType_min", "column_0": "1.0"},
                ),
                (
                    {"column": "DoubleType_max", "column_0": None},
                    {"column": "DoubleType_max", "column_0": "1.0"},
                ),
            ],
            orient="row",
            schema=coalesced_ne_non_coalesced.schema,
        ),
    )

    dfiles = [x.file.file_path for x in tbl.scan().plan_files()]
    assert len(dfiles) == 1

    Path(dfiles[0].removeprefix("file://")).unlink()

    expect_file_not_found_err = pytest.raises(
        OSError,
        match=(
            "The system cannot find the file specified"
            if sys.platform == "win32"
            else "No such file or directory"
        ),
    )

    with expect_file_not_found_err:
        pl.scan_iceberg(tbl, reader_override="native").collect()

    iceberg_table_filter_seen = False

    def ensure_filter_skips_file(filter_expr: pl.Expr) -> None:
        nonlocal iceberg_table_filter_seen

        with plmonkeypatch.context() as cx:
            cx.setenv("POLARS_VERBOSE", "1")
            capfd.readouterr()

            assert_frame_equal(
                pl.scan_iceberg(tbl, reader_override="native").filter(filter_expr),
                pl.LazyFrame(schema=pl_schema),
            )

            capture = capfd.readouterr().err

            if "iceberg_table_filter: Some(<redacted>)" in capture:
                assert "allows skipping 0 / 0 files" in capture
                assert (
                    "apply_scan_predicate_to_scan_ir: PredicateFileSkip { no_residual_predicate: false, original_len: 0 }"
                    in capture
                )

                # Scanning with pyiceberg can also skip the file if the predicate
                # can be converted.
                assert_frame_equal(
                    pl.scan_iceberg(tbl, reader_override="pyiceberg").filter(
                        filter_expr
                    ),
                    pl.LazyFrame(schema=pl_schema),
                )

                iceberg_table_filter_seen = True
            else:
                assert "allows skipping 1 / 1 files" in capture
                assert (
                    "apply_scan_predicate_to_scan_ir: PredicateFileSkip { no_residual_predicate: false, original_len: 1 }"
                    in capture
                )

            capfd.readouterr()

            assert_frame_equal(
                pl.scan_iceberg(tbl, reader_override="native")
                .with_row_index()
                .filter(filter_expr),
                pl.LazyFrame(schema=pl_schema).with_row_index(),
            )

            capture = capfd.readouterr().err

            assert "iceberg_table_filter: Some(<redacted>)" not in capture

    # Check different operators
    ensure_filter_skips_file(pl.col("IntegerType") > 1)
    ensure_filter_skips_file(pl.col("IntegerType") != 1)
    ensure_filter_skips_file(pl.col("IntegerType").is_in([0]))

    # Ensure `use_metadata_statistics=False` does not skip based on statistics
    with expect_file_not_found_err:
        pl.scan_iceberg(
            tbl,
            reader_override="native",
            use_metadata_statistics=False,
        ).filter(pl.col("IntegerType") > 1).collect()

    with expect_file_not_found_err:
        pickle.loads(
            pickle.dumps(
                pl.scan_iceberg(
                    tbl,
                    reader_override="native",
                    use_metadata_statistics=False,
                ).filter(pl.col("IntegerType") > 1)
            )
        ).collect()

    # Check different types
    ensure_filter_skips_file(pl.col("BooleanType") < True)
    ensure_filter_skips_file(pl.col("IntegerType") < 1)
    ensure_filter_skips_file(pl.col("LongType") < 1)
    ensure_filter_skips_file(pl.col("FloatType") < 1.0)
    ensure_filter_skips_file(pl.col("DoubleType") < 1.0)
    ensure_filter_skips_file(pl.col("DateType") < datetime.date(2025, 1, 1))
    ensure_filter_skips_file(pl.col("TimeType") < datetime.time(11, 30))
    ensure_filter_skips_file(pl.col("TimestampType") < datetime.datetime(2025, 1, 1))
    ensure_filter_skips_file(
        pl.col("TimestamptzType")
        < pl.lit(datetime.datetime(2025, 1, 1), dtype=pl.Datetime("ms", "UTC"))
    )
    ensure_filter_skips_file(pl.col("StringType") < "A")
    ensure_filter_skips_file(pl.col("BinaryType") < b"A")
    ensure_filter_skips_file(pl.col("DecimalType") < D("1.00"))
    ensure_filter_skips_file(
        pl.col("DecimalTypeLargeValue") < D("73377733337777733333377777773333333377")
    )
    ensure_filter_skips_file(
        pl.col("DecimalTypeLargeNegativeValue")
        < D("-73377733337777733333377777773333333377")
    )
    ensure_filter_skips_file(pl.col("FixedType") < b"A")

    # Check row index. It should have a null_count statistic column of 0.
    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native")
        .with_row_index()
        .filter(pl.col("index").is_null()),
        pl.LazyFrame(schema={"index": pl.get_index_type(), **pl_schema}),
    )

    assert iceberg_table_filter_seen


@pytest.mark.write_disk
def test_scan_iceberg_categorical_24140(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    next_field_id = partial(next, itertools.count(1))

    iceberg_schema = IcebergSchema(
        NestedField(
            next_field_id(),
            "values",
            StringType(),
        ),
    )

    tbl = catalog.create_table("namespace.table", iceberg_schema)

    df = pl.DataFrame(
        {"values": "A"},
        schema={"values": pl.Categorical()},
    )

    arrow_tbl = df.to_arrow()

    arrow_type = arrow_tbl.schema.field("values").type
    assert arrow_type.index_type == pa.uint32()
    assert arrow_type.value_type == pa.large_string()

    tbl.append(arrow_tbl)

    expect = pl.DataFrame({"values": "A"}, schema={"values": pl.String})

    assert_frame_equal(
        pl.scan_iceberg(tbl, reader_override="native").collect(),
        expect,
    )


@pytest.mark.write_disk
def test_scan_iceberg_fast_count(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("namespace")

    catalog.create_table(
        "namespace.table",
        IcebergSchema(NestedField(1, "a", LongType())),
    )

    tbl = catalog.load_table("namespace.table")

    pl.DataFrame({"a": [0, 1, 2, 3, 4]}).write_iceberg(tbl, mode="append")

    assert (
        pl.scan_iceberg(tbl, reader_override="native", use_metadata_statistics=True)
        .select(pl.len())
        .collect()
        .item()
        == 5
    )

    assert (
        pl.scan_iceberg(tbl, reader_override="native", use_metadata_statistics=True)
        .filter(pl.col("a") <= 2)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    assert (
        pl.scan_iceberg(tbl, reader_override="native", use_metadata_statistics=True)
        .head(3)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    assert (
        pl.scan_iceberg(tbl, reader_override="native", use_metadata_statistics=True)
        .slice(1, 3)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    dfiles = [*tbl.scan().plan_files()]

    assert len(dfiles) == 1

    p = dfiles[0].file.file_path.removeprefix("file://")

    # Overwrite the data file with one that has a different number of rows
    pq.write_table(
        pa.Table.from_pydict(
            {"a": [0, 1, 2]},
            schema=schema_to_pyarrow(tbl.schema()),
        ),
        p,
    )

    # `use_metadata_statistics=False` should disable sourcing the row count from
    # Iceberg metadata.
    assert (
        pl.scan_iceberg(tbl, reader_override="native", use_metadata_statistics=False)
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    assert (
        pickle.loads(
            pickle.dumps(
                pl.scan_iceberg(
                    tbl, reader_override="native", use_metadata_statistics=False
                ).select(pl.len())
            )
        )
        .collect()
        .item()
        == 3
    )

    Path(p).unlink()

    with pytest.raises(
        OSError,
        match=(
            "The system cannot find the file specified"
            if sys.platform == "win32"
            else "No such file or directory"
        ),
    ):
        pl.scan_iceberg(tbl, reader_override="native").collect()

    # `select(len())` should be able to return the result from the Iceberg metadata
    # without looking at the underlying data files.
    assert (
        pl.scan_iceberg(tbl, reader_override="native").select(pl.len()).collect().item()
        == 5
    )


def test_scan_iceberg_idxsize_limit() -> None:
    if isinstance(pl.get_index_type(), pl.UInt64):
        assert (
            pl.scan_parquet([b""], schema={}, _row_count=(1 << 32, 0))
            .select(pl.len())
            .collect()
            .item()
            == 1 << 32
        )

        return

    f = io.BytesIO()

    pl.DataFrame({"x": 1}).write_parquet(f)

    q = pl.scan_parquet([f.getvalue()], schema={"x": pl.Int64}, _row_count=(1 << 32, 0))

    assert_frame_equal(q.collect(), pl.DataFrame({"x": 1}))

    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"row count \(4294967296\) exceeded maximum supported of 4294967295.*Consider installing 'polars\[rt64\]'.",
    ):
        q.select(pl.len()).collect()


@pytest.mark.write_disk
def test_iceberg_filter_bool_26474(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "test", uri="sqlite:///:memory:", warehouse=format_file_uri_iceberg(tmp_path)
    )
    catalog.create_namespace("default")
    tbl = catalog.create_table(
        "default.test",
        IcebergSchema(
            NestedField(1, "id", LongType()),
            NestedField(2, "foo", BooleanType()),
        ),
    )

    schema = {"id": pl.Int64, "foo": pl.Boolean}

    dfs = [
        pl.DataFrame({"id": [1], "foo": [True]}, schema=schema),
        pl.DataFrame({"id": [2], "foo": [False]}, schema=schema),
        pl.DataFrame({"id": [3], "foo": [None]}, schema=schema),
    ]

    for df in dfs:
        df.write_iceberg(tbl, mode="append")

    assert sum(1 for _ in tbl.scan().plan_files()) == 3

    dfs_concat = pl.concat(dfs)

    for predicate in [
        pl.col("foo"),
        ~pl.col("foo"),
        pl.col("foo") & pl.col("foo"),
        pl.col("foo") | pl.col("foo"),
        pl.col("foo") ^ pl.col("foo"),
        pl.col("foo") & ~pl.col("foo"),
        pl.col("foo") | ~pl.col("foo"),
        pl.col("foo") ^ pl.col("foo"),
        pl.col("foo") & pl.col("foo") | pl.col("foo"),
        pl.col("foo") | pl.col("foo") & pl.col("foo"),
        pl.col("foo") == True,  # noqa: E712
        pl.col("foo") == False,  # noqa: E712
    ]:
        assert_frame_equal(
            pl.scan_iceberg(tbl).filter(predicate).collect(),
            dfs_concat.filter(predicate),
            check_row_order=False,
        )
