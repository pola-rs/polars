# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
import itertools
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.types import (
    IntegerType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
    TimestampType,
)

import polars as pl
from polars.io.iceberg._utils import _convert_predicate, _to_ast
from polars.io.iceberg.dataset import IcebergDataset
from polars.testing import assert_frame_equal


@pytest.fixture
def iceberg_path(io_files_path: Path) -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path(__file__).parent.resolve()

    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")

    iceberg_path = io_files_path / "iceberg-table" / "metadata" / "v2.metadata.json"
    return f"file://{iceberg_path.resolve()}"


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
@pytest.mark.ci_only
class TestIcebergScanIO:
    """Test coverage for `iceberg` scan ops."""

    def test_scan_iceberg_plain(self, iceberg_path: str) -> None:
        df = pl.scan_iceberg(iceberg_path)
        assert len(df.collect()) == 3
        assert df.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_snapshot_id(self, iceberg_path: str) -> None:
        df = pl.scan_iceberg(iceberg_path, snapshot_id=7051579356916758811)
        assert len(df.collect()) == 3
        assert df.collect_schema() == {
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
        from pyiceberg.expressions import IsNull

        expr = _to_ast("(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == IsNull("id")

    def test_is_not_null_expression(self) -> None:
        from pyiceberg.expressions import IsNull, Not

        expr = _to_ast("~(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == Not(IsNull("id"))

    def test_isin_expression(self) -> None:
        from pyiceberg.expressions import In, literal

        expr = _to_ast("(pa.compute.field('id')).isin([1,2,3])")
        assert _convert_predicate(expr) == In(
            "id", {literal(1), literal(2), literal(3)}
        )

    def test_parse_combined_expression(self) -> None:
        from pyiceberg.expressions import (
            And,
            EqualTo,
            GreaterThan,
            In,
            Or,
            Reference,
            literal,
        )

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
        from pyiceberg.expressions import GreaterThan

        expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")

    def test_parse_gteq(self) -> None:
        from pyiceberg.expressions import GreaterThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")

    def test_parse_eq(self) -> None:
        from pyiceberg.expressions import EqualTo

        expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
        assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")

    def test_parse_lt(self) -> None:
        from pyiceberg.expressions import LessThan

        expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
        assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")

    def test_parse_lteq(self) -> None:
        from pyiceberg.expressions import LessThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
        assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")

    def test_compare_boolean(self) -> None:
        from pyiceberg.expressions import EqualTo

        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(True))")
        assert _convert_predicate(expr) == EqualTo("ts", True)

        expr = _to_ast("(pa.compute.field('ts') == pa.compute.scalar(False))")
        assert _convert_predicate(expr) == EqualTo("ts", False)


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
        "default", uri="sqlite:///:memory:", warehouse=f"file://{tmp_path}"
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
        warehouse=f"file://{tmp_path}",
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

    file_paths = [x.file.file_path for x in tbl.scan().plan_files()]
    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={
            "row_index_in_file": pl.Int32,
            "file_path_in_file": pl.String,
        },
        _column_mapping=("iceberg-column-mapping", IcebergDataset(tbl).arrow_schema()),
        include_file_paths="file_path",
        row_index_name="row_index",
        row_index_offset=3,
    )

    assert_frame_equal(
        q.collect().with_columns(
            # To pass Windows CI
            pl.col("file_path").map_elements(
                lambda x: str(Path(x).resolve()),
                return_dtype=pl.String,
            )
        ),
        pl.DataFrame(
            {
                "row_index": [3, 4, 5, 6, 7],
                "row_index_in_file": [0, 1, 2, 3, 4],
                "file_path_in_file": None,
                "file_path": str(Path(file_paths[0]).resolve()),
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
def test_scan_iceberg_extra_columns(tmp_path: Path) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=f"file://{tmp_path}",
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

    file_paths = [x.file.file_path for x in tbl.scan().plan_files()]
    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Int32},
        _column_mapping=("iceberg-column-mapping", IcebergDataset(tbl).arrow_schema()),
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
        _column_mapping=("iceberg-column-mapping", IcebergDataset(tbl).arrow_schema()),
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
        warehouse=f"file://{tmp_path}",
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

    file_paths = [x.file.file_path for x in tbl.scan().plan_files()]
    assert len(file_paths) == 1

    q = pl.scan_parquet(
        file_paths,
        schema={"a": pl.Struct({"a": pl.Int32})},
        _column_mapping=("iceberg-column-mapping", IcebergDataset(tbl).arrow_schema()),
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
        _column_mapping=("iceberg-column-mapping", IcebergDataset(tbl).arrow_schema()),
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
        warehouse=f"file://{tmp_path}",
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
        warehouse=f"file://{tmp_path}",
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
        warehouse=f"file://{tmp_path}",
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
        warehouse=f"file://{tmp_path}",
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


def test_scan_iceberg_parquet_prefilter_with_column_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    catalog = SqlCatalog(
        "default",
        uri="sqlite:///:memory:",
        warehouse=f"file://{tmp_path}",
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

    q = pl.scan_iceberg(tbl, reader_override="native").filter(pl.col("column_3") == 5)

    with monkeypatch.context() as cx:
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
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 0 / 1 row groups" in capture
    )
    # Second file
    assert (
        "[ParquetFileReader]: Predicate pushdown: reading 1 / 1 row groups" in capture
    )
    assert (
        "[ParquetFileReader]: Pre-filtered decode enabled (1 live, 1 non-live)"
        in capture
    )
