//! `iceberg.v1` output → `DslPlan::Scan` (see `polars_io_ext_ffi::iceberg_v1` for the layout).
use std::sync::Arc;

use polars::prelude::default_values::DefaultFieldValues;
use polars::prelude::deletion::DeletionFilesList;
use polars::prelude::{
    ArrowDataType, CastColumnsPolicy, ColumnMapping, DataFrame, DataType, DslBuilder, DslPlan,
    ExtraColumnsPolicy, Field, IDX_DTYPE, MissingColumnsPolicy, ParquetOptions, PlIndexMap,
    ScanSources, Schema, Series, TableStatistics, UnifiedScanArgs,
};
use polars_arrow::datatypes::ArrowSchema as PolarsArrowSchema;
use polars_arrow::ffi;
use polars_buffer::Buffer;
use polars_core::prelude::{Column, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::iceberg::IcebergSchema;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_io::HiveOptions;
use polars_io::cloud::CloudOptions;
use polars_io_ext_ffi::common::{ArrowArrayStream, ArrowSchema};
use polars_io_ext_ffi::iceberg_v1::{Output, OutputHeader};
use polars_plan::dsl::default_values::IcebergDefaultFieldValues;
use polars_plan::dsl::deletion::IcebergDeletes;
use polars_utils::aliases::PlIndexMapHashable;
use polars_utils::pl_path::PlRefPath;

pub(super) struct ResolvedScan {
    pub schema: Schema,
    pub arrow_schema: PolarsArrowSchema,
    pub paths: Vec<PlRefPath>,
    pub sizes: Vec<u64>,
    pub row_count: Option<(u64, u64)>,
    pub default_values: Option<DefaultFieldValues>,
    pub deletion_files: Option<DeletionFilesList>,
    pub table_statistics: Option<TableStatistics>,
}

/// Import an Arrow struct schema.
fn import_arrow_schema(schema: ArrowSchema) -> PolarsResult<PolarsArrowSchema> {
    polars_ensure!(
        !schema.is_released(),
        ComputeError: "iceberg plugin returned a released schema"
    );

    let schema: ffi::ArrowSchema = unsafe { schema.transmute_into() };
    let field = unsafe { ffi::import_field_from_c(&schema)? };

    let ArrowDataType::Struct(fields) = field.dtype else {
        polars_bail!(
            ComputeError:
            "iceberg plugin returned a non-struct schema: {:?}", field.dtype
        )
    };

    Ok(fields.into_iter().map(|f| (f.name.clone(), f)).collect())
}

pub(super) fn import_output(output: Output) -> PolarsResult<ResolvedScan> {
    let Output {
        header,
        schema,
        files,
        table_values,
    } = output;

    let header = OutputHeader::from_ffi(&header);
    let OutputHeader {
        row_count,
        constant_errors,
        statistics: has_statistics,
    } = header;

    let arrow_schema = import_arrow_schema(schema)?;
    let schema: Schema = arrow_schema.iter_values().map(Field::from).collect();

    let files = import_struct_stream(files)?;
    let table_values = import_struct_stream(table_values)?;

    let n = files.as_ref().map_or(0, |f| f.height());
    let file_columns: &[Column] = files.as_ref().map_or(&[], |f| f.columns());
    let file_column = |name: &str| file_columns.iter().find(|c| c.name() == name);
    let required = |name: &str| {
        file_column(name).ok_or_else(
            || polars_err!(ComputeError: "iceberg plugin files: missing column '{name}'"),
        )
    };

    let (paths, sizes, record_counts) = if n == 0 {
        (vec![], vec![], None)
    } else {
        let path_col = required("path")?.as_materialized_series();
        let size_col = required("size")?.cast(&DataType::UInt64)?;
        polars_ensure!(
            path_col.null_count() == 0 && size_col.null_count() == 0,
            ComputeError: "iceberg plugin files: 'path' and 'size' must not contain nulls"
        );
        let paths = path_col
            .str()?
            .iter()
            .flatten()
            .map(PlRefPath::new)
            .collect();
        let sizes = size_col.u64()?.into_no_null_iter().collect();
        let record_counts = file_column("record_count")
            .map(|c| c.cast(&DataType::UInt64))
            .transpose()?;
        (paths, sizes, record_counts)
    };

    // Per-source constants and table-level default values.
    let mut identity_values = PlIndexMap::default();
    if let Some(constants) = file_column("constants") {
        for field in constants.struct_()?.fields_as_series() {
            let id = parse_field_id(field.name())?;
            identity_values.insert(id, Ok(field.into_column()));
        }
    }
    for (id, msg) in constant_errors {
        identity_values.insert(id, Err(msg));
    }

    let mut initial_defaults = PlIndexMap::default();
    if let Some(table_values) = &table_values
        && let Some(defaults) = table_values
            .columns()
            .iter()
            .find(|c| c.name() == "initial_defaults")
    {
        for field in defaults.struct_()?.fields_as_series() {
            let id = parse_field_id(field.name())?;
            let value = Scalar::new(field.dtype().clone(), field.get(0)?.into_static());
            initial_defaults.insert(id, value);
        }
    }

    let default_values = IcebergDefaultFieldValues {
        identity_transformed_partition_fields: PlIndexMapHashable(identity_values),
        initial_defaults: PlIndexMapHashable(initial_defaults),
    };
    let default_values =
        (!default_values.is_empty()).then(|| DefaultFieldValues::Iceberg(Arc::new(default_values)));

    // Deletion files.
    let mut deletes = PlIndexMap::default();
    if let Some(deletes_col) = file_column("deletes") {
        let lists = deletes_col.list()?;
        for (i, entries) in lists.series_iter().enumerate() {
            let Some(entries) = entries else { continue };
            if entries.is_empty() {
                continue;
            }
            let entries = entries.struct_()?.fields_as_series();
            let kinds = entries
                .iter()
                .find(|s| s.name() == "kind")
                .ok_or_else(|| polars_err!(ComputeError: "iceberg plugin deletes: no 'kind'"))?
                .str()?
                .clone();
            let delete_paths = entries
                .iter()
                .find(|s| s.name() == "path")
                .ok_or_else(|| polars_err!(ComputeError: "iceberg plugin deletes: no 'path'"))?
                .str()?
                .clone();

            let mut position = vec![];
            let mut dv = None;
            for (kind, path) in kinds.iter().zip(delete_paths.iter()) {
                let (Some(kind), Some(path)) = (kind, path) else {
                    polars_bail!(ComputeError: "iceberg plugin deletes: null kind or path")
                };
                match kind {
                    "position" => position.push(PlRefPath::new(path)),
                    "deletion_vector" => dv = Some(PlRefPath::new(path)),
                    other => {
                        polars_bail!(ComputeError: "iceberg plugin deletes: unknown kind {other:?}")
                    },
                }
            }
            let value = match dv {
                Some(dv) => IcebergDeletes::DeletionVector(dv),
                None => IcebergDeletes::PositionDeletes(Buffer::from(position)),
            };
            deletes.insert(i, value);
        }
    }
    let deletion_files =
        DeletionFilesList::filter_empty(Some(DeletionFilesList::Iceberg(Arc::new(deletes))));

    // Statistics: `len`, then `<col>_nc` / `_min` / `_max`.
    let table_statistics = if has_statistics {
        let len = match &record_counts {
            Some(rc) => rc.cast(&DataType::UInt32)?.with_name("len".into()),
            None => Column::new_empty("len".into(), &DataType::UInt32),
        };
        let mut columns = vec![len];
        if let Some(stats) = file_column("stats") {
            for field in stats.struct_()?.fields_as_series() {
                let field = if field.name().ends_with("_nc") {
                    let dtype = idx_null_count_dtype(field.dtype());
                    field.cast(&dtype)?
                } else {
                    field
                };
                columns.push(field.into_column());
            }
        }
        Some(TableStatistics(Arc::new(DataFrame::new(n, columns)?)))
    } else {
        None
    };

    Ok(ResolvedScan {
        schema,
        arrow_schema,
        paths,
        sizes,
        row_count,
        default_values,
        deletion_files,
        table_statistics,
    })
}

fn parse_field_id(name: &str) -> PolarsResult<u32> {
    name.parse()
        .map_err(|_| polars_err!(ComputeError: "iceberg plugin: invalid field ID {name:?}"))
}

/// Null counts use the index type (per leaf for structs).
fn idx_null_count_dtype(dtype: &DataType) -> DataType {
    match dtype {
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Field::new(f.name().clone(), idx_null_count_dtype(f.dtype())))
                .collect(),
        ),
        _ => IDX_DTYPE,
    }
}

/// Read a stream of struct arrays into a `DataFrame` of its fields. `None` if the stream's
/// struct has no fields.
fn import_struct_stream(stream: ArrowArrayStream) -> PolarsResult<Option<DataFrame>> {
    let stream: ffi::ArrowArrayStream = unsafe { stream.transmute_into() };
    let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream))? };

    let mut arrays = vec![];
    while let Some(array) = unsafe { reader.next() } {
        arrays.push(array?);
    }

    let field = reader.field();
    if arrays.is_empty() {
        // Keep the columns of empty tables, e.g. the statistics of a scan without files.
        let ArrowDataType::Struct(fields) = &field.dtype else {
            return Ok(None);
        };
        if fields.is_empty() {
            return Ok(None);
        }
        arrays.push(polars_arrow::array::new_empty_array(field.dtype.clone()));
    }

    let series = Series::try_from((field, arrays))?;
    let height = series.len();
    let columns = series
        .struct_()?
        .fields_as_series()
        .into_iter()
        .map(IntoColumn::into_column)
        .collect();
    Ok(Some(DataFrame::new(height, columns)?))
}

/// Build the parquet scan for a resolved dataset. Mirrors the `scan_parquet(...)` call of
/// `_NativeIcebergScanData.to_lazyframe()`, with the schema supplied so no footer is read during
/// planning.
pub(super) fn build_scan(
    scan: ResolvedScan,
    cloud_options: Option<CloudOptions>,
    cast_columns_policy: CastColumnsPolicy,
) -> PolarsResult<DslPlan> {
    let ResolvedScan {
        schema,
        arrow_schema,
        paths,
        sizes,
        row_count,
        default_values,
        deletion_files,
        table_statistics,
    } = scan;

    let column_mapping =
        ColumnMapping::Iceberg(Arc::new(IcebergSchema::from_arrow_schema(&arrow_schema)?));

    let options = ParquetOptions {
        schema: Some(Arc::new(schema)),
        ..Default::default()
    };

    let unified_scan_args = UnifiedScanArgs {
        cloud_options,
        hive_options: HiveOptions::new_disabled(),
        glob: false,
        expand_paths: false,
        column_mapping: Some(column_mapping),
        default_values,
        deletion_files,
        table_statistics,
        cast_columns_policy,
        missing_columns_policy: MissingColumnsPolicy::Insert,
        extra_columns_policy: ExtraColumnsPolicy::Ignore,
        row_count,
        source_sizes: Some(Buffer::from(sizes)),
        ..Default::default()
    };

    Ok(DslBuilder::scan_parquet(
        ScanSources::Paths(Buffer::from(paths)),
        options,
        unified_scan_args,
    )?
    .build())
}
