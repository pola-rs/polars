//! Import the Delta plugin output into a native multi-file Parquet scan.
use std::sync::Arc;

use polars::prelude::*;
use polars_arrow::datatypes::ArrowSchema as PolarsArrowSchema;
use polars_arrow::ffi;
use polars_buffer::Buffer;
use polars_core::prelude::IntoColumn;
use polars_core::schema::iceberg::IcebergSchema;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_io::HiveOptions;
use polars_io::cloud::CloudOptions;
use polars_io_ext_ffi::common::{ArrowArrayStream, ArrowSchema};
use polars_io_ext_ffi::delta_v1::{Output, OutputHeader};
use polars_plan::dsl::deletion::{
    DeletionFilesList, DeltaDeletionVector, DeltaDeletionVectorLocation,
};
use polars_plan::dsl::{HivePartitions, StatisticsBounds};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_path::PlRefPath;
/// Import an Arrow struct schema.
fn import_arrow_schema(schema: ArrowSchema) -> PolarsResult<PolarsArrowSchema> {
    polars_ensure!(
        !schema.is_released(),
        ComputeError: "delta plugin returned a released schema"
    );

    let schema: ffi::ArrowSchema = unsafe { schema.transmute_into() };
    let field = unsafe { ffi::import_field_from_c(&schema)? };

    let ArrowDataType::Struct(fields) = field.dtype else {
        polars_bail!(
            ComputeError:
            "delta plugin returned a non-struct schema: {:?}", field.dtype
        )
    };

    Ok(fields.into_iter().map(|f| (f.name.clone(), f)).collect())
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

pub(super) fn build_scan(
    output: Output,
    cloud_options: Option<CloudOptions>,
) -> PolarsResult<(DslPlan, i64)> {
    let Output {
        header,
        schema,
        files,
    } = output;
    let header = OutputHeader::from_ffi(&header);
    let arrow_schema = import_arrow_schema(schema)?;
    let schema: Schema = arrow_schema.iter_values().map(Field::from).collect();
    let files = import_struct_stream(files)?
        .ok_or_else(|| polars_err!(ComputeError: "delta plugin omitted files schema"))?;
    let n = files.height();
    let paths = files.column("path")?.str()?;
    let sizes = files.column("size")?.u64()?;
    polars_ensure!(paths.null_count()==0 && sizes.null_count()==0, ComputeError: "delta plugin returned null path/size");
    let paths: Vec<_> = paths.iter().flatten().map(PlRefPath::new).collect();
    let sizes: Vec<_> = sizes.into_no_null_iter().collect();
    let partitions = files
        .column("partition_values")
        .ok()
        .map(|c| {
            DataFrame::new(
                n,
                c.struct_()?
                    .fields_as_series()
                    .into_iter()
                    .map(IntoColumn::into_column)
                    .collect(),
            )
        })
        .transpose()?
        .filter(|df| df.width() > 0);
    let hive_schema = partitions.as_ref().map(|df| df.schema().clone());
    let mut deletes = PlIndexMap::default();
    if let Ok(column) = files.column("deletion_vector") {
        let dv = column.struct_()?;
        let fields = dv.fields_as_series();
        let field = |name: &str| {
            fields
                .iter()
                .find(|s| s.name() == name)
                .ok_or_else(|| polars_err!(ComputeError: "delta plugin DV missing {name}"))
        };
        let path = field("path")?.str()?;
        let inline = field("inline")?.binary()?;
        let offsets = field("offset")?.u64()?;
        let sizes = field("size_in_bytes")?.u32()?;
        let cardinalities = field("cardinality")?.u64()?;
        let is_null = column.is_null();
        for i in 0..n {
            if is_null.get(i) == Some(true) {
                continue;
            }
            let location = match (path.get(i), inline.get(i)) {
                (Some(path), None) => DeltaDeletionVectorLocation::File {
                    path: PlRefPath::new(path),
                    offset: offsets
                        .get(i)
                        .ok_or_else(|| polars_err!(ComputeError: "delta plugin null DV offset"))?,
                },
                (None, Some(bytes)) => {
                    DeltaDeletionVectorLocation::Inline(Buffer::from(bytes.to_vec()))
                },
                _ => {
                    polars_bail!(ComputeError: "delta plugin DV must contain exactly one of path/inline")
                },
            };
            let size_in_bytes = sizes
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "delta plugin null DV size"))?;
            let cardinality = cardinalities
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "delta plugin null DV cardinality"))?;
            deletes.insert(
                i,
                DeltaDeletionVector {
                    location,
                    size_in_bytes,
                    cardinality,
                },
            );
        }
    }
    let table_statistics = if header.statistics {
        let len = files
            .column("record_count")?
            .cast(&DataType::UInt32)?
            .with_name("len".into());
        let mut columns = vec![len];
        if let Ok(stats) = files.column("stats") {
            for field in stats.struct_()?.fields_as_series() {
                let field = if field.name().ends_with("_nc") {
                    field.cast(&idx_null_count_dtype(field.dtype()))?
                } else {
                    field
                };
                columns.push(field.into_column());
            }
        }
        Some(TableStatistics(
            Arc::new(DataFrame::new(n, columns)?),
            StatisticsBounds::Loose,
        ))
    } else {
        None
    };
    let column_mapping = match header.column_mapping_mode.as_str() {
        "none" => None,
        mode @ ("name" | "id") => {
            let mut ids = IcebergSchema::from_arrow_schema(&arrow_schema)?;
            if mode == "name" {
                let mut names = PlHashMap::default();
                collect_physical_names(arrow_schema.iter_values(), &mut names)?;
                ids.apply_name_mapping(&names);
            }
            Some(ColumnMapping::Delta {
                schema: Arc::new(ids),
                by_name: mode == "name",
            })
        },
        mode => polars_bail!(ComputeError: "delta plugin invalid mapping mode {mode:?}"),
    };
    // Same as `ScanCastOptions._default_iceberg()`, which the deltalake path uses, plus
    // type widening.
    let cast_columns_policy = CastColumnsPolicy {
        integer_upcast: true,
        float_upcast: true,
        float_downcast: true,
        delta_type_widening: true,
        datetime_nanoseconds_downcast: true,
        datetime_milliseconds_upcast: true,
        datetime_convert_timezone: true,
        categorical_to_string: true,
        missing_struct_fields: MissingColumnsPolicy::Insert,
        extra_struct_fields: ExtraColumnsPolicy::Ignore,
        ..Default::default()
    };
    let options = ParquetOptions {
        schema: Some(Arc::new(schema)),
        ..Default::default()
    };
    let args = UnifiedScanArgs {
        cloud_options,
        glob: false,
        expand_paths: false,
        column_mapping,
        hive_options: HiveOptions {
            enabled: Some(partitions.is_some()),
            schema: hive_schema,
            ..HiveOptions::new_disabled()
        },
        hive_partitions: partitions.map(|df| HivePartitions(Arc::new(df))),
        deletion_files: DeletionFilesList::filter_empty(Some(DeletionFilesList::Delta(Arc::new(
            deletes,
        )))),
        table_statistics,
        cast_columns_policy,
        missing_columns_policy: MissingColumnsPolicy::Insert,
        extra_columns_policy: ExtraColumnsPolicy::Ignore,
        row_count: header.row_count,
        source_sizes: Some(Buffer::from(sizes)),
        ..Default::default()
    };
    Ok((
        DslBuilder::scan_parquet(ScanSources::Paths(Buffer::from(paths)), options, args)?.build(),
        header.resolved_version,
    ))
}

fn collect_physical_names<'a>(
    fields: impl IntoIterator<Item = &'a polars_arrow::datatypes::Field>,
    out: &mut PlHashMap<u32, Vec<PlSmallStr>>,
) -> PolarsResult<()> {
    for field in fields {
        if let Some(md) = &field.metadata {
            if let (Some(id), Some(name)) = (
                md.get("PARQUET:field_id"),
                md.get("delta.columnMapping.physicalName"),
            ) {
                let id = id
                    .parse()
                    .map_err(|_| polars_err!(ComputeError: "delta plugin invalid field ID"))?;
                out.insert(id, vec![name.as_str().into()]);
            }
        }
        match &field.dtype {
            ArrowDataType::Struct(fields) => collect_physical_names(fields, out)?,
            ArrowDataType::List(inner)
            | ArrowDataType::LargeList(inner)
            | ArrowDataType::Map(inner, _) => collect_physical_names([inner.as_ref()], out)?,
            _ => {},
        }
    }
    Ok(())
}
