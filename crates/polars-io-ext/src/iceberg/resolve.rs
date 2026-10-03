//! The `plan` entry point: `resolve()` turns the planned files into the output contract of [`crate::iceberg::output`]. The
//! per-file data (identity-partition constants, `initial-default`s, statistics, deletes, row
//! count) follows the Python resolver (`IcebergScanResolver._to_dataset_scan_impl`).
use std::time::Instant;

use polars_arrow::array::{Array, PrimitiveArray, new_null_array};
use polars_utils::aliases::{PlHashMap, PlHashSet};

use crate::iceberg::arrow_types::{null_count_dtype, table_fields, value_dtype};
use crate::iceberg::avro::Datum;
use crate::iceberg::error::{IcebergResult, err_invalid_data, err_not_implemented};
use crate::iceberg::expr::bind;
use crate::iceberg::host::{Host, Storage};
use crate::iceberg::output::{DeleteKind, DeleteRef, FilesTable, Resolved};
use crate::iceberg::planner::{FileTask, PlanOptions, plan_files, resolve_selection};
use crate::iceberg::prune::Pruner;
use crate::iceberg::request::{Request, selection};
use crate::iceberg::spec::{NestedField, PrimitiveType, Schema, Table, Transform, Type};
use crate::iceberg::values::{
    bounds_array, bounds_supported, initial_default_array, partition_type_change_allowed,
    partition_values_array,
};

async fn load_table(host: &Host, request: &Request) -> IcebergResult<(Storage, Table)> {
    let location = request.metadata_location.as_str();
    let storage = host.get_storage(location).await?;

    if let Some(fail) = request.testing_fail.as_deref() {
        match fail {
            "panic" => panic!("testing panic requested"),
            "io" => {
                let missing = format!("{location}.polars-iceberg-testing-missing");
                storage.head(&missing).await?;
                unreachable!("{missing} must not exist");
            },
            other => {
                return Err(err_invalid_data(format!(
                    "unknown testing.fail value: {other:?}"
                )));
            },
        }
    }

    let bytes = storage.get(location).await?;
    let table = Table::parse(&bytes)
        .map_err(|e| err_invalid_data(format!("{location}: {}", e.message())))?;
    Ok((storage, table))
}

pub async fn resolve(host: Host, request: &Request) -> IcebergResult<Resolved> {
    let (storage, table) = load_table(&host, request).await?;
    let selection = selection(request);
    let resolved = resolve_selection(&table, &selection)?;
    let query = request;

    host.debug(&format!(
        "polars-iceberg: resolve(): snapshot ID: {:?}, from snapshot ID exclusive: {:?}, \
        to snapshot ID inclusive: {:?}, version key: {:?}, \
        limit: {:?}, projection: {:?}, filter_columns: {:?}, use_metadata_statistics: {}",
        selection.snapshot_id,
        selection.from_snapshot_id_exclusive,
        selection.to_snapshot_id_inclusive,
        resolved.version_key,
        query.limit,
        query.projection,
        query.filter_columns,
        request.use_metadata_statistics,
    ));

    let schema = resolved.schema.clone();

    let projected: Vec<&NestedField> = match &query.projection {
        None => schema.fields.iter().collect(),
        Some(names) => schema.select(names),
    };
    let projected_schema = Schema::new(schema.schema_id, projected.into_iter().cloned().collect());

    let stats_fields: Option<Vec<&NestedField>> = query
        .filter_columns
        .as_ref()
        .filter(|_| request.use_metadata_statistics)
        .map(|columns| schema.select(columns));

    let pruner_schema = table.current_schema()?.clone();
    let row_filter = query
        .row_filter
        .as_deref()
        .map(serde_json::from_str::<serde_json::Value>)
        .transpose()
        .map_err(|e| err_invalid_data(format!("invalid row_filter JSON: {e}")))?;
    let pruner = row_filter.as_ref().map(|json| {
        let bound = bind(json, &pruner_schema);
        host.debug(&format!(
            "polars-iceberg: resolve(): bound row filter: {bound:?}"
        ));
        Pruner::new(bound, &table.specs)
    });

    let options = PlanOptions {
        stats_field_ids: stats_fields
            .iter()
            .flatten()
            .map(|f| f.id)
            .collect::<PlHashSet<_>>(),
        pruner,
        pruner_schema,
    };

    let start = Instant::now();
    let (tasks, plan_stats) = plan_files(&storage, &table, &selection, &resolved, &options).await?;
    host.debug(&format!(
        "polars-iceberg: resolve(): planned {} files ({:.3}s), pruned {} / {} manifests and \
        {} data files",
        tasks.len(),
        start.elapsed().as_secs_f64(),
        plan_stats.manifests_pruned,
        plan_stats.manifests,
        plan_stats.data_files_pruned,
    ));

    let mut files = FilesTable::default();
    let mut total_physical_rows: u64 = 0;
    let mut total_deleted_rows: u64 = 0;
    let mut num_position_delete_files = 0;
    let mut num_deletion_vectors = 0;

    for task in &tasks {
        let file = &task.file;
        if file.file_format != "PARQUET" {
            return Err(err_not_implemented(format!(
                "non-parquet data file format: {} ({})",
                file.file_format, file.file_path
            )));
        }

        let mut position_deletes = vec![];
        let mut deletion_vector = None;
        let mut position_delete_rows: u64 = 0;
        let mut deletion_vector_rows: u64 = 0;

        for delete in &task.deletes {
            match delete.file_format.as_str() {
                "PARQUET" => {
                    position_deletes.push(DeleteRef {
                        kind: DeleteKind::Position,
                        path: normalize_path(&delete.file_path),
                    });
                    position_delete_rows += delete.record_count as u64;
                },
                "PUFFIN" => {
                    if deletion_vector.is_some() {
                        return Err(err_not_implemented(format!(
                            "multiple deletion vectors associated with one data file ({})",
                            file.file_path
                        )));
                    }
                    deletion_vector = Some(DeleteRef {
                        kind: DeleteKind::DeletionVector,
                        path: normalize_path(&delete.file_path),
                    });
                    deletion_vector_rows += delete.record_count as u64;
                },
                other => {
                    return Err(err_not_implemented(format!(
                        "deletion file format {other} ({})",
                        delete.file_path
                    )));
                },
            }
        }

        // A deletion vector supersedes position delete files for the same data file.
        let deletes = match deletion_vector {
            Some(dv) => {
                total_deleted_rows += deletion_vector_rows;
                num_deletion_vectors += 1;
                vec![dv]
            },
            None => {
                total_deleted_rows += position_delete_rows;
                num_position_delete_files += position_deletes.len();
                position_deletes
            },
        };

        total_physical_rows += file.record_count as u64;
        files.paths.push(normalize_path(&file.file_path));
        files.sizes.push(file.file_size_in_bytes as u64);
        files.record_counts.push(file.record_count as u64);
        files.deletes.push(deletes);
    }

    // Identity-partition constants of the projected fields.
    let partition_values = PartitionValues::build(&table, &projected_schema, &tasks);

    let mut constant_errors = vec![];
    for (field_id, values) in &partition_values.columns {
        match values {
            Ok(datums) => {
                let ty = &projected_schema.field_by_id(*field_id).unwrap().field_type;
                let refs: Vec<Option<&Datum>> = datums.iter().map(Option::as_ref).collect();
                match partition_values_array(ty, &refs) {
                    Ok(array) => files.constants.push((*field_id, array)),
                    Err(e) => constant_errors
                        .push((*field_id, format!("failed to load partition values: {e}"))),
                }
            },
            Err(msg) => constant_errors.push((*field_id, msg.clone())),
        }
    }

    // `initial-default` values of all projected fields, including nested ones.
    let mut initial_defaults = vec![];
    let mut default_ids: Vec<i32> = projected_schema.field_ids().collect();
    default_ids.sort_unstable();
    for id in default_ids {
        let field = projected_schema.field_by_id(id).unwrap();
        if let Some(json) = &field.initial_default {
            initial_defaults.push((id, initial_default_array(&field.field_type, json)?));
        }
    }

    // Statistics of the filter columns.
    if let Some(stats_fields) = &stats_fields {
        let mut stats = vec![];
        for field in stats_fields {
            let constants = match partition_values.columns.get(&field.id) {
                Some(Err(msg)) => {
                    return Err(err_invalid_data(format!(
                        "statistics load failure for filter column: {msg}"
                    )));
                },
                Some(Ok(v)) => Some(v.as_slice()),
                None => None,
            };
            stats.extend(column_statistics(&table, field, &tasks, constants)?);
        }
        files.stats = Some(stats);
    }

    let row_count = (request.use_metadata_statistics
        && (request.fast_deletion_count || total_deleted_rows == 0))
        .then_some((total_physical_rows, total_deleted_rows));

    host.debug(&format!(
        "polars-iceberg: resolve(): native scan_parquet(): num_sources: {}, snapshot ID: {:?}, \
        schema ID: {}, num_position_delete_files: {num_position_delete_files}, \
        num_deletion_vectors: {num_deletion_vectors}",
        files.paths.len(),
        resolved.snapshot.map(|s| s.snapshot_id),
        schema.schema_id,
    ));

    Ok(Resolved {
        schema: table_fields(&schema),
        files,
        row_count,
        constant_errors,
        initial_defaults,
    })
}

/// PyIceberg on Windows uses `file://C:/` rather than `file:///C:/`.
fn normalize_path(path: &str) -> String {
    match path.strip_prefix("file://") {
        Some(rest) if !rest.starts_with('/') => format!("file:///{rest}"),
        _ => path.to_owned(),
    }
}

/// Identity-partition values of projected fields, one value per file. Mirrors
/// `IdentityTransformedPartitionValuesBuilder`.
struct PartitionValues {
    /// Source field ID → per-file values, or the reason they cannot be used.
    columns: PlHashMap<i32, Result<Vec<Option<Datum>>, String>>,
}

impl PartitionValues {
    fn build(table: &Table, projected: &Schema, tasks: &[FileTask]) -> Self {
        let projected_ids: PlHashSet<i32> = projected.field_ids().collect();

        // spec ID → [(index in partition tuple, source field ID)]
        let mut identity_fields: PlHashMap<i32, Vec<(usize, i32)>> = PlHashMap::default();
        let mut columns: PlHashMap<i32, Result<Vec<Option<Datum>>, String>> = PlHashMap::default();

        for (spec_id, spec) in &table.specs {
            let fields = spec
                .fields
                .iter()
                .enumerate()
                .filter(|(_, f)| {
                    f.transform == Transform::Identity && projected_ids.contains(&f.source_id)
                })
                .map(|(i, f)| (i, f.source_id))
                .collect::<Vec<_>>();
            for (_, source_id) in &fields {
                columns.insert(*source_id, Ok(vec![]));
            }
            identity_fields.insert(*spec_id, fields);
        }

        for (field_id, column) in columns.iter_mut() {
            let projected_type = &projected.field_by_id(*field_id).unwrap().field_type;
            if !projected_type.is_primitive() {
                *column = Err(format!("non-primitive type: {projected_type:?}"));
            }
            for schema in table.schemas.values() {
                if let Some(other) = schema.field_by_id(*field_id)
                    && !partition_type_change_allowed(projected_type, &other.field_type)
                {
                    *column = Err(format!(
                        "unsupported type change: from: {:?}, to: {projected_type:?}",
                        other.field_type
                    ));
                }
            }
        }

        let n = tasks.len();
        for (i, task) in tasks.iter().enumerate() {
            let Some(fields) = identity_fields.get(&task.spec_id) else {
                for column in columns.values_mut() {
                    *column = Err(format!("partition spec ID not found: {}", task.spec_id));
                }
                continue;
            };
            for (index, source_id) in fields {
                if let Some(Ok(values)) = columns.get_mut(source_id) {
                    values.resize(i, None);
                    values.push(task.partition.get(*index).cloned().flatten());
                }
            }
        }

        for values in columns.values_mut().flatten() {
            values.resize(n, None);
        }

        Self { columns }
    }
}

/// `{name}_nc`, `{name}_min`, `{name}_max` for one filter column. Mirrors
/// `IcebergColumnStatisticsLoader`: identity-partition values take precedence over bounds.
fn column_statistics(
    table: &Table,
    field: &NestedField,
    tasks: &[FileTask],
    constants: Option<&[Option<Datum>]>,
) -> IcebergResult<Vec<(String, Box<dyn Array>)>> {
    let name = &field.name;
    let ty = &field.field_type;
    let n = tasks.len();

    let null_counts: Box<dyn Array> = if matches!(ty, Type::Struct(_)) {
        new_null_array(null_count_dtype(ty), n)
    } else {
        PrimitiveArray::<u64>::from(
            tasks
                .iter()
                .map(|t| t.file.null_value_count(field.id).map(|v| v as u64))
                .collect::<Vec<_>>(),
        )
        .boxed()
    };

    let all_types: Vec<&Type> = table
        .schemas
        .values()
        .filter_map(|s| s.field_by_id(field.id).map(|f| &f.field_type))
        .collect();

    let (min, max) = if bounds_supported(ty, &all_types) {
        let constant_bytes: Vec<Option<Vec<u8>>> = (0..n)
            .map(|i| {
                constants
                    .and_then(|c| c[i].as_ref())
                    .map(|d| datum_to_bytes(d, ty))
            })
            .collect();
        let bounds = |lower: bool| -> IcebergResult<Box<dyn Array>> {
            let values: Vec<Option<&[u8]>> = tasks
                .iter()
                .zip(&constant_bytes)
                .map(|(t, c)| {
                    c.as_deref().or_else(|| {
                        if lower {
                            t.file.lower_bound(field.id)
                        } else {
                            t.file.upper_bound(field.id)
                        }
                    })
                })
                .collect();
            bounds_array(ty, &values)
        };
        (bounds(true)?, bounds(false)?)
    } else {
        let values = match constants {
            Some(c) => {
                let refs: Vec<Option<&Datum>> = c.iter().map(Option::as_ref).collect();
                partition_values_array(ty, &refs).map_err(err_invalid_data)?
            },
            None => new_null_array(value_dtype(ty), n),
        };
        (values.clone(), values)
    };

    Ok(vec![
        (format!("{name}_nc"), null_counts),
        (format!("{name}_min"), min),
        (format!("{name}_max"), max),
    ])
}

/// Iceberg single-value binary serialization of a partition value.
fn datum_to_bytes(d: &Datum, ty: &Type) -> Vec<u8> {
    match d {
        Datum::Bool(b) => vec![u8::from(*b)],
        Datum::Int(v) => match ty {
            // Promoted int → long partition values.
            Type::Primitive(PrimitiveType::Long) => i64::from(*v).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        },
        Datum::Long(v) => v.to_le_bytes().to_vec(),
        Datum::Float(v) => v.to_le_bytes().to_vec(),
        Datum::Double(v) => v.to_le_bytes().to_vec(),
        Datum::String(s) => s.as_bytes().to_vec(),
        Datum::Bytes(b) => b.clone(),
    }
}
