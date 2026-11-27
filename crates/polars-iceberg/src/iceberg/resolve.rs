//! The `plan` entry point: `resolve()` turns the planned files into the output contract of [`crate::iceberg::output`]. The
//! per-file data (identity-partition constants, `initial-default`s, statistics, deletes, row
//! count) follows the Python resolver (`IcebergScanResolver._to_dataset_scan_impl`).
use std::borrow::Cow;
use std::time::Instant;

use polars_arrow::array::{Array, BooleanArray, PrimitiveArray, new_null_array};
use polars_arrow::compute::concatenate::concatenate;
use polars_utils::aliases::{PlHashMap, PlHashSet};

use crate::iceberg::arrow_types::{null_count_dtype, table_fields, value_dtype};
use crate::iceberg::avro::Datum;
use crate::iceberg::error::{IcebergResult, err_invalid_data, err_not_implemented, with_context};
use crate::iceberg::expr::bind;
use crate::iceberg::host::{Host, Storage, normalize_path};
use crate::iceberg::json;
use crate::iceberg::output::{DeleteKind, DeleteRef, FilesTable, Resolved};
use crate::iceberg::planner::{
    FileTask, PlanOptions, plan_files, referenced_data_file, resolve_selection,
};
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
    let table = decompress_metadata(&bytes)
        .and_then(|bytes| Table::parse(&bytes))
        .map_err(|e| with_context(e, location))?;
    Ok((storage, table))
}

/// Table metadata may be gzip-compressed (`*.gz.metadata.json`, `write.metadata.compression-codec`
/// = `gzip`); detected by the gzip magic bytes, as Java and PyIceberg do.
fn decompress_metadata(bytes: &[u8]) -> IcebergResult<Cow<'_, [u8]>> {
    if !bytes.starts_with(&[0x1F, 0x8B]) {
        return Ok(Cow::Borrowed(bytes));
    }
    crate::iceberg::avro::read_to_end_limited(
        flate2::read::MultiGzDecoder::new(bytes),
        bytes.len() * 8,
        crate::iceberg::avro::MAX_DECOMPRESSED_BYTES,
        "gzip-compressed table metadata",
    )
    .map(Cow::Owned)
}

pub async fn resolve(host: Host, request: &Request) -> IcebergResult<Resolved> {
    let (storage, table) = load_table(&host, request).await?;
    let selection = selection(request);
    let resolved = resolve_selection(&table, &selection)?;
    let query = request;

    host.debug(&format!(
        "polars-iceberg: resolve(): snapshot ID: {:?}, from snapshot ID exclusive: {:?}, \
        to snapshot ID inclusive: {:?}, version key: {:?}, \
        limit: {:?}, projection: {:?}, filter_columns: {:?}, statistics_columns: {:?}, \
        use_metadata_statistics: {}",
        selection.snapshot_id,
        selection.from_snapshot_id_exclusive,
        selection.to_snapshot_id_inclusive,
        resolved.version_key,
        query.limit,
        query.projection,
        query.filter_columns,
        query.statistics_columns,
        request.use_metadata_statistics,
    ));

    let schema = resolved.schema.clone();

    let projected: Vec<&NestedField> = match &query.projection {
        None => schema.fields.iter().collect(),
        Some(names) => schema.select(names),
    };
    let projected_schema = Schema::new(schema.schema_id, projected.into_iter().cloned().collect());

    // Statistics of columns that are not filtered on are best effort.
    let best_effort_columns: Vec<String> = query
        .statistics_columns
        .iter()
        .flatten()
        .filter(|c| !query.filter_columns.iter().flatten().any(|f| f == *c))
        .cloned()
        .collect();
    let best_effort_fields = schema.select(&best_effort_columns);
    let best_effort_ids: PlHashSet<i32> = best_effort_fields.iter().map(|f| f.id).collect();
    let stats_fields: Option<Vec<&NestedField>> = (request.use_metadata_statistics
        && (query.filter_columns.is_some() || !best_effort_fields.is_empty()))
    .then(|| {
        let mut fields = schema.select(query.filter_columns.as_deref().unwrap_or_default());
        fields.extend(best_effort_fields);
        fields
    });

    // Filter column names refer to the scanned schema (the snapshot's schema when time
    // travelling), not the current one: a column may have been renamed since.
    let pruner_schema = schema.clone();
    let row_filter = query
        .row_filter
        .as_deref()
        .map(|s| json::from_slice::<serde_json::Value>(s.as_bytes(), "row_filter JSON"))
        .transpose()?;
    let pruner = row_filter.as_ref().map(|json| {
        let bound = bind(json, &pruner_schema);
        host.debug(&format!(
            "polars-iceberg: resolve(): bound row filter: {}",
            bound.redacted()
        ));
        Pruner::new(bound, &table.specs, &table.schemas)
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
    // `None` if a file's record count is unknown.
    let mut total_physical_rows: Option<u64> = Some(0);
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
        let mut position_delete_rows: u64 = 0;
        let mut deletion_vector = None;
        let mut deletion_vector_rows: u64 = 0;

        for delete in &task.deletes {
            match delete.file_format.as_str() {
                "PARQUET" => {
                    // Polars reads position delete files of one data file only. This also keeps
                    // the deleted row count exact: the rows of a delete file scoped to a
                    // partition may belong to data files that are no longer live.
                    if referenced_data_file(delete).is_none() {
                        return Err(err_not_implemented(format!(
                            "position delete file not limited to one data file ({})",
                            delete.file_path
                        )));
                    }
                    position_deletes.push(DeleteRef {
                        kind: DeleteKind::Position,
                        path: normalize_path(&delete.file_path),
                        content_offset: None,
                    });
                    position_delete_rows = position_delete_rows
                        .saturating_add(non_negative(delete.record_count, "record_count")?);
                },
                "PUFFIN" => {
                    // A deletion vector must reference its data file (spec). Without it, it would
                    // be associated with every data file of its partition.
                    if referenced_data_file(delete).is_none() {
                        return Err(err_not_implemented(format!(
                            "deletion vector without referenced data file ({})",
                            delete.file_path
                        )));
                    }
                    if deletion_vector.is_some() {
                        return Err(err_not_implemented(format!(
                            "multiple deletion vectors associated with one data file ({})",
                            file.file_path
                        )));
                    }
                    deletion_vector = Some(DeleteRef {
                        kind: DeleteKind::DeletionVector,
                        path: normalize_path(&delete.file_path),
                        // A Puffin file can hold several deletion vectors of one data file
                        // (e.g. a live and a replaced one); the offset selects the blob.
                        content_offset: delete.content_offset,
                    });
                    deletion_vector_rows = deletion_vector_rows
                        .saturating_add(non_negative(delete.record_count, "record_count")?);
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
                total_deleted_rows = total_deleted_rows.saturating_add(deletion_vector_rows);
                num_deletion_vectors += 1;
                vec![dv]
            },
            None => {
                total_deleted_rows = total_deleted_rows.saturating_add(position_delete_rows);
                num_position_delete_files += position_deletes.len();
                position_deletes
            },
        };

        total_physical_rows = add_record_count(total_physical_rows, file.record_count);
        files.paths.push(normalize_path(&file.file_path));
        files
            .sizes
            .push(non_negative(file.file_size_in_bytes, "file_size_in_bytes")?);
        // A negative (unknown) count becomes `u64::MAX`, which the host's `len` statistic
        // turns into null.
        files
            .record_counts
            .push(u64::try_from(file.record_count).unwrap_or(u64::MAX));
        files.deletes.push(deletes);
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

    // Identity-partition constants of the projected fields.
    let partition_values = PartitionValues::build(&table, &projected_schema, &tasks);

    files.constants_present = partition_values
        .present
        .iter()
        .map(|(id, present)| (*id, BooleanArray::from_slice(present).boxed()))
        .collect();

    let mut constant_errors = vec![];
    for (field_id, values) in &partition_values.columns {
        match values {
            Ok(datums) => {
                let ty = &projected_schema.field_by_id(*field_id).unwrap().field_type;
                let refs: Vec<Option<&Datum>> = datums.iter().map(Option::as_ref).collect();
                let array = partition_values_array(ty, &refs).and_then(|array| {
                    // A null value of a spec with the identity field is a null; files of specs
                    // without it take the `initial-default` (the host only falls back to it for
                    // sources past the end of the constants).
                    match initial_defaults.iter().find(|(id, _)| id == field_id) {
                        Some((_, default)) => fill_absent(
                            array,
                            default.as_ref(),
                            &partition_values.present[field_id],
                        ),
                        None => Ok(array),
                    }
                });
                match array {
                    Ok(array) => files.constants.push((*field_id, array)),
                    Err(e) => constant_errors
                        .push((*field_id, format!("failed to load partition values: {e}"))),
                }
            },
            Err(msg) => constant_errors.push((*field_id, msg.clone())),
        }
    }

    // Statistics of the filter and statistics columns.
    if let Some(stats_fields) = &stats_fields {
        let mut stats = vec![];
        for field in stats_fields {
            let field_stats = match partition_values.columns.get(&field.id) {
                Some(Err(msg)) => Err(err_invalid_data(format!(
                    "statistics load failure for filter column: {msg}"
                ))),
                Some(Ok(v)) => column_statistics(&table, field, &tasks, Some(v.as_slice())),
                None => column_statistics(&table, field, &tasks, None),
            };
            match field_stats {
                Ok(field_stats) => stats.extend(field_stats),
                Err(e) if best_effort_ids.contains(&field.id) => {
                    host.debug(&format!(
                        "polars-iceberg: resolve(): statistics load failed for column {:?}: \
                        {}",
                        field.name,
                        e.message()
                    ));
                    stats.extend(null_column_statistics(field, tasks.len()));
                },
                Err(e) => return Err(e),
            }
        }
        files.stats = Some(stats);
    }

    let row_count = total_physical_rows
        .filter(|_| {
            request.use_metadata_statistics
                && (request.fast_deletion_count || total_deleted_rows == 0)
        })
        .map(|rows| (rows, total_deleted_rows));

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

/// Adds a file's record count to a running total; `None` once a count is unknown (negative: some
/// format v1 writers wrote -1) or the total overflows.
fn add_record_count(total: Option<u64>, record_count: i64) -> Option<u64> {
    total?.checked_add(u64::try_from(record_count).ok()?)
}

/// Identity-partition values of projected fields, one value per file. Mirrors
/// `IdentityTransformedPartitionValuesBuilder`.
struct PartitionValues {
    /// Source field ID → per-file values, or the reason they cannot be used.
    columns: PlHashMap<i32, Result<Vec<Option<Datum>>, String>>,
    /// Source field ID → per-file: whether the file's partition spec has the identity field.
    present: PlHashMap<i32, Vec<bool>>,
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
        let mut present: PlHashMap<i32, Vec<bool>> =
            columns.keys().map(|id| (*id, vec![false; n])).collect();
        for (i, task) in tasks.iter().enumerate() {
            let Some(fields) = identity_fields.get(&task.spec_id) else {
                for column in columns.values_mut() {
                    *column = Err(format!("partition spec ID not found: {}", task.spec_id));
                }
                continue;
            };
            for (index, source_id) in fields {
                present.get_mut(source_id).unwrap()[i] = true;
                if let Some(Ok(values)) = columns.get_mut(source_id) {
                    values.resize(i, None);
                    values.push(task.partition.get(*index).cloned().flatten());
                }
            }
        }

        for values in columns.values_mut().flatten() {
            values.resize(n, None);
        }

        Self { columns, present }
    }
}

/// `values` with the entries where `present` is false replaced by `default` (length 1).
fn fill_absent(
    values: Box<dyn Array>,
    default: &dyn Array,
    present: &[bool],
) -> Result<Box<dyn Array>, String> {
    if present.iter().all(|p| *p) {
        return Ok(values);
    }
    let mut parts: Vec<Box<dyn Array>> = vec![];
    let mut start = 0;
    while start < present.len() {
        let is_present = present[start];
        let end = present[start..]
            .iter()
            .position(|p| *p != is_present)
            .map_or(present.len(), |len| start + len);
        if is_present {
            parts.push(values.sliced(start, end - start));
        } else {
            parts.extend((start..end).map(|_| default.to_boxed()));
        }
        start = end;
    }
    let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
    concatenate(&refs).map_err(|e| e.to_string())
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
                .map(|t| {
                    t.file
                        .null_value_count(field.id)
                        .and_then(|v| u64::try_from(v).ok())
                })
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

/// Null `{name}_nc`, `{name}_min`, `{name}_max` for a column whose statistics cannot be loaded.
fn null_column_statistics(field: &NestedField, n: usize) -> Vec<(String, Box<dyn Array>)> {
    let name = &field.name;
    let ty = &field.field_type;
    vec![
        (
            format!("{name}_nc"),
            new_null_array(null_count_dtype(ty), n),
        ),
        (format!("{name}_min"), new_null_array(value_dtype(ty), n)),
        (format!("{name}_max"), new_null_array(value_dtype(ty), n)),
    ]
}

/// Iceberg single-value binary serialization of a partition value.
fn datum_to_bytes(d: &Datum, ty: &Type) -> Vec<u8> {
    match d {
        Datum::Bool(b) => vec![u8::from(*b)],
        // Decimal bounds are big-endian; a writer can store a decimal partition value as an
        // Avro int or long.
        Datum::Int(v) if matches!(ty, Type::Primitive(PrimitiveType::Decimal { .. })) => {
            i128::from(*v).to_be_bytes().to_vec()
        },
        Datum::Long(v) if matches!(ty, Type::Primitive(PrimitiveType::Decimal { .. })) => {
            i128::from(*v).to_be_bytes().to_vec()
        },
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

fn non_negative(v: i64, name: &str) -> IcebergResult<u64> {
    u64::try_from(v).map_err(|_| err_invalid_data(format!("negative manifest {name}: {v}")))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use polars_io_ext_ffi::common::FfiErrorKind;

    use super::*;
    use crate::iceberg::error::err_not_implemented;

    #[test]
    fn int_and_long_decimal_partition_values_are_big_endian() {
        let decimal = Type::Primitive(PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        });
        for d in [Datum::Int(-150), Datum::Long(-150)] {
            let bytes = datum_to_bytes(&d, &decimal);
            assert_eq!(
                crate::iceberg::values::decimal_from_be_bytes(&bytes, 9).unwrap(),
                -150
            );
        }
    }

    #[test]
    fn gzip_metadata_is_decompressed() {
        let json = br#"{"format-version": 2}"#;
        assert_eq!(&*decompress_metadata(json).unwrap(), json);

        let mut encoder = flate2::write::GzEncoder::new(vec![], flate2::Compression::default());
        encoder.write_all(json).unwrap();
        let gz = encoder.finish().unwrap();
        assert_eq!(&*decompress_metadata(&gz).unwrap(), json);
    }

    #[test]
    fn unknown_record_count_has_no_total() {
        assert_eq!(add_record_count(Some(10), 5), Some(15));
        assert_eq!(add_record_count(Some(10), 0), Some(10));
        assert_eq!(add_record_count(Some(10), -1), None);
        assert_eq!(add_record_count(None, 5), None);
        assert_eq!(add_record_count(Some(u64::MAX), 1), None);
    }

    #[test]
    fn context_keeps_error_kind() {
        let err = with_context(err_not_implemented("Avro codec 'bzip2'"), "manifest m.avro");
        assert_eq!(err.kind(), FfiErrorKind::NOT_IMPLEMENTED);
        assert_eq!(
            err.message(),
            "iceberg: manifest m.avro: unsupported: Avro codec 'bzip2'"
        );
    }
}
