//! Scan planning: snapshot selection, manifest list and manifest reading, and assignment of
//! delete files to data files.
//!
//! Semantics follow PyIceberg's `DataScan` / `IncrementalAppendScan` planning (as used by the
//! Python resolver), including file order: manifests in manifest-list order, entries in manifest
//! order.
use std::borrow::Cow;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use polars_utils::aliases::{PlHashMap, PlHashSet};

use crate::iceberg::avro::Datum;
use crate::iceberg::error::{
    IcebergResult, err_invalid_data, err_invalid_input, err_not_implemented, with_context,
};
use crate::iceberg::host::Storage;
use crate::iceberg::manifest::{
    CONTENT_DATA, CONTENT_EQUALITY_DELETES, CONTENT_POSITION_DELETES, DELETE_FILE_PATH_FIELD_ID,
    DataFile, MANIFEST_CONTENT_DATA, MANIFEST_CONTENT_DELETES, ManifestFile, STATUS_ADDED,
    parse_manifest, parse_manifest_list,
};
use crate::iceberg::prune::Pruner;
use crate::iceberg::spec::{
    PartitionSpec, PrimitiveType, Schema, Snapshot, Table, Transform, Type,
};

/// Maximum number of manifests fetched and decoded concurrently.
const MANIFEST_CONCURRENCY: usize = 64;

/// What to scan, as requested by the user.
#[derive(Debug, Default, Clone)]
pub struct SnapshotSelection {
    pub snapshot_id: Option<i64>,
    pub from_snapshot_id_exclusive: Option<i64>,
    pub to_snapshot_id_inclusive: Option<i64>,
}

impl SnapshotSelection {
    pub fn is_incremental(&self) -> bool {
        self.from_snapshot_id_exclusive.is_some() || self.to_snapshot_id_inclusive.is_some()
    }
}

/// The schema and version key of a scan, resolved without reading any manifest.
pub struct ResolvedSelection<'a> {
    pub schema: Arc<Schema>,
    pub version_key: String,
    /// The snapshot to scan (non-incremental scans), `None` for an empty table.
    pub snapshot: Option<&'a Snapshot>,
}

/// Resolve the scanned schema and version key. Mirrors `IcebergScanResolver`, including the
/// version key format, so that both paths cache the same way.
pub fn resolve_selection<'a>(
    table: &'a Table,
    selection: &SnapshotSelection,
) -> IcebergResult<ResolvedSelection<'a>> {
    if let Some(snapshot_id) = selection.snapshot_id {
        let (snapshot, schema_id) = table.snapshot_for_scan(snapshot_id)?;
        return Ok(ResolvedSelection {
            schema: table.schema_by_id(schema_id)?.clone(),
            version_key: snapshot.snapshot_id.to_string(),
            snapshot: Some(snapshot),
        });
    }

    let schema = table.current_schema()?.clone();
    let current = table.current_snapshot();

    let version_key = if selection.is_incremental() {
        let end = selection
            .to_snapshot_id_inclusive
            .or(current.map(|s| s.snapshot_id));
        format!(
            "incremental:{}:{}:schema:{}",
            py_opt(selection.from_snapshot_id_exclusive),
            py_opt(end),
            schema.schema_id
        )
    } else {
        current.map_or_else(String::new, |s| s.snapshot_id.to_string())
    };

    Ok(ResolvedSelection {
        schema,
        version_key,
        snapshot: current,
    })
}

/// Python's `str(Optional[int])`.
fn py_opt(v: Option<i64>) -> String {
    v.map_or_else(|| "None".to_owned(), |v| v.to_string())
}

/// A data file to scan, with the delete files that apply to it.
pub struct FileTask {
    pub spec_id: i32,
    /// Partition tuple, in the order of the fields of partition spec `spec_id`.
    pub partition: Vec<Option<Datum>>,
    pub file: DataFile,
    pub deletes: Vec<Arc<DataFile>>,
}

pub struct PlanOptions {
    /// Field IDs whose column statistics are needed.
    pub stats_field_ids: PlHashSet<i32>,
    /// Row filter for pruning manifests and data files, bound to `pruner_schema`.
    pub pruner: Option<Pruner>,
    /// Schema the pruner's filter is bound to (the scanned schema).
    pub pruner_schema: Arc<Schema>,
}

#[derive(Debug, Default)]
pub struct PlanStats {
    pub manifests: usize,
    pub manifests_pruned: usize,
    pub data_files_pruned: usize,
}

pub async fn plan_files(
    storage: &Storage,
    table: &Table,
    selection: &SnapshotSelection,
    resolved: &ResolvedSelection<'_>,
    options: &PlanOptions,
) -> IcebergResult<(Vec<FileTask>, PlanStats)> {
    if selection.is_incremental() {
        return plan_incremental(storage, table, selection, options).await;
    }

    let Some(snapshot) = resolved.snapshot else {
        return Ok((vec![], PlanStats::default()));
    };

    let manifests = read_manifest_list(storage, snapshot).await?;
    plan_manifests(storage, table, manifests, options, |_| true).await
}

async fn plan_incremental(
    storage: &Storage,
    table: &Table,
    selection: &SnapshotSelection,
    options: &PlanOptions,
) -> IcebergResult<(Vec<FileTask>, PlanStats)> {
    if selection.from_snapshot_id_exclusive.is_none()
        && selection.to_snapshot_id_inclusive.is_none()
        && table.current_snapshot().is_none()
    {
        return Ok((vec![], PlanStats::default()));
    }

    let to_snapshot = match selection.to_snapshot_id_inclusive {
        Some(id) => table.snapshot_by_id(id).ok_or_else(|| {
            err_invalid_input(format!("End snapshot not found in table metadata: {id}"))
        })?,
        None => table.current_snapshot().ok_or_else(|| {
            err_invalid_input("End snapshot is not set and table has no current snapshot")
        })?,
    };

    if let Some(from) = selection.from_snapshot_id_exclusive
        && !table
            .ancestors(to_snapshot)
            .any(|s| s.parent_snapshot_id == Some(from))
    {
        return Err(err_invalid_input(format!(
            "Starting snapshot (exclusive) {from} is not a parent ancestor of end snapshot {}",
            to_snapshot.snapshot_id
        )));
    }

    let append_snapshots: Vec<&Snapshot> = table
        .ancestors(to_snapshot)
        .take_while(|s| Some(s.snapshot_id) != selection.from_snapshot_id_exclusive)
        .filter(|s| s.operation() == Some("append"))
        .collect();

    let append_ids: PlHashSet<i64> = append_snapshots.iter().map(|s| s.snapshot_id).collect();

    let mut seen = PlHashSet::default();
    let mut manifests = vec![];
    for snapshot in &append_snapshots {
        for m in read_manifest_list(storage, snapshot).await? {
            if m.content == MANIFEST_CONTENT_DATA
                && m.added_snapshot_id
                    .is_some_and(|id| append_ids.contains(&id))
                && seen.insert(m.path.clone())
            {
                manifests.push(m);
            }
        }
    }

    plan_manifests(storage, table, manifests, options, |entry| {
        entry.status == STATUS_ADDED && entry.snapshot_id.is_some_and(|id| append_ids.contains(&id))
    })
    .await
}

async fn read_manifest_list(
    storage: &Storage,
    snapshot: &Snapshot,
) -> IcebergResult<Vec<ManifestFile>> {
    let Some(path) = &snapshot.manifest_list else {
        return Err(err_not_implemented(
            "snapshots without a manifest list (format v1 inline manifests)",
        ));
    };
    let bytes = storage.get(path).await?;
    parse_manifest_list(&bytes).map_err(|e| with_context(e, format_args!("manifest list {path}")))
}

struct EntryView {
    status: i32,
    snapshot_id: Option<i64>,
}

async fn plan_manifests(
    storage: &Storage,
    table: &Table,
    manifests: Vec<ManifestFile>,
    options: &PlanOptions,
    entry_filter: impl Fn(&EntryView) -> bool,
) -> IcebergResult<(Vec<FileTask>, PlanStats)> {
    let mut stats = PlanStats {
        manifests: manifests.len(),
        ..Default::default()
    };
    let pruner = options.pruner.as_ref().filter(|p| !p.is_trivial());
    let schema = options.pruner_schema.as_ref();

    // Partition-summary pruning of data manifests.
    let manifests: Vec<ManifestFile> = match pruner {
        None => manifests,
        Some(pruner) => manifests
            .into_iter()
            .filter(|m| {
                let keep = m.content != MANIFEST_CONTENT_DATA
                    || table
                        .specs
                        .get(&m.spec_id)
                        .is_none_or(|spec| pruner.manifest_might_match(m, spec, schema));
                if !keep {
                    stats.manifests_pruned += 1;
                }
                keep
            })
            .collect(),
    };

    // Delete manifests older than all data are irrelevant (`_check_sequence_number`).
    let min_data_sequence_number = manifests
        .iter()
        .filter(|m| m.content == MANIFEST_CONTENT_DATA)
        .map(|m| m.min_sequence_number)
        .min()
        .unwrap_or(0);

    let manifests: Vec<ManifestFile> = manifests
        .into_iter()
        .filter(|m| {
            m.content == MANIFEST_CONTENT_DATA
                || (m.content == MANIFEST_CONTENT_DELETES
                    && m.sequence_number >= min_data_sequence_number)
        })
        .collect();

    let mut stats_field_ids = options.stats_field_ids.clone();
    if let Some(pruner) = pruner {
        stats_field_ids.extend(pruner.field_ids());
    }
    let stats_field_ids = Arc::new(stats_field_ids);

    let decoded: Vec<_> = futures::stream::iter(manifests.into_iter().map(|manifest| {
        let storage = storage.clone();
        let stats_field_ids = stats_field_ids.clone();
        async move {
            let bytes = storage.get(&manifest.path).await?;
            let path = manifest.path.clone();
            let decoded = crate::iceberg::runtime::spawn(async move {
                parse_manifest(&bytes, &stats_field_ids)
                    .map_err(|e| with_context(e, format_args!("manifest {path}")))
            })
            .await
            .map_err(|e| err_invalid_data(format!("manifest decode task failed: {e}")))??;
            IcebergResult::Ok((manifest, decoded))
        }
    }))
    .buffered(MANIFEST_CONCURRENCY)
    .try_collect()
    .await?;

    let mut data_files = vec![];
    let mut delete_index = DeleteFileIndex::default();

    for (manifest, decoded) in decoded {
        let spec = table.specs.get(&manifest.spec_id).ok_or_else(|| {
            err_invalid_data(format!("partition spec {} not found", manifest.spec_id))
        })?;
        let partition_order = partition_order(&decoded.partition_field_ids, spec)
            .map_err(|e| with_context(e, format_args!("manifest {}", manifest.path)))?;
        let decimal_fields = decimal_partition_fields(table, spec);

        for mut entry in decoded.entries {
            // Inheritance from the manifest list (`_inherit_from_manifest`).
            if entry.snapshot_id.is_none() {
                entry.snapshot_id = manifest.added_snapshot_id;
            }
            let inherit = manifest.sequence_number == 0 || entry.status == STATUS_ADDED;
            if entry.sequence_number.is_none() && inherit {
                entry.sequence_number = Some(manifest.sequence_number);
            }

            if !entry_filter(&EntryView {
                status: entry.status,
                snapshot_id: entry.snapshot_id,
            }) {
                continue;
            }

            let mut file = entry.file;
            file.partition = partition_order
                .iter()
                .zip(&decimal_fields)
                .map(|(i, is_decimal)| {
                    let value = file.partition.get(*i).cloned().flatten();
                    match value {
                        Some(Datum::Bytes(b)) if *is_decimal => {
                            Some(Datum::Bytes(minimal_twos_complement(&b).to_vec()))
                        },
                        v => v,
                    }
                })
                .collect();
            let sequence_number = entry.sequence_number.unwrap_or(0);

            match file.content {
                CONTENT_DATA => {
                    if let Some(pruner) = pruner
                        && !pruner.file_might_match(
                            manifest.spec_id,
                            spec,
                            schema,
                            &file.partition,
                            &file,
                        )
                    {
                        stats.data_files_pruned += 1;
                        continue;
                    }
                    data_files.push((manifest.spec_id, sequence_number, file))
                },
                // Empty delete files (and deletion vectors) delete nothing; PyIceberg also
                // skips them.
                CONTENT_POSITION_DELETES if file.record_count == 0 => {},
                CONTENT_POSITION_DELETES => {
                    delete_index.add(manifest.spec_id, sequence_number, file)
                },
                CONTENT_EQUALITY_DELETES => {
                    return Err(err_not_implemented(format!(
                        "equality delete files ({})",
                        file.file_path
                    )));
                },
                other => {
                    return Err(err_invalid_data(format!(
                        "unknown data file content {other} ({})",
                        file.file_path
                    )));
                },
            }
        }
    }

    let tasks = data_files
        .into_iter()
        .map(|(spec_id, sequence_number, file)| FileTask {
            deletes: delete_index.for_data_file(spec_id, sequence_number, &file),
            spec_id,
            partition: file.partition.clone(),
            file,
        })
        .collect();
    Ok((tasks, stats))
}

/// For each field of `spec`, the index of its value in the decoded partition tuple. Matches by
/// field ID when the writer recorded them, otherwise by position. A spec field without a value in
/// the tuple is refused: treating it as null would prune and fill in constants wrongly.
fn partition_order(
    decoded_field_ids: &[Option<i32>],
    spec: &PartitionSpec,
) -> IcebergResult<Vec<usize>> {
    let by_id = decoded_field_ids.iter().all(|id| id.is_some()) && !decoded_field_ids.is_empty();
    spec.fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let index = if by_id {
                decoded_field_ids
                    .iter()
                    .position(|id| *id == Some(field.field_id))
            } else {
                (i < decoded_field_ids.len()).then_some(i)
            };
            index.ok_or_else(|| {
                err_not_implemented(format!(
                    "partition field {} not found in manifest partition tuple",
                    field.field_id
                ))
            })
        })
        .collect()
}

/// For each field of `spec`, whether its values are decimals.
fn decimal_partition_fields(table: &Table, spec: &PartitionSpec) -> Vec<bool> {
    spec.fields
        .iter()
        .map(|field| {
            matches!(
                field.transform,
                Transform::Identity | Transform::Truncate(_)
            ) && table.schemas.values().any(|schema| {
                schema.field_by_id(field.source_id).is_some_and(|f| {
                    matches!(f.field_type, Type::Primitive(PrimitiveType::Decimal { .. }))
                })
            })
        })
        .collect()
}

/// Strip redundant sign-extension bytes from a big-endian two's-complement integer. Decimal
/// partition values are Avro `fixed` sized by the precision, so widening the precision (e.g.
/// `decimal(9, 2)` → `decimal(10, 2)`) changes the encoding of equal values: data and delete
/// files written before and after must still match by partition.
fn minimal_twos_complement(be: &[u8]) -> &[u8] {
    let redundant = be
        .windows(2)
        .take_while(|w| (w[0] == 0x00 && w[1] & 0x80 == 0) || (w[0] == 0xFF && w[1] & 0x80 != 0))
        .count();
    &be[redundant..]
}

/// Position delete files (including deletion vectors) indexed by partition and by referenced
/// data file. Mirrors PyIceberg's `DeleteFileIndex`.
#[derive(Default)]
struct DeleteFileIndex {
    by_partition: PlHashMap<(i32, String), PositionDeletes>,
    by_path: PlHashMap<String, PositionDeletes>,
}

#[derive(Default)]
struct PositionDeletes {
    /// `(sequence number, file)`, sorted by sequence number once indexing is done.
    files: Vec<(i64, Arc<DataFile>)>,
    sorted: bool,
}

impl PositionDeletes {
    fn add(&mut self, sequence_number: i64, file: Arc<DataFile>) {
        self.files.push((sequence_number, file));
        self.sorted = false;
    }

    /// Delete files with a sequence number `>=` the data file's.
    fn filter_by_seq(&mut self, sequence_number: i64) -> impl Iterator<Item = &Arc<DataFile>> {
        if !self.sorted {
            self.files.sort_by_key(|(seq, _)| *seq);
            self.sorted = true;
        }
        let start = self
            .files
            .partition_point(|(seq, _)| *seq < sequence_number);
        self.files[start..].iter().map(|(_, f)| f)
    }
}

impl DeleteFileIndex {
    fn add(&mut self, spec_id: i32, sequence_number: i64, file: DataFile) {
        let file = Arc::new(file);
        match referenced_data_file(&file) {
            Some(path) => self.by_path.entry(path).or_default(),
            None => self
                .by_partition
                .entry((spec_id, partition_key(&file.partition)))
                .or_default(),
        }
        .add(sequence_number, file);
    }

    fn for_data_file(
        &mut self,
        spec_id: i32,
        sequence_number: i64,
        data_file: &DataFile,
    ) -> Vec<Arc<DataFile>> {
        let mut out = vec![];
        if self.by_partition.is_empty() && self.by_path.is_empty() {
            return out;
        }

        if let Some(deletes) = self
            .by_partition
            .get_mut(&(spec_id, partition_key(&data_file.partition)))
        {
            out.extend(
                deletes
                    .filter_by_seq(sequence_number)
                    .filter(|d| applies_to_data_file(d, &data_file.file_path))
                    .cloned(),
            );
        }

        if let Some(deletes) = self.by_path.get_mut(&data_file.file_path) {
            out.extend(deletes.filter_by_seq(sequence_number).cloned());
        }

        // A delete file listed in several manifests applies once. Deletion vectors in one
        // Puffin file differ by offset.
        if out.len() > 1 {
            let mut seen = PlHashSet::default();
            out.retain(|d| seen.insert((d.file_path.clone(), d.content_offset)));
        }

        out
    }
}

/// Key matching partition tuples by value. Partition values are compared after type promotion
/// (int -> long, float -> double): a file written before the promotion stores the narrower Avro
/// type in the same spec.
fn partition_key(partition: &[Option<Datum>]) -> String {
    let promoted = partition.iter().map(|v| match v {
        Some(Datum::Int(i)) => Some(Cow::Owned(Datum::Long(i64::from(*i)))),
        Some(Datum::Float(f)) => Some(Cow::Owned(Datum::Double(f64::from(*f)))),
        v => v.as_ref().map(Cow::Borrowed),
    });
    format!("{:?}", promoted.collect::<Vec<_>>())
}

/// The data file a delete file applies to, if it is limited to one.
pub fn referenced_data_file(file: &DataFile) -> Option<String> {
    if let Some(path) = &file.referenced_data_file {
        return Some(path.clone());
    }
    let lower = file.lower_bound(DELETE_FILE_PATH_FIELD_ID)?;
    let upper = file.upper_bound(DELETE_FILE_PATH_FIELD_ID)?;
    (lower == upper)
        .then(|| std::str::from_utf8(lower).ok().map(str::to_owned))
        .flatten()
}

/// Whether the `file_path` bounds of a position delete file can include `data_path`.
fn applies_to_data_file(delete: &DataFile, data_path: &str) -> bool {
    let (Some(lower), Some(upper)) = (
        delete.lower_bound(DELETE_FILE_PATH_FIELD_ID),
        delete.upper_bound(DELETE_FILE_PATH_FIELD_ID),
    ) else {
        return true;
    };
    let path = data_path.as_bytes();
    lower <= path && path <= upper
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iceberg::spec::PartitionField;

    #[test]
    fn decimal_partition_keys_ignore_precision_width() {
        // 1.00 as `decimal(9, 2)` (4 bytes) and `decimal(10, 2)` (5 bytes).
        let narrow = [0x00, 0x00, 0x00, 0x64];
        let wide = [0x00, 0x00, 0x00, 0x00, 0x64];
        assert_eq!(minimal_twos_complement(&narrow), [0x64]);
        assert_eq!(minimal_twos_complement(&wide), [0x64]);

        // -1.00
        assert_eq!(minimal_twos_complement(&[0xFF, 0xFF, 0xFF, 0x9C]), [0x9C]);
        assert_eq!(
            minimal_twos_complement(&[0xFF, 0xFF, 0x00, 0x9C]),
            [0xFF, 0x00, 0x9C]
        );
        // 128 needs a leading zero byte.
        assert_eq!(minimal_twos_complement(&[0x00, 0x00, 0x80]), [0x00, 0x80]);
        assert_eq!(minimal_twos_complement(&[0x00]), [0x00]);
        assert_eq!(minimal_twos_complement(&[]), [] as [u8; 0]);
    }

    #[test]
    fn duplicate_delete_files_apply_once() {
        let delete = |path: &str, content_offset| DataFile {
            content: 1,
            file_path: path.into(),
            content_offset,
            ..Default::default()
        };
        let mut index = DeleteFileIndex::default();
        index.add(0, 2, delete("d1.parquet", None));
        index.add(0, 3, delete("d1.parquet", None));
        index.add(0, 2, delete("d2.parquet", None));
        index.add(0, 2, delete("dv.puffin", Some(4)));
        index.add(0, 2, delete("dv.puffin", Some(40)));

        let data_file = DataFile {
            file_path: "data.parquet".into(),
            ..Default::default()
        };
        let mut paths = index
            .for_data_file(0, 1, &data_file)
            .iter()
            .map(|d| (d.file_path.clone(), d.content_offset))
            .collect::<Vec<_>>();
        paths.sort();
        assert_eq!(
            paths,
            [
                ("d1.parquet".to_string(), None),
                ("d2.parquet".to_string(), None),
                ("dv.puffin".to_string(), Some(4)),
                ("dv.puffin".to_string(), Some(40)),
            ]
        );
    }

    #[test]
    fn partition_order_refuses_missing_fields() {
        let field = |field_id| PartitionField {
            source_id: 1,
            field_id,
            transform: Transform::Identity,
        };
        let spec = PartitionSpec {
            fields: vec![field(1000), field(1001)],
        };

        assert_eq!(
            partition_order(&[Some(1001), Some(1000)], &spec).unwrap(),
            [1, 0]
        );
        assert_eq!(partition_order(&[None, None], &spec).unwrap(), [0, 1]);
        assert!(partition_order(&[Some(1000), Some(1002)], &spec).is_err());
        assert!(partition_order(&[None], &spec).is_err());
        assert!(partition_order(&[], &spec).is_err());
        assert!(
            partition_order(&[], &PartitionSpec { fields: vec![] })
                .unwrap()
                .is_empty()
        );
    }
}
