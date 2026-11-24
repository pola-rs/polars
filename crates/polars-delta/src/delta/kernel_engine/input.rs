//! Validate action sets before Kernel's snapshot replay discards rows and action columns.

use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::arrow::array::{
    Array, ArrayRef, Int32Array, RecordBatch, StringArray, StructArray, UInt64Array,
};
use delta_kernel::arrow::compute::take;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::object_store::{self, GetResultPayload};
use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
use delta_kernel::{EngineData, FileDataReadResultIterator, KernelError, KernelResult};
use futures::stream;
use polars_utils::aliases::PlHashSet;
use polars_utils::pl_str::PlSmallStr;

use super::HostError;
use crate::delta::actions;

pub(super) fn is_commit(name: &str) -> bool {
    name.strip_suffix(".json")
        .is_some_and(|version| version.len() == 20 && version.bytes().all(|b| b.is_ascii_digit()))
}

pub(super) fn is_log_json(name: &str) -> bool {
    is_commit(name) || (name.ends_with(".json") && name.contains(".checkpoint."))
}

pub(super) fn schema_only_payload(
    name: &str,
    bytes: Bytes,
) -> object_store::Result<(GetResultPayload, u64)> {
    // Arrow tokenizes the entire line even when its only action is absent from the projection.
    // Skip canonical file-action lines and validate the surviving lines before decoding.
    let commit = actions::parse_json(&bytes, true).map_err(HostError::from)?;
    if is_commit(name) {
        commit.validate_commit(false).map_err(HostError::from)?;
    }
    let ranges: Vec<_> = actions::schema_only_json(&bytes).collect();
    let len = ranges.iter().map(|range| range.len() as u64).sum();
    let payload = stream::iter(ranges.into_iter().map(move |range| Ok(bytes.slice(range))));
    Ok((GetResultPayload::Stream(Box::pin(payload)), len))
}

#[derive(Eq, Hash, PartialEq)]
struct DvKey {
    storage_type: PlSmallStr,
    path_or_inline: PlSmallStr,
    offset: Option<i32>,
}

#[derive(Default)]
pub(super) struct FileKeys(PlHashSet<(PlSmallStr, Option<DvKey>)>);

impl FileKeys {
    pub(super) fn check_record(&mut self, record: &RecordBatch) -> KernelResult<()> {
        if let (Some(add), Some(remove)) = (
            record.column_by_name("add"),
            record.column_by_name("remove"),
        ) && add.null_count() < add.len()
            && remove.null_count() < remove.len()
            && (0..record.num_rows()).any(|row| add.is_valid(row) && remove.is_valid(row))
        {
            return Err(KernelError::generic("multiple actions in one row"));
        }
        for name in ["add", "remove"] {
            if let Some(action) = record.column_by_name(name).and_then(struct_array) {
                self.check_action(action)?;
            }
        }
        Ok(())
    }

    fn check_action(&mut self, action: &StructArray) -> KernelResult<()> {
        let Some(paths) = string_column(action, "path") else {
            return Ok(());
        };
        let dv = action
            .column_by_name("deletionVector")
            .and_then(struct_array);
        let storage_type = dv.and_then(|dv| string_column(dv, "storageType"));
        let path_or_inline = dv.and_then(|dv| string_column(dv, "pathOrInlineDv"));
        let offset = dv
            .and_then(|dv| dv.column_by_name("offset"))
            .and_then(|array| array.as_any().downcast_ref::<Int32Array>());
        for row in 0..action.len() {
            if action.is_null(row) || paths.is_null(row) {
                continue;
            }
            let key = if dv.is_some_and(|dv| dv.is_valid(row)) {
                let (Some(storage_type), Some(path_or_inline)) = (storage_type, path_or_inline)
                else {
                    continue;
                };
                if storage_type.is_null(row) || path_or_inline.is_null(row) {
                    // Leave malformed descriptors to Kernel's normal action validation.
                    continue;
                }
                Some(DvKey {
                    storage_type: storage_type.value(row).into(),
                    path_or_inline: path_or_inline.value(row).into(),
                    offset: offset.and_then(|a| a.is_valid(row).then(|| a.value(row))),
                })
            } else {
                None
            };
            if !self.0.insert((paths.value(row).into(), key)) {
                return Err(KernelError::generic(
                    "duplicate add/remove logical file in action set",
                ));
            }
        }
        Ok(())
    }
}

fn struct_array(array: &ArrayRef) -> Option<&StructArray> {
    array.as_any().downcast_ref()
}

fn string_column<'a>(array: &'a StructArray, name: &str) -> Option<&'a StringArray> {
    array.column_by_name(name)?.as_any().downcast_ref()
}

const PRESENCE_LEAVES: [(&str, &str, DataType); 5] = [
    ("add", "path", DataType::STRING),
    ("metaData", "id", DataType::STRING),
    ("protocol", "minReaderVersion", DataType::INTEGER),
    ("checkpointMetadata", "version", DataType::LONG),
    ("sidecar", "path", DataType::STRING),
];

const COMMIT_ONLY_ROOTS: [&str; 5] = ["remove", "commitInfo", "txn", "cdc", "domainMetadata"];

fn checkpoint_roots() -> impl Iterator<Item = &'static str> {
    PRESENCE_LEAVES.iter().map(|(root, _, _)| *root)
}

fn commit_roots() -> impl Iterator<Item = &'static str> {
    checkpoint_roots().chain(COMMIT_ONLY_ROOTS)
}

#[derive(Default)]
pub(super) struct CommitActions {
    keys: FileKeys,
    seen: [bool; 2],
}

impl CommitActions {
    pub(super) fn check(
        &mut self,
        batch: Box<dyn EngineData>,
        original_columns: &[usize],
    ) -> KernelResult<Box<dyn EngineData>> {
        let batch = ArrowEngineData::try_from_engine_data(batch)?;
        let record = batch.record_batch();
        if has_mixed_actions(record, commit_roots()) {
            return Err(KernelError::generic("multiple actions in one row"));
        }
        let count = |name| {
            let root = record.column_by_name(name).unwrap();
            root.len() - root.null_count()
        };
        if count("sidecar") > 0 || count("checkpointMetadata") > 0 {
            return Err(KernelError::generic(
                "checkpoint-only action in Delta commit",
            ));
        }
        for (root, seen) in ["protocol", "metaData"].into_iter().zip(&mut self.seen) {
            let count = count(root);
            if count > 1 || (count > 0 && *seen) {
                return Err(KernelError::generic(
                    "duplicate metadata/protocol/checkpointMetadata",
                ));
            }
            *seen |= count > 0;
        }
        self.keys.check_record(record)?;
        Ok(Box::new(ArrowEngineData::new(
            record.project(original_columns)?,
        )))
    }
}

pub(super) fn commit_projection(schema: &SchemaRef) -> KernelResult<SchemaRef> {
    // Arrow JSON preserves each object's root validity with an empty struct.
    let presence = StructType::try_new([])?;
    let fields = commit_roots()
        .filter(|root| schema.field(root).is_none())
        .map(|root| StructField::nullable(root, presence.clone()));
    let fields = schema.fields().cloned().chain(fields);
    Ok(Arc::new(StructType::try_new(fields)?))
}

/// One projected leaf per action is enough to recover the root validity bitmap. Add these
/// roots to the file pass, where there can be more rows than snapshot metadata discovery visits.
pub(super) fn checkpoint_projection(schema: &SchemaRef) -> KernelResult<SchemaRef> {
    let mut fields = schema.fields().cloned().collect::<Vec<_>>();
    for (root, leaf, dtype) in PRESENCE_LEAVES {
        if schema.field(root).is_none() {
            let action = StructType::try_new([StructField::nullable(leaf, dtype)])?;
            fields.push(StructField::nullable(root, action));
        }
    }
    Ok(Arc::new(StructType::try_new(fields)?))
}

fn has_mixed_actions<'a>(batch: &RecordBatch, roots: impl IntoIterator<Item = &'a str>) -> bool {
    let roots = roots
        .into_iter()
        .filter_map(|root| batch.column_by_name(root))
        .collect::<Vec<_>>();
    if roots
        .iter()
        .filter(|root| root.null_count() < root.len())
        .count()
        < 2
    {
        return false;
    }
    (0..batch.num_rows()).any(|row| {
        roots
            .iter()
            .filter(|root| root.is_valid(row))
            .nth(1)
            .is_some()
    })
}

pub(super) fn check_checkpoint_metadata(
    batches: FileDataReadResultIterator,
) -> KernelResult<FileDataReadResultIterator> {
    let mut candidates: [Option<Box<dyn EngineData>>; 2] = [None, None];
    let mut seen = [false; 2];
    for batch in batches {
        let batch = ArrowEngineData::try_from_engine_data(batch?)?;
        let record = batch.record_batch();
        if has_mixed_actions(record, checkpoint_roots()) {
            return Err(KernelError::generic("multiple actions in one row"));
        }
        // Kernel would stop once it has both actions. Check every batch across all
        // checkpoint parts before yielding either, retaining at most two rows.
        for ((root, leaf), (candidate, seen)) in
            [("protocol", "minReaderVersion"), ("metaData", "id")]
                .into_iter()
                .zip(candidates.iter_mut().zip(&mut seen))
        {
            let Some(root) = record.column_by_name(root).and_then(struct_array) else {
                continue;
            };
            let populated = root.len() - root.null_count();
            if populated == 0 {
                continue;
            }
            if *seen || populated > 1 {
                return Err(KernelError::generic(
                    "duplicate metadata/protocol/checkpointMetadata",
                ));
            }
            *seen = true;
            let Some(leaf) = root.column_by_name(leaf) else {
                continue;
            };
            let Some(row) = (0..root.len()).find(|&row| root.is_valid(row) && leaf.is_valid(row))
            else {
                continue;
            };
            let index = UInt64Array::from(vec![row as u64]);
            let row = record
                .columns()
                .iter()
                .map(|column| take(column.as_ref(), &index, None))
                .collect::<Result<Vec<_>, _>>()?;
            let row = RecordBatch::try_new(record.schema(), row)?;
            *candidate = Some(Box::new(ArrowEngineData::new(row)));
        }
    }
    Ok(Box::new(candidates.into_iter().flatten().map(Ok)))
}

pub(super) fn check_checkpoint(
    batch: Box<dyn EngineData>,
    original_columns: &[usize],
    is_sidecar: bool,
) -> KernelResult<Box<dyn EngineData>> {
    let batch = ArrowEngineData::try_from_engine_data(batch)?;
    let record = batch.record_batch();
    if has_mixed_actions(record, checkpoint_roots()) {
        return Err(KernelError::generic("multiple actions in one row"));
    }
    if is_sidecar
        && ["metaData", "protocol", "checkpointMetadata", "sidecar"]
            .iter()
            .filter_map(|name| record.column_by_name(name))
            .any(|root| root.null_count() < root.len())
    {
        return Err(KernelError::invalid_checkpoint(
            "non-file action in sidecar",
        ));
    }
    Ok(Box::new(ArrowEngineData::new(
        record.project(original_columns)?,
    )))
}

/// A missing required metadata field stops Parquet before it can emit even the first batch.
/// Validate action occupancy in a narrow projection before decoding full actions.
pub(super) fn checkpoint_error(
    error: KernelError,
    inspect: impl FnOnce() -> KernelResult<FileDataReadResultIterator>,
) -> KernelError {
    fn missing_format(error: &KernelError) -> bool {
        match error {
            KernelError::Backtraced { source, .. } => missing_format(source),
            KernelError::Generic(message) => {
                message
                    == "Requested field not found in parquet schema, and field is not nullable: format"
            },
            _ => false,
        }
    }
    if missing_format(&error)
        && let Ok(batches) = inspect()
        && batches
            .map(|batch| ArrowEngineData::try_from_engine_data(batch?))
            .take_while(KernelResult::is_ok)
            .flatten()
            .any(|batch| has_mixed_actions(batch.record_batch(), checkpoint_roots()))
    {
        return KernelError::generic("multiple actions in one row");
    }
    error
}
