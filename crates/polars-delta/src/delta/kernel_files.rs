//! Bridge Kernel's selected add batches to the columnar file source used by the exporter.

use delta_kernel::arrow::array::{Array as _, RecordBatch, StructArray as KernelStruct};
use delta_kernel::arrow::ffi::to_ffi;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::scan::ScanMetadata;
use polars_arrow::array::{Array, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_arrow::ffi;
use polars_io_ext_ffi::common::{ArrowArray, ArrowSchema};

use super::error::{Result, invalid};
use super::files::{CheckpointFiles, Source, child};

fn import(batch: RecordBatch) -> Result<Box<dyn Array>> {
    let (array, schema) = to_ffi(&KernelStruct::from(batch).into_data()).map_err(invalid)?;
    // Both implementations use the Arrow C Data Interface layout. The exported array owns
    // its buffers through its release callback, which the Polars import retains.
    unsafe {
        let schema: ffi::ArrowSchema = ArrowSchema::transmute_from(schema).transmute_into();
        let field = ffi::import_field_from_c(&schema).map_err(invalid)?;
        let array: ffi::ArrowArray = ArrowArray::transmute_from(array).transmute_into();
        ffi::import_array_from_c(array, field.dtype).map_err(invalid)
    }
}

fn drop_unused_stats(batch: &mut RecordBatch, selection: &[bool], only_json_commits: bool) {
    let unused = if only_json_commits {
        // The original text retains Boolean and exact decimal bounds for JSON commits.
        "stats_parsed"
    } else if batch.column_by_name("stats_parsed").is_some_and(|typed| {
        typed.as_any().is::<KernelStruct>()
            && (typed.null_count() == 0
                || (0..typed.len()).all(|i| typed.is_valid(i) || selection.get(i) == Some(&false)))
    }) {
        // A valid native struct wins even when one of its children is null.
        "stats"
    } else {
        return;
    };
    if let Ok(index) = batch.schema_ref().index_of(unused) {
        batch.remove_column(index);
    }
}

pub(super) fn append(
    batch: ScanMetadata,
    only_json_commits: bool,
    sources: &mut Vec<Source>,
    files: &mut Vec<(u32, u32)>,
) -> Result<()> {
    let (data, selection) = batch.scan_files.into_parts();
    let data = ArrowEngineData::try_from_engine_data(data).map_err(invalid)?;
    let len = data.record_batch().num_rows();
    if selection.len() == len && !selection.contains(&true) {
        return Ok(());
    }
    let source = u32::try_from(sources.len()).map_err(invalid)?;
    u32::try_from(len - 1).map_err(invalid)?;

    let mut batch = data.into();
    // Imported buffers share the C Data owner of the whole batch. Drop any unused stats
    // column before exporting so surviving columns do not keep its buffers alive.
    drop_unused_stats(&mut batch, &selection, only_json_commits);
    let array = import(batch)?;
    let rows = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| invalid("Kernel scan batch is not a struct"))?;
    let constants = child(rows, "fileConstantValues")
        .and_then(|a| a.as_any().downcast_ref::<StructArray>())
        .ok_or_else(|| invalid("Kernel scan batch is missing file constants"))?;

    let selected = |i| selection.get(i).copied().unwrap_or(true);
    // Keep nullable DV and stats structs, their child buffers, and row positions intact.
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (parent, name) in [
        (rows, "path"),
        (rows, "size"),
        (constants, "partitionValues"),
        (rows, "partitionValues_parsed"),
        (rows, "deletionVector"),
        (rows, "stats"),
        (rows, "stats_parsed"),
    ] {
        if let Some(array) = child(parent, name) {
            fields.push(Field::new(name.into(), array.dtype().clone(), true));
            columns.push(array.to_boxed());
        }
    }
    // The evaluator emits typed partition values from a direct checkpoint column only.
    let validity = selection
        .contains(&false)
        .then(|| Bitmap::from_iter((0..len).map(selected)));
    let adds = StructArray::try_new(ArrowDataType::Struct(fields), len, columns, validity)
        .map_err(invalid)?;
    let adds = CheckpointFiles::new(&adds)?;

    sources.push(Source::Checkpoint(Box::new(adds)));
    files.extend(
        (0..len)
            .filter(|&i| selected(i))
            .map(|i| (source, i as u32)),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use delta_kernel::FilteredEngineData;
    use delta_kernel::arrow::array::builder::{MapBuilder, StringBuilder};
    use delta_kernel::arrow::array::{ArrayRef, BooleanArray, Int32Array, Int64Array, StringArray};
    use delta_kernel::arrow::buffer::NullBuffer;
    use delta_kernel::arrow::datatypes::Field as KernelField;

    use super::*;
    use crate::delta::values::Sv;

    fn struct_array<const N: usize>(columns: [(&str, ArrayRef); N]) -> KernelStruct {
        KernelStruct::from(
            columns
                .into_iter()
                .map(|(name, a)| {
                    (
                        Arc::new(KernelField::new(name, a.data_type().clone(), true)),
                        a,
                    )
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn selected_kernel_rows_keep_nullable_children() {
        let mut partitions = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        for value in [None, Some("ignored"), Some("")] {
            partitions.keys().append_value("p");
            partitions.values().append_option(value);
            partitions.append(true).unwrap();
        }
        let constants = struct_array([("partitionValues", Arc::new(partitions.finish()))]);
        let dv = struct_array([
            (
                "storageType",
                Arc::new(StringArray::from(vec!["p", "p", "p"])),
            ),
            (
                "pathOrInlineDv",
                Arc::new(StringArray::from(vec!["/a", "/b", "/c"])),
            ),
            (
                "offset",
                Arc::new(Int32Array::from(vec![None, Some(4), Some(0)])),
            ),
        ]);
        let typed = struct_array([("numRecords", Arc::new(Int64Array::from(vec![1, 2, 3])))]);
        let (typed_fields, typed_columns, _) = typed.into_parts();
        let typed = KernelStruct::new(
            typed_fields,
            typed_columns,
            Some(NullBuffer::from(vec![true, false, false])),
        );
        let batch = RecordBatch::try_from_iter([
            (
                "path",
                Arc::new(StringArray::from(vec![Some("a%20b"), None, Some("c")])) as ArrayRef,
            ),
            ("size", Arc::new(Int64Array::from(vec![1, -1, 3]))),
            ("fileConstantValues", Arc::new(constants)),
            ("deletionVector", Arc::new(dv)),
            ("stats_parsed", Arc::new(typed)),
            (
                "stats",
                Arc::new(StringArray::from(vec![
                    None,
                    None,
                    Some(r#"{"numRecords":4}"#),
                ])),
            ),
        ])
        .unwrap();
        let mut only_native = batch.clone();
        drop_unused_stats(&mut only_native, &[true, false, false], false);
        assert!(only_native.column_by_name("stats").is_none());
        let scan_files =
            FilteredEngineData::try_new(Box::new(ArrowEngineData::new(batch)), vec![true, false])
                .unwrap();
        let mut sources = Vec::new();
        let mut files = Vec::new();
        append(
            ScanMetadata {
                scan_files,
                scan_file_transforms: Vec::new(),
            },
            false,
            &mut sources,
            &mut files,
        )
        .unwrap();

        let source = &sources[0];
        assert_eq!(files, [(0, 0), (0, 2)]);
        assert_eq!(source.path(0), "a%20b");
        assert_eq!(source.partition_raw(0, "p"), Some(None));
        assert_eq!(source.partition_raw(2, "p"), Some(Some("")));
        assert_eq!(source.dv(0).unwrap().unwrap().id.offset, None);
        assert_eq!(source.dv(2).unwrap().unwrap().id.offset, Some(0));
        let lookup = source.stats_lookup(&[]);
        assert_eq!(
            source.file_stats(0, &[], &lookup).unwrap().num_records,
            Some(1)
        );
        assert_eq!(
            source.file_stats(2, &[], &lookup).unwrap().num_records,
            Some(4)
        );
    }

    #[test]
    fn kernel_uses_original_stats_only_for_proven_json_commits() {
        let mut partitions = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        let raw = [
            Some(r#"{"numRecords":1,"minValues":{"b":false,"typed":false,"decimal":1.001}}"#),
            None,
        ];
        for _ in raw {
            partitions.append(true).unwrap();
        }
        let constants = struct_array([("partitionValues", Arc::new(partitions.finish()))]);
        let typed_bound = Arc::new(struct_array([(
            "typed",
            Arc::new(BooleanArray::from(vec![true; raw.len()])),
        )]));
        let typed = struct_array([
            (
                "numRecords",
                Arc::new(Int64Array::from(vec![None, Some(1)])),
            ),
            ("minValues", typed_bound),
        ]);
        let batch = RecordBatch::try_from_iter([
            (
                "path",
                Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
            ),
            ("size", Arc::new(Int64Array::from(vec![1; raw.len()]))),
            ("fileConstantValues", Arc::new(constants)),
            ("stats_parsed", Arc::new(typed)),
            ("stats", Arc::new(StringArray::from(raw.to_vec()))),
        ])
        .unwrap();
        // The same output from a checkpoint can hide a conflicting native Boolean bound.
        // Even a well-formed JSON pair with the same count must be ignored in that case.
        for only_json_commits in [true, false] {
            let data = Box::new(ArrowEngineData::new(batch.clone()));
            let scan_files = FilteredEngineData::with_all_rows_selected(data);
            let mut sources = Vec::new();
            let mut files = Vec::new();
            append(
                ScanMetadata {
                    scan_files,
                    scan_file_transforms: Vec::new(),
                },
                only_json_commits,
                &mut sources,
                &mut files,
            )
            .unwrap();

            let source = &sources[0];
            let wanted = ["b", "typed", "decimal"];
            let lookup = source.stats_lookup(&wanted);
            let stats = source.file_stats(0, &wanted, &lookup).unwrap();
            if only_json_commits {
                assert_eq!(stats.num_records, Some(1));
                assert_eq!(
                    stats.min,
                    [Sv::Bool(false), Sv::Bool(false), Sv::Num("1.001".into())]
                );
                assert!(source.file_stats(1, &wanted, &lookup).is_none());
            } else {
                assert_eq!(stats.num_records, None);
                assert_eq!(stats.min, [Sv::Null, Sv::Bool(true), Sv::Null]);
                assert_eq!(
                    source.file_stats(1, &wanted, &lookup).unwrap().num_records,
                    Some(1)
                );
            }
        }
    }
}
