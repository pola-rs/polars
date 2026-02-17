use std::io::Cursor;
use std::sync::Arc;

use arrow::array::TryExtend;
use arrow::io::ipc::read::{Dictionaries, ProjectionInfo};
use arrow::io::ipc::write::KeyValueRef;
use polars_core::chunked_array::flags::StatisticsFlags;
use polars_core::frame::DataFrame;
use polars_core::prelude::PlHashMap;
use polars_core::schema::Schema;
use polars_core::utils::arrow::io::ipc::read::common::apply_projection;
use polars_core::utils::arrow::io::ipc::read::{BlockReader, FileMetadata, read_batch};
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::bool::UnsafeBool;

use super::record_batch_data_fetch::RecordBatchData;
use crate::nodes::io_sinks::writers::interface::IPC_RW_RECORD_BATCH_FLAGS_KEY;

pub(super) struct RecordBatchDecoder {
    pub(super) file_metadata: Arc<FileMetadata>,
    pub(super) pl_schema: Arc<Schema>,
    pub(super) projection_info: Arc<Option<ProjectionInfo>>,
    pub(super) dictionaries: Arc<Option<Dictionaries>>,
    pub(super) row_index: Option<RowIndex>,
    pub(super) read_statistics_flags: bool,
    pub(super) checked: UnsafeBool,
}

impl RecordBatchDecoder {
    pub(super) async fn record_batch_data_to_df(
        &self,
        record_batch_data: RecordBatchData,
        // Rows as requested, relative to the start of the Record Batch.
        slice_offset: usize,
        slice_len: usize,
    ) -> PolarsResult<DataFrame> {
        let file_metadata = self.file_metadata.clone();
        let pl_schema = self.pl_schema.clone();
        let projection_info = self.projection_info.as_ref().clone();
        let bytes = record_batch_data.fetched_bytes;
        let block_index = record_batch_data.block_index;

        let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));
        let dictionaries = self.dictionaries.as_ref().as_ref().unwrap();

        let mut data_scratch = Vec::new();
        let mut message_scratch = Vec::new();

        let limit = Some(slice_offset + slice_len);

        // Extract statistics flags from the metadata
        let flags = self
            .read_statistics_flags
            .then(|| reader.record_batch_custom_metadata(&mut message_scratch))
            .transpose()?
            .map(|meta| get_flags(&meta))
            .transpose()?;

        // Create the DataFrame with the appropriate schema based on the data.
        // @NOTE: This empty schema code path is relied upon for `select(pl.len())`
        let mut df = if pl_schema.is_empty() {
            DataFrame::empty_with_height(slice_len)
        } else {
            let chunk = read_batch(
                &mut reader.reader,
                dictionaries,
                &file_metadata.clone(),
                projection_info.as_ref().map(|x| x.columns.as_ref()),
                limit,
                block_index,
                true,
                &mut message_scratch,
                &mut data_scratch,
                self.checked,
            );

            // Apply projection.
            let (chunk, flags) = if let Some(ProjectionInfo { columns, map, .. }) = &projection_info
            {
                let chunk = chunk.map(|chunk| apply_projection(chunk, map));
                let flags = flags.map(|some_flags| project_flags(some_flags, columns, map));
                (chunk, flags)
            } else {
                (chunk, flags)
            };

            let mut df = DataFrame::empty_with_arc_schema(pl_schema.clone());
            df.try_extend(Some(chunk))?;

            // Update df with flags.
            if let Some(Some(flags)) = flags {
                polars_ensure!(flags.len() == df.columns().len(),
                    ComputeError: "IPC metadata flags count ({}) does not match number of columns ({})",
                    flags.len(), df.columns().len()
                );
                unsafe {
                    df.columns_mut().iter_mut().zip(flags).for_each(|(c, f)| {
                        if let Some(f) = f {
                            c.set_flags(f);
                        }
                    })
                }
            }
            df.slice(i64::try_from(slice_offset).unwrap(), slice_len)
        };

        if let Some(RowIndex { name, offset }) = &self.row_index {
            let current_row_offset = record_batch_data
                .row_offset
                .expect("row_index expects row_offset to be provided");
            let offset = current_row_offset + slice_offset as IdxSize + *offset;
            df = df.with_row_index(name.clone(), Some(offset))?;
        };

        Ok(df)
    }
}

fn get_flags(
    metadata: &Option<Vec<KeyValueRef>>,
) -> PolarsResult<Option<Vec<Option<StatisticsFlags>>>> {
    if let Some(metadata) = metadata {
        for kv in metadata {
            if let Some(key) = kv.key()?
                && key == IPC_RW_RECORD_BATCH_FLAGS_KEY
            {
                match kv.value()? {
                    Some(s) => {
                        let flags: Vec<u32> = serde_json::from_str(s)
                            .map_err(|e| polars_err!(ComputeError: "Unable to parse IPC statistics flags: {}", e))?;
                        let flags = flags
                            .iter()
                            .map(|f| StatisticsFlags::from_bits(*f))
                            .collect();
                        return Ok(Some(flags));
                    },
                    None => {
                        polars_bail!(ComputeError: "Expected IPC statistics flags value, found None")
                    },
                }
            }
        }
    };

    Ok(None)
}

fn project_flags(
    flags: Option<Vec<Option<StatisticsFlags>>>,
    columns: &[usize],
    map: &PlHashMap<usize, usize>,
) -> Option<Vec<Option<StatisticsFlags>>> {
    if let Some(inner) = flags {
        let cols: Vec<_> = columns.iter().map(|i| inner[*i]).collect();
        let mut out = cols.clone();

        // NOTE. Because of the way the projection map is generated, the scenario
        // where old != new is not reachable at the time of writing, and therefore
        // not tested.
        map.iter().for_each(|(old, new)| {
            out[*new] = cols[*old];
        });
        Some(out)
    } else {
        flags
    }
}
