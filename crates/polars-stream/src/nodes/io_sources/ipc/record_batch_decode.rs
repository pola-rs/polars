use std::io::Cursor;
use std::sync::Arc;

use arrow::array::TryExtend;
use arrow::io::ipc::read::{Dictionaries, ProjectionInfo};
use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_core::utils::arrow::io::ipc::read::common::apply_projection;
use polars_core::utils::arrow::io::ipc::read::{BlockReader, FileMetadata, read_batch};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_utils::IdxSize;

use super::record_batch_data_fetch::RecordBatchData;

pub(super) struct RecordBatchDecoder {
    pub(super) file_metadata: Arc<FileMetadata>,
    pub(super) pl_schema: Arc<Schema>,
    pub(super) projection_info: Arc<Option<ProjectionInfo>>,
    pub(super) dictionaries: Arc<Option<Dictionaries>>,
    pub(super) row_index: Option<RowIndex>,
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
            );

            let chunk = if let Some(ProjectionInfo { map, .. }) = &projection_info {
                // re-order according to projection
                chunk.map(|chunk| apply_projection(chunk, map))
            } else {
                chunk
            };

            let mut df = DataFrame::empty_with_arc_schema(self.pl_schema.clone());
            df.try_extend(Some(chunk))?;

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
