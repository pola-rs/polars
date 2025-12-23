use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::TryExtend;
use arrow::io::ipc::read::{Dictionaries, ProjectionInfo};
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_core::utils::arrow::io::ipc::read::common::apply_projection;
use polars_core::utils::arrow::io::ipc::read::{BlockReader, FileMetadata, read_batch};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_utils::IdxSize;

use super::record_batch_data_fetch::RecordBatchData;

pub(super) struct RecordBatchDecoder {
    pub(super) file_metadata: Arc<FileMetadata>,
    pub(super) projection_info: Arc<Option<ProjectionInfo>>,
    pub(super) dictionaries: Arc<Option<Dictionaries>>,
    pub(super) row_index: Option<RowIndex>,
    pub(super) slice_range: Range<usize>,
}

impl RecordBatchDecoder {
    pub(super) async fn record_batch_data_to_df(
        &self,
        record_batch_data: RecordBatchData,
        row_range: Range<IdxSize>,
    ) -> PolarsResult<DataFrame> {
        let file_metadata = self.file_metadata.clone();
        let projection_info = self.projection_info.as_ref().clone();
        let bytes = record_batch_data.fetched_bytes;

        let num_rows = record_batch_data.num_rows;
        debug_assert_eq!(row_range.end as usize - row_range.start as usize, num_rows);

        let slice_range = self.slice_range.clone();

        if slice_range.start >= row_range.end as usize
            || slice_range.end <= row_range.start as usize
        {
            return Ok(DataFrame::empty());
        }

        let schema = projection_info.as_ref().as_ref().map_or(
            file_metadata.schema.as_ref(),
            |ProjectionInfo { schema, .. }| schema,
        );
        let pl_schema = schema
            .iter()
            .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
            .collect::<Schema>();

        let mut data_scratch = Vec::new();
        let mut message_scratch = Vec::new();

        let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));

        let dictionaries = self.dictionaries.as_ref().as_ref().unwrap();
        let length = reader.record_batch_num_rows(&mut message_scratch)?;

        let limit = if slice_range.end != usize::MAX {
            Some(slice_range.end as usize - row_range.start as usize)
        } else {
            None
        };

        // Create the DataFrame with the appropriate schema based on the data.
        let mut df = if pl_schema.is_empty() {
            DataFrame::empty_with_height(length)
        } else {
            let chunk = read_batch(
                &mut reader.reader,
                dictionaries,
                &file_metadata.clone(),
                projection_info.as_ref().map(|x| x.columns.as_ref()),
                limit,
                0,
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

            let mut df = DataFrame::empty_with_schema(&pl_schema);
            df.try_extend(Some(chunk))?;

            let slice_start = std::cmp::max(slice_range.start as usize, row_range.start as usize)
                - row_range.start as usize;
            df = df.slice(i64::try_from(slice_start).unwrap(), slice_range.len());
            df
        };

        if let Some(RowIndex { name, offset }) = &self.row_index {
            let offset = slice_range.start as IdxSize + *offset;
            df = df.with_row_index(name.clone(), Some(offset))?;
        };

        Ok(df)
    }
}
