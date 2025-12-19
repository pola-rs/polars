use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{Array, TryExtend};
use arrow::io::ipc::read::{Dictionaries, ProjectionInfo};
use arrow::record_batch::RecordBatchT;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileReader, read_batch, record_batch_num_rows,
};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_utils::IdxSize;
use polars_utils::mmap::MemSlice;

use super::record_batch_data_fetch::RecordBatchData;
// use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;

pub(super) struct RecordBatchDecoder {
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
        // dbg!("start RBD_to_df"); //kdn

        let file_metadata = record_batch_data.file_metadata;
        let projection_info = self.projection_info.as_ref().clone();
        let bytes = record_batch_data.fetched_bytes;
        let idx = record_batch_data.idx;

        let num_rows = record_batch_data.num_rows;
        debug_assert_eq!(row_range.end as usize - row_range.start as usize, num_rows);
        let slice_range = self.slice_range.clone();

        // let pl_schema = file_metadata
        //     .schema
        //     .iter()
        //     .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
        //     .collect::<Schema>();

        let schema = projection_info.as_ref().as_ref().map_or(
            file_metadata.schema.as_ref(),
            |ProjectionInfo { schema, .. }| schema,
        );
        let pl_schema = schema
            .iter()
            .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
            .collect::<Schema>();

        // Amortize allocations -- move to Decoder?
        let mut data_scratch = Vec::new();
        let mut message_scratch = Vec::new();

        // BlockReader
        // kdn TODO: re-use BlockReader ?
        let mut reader = BlockReader::new_with_projection_info(
            Cursor::new(bytes.as_ref()),
            file_metadata.as_ref().clone(),
            projection_info.clone(),
        );

        // kdn TODO: fix this? - we are not re-using scratches
        reader.set_scratches((
            std::mem::take(&mut data_scratch),
            std::mem::take(&mut message_scratch),
        ));

        let dictionaries = self.dictionaries.as_ref().clone().unwrap(); //kdn TODO error handling, no clone

        let length = record_batch_num_rows(
            &mut reader.reader,
            &file_metadata,
            0,
            true,
            &mut message_scratch,
        )?;

        // Create the DataFrame with the appropriate schema and append all the record
        // batches to it. This will perform schema validation as well.
        //kdn TODO comment
        let mut df = if pl_schema.is_empty() {
            DataFrame::empty_with_height(length)
        } else {
            let chunk = read_batch(
                &mut reader.reader,
                &dictionaries,
                &file_metadata.clone(),
                projection_info.as_ref().map(|x| x.columns.as_ref()),
                None, //kdn TODO remaining
                0,
                true,
                &mut message_scratch,
                &mut data_scratch,
            );

            // kdn TODO - projection
            // let chunk = if let Some(ProjectionInfo { map, .. }) = &self.projection_info {
            //     // re-order according to projection
            //     chunk.map(|chunk| apply_projection(chunk, map))
            // } else {
            //     chunk
            // };

            let mut df = DataFrame::empty_with_schema(&pl_schema);
            df.try_extend(Some(chunk))?;

            let slice_start = slice_range.start as i64 - row_range.start as i64; //kdn TODO review boundary check
            debug_assert!((row_range.start as u64) < (slice_range.len() as u64)); //kdn TODO can be equal?
            let slice_len = slice_range.len() - row_range.start as usize;
            df = df.slice(slice_start, slice_len);
            df
        };

        // kdn TODO check slice logic
        if let Some(RowIndex { name, offset }) = &self.row_index {
            let offset = slice_range.start as IdxSize + *offset;
            df = df.with_row_index(name.clone(), Some(offset))?;
        };

        // let fr = FileReader::new(
        //     Cursor::new(bytes.as_ref()),
        //     file_metadata.as_ref().clone(),
        //     None,
        //     None,
        // );
        // df.try_extend(fr.by_ref().next())?;

        // (data_scratch, message_scratch) = reader.take_scratches();
        // df = df.slice(slice.start as i64, slice.len());

        // todo!();
        Ok(df)
    }
}
