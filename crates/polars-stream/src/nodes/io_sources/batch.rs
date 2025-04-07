//! Reads batches from a `dyn Fn`

use async_trait::async_trait;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor::{JoinHandle, TaskPriority, spawn};
use crate::execute::StreamingExecutionState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks,
};

pub mod builder {

    use std::sync::{Arc, Mutex};

    use polars_utils::pl_str::PlSmallStr;

    use super::BatchFnReader;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::FileReader;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;

    pub struct BatchFnReaderBuilder {
        pub name: PlSmallStr,
        pub reader: Mutex<Option<BatchFnReader>>,
    }

    impl FileReaderBuilder for BatchFnReaderBuilder {
        fn reader_name(&self) -> &str {
            &self.name
        }

        fn reader_capabilities(&self) -> ReaderCapabilities {
            ReaderCapabilities::empty()
        }

        fn build_file_reader(
            &self,
            _source: polars_plan::prelude::ScanSource,
            _cloud_options: Option<Arc<polars_io::cloud::CloudOptions>>,
            scan_source_idx: usize,
        ) -> Box<dyn FileReader> {
            assert_eq!(scan_source_idx, 0);

            Box::new(
                self.reader
                    .try_lock()
                    .unwrap()
                    .take()
                    .expect("BatchFnReaderBuilder called more than once"),
            ) as Box<dyn FileReader>
        }
    }

    impl std::fmt::Debug for BatchFnReaderBuilder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("BatchFnReaderBuilder: name: ")?;
            f.write_str(&self.name)?;

            Ok(())
        }
    }
}

pub type GetBatchFn =
    Box<dyn Fn(&StreamingExecutionState) -> PolarsResult<Option<DataFrame>> + Send + Sync>;

/// Wraps `GetBatchFn` to support peeking.
pub struct GetBatchState {
    func: GetBatchFn,
    peek: Option<DataFrame>,
}

impl GetBatchState {
    pub fn peek(&mut self, state: &StreamingExecutionState) -> PolarsResult<Option<&DataFrame>> {
        if self.peek.is_none() {
            self.peek = (self.func)(state)?;
        }

        Ok(self.peek.as_ref())
    }

    pub fn next(&mut self, state: &StreamingExecutionState) -> PolarsResult<Option<DataFrame>> {
        if let Some(df) = self.peek.take() {
            Ok(Some(df))
        } else {
            (self.func)(state)
        }
    }
}

impl From<GetBatchFn> for GetBatchState {
    fn from(func: GetBatchFn) -> Self {
        Self { func, peek: None }
    }
}

pub struct BatchFnReader {
    pub name: PlSmallStr,
    pub output_schema: Option<SchemaRef>,
    pub get_batch_state: Option<GetBatchState>,
    pub verbose: bool,
}

#[async_trait]
impl FileReader for BatchFnReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let BeginReadArgs {
            projected_schema: _,
            row_index: None,
            pre_slice: None,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines: _,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        // Must send this first before we `take()` the GetBatchState.
        if let Some(mut file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.try_send(self._file_schema()?);
        }

        let mut get_batch_state = self
            .get_batch_state
            .take()
            // If this is ever needed we can buffer
            .expect("unimplemented: BatchFnReader called more than once");

        // FIXME: Propagate this from BeginReadArgs.
        let exec_state = StreamingExecutionState::default();

        let verbose = self.verbose;

        if verbose {
            eprintln!("[BatchFnReader]: name: {}", self.name);
        }

        let (mut morsel_sender, morsel_rx) = FileReaderOutputSend::new_serial();

        let handle = spawn(TaskPriority::Low, async move {
            let mut seq: u64 = 0;
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            let mut n_rows_seen: usize = 0;

            while let Some(df) = get_batch_state.next(&exec_state)? {
                n_rows_seen = n_rows_seen.saturating_add(df.height());

                if morsel_sender
                    .send_morsel(Morsel::new(df, MorselSeq::new(seq), source_token.clone()))
                    .await
                    .is_err()
                {
                    break;
                };
                seq = seq.saturating_add(1);
            }

            if let Some(mut row_position_on_end_tx) = row_position_on_end_tx {
                let n_rows_seen = IdxSize::try_from(n_rows_seen)
                    .map_err(|_| polars_err!(bigidx, ctx = "batch reader", size = n_rows_seen))?;

                _ = row_position_on_end_tx.try_send(n_rows_seen)
            }

            if let Some(mut n_rows_in_file_tx) = n_rows_in_file_tx {
                if verbose {
                    eprintln!("[BatchFnReader]: read to end for full row count");
                }

                while let Some(df) = get_batch_state.next(&exec_state)? {
                    n_rows_seen = n_rows_seen.saturating_add(df.height());
                }

                let n_rows_seen = IdxSize::try_from(n_rows_seen)
                    .map_err(|_| polars_err!(bigidx, ctx = "batch reader", size = n_rows_seen))?;

                _ = n_rows_in_file_tx.try_send(n_rows_seen)
            }

            Ok(())
        });

        Ok((morsel_rx, handle))
    }
}

impl BatchFnReader {
    pub fn _file_schema(&mut self) -> PolarsResult<SchemaRef> {
        if self.output_schema.is_none() {
            let exec_state = StreamingExecutionState::default();

            let schema =
                if let Some(df) = self.get_batch_state.as_mut().unwrap().peek(&exec_state)? {
                    df.schema().clone()
                } else {
                    SchemaRef::default()
                };

            self.output_schema = Some(schema);
        }

        Ok(self.output_schema.clone().unwrap())
    }
}
