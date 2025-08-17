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
use crate::nodes::io_sources::multi_scan::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use crate::nodes::io_sources::multi_scan::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks,
};

pub mod builder {
    use std::sync::{Arc, Mutex};

    use polars_utils::pl_str::PlSmallStr;

    use super::BatchFnReader;
    use crate::execute::StreamingExecutionState;
    use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
    use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
    use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

    pub struct BatchFnReaderBuilder {
        pub name: PlSmallStr,
        pub reader: Mutex<Option<BatchFnReader>>,
        pub execution_state: Mutex<Option<StreamingExecutionState>>,
    }

    impl FileReaderBuilder for BatchFnReaderBuilder {
        fn reader_name(&self) -> &str {
            &self.name
        }

        fn reader_capabilities(&self) -> ReaderCapabilities {
            ReaderCapabilities::empty()
        }

        fn set_execution_state(&self, execution_state: &StreamingExecutionState) {
            *self.execution_state.lock().unwrap() = Some(execution_state.clone());
        }

        fn build_file_reader(
            &self,
            _source: polars_plan::prelude::ScanSource,
            _cloud_options: Option<Arc<polars_io::cloud::CloudOptions>>,
            scan_source_idx: usize,
        ) -> Box<dyn FileReader> {
            assert_eq!(scan_source_idx, 0);

            let mut reader = self
                .reader
                .try_lock()
                .unwrap()
                .take()
                .expect("BatchFnReaderBuilder called more than once");

            reader.execution_state = Some(self.execution_state.lock().unwrap().clone().unwrap());

            Box::new(reader) as Box<dyn FileReader>
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

pub use get_batch_state::GetBatchState;

mod get_batch_state {
    use polars_io::pl_async::get_runtime;

    use super::{DataFrame, GetBatchFn, PolarsResult, StreamingExecutionState};

    /// Wraps `GetBatchFn` to support peeking.
    pub struct GetBatchState {
        func: GetBatchFn,
        peek: Option<DataFrame>,
    }

    impl GetBatchState {
        pub async fn next(
            mut slf: Self,
            execution_state: StreamingExecutionState,
        ) -> PolarsResult<(Self, Option<DataFrame>)> {
            get_runtime()
                .spawn_blocking({
                    move || unsafe { slf.next_impl(&execution_state).map(|x| (slf, x)) }
                })
                .await
                .unwrap()
        }

        pub async fn peek(
            mut slf: Self,
            execution_state: StreamingExecutionState,
        ) -> PolarsResult<(Self, Option<DataFrame>)> {
            get_runtime()
                .spawn_blocking({
                    move || unsafe { slf.peek_impl(&execution_state).map(|x| (slf, x)) }
                })
                .await
                .unwrap()
        }

        /// # Safety
        /// This may deadlock if the caller is an async executor thread, as the `GetBatchFn` may
        /// be a Python function that re-enters the streaming engine before returning.
        pub unsafe fn peek_impl(
            &mut self,
            state: &StreamingExecutionState,
        ) -> PolarsResult<Option<DataFrame>> {
            if self.peek.is_none() {
                self.peek = (self.func)(state)?;
            }

            Ok(self.peek.clone())
        }

        /// # Safety
        /// This may deadlock if the caller is an async executor thread, as the `GetBatchFn` may
        /// be a Python function that re-enters the streaming engine before returning.
        unsafe fn next_impl(
            &mut self,
            state: &StreamingExecutionState,
        ) -> PolarsResult<Option<DataFrame>> {
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
}

pub struct BatchFnReader {
    pub name: PlSmallStr,
    pub output_schema: Option<SchemaRef>,
    pub get_batch_state: Option<GetBatchState>,
    pub execution_state: Option<StreamingExecutionState>,
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
            projection: _,
            row_index: None,
            pre_slice: None,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines: _,
            callbacks:
                FileReaderCallbacks {
                    mut file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        let execution_state = self.execution_state().clone();

        if file_schema_tx.is_some() && self.output_schema.is_some() {
            _ = file_schema_tx
                .take()
                .unwrap()
                .try_send(self.output_schema.clone().unwrap());
        }

        let mut get_batch_state = self
            .get_batch_state
            .take()
            // If this is ever needed we can buffer
            .expect("unimplemented: BatchFnReader called more than once");

        let verbose = self.verbose;

        if verbose {
            eprintln!("[BatchFnReader]: name: {}", self.name);
        }

        let (mut morsel_sender, morsel_rx) = FileReaderOutputSend::new_serial();

        let handle = spawn(TaskPriority::Low, async move {
            if let Some(mut file_schema_tx) = file_schema_tx {
                let opt_df;

                (get_batch_state, opt_df) =
                    GetBatchState::peek(get_batch_state, execution_state.clone()).await?;

                _ = file_schema_tx
                    .try_send(opt_df.map(|df| df.schema().clone()).unwrap_or_default())
            }

            let mut seq: u64 = 0;
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            let mut n_rows_seen: usize = 0;

            loop {
                let opt_df;

                (get_batch_state, opt_df) =
                    GetBatchState::next(get_batch_state, execution_state.clone()).await?;

                let Some(df) = opt_df else {
                    break;
                };

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

                loop {
                    let opt_df;

                    (get_batch_state, opt_df) =
                        GetBatchState::next(get_batch_state, execution_state.clone()).await?;

                    let Some(df) = opt_df else {
                        break;
                    };

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
    /// # Panics
    /// Panics if `self.execution_state` is `None`.
    fn execution_state(&self) -> &StreamingExecutionState {
        self.execution_state.as_ref().unwrap()
    }
}
