use std::cmp::Reverse;
use std::pin::Pin;

use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::SerWriter;
use polars_io::cloud::CloudOptions;
use polars_io::prelude::{CsvWriter, CsvWriterOptions};
use polars_plan::dsl::{SinkOptions, SinkTarget};
use polars_utils::priority::Priority;

use super::{SinkInputPort, SinkNode};
use crate::async_executor::spawn;
use crate::async_primitives::connector::{Receiver, Sender, connector};
use crate::async_primitives::linearizer::Linearizer;
use crate::execute::StreamingExecutionState;
use crate::morsel::MorselSeq;
use crate::nodes::io_sinks::parallelize_receive_task;
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::{JoinHandle, TaskPriority};
use crate::utils::task_handles_ext::AbortOnDropHandle;

type IOSend = Linearizer<Priority<Reverse<MorselSeq>, Vec<u8>>>;

pub struct CsvSinkNode {
    target: SinkTarget,
    schema: SchemaRef,
    sink_options: SinkOptions,
    write_options: CsvWriterOptions,
    cloud_options: Option<CloudOptions>,

    io_tx: Option<Sender<IOSend>>,
    io_task: Option<AbortOnDropHandle<PolarsResult<()>>>,
}
impl CsvSinkNode {
    pub fn new(
        target: SinkTarget,
        schema: SchemaRef,
        sink_options: SinkOptions,
        write_options: CsvWriterOptions,
        cloud_options: Option<CloudOptions>,
    ) -> Self {
        Self {
            target,
            schema,
            sink_options,
            write_options,
            cloud_options,

            io_tx: None,
            io_task: None,
        }
    }
}

impl SinkNode for CsvSinkNode {
    fn name(&self) -> &str {
        "csv-sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }

    fn initialize(&mut self, _state: &StreamingExecutionState) -> PolarsResult<()> {
        let (io_tx, mut io_rx) = connector::<Linearizer<Priority<Reverse<MorselSeq>, Vec<u8>>>>();

        // IO task.
        //
        // Task that will actually do write to the target file.
        let target = self.target.clone();
        let sink_options = self.sink_options.clone();
        let schema = self.schema.clone();
        let options = self.write_options.clone();
        let cloud_options = self.cloud_options.clone();
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::io::AsyncWriteExt;

            let mut file = target
                .open_into_writeable_async(&sink_options, cloud_options.as_ref())
                .await?;

            // Write the header
            if options.include_header || options.include_bom {
                let mut writer = CsvWriter::new(&mut *file)
                    .include_bom(options.include_bom)
                    .include_header(options.include_header)
                    .with_separator(options.serialize_options.separator)
                    .with_line_terminator(options.serialize_options.line_terminator.clone())
                    .with_quote_char(options.serialize_options.quote_char)
                    .with_datetime_format(options.serialize_options.datetime_format.clone())
                    .with_date_format(options.serialize_options.date_format.clone())
                    .with_time_format(options.serialize_options.time_format.clone())
                    .with_float_scientific(options.serialize_options.float_scientific)
                    .with_float_precision(options.serialize_options.float_precision)
                    .with_decimal_comma(options.serialize_options.decimal_comma)
                    .with_null_value(options.serialize_options.null.clone())
                    .with_quote_style(options.serialize_options.quote_style)
                    .with_compression(options.compression)
                    .n_threads(1) // Disable rayon parallelism
                    .batched(&schema)?;
                writer.write_batch(&DataFrame::empty_with_schema(&schema))?;
            }

            let mut file = file.try_into_async_writeable()?;

            while let Ok(mut lin_rx) = io_rx.recv().await {
                while let Some(Priority(_, buffer)) = lin_rx.get().await {
                    file.write_all(&buffer).await?;
                }
            }

            file.sync_on_close(sink_options.sync_on_close).await?;
            file.close().await?;

            PolarsResult::Ok(())
        });

        self.io_tx = Some(io_tx);
        self.io_task = Some(AbortOnDropHandle(io_task));

        Ok(())
    }

    fn spawn_sink(
        &mut self,
        recv_port_rx: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let io_tx = self
            .io_tx
            .take()
            .expect("not initialized / spawn called more than once");
        let pass_rxs = parallelize_receive_task(
            join_handles,
            recv_port_rx,
            state.num_pipelines,
            self.sink_options.maintain_order,
            io_tx,
        );

        // 16MB
        const DEFAULT_ALLOCATION_SIZE: usize = 1 << 24;

        // Encode task.
        //
        // Task encodes the columns into their corresponding CSV encoding.
        join_handles.extend(pass_rxs.into_iter().map(|mut pass_rx| {
            let schema = self.schema.clone();
            let options = self.write_options.clone();

            spawn(TaskPriority::High, async move {
                // Amortize the allocations over time. If we see that we need to do way larger
                // allocations, we adjust to that over time.
                let mut allocation_size = DEFAULT_ALLOCATION_SIZE;
                let options = options.clone();

                while let Ok((mut rx, mut lin_tx)) = pass_rx.recv().await {
                    while let Ok(morsel) = rx.recv().await {
                        let (df, seq, _, consume_token) = morsel.into_inner();

                        let mut buffer = Vec::with_capacity(allocation_size);
                        let mut writer = CsvWriter::new(&mut buffer)
                            .include_bom(false) // Handled once in the IO task.
                            .include_header(false) // Handled once in the IO task.
                            .with_separator(options.serialize_options.separator)
                            .with_line_terminator(options.serialize_options.line_terminator.clone())
                            .with_quote_char(options.serialize_options.quote_char)
                            .with_datetime_format(options.serialize_options.datetime_format.clone())
                            .with_date_format(options.serialize_options.date_format.clone())
                            .with_time_format(options.serialize_options.time_format.clone())
                            .with_float_scientific(options.serialize_options.float_scientific)
                            .with_float_precision(options.serialize_options.float_precision)
                            .with_decimal_comma(options.serialize_options.decimal_comma)
                            .with_null_value(options.serialize_options.null.clone())
                            .with_quote_style(options.serialize_options.quote_style)
                            .with_compression(options.compression)
                            .n_threads(1) // Disable rayon parallelism
                            .batched(&schema)?;

                        writer.write_batch(&df)?;

                        allocation_size = allocation_size.max(buffer.len());
                        if lin_tx.insert(Priority(Reverse(seq), buffer)).await.is_err() {
                            return Ok(());
                        }
                        drop(consume_token); // Keep the consume_token until here to increase the
                        // backpressure.
                    }
                }

                PolarsResult::Ok(())
            })
        }));
    }

    fn finalize(
        &mut self,
        _state: &StreamingExecutionState,
    ) -> Option<Pin<Box<dyn Future<Output = PolarsResult<()>> + Send>>> {
        // If we were never spawned, we need to make sure that the `tx` is taken. This signals to
        // the IO task that it is done and prevents deadlocks.
        drop(self.io_tx.take());

        let io_task = self
            .io_task
            .take()
            .expect("not initialized / finish called more than once");

        // Wait for the IO task to complete.
        Some(Box::pin(async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }))
    }
}
