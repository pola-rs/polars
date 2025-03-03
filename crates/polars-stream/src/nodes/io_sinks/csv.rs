use std::cmp::Reverse;
use std::path::PathBuf;

use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::prelude::{CsvWriter, CsvWriterOptions};
use polars_io::SerWriter;
use polars_utils::priority::Priority;

use super::{SinkNode, SinkRecvPort};
use crate::async_executor::spawn;
use crate::async_primitives::linearizer::Linearizer;
use crate::nodes::io_sinks::DEFAULT_SINK_LINEARIZER_BUFFER_SIZE;
use crate::nodes::{JoinHandle, MorselSeq, TaskPriority};

type Linearized = Priority<Reverse<MorselSeq>, Vec<u8>>;
pub struct CsvSinkNode {
    path: PathBuf,
    schema: SchemaRef,
    write_options: CsvWriterOptions,
}
impl CsvSinkNode {
    pub fn new(schema: SchemaRef, path: PathBuf, write_options: CsvWriterOptions) -> Self {
        Self {
            path,
            schema,
            write_options,
        }
    }
}

impl SinkNode for CsvSinkNode {
    fn name(&self) -> &str {
        "csv_sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }

    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: SinkRecvPort,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        // .. -> Encode task
        let rxs = recv_ports_recv.parallel(join_handles);
        // Encode tasks -> IO task
        let (mut lin_rx, lin_txs) =
            Linearizer::<Linearized>::new(num_pipelines, DEFAULT_SINK_LINEARIZER_BUFFER_SIZE);

        // 16MB
        const DEFAULT_ALLOCATION_SIZE: usize = 1 << 24;

        // Encode task.
        //
        // Task encodes the columns into their corresponding CSV encoding.
        join_handles.extend(rxs.into_iter().zip(lin_txs).map(|(mut rx, mut lin_tx)| {
            let schema = self.schema.clone();
            let options = self.write_options.clone();

            spawn(TaskPriority::High, async move {
                // Amortize the allocations over time. If we see that we need to do way larger
                // allocations, we adjust to that over time.
                let mut allocation_size = DEFAULT_ALLOCATION_SIZE;
                let options = options.clone();

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
                        .with_null_value(options.serialize_options.null.clone())
                        .with_quote_style(options.serialize_options.quote_style)
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

                PolarsResult::Ok(())
            })
        }));

        // IO task.
        //
        // Task that will actually do write to the target file.
        let path = self.path.clone();
        let schema = self.schema.clone();
        let include_header = self.write_options.include_header;
        let include_bom = self.write_options.include_bom;
        let io_task = polars_io::pl_async::get_runtime().spawn(async move {
            use tokio::fs::OpenOptions;
            use tokio::io::AsyncWriteExt;

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.as_path())
                .await
                .map_err(|err| polars_utils::_limit_path_len_io_err(path.as_path(), err))?;

            // Write the header
            if include_header || include_bom {
                let mut std_file = file.into_std().await;
                let mut writer = CsvWriter::new(&mut std_file)
                    .include_bom(include_bom)
                    .include_header(include_header)
                    .n_threads(1) // Disable rayon parallelism
                    .batched(&schema)?;
                writer.write_batch(&DataFrame::empty_with_schema(&schema))?;
                file = tokio::fs::File::from_std(std_file);
            }

            while let Some(Priority(_, buffer)) = lin_rx.get().await {
                file.write_all(&buffer).await?;
            }

            PolarsResult::Ok(())
        });
        join_handles.push(spawn(TaskPriority::Low, async move {
            io_task
                .await
                .unwrap_or_else(|e| Err(std::io::Error::from(e).into()))
        }));
    }
}
