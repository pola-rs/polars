use std::fs::{File, OpenOptions};
use std::path::Path;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_io::csv::write::BatchedWriter;
use polars_io::prelude::{CsvWriter, CsvWriterOptions, SerializeOptions};
use polars_io::SerWriter;

use crate::nodes::{ComputeNode, JoinHandle, PortState, TaskPriority, TaskScope};
use crate::pipe::{RecvPort, SendPort};

pub struct CsvSinkNode {
    is_finished: bool,
    writer: BatchedWriter<File>,
}

impl CsvSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        path: &Path,
        write_options: &CsvWriterOptions,
    ) -> PolarsResult<Self> {
        let file = OpenOptions::new().write(true).open(path)?;

        let CsvWriterOptions {
            include_bom,
            include_header,
            batch_size,
            maintain_order: _, // @TODO
            serialize_options,
        } = write_options;

        let SerializeOptions {
            date_format,
            time_format,
            datetime_format,
            float_scientific,
            float_precision,
            separator,
            quote_char,
            null,
            line_terminator,
            quote_style,
        } = serialize_options;

        let writer = CsvWriter::new(file)
            .include_bom(*include_bom)
            .include_header(*include_header)
            .with_batch_size(*batch_size)
            .with_date_format(date_format.clone())
            .with_time_format(time_format.clone())
            .with_datetime_format(datetime_format.clone())
            .with_float_scientific(*float_scientific)
            .with_float_precision(*float_precision)
            .with_separator(*separator)
            .with_quote_char(*quote_char)
            .with_null_value(null.clone())
            .with_line_terminator(line_terminator.clone())
            .with_quote_style(*quote_style)
            .batched(&input_schema)?;

        Ok(Self {
            is_finished: false,
            writer,
        })
    }
}

impl ComputeNode for CsvSinkNode {
    fn name(&self) -> &str {
        "csv_sink"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        if recv[0] == PortState::Done && !self.is_finished {
            // @NOTE: This function can be called afterwards multiple times. So make sure to only
            // finish the writer once.
            self.is_finished = true;
            self.writer.finish()?;
        }

        // We are always ready to receive, unless the sender is done, then we're
        // also done.
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send.is_empty());
        assert!(recv.len() == 1);
        let mut receiver = recv[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                self.writer.write_batch(&morsel.into_df())?;
            }

            Ok(())
        }));
    }
}
