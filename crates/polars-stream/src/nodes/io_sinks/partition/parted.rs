use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::prelude::row_encode::_get_rows_encoded_ca_unordered;
use polars_core::prelude::{AnyValue, Column, IntoColumn, PlHashSet};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_plan::dsl::{PartitionTargetCallback, SinkFinishCallback, SinkOptions};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::PlPath;
use polars_utils::relaxed_cell::RelaxedCell;

use super::{CreateNewSinkFn, PerPartitionSortBy};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::execute::StreamingExecutionState;
use crate::nodes::io_sinks::metrics::WriteMetrics;
use crate::nodes::io_sinks::partition::{SinkSender, open_new_sink};
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::io_sinks::{SinkInputPort, SinkNode};
use crate::nodes::{JoinHandle, Morsel, TaskPriority};

pub struct PartedPartitionSinkNode {
    input_schema: SchemaRef,
    // This is not be the same as the input_schema, e.g. when include_key=false then this will not
    // include the keys columns.
    sink_input_schema: SchemaRef,

    key_cols: Arc<[PlSmallStr]>,
    base_path: Arc<PlPath>,
    file_path_cb: Option<PartitionTargetCallback>,
    create_new: CreateNewSinkFn,
    ext: PlSmallStr,

    sink_options: SinkOptions,
    include_key: bool,

    /// The number of tasks that get used to wait for finished files. If you are write large enough
    /// files (i.e. they would be formed by multiple morsels) this should almost always be 1. But
    /// if you are writing many small files, this should scan up to allow for your threads to
    /// saturate. In any sane situation this should never go past the amount of threads you have
    /// available.
    ///
    /// This is somewhat proportional to the amount of files open at any given point.
    num_retire_tasks: usize,

    per_partition_sort_by: Option<PerPartitionSortBy>,
    partition_metrics: Arc<Mutex<Vec<Vec<WriteMetrics>>>>,
    finish_callback: Option<SinkFinishCallback>,
}

const DEFAULT_RETIRE_TASKS: usize = 1;
impl PartedPartitionSinkNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_schema: SchemaRef,
        key_cols: Arc<[PlSmallStr]>,
        base_path: Arc<PlPath>,
        file_path_cb: Option<PartitionTargetCallback>,
        create_new: CreateNewSinkFn,
        ext: PlSmallStr,
        sink_options: SinkOptions,
        include_key: bool,
        per_partition_sort_by: Option<PerPartitionSortBy>,
        finish_callback: Option<SinkFinishCallback>,
    ) -> Self {
        assert!(!key_cols.is_empty());

        let mut sink_input_schema = input_schema.clone();
        if !include_key {
            let keys_col_hm = PlHashSet::from_iter(key_cols.iter().map(|s| s.as_str()));
            sink_input_schema = Arc::new(
                sink_input_schema
                    .try_project(
                        input_schema
                            .iter_names()
                            .filter(|n| !keys_col_hm.contains(n.as_str()))
                            .cloned(),
                    )
                    .unwrap(),
            );
        }

        let num_retire_tasks =
            std::env::var("POLARS_PARTED_SINK_RETIRE_TASKS").map_or(DEFAULT_RETIRE_TASKS, |v| {
                v.parse::<usize>()
                    .expect("unable to parse POLARS_PARTED_SINK_RETIRE_TASKS")
                    .max(1)
            });

        Self {
            input_schema,
            sink_input_schema,
            key_cols,
            base_path,
            file_path_cb,
            create_new,
            ext,
            sink_options,
            num_retire_tasks,
            include_key,
            per_partition_sort_by,
            partition_metrics: Arc::new(Mutex::new(Vec::with_capacity(num_retire_tasks))),
            finish_callback,
        }
    }
}

impl SinkNode for PartedPartitionSinkNode {
    fn name(&self) -> &str {
        "partition-parted-sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        false
    }
    fn do_maintain_order(&self) -> bool {
        self.sink_options.maintain_order
    }

    fn initialize(&mut self, _state: &StreamingExecutionState) -> PolarsResult<()> {
        Ok(())
    }

    fn spawn_sink(
        &mut self,
        mut recv_port_recv: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        // Main Task -> Retire Tasks
        let (mut retire_tx, retire_rxs) = distributor_channel(self.num_retire_tasks, 1);

        // Whether an error has been observed in the retire tasks.
        let has_error_occurred = Arc::new(RelaxedCell::from(false));

        // Main Task.
        //
        // Takes the morsels coming in and passes them to underlying sink.
        let task_state = state.clone();
        let sink_input_schema = self.sink_input_schema.clone();
        let key_cols = self.key_cols.clone();
        let base_path = self.base_path.clone();
        let file_path_cb = self.file_path_cb.clone();
        let create_new = self.create_new.clone();
        let ext = self.ext.clone();
        let include_key = self.include_key;
        let retire_error = has_error_occurred.clone();
        let per_partition_sort_by = self.per_partition_sort_by.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            struct CurrentSink {
                sender: SinkSender,
                join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
                value: AnyValue<'static>,
                keys: Vec<Column>,
                node: Box<dyn SinkNode + Send>,
            }

            let verbose = config::verbose();
            let mut file_idx = 0;
            let mut current_sink_opt: Option<CurrentSink> = None;
            let mut lengths = Vec::new();

            while let Ok((outcome, recv_port)) = recv_port_recv.recv().await {
                let mut recv_port = recv_port.serial();
                while let Ok(morsel) = recv_port.recv().await {
                    let (mut df, seq, source_token, consume_token) = morsel.into_inner();
                    if df.height() == 0 {
                        continue;
                    }

                    let mut c = if key_cols.len() == 1 {
                        let idx = df.try_get_column_index(&key_cols[0])?;
                        df.get_columns()[idx].clone()
                    } else {
                        let columns = df.select_columns(key_cols.iter().cloned())?;
                        _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, &columns)?.into_column()
                    };

                    lengths.clear();
                    polars_ops::series::rle_lengths(&c, &mut lengths)?;

                    for &length in &lengths {
                        if retire_error.load() {
                            return Ok(());
                        }

                        let mut parted_df;
                        let parted_c;
                        (parted_df, df) = df.split_at(length as i64);
                        (parted_c, c) = c.split_at(length as i64);

                        let value = parted_c.get(0).unwrap().into_static();

                        // If we have a sink open that does not match the value, close it.
                        if let Some(current_sink) = current_sink_opt.take() {
                            if current_sink.value != value {
                                drop(current_sink.sender);
                                if retire_tx
                                    .send((
                                        current_sink.join_handles,
                                        current_sink.node,
                                        current_sink.keys,
                                    ))
                                    .await
                                    .is_err()
                                {
                                    return Ok(());
                                };
                            } else {
                                current_sink_opt = Some(current_sink);
                            }
                        }

                        let current_sink = match current_sink_opt.as_mut() {
                            Some(c) => c,
                            None => {
                                let keys = parted_df.select_columns(key_cols.iter().cloned())?;
                                let result = open_new_sink(
                                    base_path.as_ref().as_ref(),
                                    file_path_cb.as_ref(),
                                    super::default_by_key_file_path_cb,
                                    file_idx,
                                    file_idx,
                                    0,
                                    Some(keys.as_slice()),
                                    &create_new,
                                    sink_input_schema.clone(),
                                    "parted",
                                    ext.as_str(),
                                    verbose,
                                    &task_state,
                                    per_partition_sort_by.as_ref(),
                                )
                                .await?;
                                file_idx += 1;
                                let Some((join_handles, sender, node)) = result else {
                                    return Ok(());
                                };

                                current_sink_opt.insert(CurrentSink {
                                    sender,
                                    value,
                                    join_handles,
                                    node,
                                    keys,
                                })
                            },
                        };

                        if !include_key {
                            parted_df = parted_df.drop_many(key_cols.iter().cloned());
                        }

                        if current_sink
                            .sender
                            .send(Morsel::new(parted_df, seq, source_token.clone()))
                            .await
                            .is_err()
                        {
                            return Ok(());
                        };
                    }

                    drop(consume_token);
                }

                outcome.stopped();
            }

            if let Some(current_sink) = current_sink_opt.take() {
                drop(current_sink.sender);
                if retire_tx
                    .send((
                        current_sink.join_handles,
                        current_sink.node,
                        current_sink.keys,
                    ))
                    .await
                    .is_err()
                {
                    return Ok(());
                };
            }

            Ok(())
        }));

        // Retire Tasks.
        //
        // If a file is finished someone needs to wait for the sink tasks to finish. Since we don't
        // want to block the main task, we do it in separate tasks. Usually this is only 1 task,
        // but it can be scaled up using an environment variable.
        let has_error_occurred = &has_error_occurred;
        join_handles.extend(retire_rxs.into_iter().map(|mut retire_rx| {
            let global_partition_metrics = self.partition_metrics.clone();
            let has_error_occurred = has_error_occurred.clone();
            let task_state = state.clone();

            spawn(TaskPriority::High, async move {
                let mut partition_metrics = Vec::new();

                while let Ok((mut join_handles, mut node, keys)) = retire_rx.recv().await {
                    while let Some(ret) = join_handles.next().await {
                        ret.inspect_err(|_| {
                            has_error_occurred.store(true);
                        })?;
                    }
                    if let Some(mut metrics) = node.get_metrics()? {
                        metrics.keys = Some(
                            keys.into_iter()
                                .map(|c| c.get(0).unwrap().into_static())
                                .collect(),
                        );
                        partition_metrics.push(metrics);
                    }
                    if let Some(finalize) = node.finalize(&task_state) {
                        finalize.await?;
                    }
                }

                {
                    let mut global_written_partitions = global_partition_metrics.lock().unwrap();
                    global_written_partitions.push(partition_metrics);
                }

                Ok(())
            })
        }));
    }

    fn finalize(
        &mut self,
        _state: &StreamingExecutionState,
    ) -> Option<Pin<Box<dyn Future<Output = PolarsResult<()>> + Send>>> {
        let finish_callback = self.finish_callback.clone();
        let partition_metrics = self.partition_metrics.clone();
        let sink_input_schema = self.sink_input_schema.clone();
        let input_schema = self.input_schema.clone();
        let key_cols = self.key_cols.clone();

        Some(Box::pin(async move {
            if let Some(finish_callback) = &finish_callback {
                let mut written_partitions = partition_metrics.lock().unwrap();
                let written_partitions =
                    std::mem::take::<Vec<Vec<WriteMetrics>>>(written_partitions.as_mut())
                        .into_iter()
                        .flatten()
                        .collect();
                let df = WriteMetrics::collapse_to_df(
                    written_partitions,
                    &sink_input_schema,
                    Some(&input_schema.try_project(key_cols.iter()).unwrap()),
                );
                finish_callback.call(df)?;
            }
            Ok(())
        }))
    }
}
