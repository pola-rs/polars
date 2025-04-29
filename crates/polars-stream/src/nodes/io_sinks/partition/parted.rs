use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::prelude::row_encode::_get_rows_encoded_ca_unordered;
use polars_core::prelude::{AnyValue, IntoColumn, PlHashSet};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_plan::dsl::{PartitionTargetCallback, SinkOptions};
use polars_utils::pl_str::PlSmallStr;

use super::CreateNewSinkFn;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::execute::StreamingExecutionState;
use crate::nodes::io_sinks::partition::{SinkSender, open_new_sink};
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::io_sinks::{SinkInputPort, SinkNode};
use crate::nodes::{JoinHandle, Morsel, TaskPriority};

pub struct PartedPartitionSinkNode {
    // This is not be the same as the input_schema, e.g. when include_key=false then this will not
    // include the keys columns.
    sink_input_schema: SchemaRef,

    key_cols: Arc<[PlSmallStr]>,
    base_path: Arc<PathBuf>,
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
}

const DEFAULT_RETIRE_TASKS: usize = 1;
impl PartedPartitionSinkNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_schema: SchemaRef,
        key_cols: Arc<[PlSmallStr]>,
        base_path: Arc<PathBuf>,
        file_path_cb: Option<PartitionTargetCallback>,
        create_new: CreateNewSinkFn,
        ext: PlSmallStr,
        sink_options: SinkOptions,
        include_key: bool,
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
            sink_input_schema,
            key_cols,
            base_path,
            file_path_cb,
            create_new,
            ext,
            sink_options,
            num_retire_tasks,
            include_key,
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

    fn spawn_sink(
        &mut self,
        mut recv_port_recv: Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        // Main Task -> Retire Tasks
        let (mut retire_tx, retire_rxs) = distributor_channel(self.num_retire_tasks, 1);

        // Whether an error has been observed in the retire tasks.
        let has_error_occurred = Arc::new(AtomicBool::new(false));

        // Main Task.
        //
        // Takes the morsels coming in and passes them to underlying sink.
        let state = state.clone();
        let sink_input_schema = self.sink_input_schema.clone();
        let key_cols = self.key_cols.clone();
        let base_path = self.base_path.clone();
        let file_path_cb = self.file_path_cb.clone();
        let create_new = self.create_new.clone();
        let ext = self.ext.clone();
        let include_key = self.include_key;
        let retire_error = has_error_occurred.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            struct CurrentSink {
                sender: SinkSender,
                join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
                value: AnyValue<'static>,
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
                        if retire_error.load(Ordering::Relaxed) {
                            return Ok(());
                        }

                        let mut parted_df;
                        let parted_c;
                        (parted_df, df) = df.split_at(length as i64);
                        (parted_c, c) = c.split_at(length as i64);

                        let value = parted_c.get(0).unwrap().into_static();

                        // If we have a sink open that does not match the value, close it.
                        if let Some(mut current_sink) = current_sink_opt.take() {
                            if current_sink.value != value {
                                let join_handles = std::mem::take(&mut current_sink.join_handles);
                                drop(current_sink_opt.take());
                                if retire_tx.send(join_handles).await.is_err() {
                                    return Ok(());
                                };
                            } else {
                                current_sink_opt = Some(current_sink);
                            }
                        }

                        let current_sink = match current_sink_opt.as_mut() {
                            Some(c) => c,
                            None => {
                                let result = open_new_sink(
                                    base_path.as_path(),
                                    file_path_cb.as_ref(),
                                    super::default_by_key_file_path_cb,
                                    file_idx,
                                    file_idx,
                                    0,
                                    Some(
                                        parted_df
                                            .select_columns(key_cols.iter().cloned())?
                                            .as_slice(),
                                    ),
                                    &create_new,
                                    sink_input_schema.clone(),
                                    "parted",
                                    ext.as_str(),
                                    verbose,
                                    &state,
                                )
                                .await?;
                                file_idx += 1;
                                let Some((join_handles, sender)) = result else {
                                    return Ok(());
                                };

                                current_sink_opt.insert(CurrentSink {
                                    sender,
                                    value,
                                    join_handles,
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

            if let Some(mut current_sink) = current_sink_opt.take() {
                drop(current_sink.sender);
                while let Some(res) = current_sink.join_handles.next().await {
                    res?;
                }
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
            let has_error_occurred = has_error_occurred.clone();
            spawn(TaskPriority::High, async move {
                while let Ok(mut join_handles) = retire_rx.recv().await {
                    while let Some(ret) = join_handles.next().await {
                        ret.inspect_err(|_| {
                            has_error_occurred.store(true, Ordering::Relaxed);
                        })?;
                    }
                }

                Ok(())
            })
        }));
    }
}
