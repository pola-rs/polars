use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_plan::dsl::SinkOptions;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, format_pl_smallstr};

use super::CreateNewSinkFn;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::execute::StreamingExecutionState;
use crate::nodes::io_sinks::partition::{SinkSender, open_new_sink};
use crate::nodes::io_sinks::{SinkInputPort, SinkNode};
use crate::nodes::{JoinHandle, Morsel, PhaseOutcome, TaskPriority};

pub struct MaxSizePartitionSinkNode {
    input_schema: SchemaRef,
    max_size: IdxSize,

    path_f_string: Arc<PathBuf>,
    create_new: CreateNewSinkFn,

    sink_options: SinkOptions,

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
impl MaxSizePartitionSinkNode {
    pub fn new(
        input_schema: SchemaRef,
        max_size: IdxSize,
        path_f_string: Arc<PathBuf>,
        create_new: CreateNewSinkFn,
        sink_options: SinkOptions,
    ) -> Self {
        assert!(max_size > 0);
        let num_retire_tasks =
            std::env::var("POLARS_MAX_SIZE_SINK_RETIRE_TASKS").map_or(DEFAULT_RETIRE_TASKS, |v| {
                v.parse::<usize>()
                    .expect("unable to parse POLARS_MAX_SIZE_SINK_RETIRE_TASKS")
                    .max(1)
            });

        Self {
            input_schema,
            max_size,
            path_f_string,
            create_new,
            sink_options,
            num_retire_tasks,
        }
    }
}

impl SinkNode for MaxSizePartitionSinkNode {
    fn name(&self) -> &str {
        "partition[max_size]"
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
        let input_schema = self.input_schema.clone();
        let max_size = self.max_size;
        let path_f_string = self.path_f_string.clone();
        let create_new = self.create_new.clone();
        let retire_error = has_error_occurred.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let part_str = PlSmallStr::from_static("part");
            let mut format_args = PlHashMap::with_capacity(1);
            format_args.insert(part_str.clone(), PlSmallStr::EMPTY);

            struct CurrentSink {
                sender: SinkSender,
                join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
                num_remaining: IdxSize,
            }

            let verbose = config::verbose();
            let mut part = 0;
            let mut current_sink_opt = None;

            while let Ok((outcome, recv_port)) = recv_port_recv.recv().await {
                let mut recv_port = recv_port.serial();
                'morsel_loop: while let Ok(mut morsel) = recv_port.recv().await {
                    while morsel.df().height() > 0 {
                        if retire_error.load(Ordering::Relaxed) {
                            return Ok(());
                        }

                        let current_sink = match current_sink_opt.as_mut() {
                            Some(c) => c,
                            None => {
                                *format_args.get_mut(&part_str).unwrap() =
                                    format_pl_smallstr!("{part}");
                                part += 1;

                                let result = open_new_sink(
                                    path_f_string.as_path(),
                                    &create_new,
                                    &format_args,
                                    input_schema.clone(),
                                    "max-size",
                                    verbose,
                                    &state,
                                )
                                .await?;
                                let Some((join_handles, sender)) = result else {
                                    return Ok(());
                                };

                                current_sink_opt.insert(CurrentSink {
                                    sender,
                                    num_remaining: max_size,
                                    join_handles,
                                })
                            },
                        };

                        // If we can send the whole morsel into sink, do that.
                        if morsel.df().height() < current_sink.num_remaining as usize {
                            current_sink.num_remaining -= morsel.df().height() as IdxSize;

                            // This sends the consume token along so that we don't start buffering here
                            // too much. The sinks are very specific about how they handle consume
                            // tokens and we want to keep that behavior.
                            let result = match &mut current_sink.sender {
                                SinkSender::Connector(s) => s.send(morsel).await.ok(),
                                SinkSender::Distributor(s) => s.send(morsel).await.ok(),
                            };

                            if result.is_none() {
                                break 'morsel_loop;
                            }
                            break;
                        }

                        // Else, we need to split up the morsel into what can be sent and what needs to
                        // be passed to the current sink and what needs to be passed to the next sink.
                        let (df, seq, source_token, consume_token) = morsel.into_inner();

                        let (final_sink_df, df) = df.split_at(current_sink.num_remaining as i64);
                        let final_sink_morsel =
                            Morsel::new(final_sink_df, seq, source_token.clone());

                        if current_sink.sender.send(final_sink_morsel).await.is_err() {
                            return Ok(());
                        };

                        let join_handles = std::mem::take(&mut current_sink.join_handles);
                        drop(current_sink_opt.take());
                        if retire_tx.send(join_handles).await.is_err() {
                            return Ok(());
                        };

                        // We consciously keep the consume token for the last sub-morsel sent.
                        morsel = Morsel::new(df, seq, source_token);
                        if let Some(consume_token) = consume_token {
                            morsel.set_consume_token(consume_token);
                        }
                    }
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
