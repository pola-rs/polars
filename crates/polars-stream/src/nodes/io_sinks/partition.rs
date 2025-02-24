use std::cmp::Reverse;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, PlHashMap};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;

use super::SinkNode;
use crate::async_executor::{spawn, AbortOnDropHandle};
use crate::async_primitives::connector::{connector, Sender};
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::io_sinks::{SinkInput, SinkInputPort, SinkRecvPort};
use crate::nodes::io_sources::PhaseOutcomeToken;
use crate::nodes::{JoinHandle, Morsel, MorselSeq, TaskPriority};
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;

pub trait PartionableSinkNode: Sized + SinkNode + Send + Sync + 'static {
    type SinkOptions: Clone + Send + Sync + 'static;

    fn new(
        path: &Path,
        input_schema: &SchemaRef,
        options: &Self::SinkOptions,
    ) -> impl Future<Output = PolarsResult<Self>> + Send + Sync;
    fn key_to_path(
        keys: &[AnyValue<'static>],
        options: &Self::SinkOptions,
    ) -> PolarsResult<PathBuf>;
}

struct OpenSinkNode {
    tx: Sender<Morsel>,
    node: Box<dyn SinkNode + Send + Sync>,
    join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,

    oneshot_tx: Sender<SinkInput>,
    wg: WaitGroup,
    outcome: PhaseOutcomeToken,
}

struct OpenPartition {
    sink_node: Result<OpenSinkNode, DataFrame>,
    seq: MorselSeq,
}

pub struct PartitionSink<T: PartionableSinkNode> {
    options: T::SinkOptions,

    num_partition_exprs: usize,

    input_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl<T: PartionableSinkNode> PartitionSink<T> {
    pub fn new(
        options: T::SinkOptions,
        num_partition_exprs: usize,
        input_schema: SchemaRef,
    ) -> Self {
        assert!(input_schema.len() > num_partition_exprs);

        let mut output_schema = input_schema.as_ref().clone();
        for _ in 0..num_partition_exprs {
            output_schema.pop();
        }
        let output_schema = Arc::new(output_schema);

        Self {
            options,

            num_partition_exprs,

            input_schema,
            output_schema,
        }
    }
}

fn part_key_colname(i: usize) -> PlSmallStr {
    format_pl_smallstr!("__PL_PARTBY_{i}")
}

const DEFAULT_MAX_OPEN_PARTITION_SINKS: usize = 16;
fn max_open_partitions() -> usize {
    std::env::var("POLARS_MAX_OPEN_PARTITION_SINKS").map_or(DEFAULT_MAX_OPEN_PARTITION_SINKS, |v| {
        v.parse::<usize>()
            .expect("unable to parse POLARS_MAX_OPEN_PARTITION_SINKS")
            .max(1)
    })
}

impl<T: PartionableSinkNode> SinkNode for PartitionSink<T> {
    fn name(&self) -> &str {
        "partition_sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }

    fn spawn_sink(
        &mut self,
        num_pipelines: usize,
        recv_ports_recv: super::SinkRecvPort,
        state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let (handle, rx_receivers) = recv_ports_recv.parallel();
        join_handles.push(handle);

        let (mut linearizer, txs) = Linearizer::<Priority<Reverse<MorselSeq>, Vec<DataFrame>>>::new(
            num_pipelines,
            DEFAULT_LINEARIZER_BUFFER_SIZE,
        );

        // Evaluate and Partition Tasks
        join_handles.extend(
            rx_receivers
                .into_iter()
                .zip(txs)
                .map(|(mut rx_receiver, mut tx)| {
                    let num_partition_exprs = self.num_partition_exprs;
                    let state = state.clone();
                    spawn(TaskPriority::High, async move {
                        while let Ok((_token, outcome, mut rx)) = rx_receiver.recv().await {
                            while let Ok(m) = rx.recv().await {
                                let (mut df, seq, _, _) = m.into_inner();
                                if df.height() == 0 {
                                    continue;
                                }

                                let partition_cols = df.get_columns()
                                    [df.width() - num_partition_exprs..]
                                    .iter()
                                    .map(|c| c.name().clone());

                                // @Incomplete: Stability
                                let partitions = df.partition_by(partition_cols, true)?;
                                if tx.insert(Priority(Reverse(seq), partitions)).await.is_err() {
                                    return Ok(());
                                }
                            }

                            outcome.stop();
                        }

                        Ok(())
                    })
                }),
        );

        let max_open_partitions = max_open_partitions();
        let options = self.options.clone();
        let num_partition_exprs = self.num_partition_exprs;
        let output_schema = self.output_schema.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut open_partitions = PlHashMap::<Vec<AnyValue<'static>>, OpenPartition>::default();

            let mut keys = Vec::with_capacity(num_partition_exprs);
            let source_token = SourceToken::new();

            while let Some(Priority(_, partitions)) = linearizer.get().await {
                for mut df in partitions {
                    assert_ne!(df.height(), 0);
                    keys.clear();
                    df.clear_schema();
                    let columns = unsafe { df.get_columns_mut() };

                    for key_column in columns.drain(columns.len() - num_partition_exprs..) {
                        let key_value = unsafe { key_column.get_unchecked(0) };
                        keys.push(key_value.into_static());
                    }

                    let open_partition = match open_partitions.get_mut(&keys) {
                        Some(open_partition) => open_partition,
                        None => {
                            let sink_node = if open_partitions.len() <= max_open_partitions {
                                let path = T::key_to_path(&keys, &options)?;
                                let (tx, rx) = connector();
                                let mut sink_node = T::new(&path, &output_schema, &options).await?;
                                let sink_input = if sink_node.is_sink_input_parallel() {
                                    SinkInputPort::Parallel(vec![rx])
                                } else {
                                    SinkInputPort::Serial(rx)
                                };
                                let (outcome, wg, sink_input) = SinkInput::from_port(sink_input);
                                let (mut oneshot_tx, oneshot_rx) = connector();
                                let port = SinkRecvPort {
                                    num_pipelines: 1,
                                    recv: oneshot_rx,
                                };

                                let mut join_handles = Vec::new();
                                let state = ExecutionState::new();
                                sink_node.spawn_sink(1, port, &state, &mut join_handles);
                                let join_handles = join_handles
                                    .into_iter()
                                    .map(AbortOnDropHandle::new)
                                    .collect();

                                if oneshot_tx.send(sink_input).await.is_err() {
                                    return Ok(());
                                }

                                Ok(OpenSinkNode {
                                    tx,
                                    node: Box::new(sink_node),
                                    join_handles,

                                    oneshot_tx,
                                    wg,
                                    outcome,
                                })
                            } else {
                                Err(DataFrame::empty_with_schema(&output_schema))
                            };

                            let mut open_partition = OpenPartition {
                                sink_node,
                                seq: MorselSeq::default(),
                            };
                            let key = std::mem::replace(
                                &mut keys,
                                Vec::with_capacity(num_partition_exprs),
                            );
                            let (_, open_partition) = unsafe {
                                open_partitions.insert_unique_unchecked(key, open_partition)
                            };

                            open_partition
                        },
                    };

                    match open_partition.sink_node.as_mut() {
                        Ok(open_sink_node) => {
                            let seq = open_partition.seq;
                            open_partition.seq = open_partition.seq.successor();
                            if open_sink_node
                                .tx
                                .send(Morsel::new(df, seq, source_token.clone()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                            assert!(!source_token.stop_requested());
                        },
                        Err(buffer_df) => buffer_df.vstack_mut_owned_unchecked(df),
                    }
                }
            }

            if open_partitions.len() > max_open_partitions {
                todo!()
            }
            for (_, open_partition) in open_partitions.into_iter() {
                match open_partition.sink_node {
                    Ok(mut sink_node) => {
                        drop(sink_node.oneshot_tx);
                        sink_node.wg.wait().await;

                        // Either the task finished or some error occurred.
                        while let Some(ret) = sink_node.join_handles.next().await {
                            ret?;
                        }
                    },
                    _ => todo!(),
                }
            }

            Ok(())
        }));
    }
}
