use std::cmp::Reverse;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, GroupsType, PlHashMap, PlIndexMap, PlIndexSet};
use polars_core::schema::SchemaRef;
use polars_core::series::IsSorted;
use polars_error::PolarsResult;
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
    seq: MorselSeq,

    oneshot_tx: Sender<SinkInput>,
    wg: WaitGroup,
    outcome: PhaseOutcomeToken,
}

pub struct PartitionSink<T: PartionableSinkNode> {
    options: T::SinkOptions,

    num_partition_exprs: usize,

    input_schema: SchemaRef,
    output_schema: SchemaRef,

    include_key: bool,
    stable: bool,
    
    /// How many partition sinks may be open / active at any given time.
    ///
    /// This is needed because platforms put limits on the amount of open files you can have at any
    /// given time. On linux for example this maximum is 1024.
    max_open_partitions: usize,
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

            include_key: false,
            stable: false,

            max_open_partitions: max_open_partitions(),
        }
    }

    pub fn with_max_open_partitions(mut self, max_open_partitions: usize) -> Self {
        self.max_open_partitions = max_open_partitions;
        max_open_partitions
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

async fn open_sink<T: PartionableSinkNode>(
    keys: &[AnyValue<'static>],
    output_schema: &SchemaRef,
    options: &T::SinkOptions,
    num_pipelines: usize,
) -> PolarsResult<(SinkInput, OpenSinkNode)> {
    let path = T::key_to_path(&keys, &options)?;
    let (tx, mut rx) = connector();
    let mut sink_node = T::new(&path, &output_schema, &options).await?;
    let mut join_handles = Vec::new();

    let sink_input = if sink_node.is_sink_input_parallel() {
        let (mut txs, mut rxs) = (0..num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();

        // Place a small distributor between the RX and the actual sink receive port. This allows
        // the sink to actually use parallelism properly.
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut rr = 0;
            while let Ok(m) = rx.recv().await {
                if txs[rr].send(m).await.is_err() {
                    break;
                }
                rr += 1;
                rr %= num_pipelines;
            }

            Ok(())
        }));
        SinkInputPort::Parallel(rxs)
    } else {
        SinkInputPort::Serial(rx)
    };
    let (outcome, wg, sink_input) = SinkInput::from_port(sink_input);
    let (mut oneshot_tx, oneshot_rx) = connector();
    let port = SinkRecvPort {
        num_pipelines,
        recv: oneshot_rx,
    };

    let state = ExecutionState::new();
    sink_node.spawn_sink(num_pipelines, port, &state, &mut join_handles);
    let join_handles = join_handles
        .into_iter()
        .map(AbortOnDropHandle::new)
        .collect();

    Ok((
        sink_input,
        OpenSinkNode {
            tx,
            node: Box::new(sink_node),
            join_handles,

            seq: MorselSeq::default(),
            oneshot_tx,
            wg,
            outcome,
        },
    ))
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
                    let stable = self.stable;
                    let include_key = self.include_key;

                    spawn(TaskPriority::High, async move {
                        while let Ok((_token, outcome, mut rx)) = rx_receiver.recv().await {
                            while let Ok(m) = rx.recv().await {
                                let (df, seq, _, _) = m.into_inner();
                                if df.height() == 0 {
                                    continue;
                                }

                                let partition_cols = df.get_columns()
                                    [df.width() - num_partition_exprs..]
                                    .iter()
                                    .map(|c| c.name().clone())
                                    .collect::<Vec<_>>();

                                let partitions =
                                    df._partition_by_impl(&partition_cols, stable, true, false)?;
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

        let max_open_partitions = self.max_open_partitions;
        let options = self.options.clone();
        let num_partition_exprs = self.num_partition_exprs;
        let output_schema = self.output_schema.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut known_partitions = PlIndexSet::<Vec<AnyValue<'static>>>::default();
            let mut open_sinks = Vec::<OpenSinkNode>::with_capacity(max_open_partitions);
            let mut buffering_partitions = Vec::<DataFrame>::new();

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

                    let idx = match known_partitions.get_index_of(&keys) {
                        None => {
                            let key = std::mem::replace(
                                &mut keys,
                                Vec::with_capacity(num_partition_exprs),
                            );
                            let (idx, _) = known_partitions.insert_full(key);
                            if idx < max_open_partitions {
                                let (sink_input, mut open_sink) =
                                    open_sink::<T>(&keys, &output_schema, &options, num_pipelines)
                                        .await?;
                                if open_sink.oneshot_tx.send(sink_input).await.is_err() {
                                    open_sinks.push(open_sink);
                                    break;
                                };
                                open_sinks.push(open_sink);
                            } else {
                                // If there are too many files open, we need to start buffering the
                                // remaining partitions.
                                buffering_partitions
                                    .push(DataFrame::empty_with_schema(&output_schema));
                            };
                            idx
                        },
                        Some(idx) => idx,
                    };

                    if idx < max_open_partitions {
                        let open_sink_node = &mut open_sinks[idx];
                        let seq = open_sink_node.seq;
                        open_sink_node.seq = seq.successor();

                        if open_sink_node
                            .tx
                            .send(Morsel::new(df, seq, source_token.clone()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                        assert!(!source_token.stop_requested());
                    } else {
                        let buffer_df = &mut buffering_partitions[idx - max_open_partitions];
                        buffer_df.vstack_mut_owned_unchecked(df);
                    }
                }
            }

            if known_partitions.is_empty() {
                return Ok(());
            }

            let mut num_open_sinks = known_partitions.len().min(max_open_partitions);
            for mut open_sink_node in open_sinks.drain(..) {
                drop(open_sink_node.oneshot_tx);
                drop(open_sink_node.tx);
                open_sink_node.wg.wait().await;

                // Either the task finished or some error occurred.
                while let Some(ret) = open_sink_node.join_handles.next().await {
                    ret?;
                }
            }

            // @Improvement: This should be better parallelized.
            for (idx, mut df) in buffering_partitions.into_iter().enumerate() {
                let keys = known_partitions
                    .get_index(idx + max_open_partitions)
                    .unwrap();
                let (sink_input, mut open_sink_node) =
                    open_sink::<T>(keys, &output_schema, &options, num_pipelines).await?;
                if open_sink_node.oneshot_tx.send(sink_input).await.is_ok() {
                    if open_sink_node
                        .tx
                        .send(Morsel::new(df, MorselSeq::default(), source_token.clone()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    assert!(!source_token.stop_requested());
                };

                drop(open_sink_node.oneshot_tx);
                drop(open_sink_node.tx);
                open_sink_node.wg.wait().await;

                // Either the task finished or some error occurred.
                while let Some(ret) = open_sink_node.join_handles.next().await {
                    ret?;
                }
            }

            Ok(())
        }));
    }
}
