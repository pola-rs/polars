use std::cmp::Reverse;
use std::collections::VecDeque;
use std::future::Future;
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use polars_core::config::{self, verbose};
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, GroupsType, PlHashMap, PlIndexMap, PlIndexSet};
use polars_core::schema::SchemaRef;
use polars_core::series::IsSorted;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use tokio::sync::{Barrier, Mutex};

use super::SinkNode;
use crate::async_executor::{spawn, AbortOnDropHandle};
use crate::async_primitives::connector::{connector, SendError, Sender};
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
    ) -> PolarsResult<Self>;
    fn key_to_path(
        keys: &[AnyValue<'static>],
        options: &Self::SinkOptions,
    ) -> PolarsResult<PathBuf>;
}

struct OpenSinkNode {
    node: Box<dyn SinkNode + Send + Sync>,
    join_handles: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
    seq: MorselSeq,

    phase_tx: Sender<SinkInput>,
    phase: Option<(WaitGroup, PhaseOutcomeToken, Vec<JoinHandle<PolarsResult<()>>>)>,
}

impl OpenSinkNode {
    async fn start_phase(&mut self, gp_txs: &mut [Option<Vec<Sender<Morsel>>>], num_pipelines: usize) -> Result<(), SinkInput> {
        assert!(self.phase.is_none());

        let mut join_handles = Vec::new();
        let (mut txs, mut rxs) = (0..num_pipelines)
            .map(|_| connector())
            .collect::<(Vec<_>, Vec<_>)>();

    let sink_input = if self.node.is_sink_input_parallel() {
        SinkInputPort::Parallel(rxs)
    } else {
        let (mut linearizer, inserters) =
            Linearizer::new(num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);
        let (mut sink_tx, sink_rx) = connector();
        join_handles.extend(
            rxs.into_iter()
                .zip(inserters)
                .map(|(mut rx, mut inserter)| {
                    spawn(TaskPriority::High, async move {
                        while let Ok(m) = rx.recv().await {
                            if inserter
                                .insert(Priority(Reverse(m.seq()), m))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Ok(())
                    })
                }),
        );

        // Place a linearizer between the RXS and the actual sink receive port.
        join_handles.push(spawn(TaskPriority::High, async move {
            while let Some(Priority(_, m)) = linearizer.get().await {
                // Specifically filter out empty dataframes here.
                if m.df().height() == 0 {
                    continue;
                }

                if sink_tx.send(m).await.is_err() {
                    break;
                }
            }
            Ok(())
        }));
        SinkInputPort::Serial(sink_rx)
    };
    for (gp_tx, tx) in gp_txs.iter_mut().zip(txs) {
        if let Some(gp_tx) = gp_tx {
            gp_tx.push(tx);
        }
    }

    let (outcome, wg, sink_input) = SinkInput::from_port(sink_input);
    self.phase_tx.send(sink_input).await?;
    self.phase = Some((wg, outcome, join_handles));
    Ok(())
    }
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
        self
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

fn open_sink<T: PartionableSinkNode>(
    keys: &[AnyValue<'static>],
    output_schema: &SchemaRef,
    options: &T::SinkOptions,
    gp_txs: &mut [Option<Vec<Sender<Morsel>>>],
    num_pipelines: usize,
) -> PolarsResult<OpenSinkNode> {
    let path = T::key_to_path(&keys, &options)?;
    let mut sink_node = T::new(&path, &output_schema, &options)?;
    let mut join_handles = Vec::new();

    let (mut phase_tx, phase_rx) = connector();
    let port = SinkRecvPort {
        num_pipelines,
        recv: phase_rx,
    };

    let state = ExecutionState::new();
    sink_node.spawn_sink(num_pipelines, port, &state, &mut join_handles);
    let join_handles = join_handles
        .into_iter()
        .map(AbortOnDropHandle::new)
        .collect();

    Ok(
        OpenSinkNode {
            node: Box::new(sink_node),
            join_handles,

            seq: MorselSeq::default(),
            phase_tx,
            phase: None,
        },
    )
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

        struct GlobalPartitions {
            known_partitions: Arc<PlIndexSet<Vec<AnyValue<'static>>>>,
            open_sinks: Vec<OpenSinkNode>,
            txs: Vec<Option<Vec<Sender<Morsel>>>>,
        }

        let global_partitions: Arc<Mutex<GlobalPartitions>> =
            Arc::new(Mutex::new(GlobalPartitions {
                known_partitions: Arc::new(Default::default()),
                open_sinks: Vec::default(),
                txs: (0..num_pipelines).map(|_| Some(Vec::new())).collect(),
            }));
        let barrier = Arc::new(Barrier::new(num_pipelines));

        // Evaluate and Partition Tasks
        join_handles.extend(
            rx_receivers
                .into_iter()
                .enumerate()
                .map(|(i, mut rx_receiver)| {
                    let num_partition_exprs = self.num_partition_exprs;
                    let stable = self.stable;
                    let include_key = self.include_key;
                    let max_open_partitions = self.max_open_partitions;
                    let output_schema = self.output_schema.clone();
                    let options = self.options.clone();
                    let global_partitions = global_partitions.clone();
                    let verbose = config::verbose();
                    let barrier = barrier.clone();

                    spawn(TaskPriority::High, async move {
                        let mut known_partitions =
                            Arc::new(PlIndexSet::<Vec<AnyValue<'static>>>::default());
                        let mut unknown_idxs = Vec::new();
                        let mut unknown_keys = Vec::new();
                        let mut unknown_morsels = Vec::new();
                        let mut sink_txs = Vec::<Sender<Morsel>>::new();
                        let mut buffering = Vec::<Option<Morsel>>::new();
                        let mut keys = Vec::new();

                        while let Ok((_token, outcome, mut rx)) = rx_receiver.recv().await {
                            { // Critical section that holds the Mutex lock.
                                let mut global_partitions = global_partitions.lock().await;
                                let GlobalPartitions { known_partitions, ref mut open_sinks, ref mut txs } = global_partitions.deref_mut();
                                if !open_sinks.is_empty() && open_sinks[0].phase.is_none() {
                                    let num_open_sinks = open_sinks.len();
                                    for txs in txs.iter_mut() {
                                        *txs = Some(Vec::with_capacity(num_open_sinks));
                                    }
                                    for open_sink in open_sinks.iter_mut() {
                                        if open_sink.start_phase(txs, num_pipelines).await.is_err() {
                                            dbg!("TODO: wait for joinhandles");
                                            return Ok(());
                                        };
                                    }
                                }
                                sink_txs.extend(txs[i].as_mut().unwrap().drain(..));
                                drop(global_partitions);
                            }

                            while let Ok(m) = rx.recv().await {
                                let (df, seq, source_token, _) = m.into_inner();
                                if df.height() == 0 {
                                    continue;
                                }

                                let partition_cols = df.get_columns()
                                    [df.width() - num_partition_exprs..]
                                    .iter()
                                    .map(|c| c.name().clone())
                                    .collect::<Vec<_>>();

                                let mut partitions =
                                    df._partition_by_impl(&partition_cols, stable, true, false)?;

                                for mut df in partitions.drain(..) {
                                    assert_ne!(df.height(), 0);
                                    keys.clear();
                                    df.clear_schema();
                                    let columns = unsafe { df.get_columns_mut() };

                                    for key_column in
                                        columns.drain(columns.len() - num_partition_exprs..)
                                    {
                                        let key_value = unsafe { key_column.get_unchecked(0) };
                                        keys.push(key_value.into_static());
                                    }

                                    let morsel = Morsel::new(df, seq, source_token.clone());
                                    let Some(idx) = known_partitions.get_index_of(&keys) else {
                                        unknown_morsels.push(morsel);
                                        unknown_keys.push(std::mem::replace(
                                            &mut keys,
                                            Vec::with_capacity(num_partition_exprs),
                                        ));
                                        continue;
                                    };

                                    debug_assert!(buffering[idx].is_none());
                                    buffering[idx] = Some(morsel);
                                }

                                if !unknown_keys.is_empty() {
                                    unknown_idxs.clear();

                                    let mut mut_known_partitions = None;

                                    { // Critical section that holds the Mutex lock.
                                        let mut global_partitions = global_partitions.lock().await;
                                        for keys in unknown_keys.drain(..) {
                                            let idx = match global_partitions
                                                .known_partitions
                                                .get_index_of(&keys)
                                            {
                                                None => {
                                                    if global_partitions.known_partitions.len()
                                                        < max_open_partitions
                                                    {
                                                        let mut open_sink =
                                                            open_sink::<T>(
                                                                &keys,
                                                                &output_schema,
                                                                &options,
                                                                &mut global_partitions.txs,
                                                                num_pipelines,
                                                            )?;
                                                        if open_sink.start_phase(&mut global_partitions.txs, num_pipelines).await.is_err() {
                                                            dbg!("TODO: wait for join handles");
                                                            return Ok(());
                                                        }
                                                        global_partitions
                                                            .open_sinks
                                                            .push(open_sink);
                                                    }
                                                    mut_known_partitions
                                                        .get_or_insert_with(|| {
                                                            global_partitions
                                                                .known_partitions
                                                                .as_ref()
                                                                .clone()
                                                        })
                                                        .insert_full(keys)
                                                        .0
                                                },
                                                Some(idx) => idx,
                                            };
                                            unknown_idxs.push(idx);
                                        }

                                        if let Some(mut_known_partitions) = mut_known_partitions {
                                            if verbose && global_partitions.known_partitions.len() <= max_open_partitions && mut_known_partitions.len() > max_open_partitions {
                                                eprintln!("[PartitionSink]: Reached the maximum open partitions, resorting to buffering remaining partitions");
                                            }

                                            global_partitions.known_partitions =
                                                Arc::new(mut_known_partitions);
                                        }
                                        known_partitions =
                                            global_partitions.known_partitions.clone();

                                        let offset = sink_txs.len();
                                        sink_txs.extend(
                                            global_partitions.txs[i].as_mut().unwrap().drain(..)
                                        );
                                        drop(global_partitions);
                                    }

                                    buffering.resize_with(known_partitions.len(), || None);
                                    for ((idx, morsel), buf) in
                                        unknown_idxs.drain(..).zip(unknown_morsels.drain(..))
                                            .zip(buffering.iter_mut())

                                    {
                                        debug_assert!(buf.is_none());
                                        *buf = Some(morsel);
                                    }
                                }

                                let mut sends = FuturesUnordered::from_iter(
                                    sink_txs.iter_mut().zip(buffering.iter_mut()).map(|(sink_tx, buf)| {
                                        let m = buf.take().unwrap_or_else(|| 
                                            Morsel::new(DataFrame::empty(), seq, source_token.clone()));
                                        sink_tx.send(m)
                                    })
                                );
                                while let Some(send) = sends.next().await {
                                    if send.is_err() {
                                        panic!("failed to send");
                                    }
                                }
                            }

                            debug_assert!(buffering.iter().all(|b| b.is_none()));

                            sink_txs.clear();
                            { // Critical section that holds the Mutex lock.
                                let mut global_partitions = global_partitions.lock().await;
                                let _txs = global_partitions.txs[i].take();
                                drop(global_partitions);
                                drop(_txs);
                            }

                            if i == 0 {
                                let mut global_partitions = global_partitions.lock().await;
                                for mut open_sink_node in global_partitions.open_sinks.iter_mut() {
                                    let (wg, outcome, join_handles) = open_sink_node.phase.take().unwrap();

                                    // @NOTE: The combination of the following two are essentially
                                    // a barrier for this open sink.
                                    for handle in join_handles {
                                        handle.await?;
                                    }
                                    wg.wait().await;

                                    if outcome.did_finish() {
                                        // Some error occurred.
                                        while let Some(ret) = open_sink_node.join_handles.next().await {
                                            ret?;
                                        }
                                    }
                                }
                                let num_open_sinks = global_partitions.open_sinks.len();
                                drop(global_partitions);
                            }

                            outcome.stop();
                        }

                        barrier.wait().await;

                        if i == 0 {
                            if verbose {
                                eprintln!("[PartitionSink]: Finished receiving morsels. Closing and flushing partitions.");
                            }

                            {
                                let mut global_partitions = global_partitions.lock().await;
                                for mut open_sink_node in global_partitions.open_sinks.drain(..) {
                                    drop(open_sink_node.phase_tx);
                                    // Either the task finished or some error occurred.
                                    while let Some(ret) = open_sink_node.join_handles.next().await {
                                        ret?;
                                    }
                                }
                                drop(global_partitions);
                            }
                        }

                        assert!(buffering.len() <= max_open_partitions && buffering.iter().all(|b| b.is_none())); // @TODO

                       // for ms in buffering {
                       //     let Ok(mut tx) = buffering_flush_rx.recv().await else {
                       //         return Ok(());
                       //     };
                       //
                       //     for m in ms {
                       //         if tx.send(m).await.is_err() {
                       //             return Ok(());
                       //         }
                       //     }
                       // }

                        Ok(())
                    })
                }),
        );
    }
}
