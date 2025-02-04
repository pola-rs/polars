use std::cmp::Reverse;
use std::future::Future;
use std::marker::PhantomData;
use std::ops::Range;

use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_mem_engine::ScanPredicate;
use polars_plan::plans::{ScanSource, ScanSources};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::IdxSize;

use super::{SourceNode, SourceOutput};
use crate::async_executor::spawn;
use crate::async_primitives::connector::{connector, Receiver};
use crate::async_primitives::linearizer::{self, Linearizer};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::io_sources::PhaseResult;
use crate::nodes::{JoinHandle, Morsel, MorselSeq, TaskPriority};
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;

#[allow(unused)]
pub enum RowRestrication {
    Slice(Range<usize>),
    Predicate(ScanPredicate),
}

pub struct MultiScanNode<T: MultiScanable> {
    name: String,
    sources: ScanSources,
    projection: Option<Bitmap>,
    _pd: PhantomData<T>,
}

impl<T: MultiScanable> MultiScanNode<T> {
    pub fn new(sources: ScanSources, projection: Option<Bitmap>) -> Self {
        Self {
            name: format!("multi-scan[{}]", T::BASE_NAME),
            sources,

            projection,

            _pd: PhantomData,
        }
    }
}

fn process_dataframe(df: DataFrame) -> PolarsResult<DataFrame> {
    Ok(df)
}

pub trait MultiScanable: SourceNode + Sized + Send + Sync {
    const BASE_NAME: &'static str;

    const DOES_PRED_PD: bool;
    const DOES_SLICE_PD: bool;
    const DOES_ROW_INDEX: bool;

    fn new(source: ScanSource) -> impl Future<Output = PolarsResult<Self>> + Send;

    fn with_projection(&mut self, projection: Option<&Bitmap>);
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestrication>);
    fn with_row_index(&mut self, row_index: Option<PlSmallStr>);

    #[allow(unused)]
    fn row_count(&mut self) -> impl Future<Output = PolarsResult<IdxSize>> + Send;
    #[allow(unused)]
    fn schema(&mut self) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
}

enum SourceInput {
    Serial(Receiver<Morsel>),
    Parallel(Vec<Receiver<Morsel>>),
}

impl<T: MultiScanable> SourceNode for MultiScanNode<T> {
    const EFFICIENT_PRED_PD: bool = T::EFFICIENT_PRED_PD;
    const EFFICIENT_SLICE_PD: bool = true;

    fn name(&self) -> &str {
        &self.name
    }

    fn is_source_output_parallel(&self) -> bool {
        false
    }

    fn spawn_source(
        &mut self,
        num_pipelines: usize,
        mut send_port_recv: Receiver<SourceOutput>,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<PlSmallStr>,
    ) -> Receiver<PhaseResult> {
        assert!(unrestricted_row_count.is_none());

        let (mut phase_result_tx, phase_result_rx) = connector();
        let num_concurrent_scans = num_pipelines;

        let sources = &self.sources;
        let projection = &self.projection;

        let (si_send, mut si_recv) = (0..num_concurrent_scans)
            .map(|_| connector::<SourceInput>())
            .collect::<(Vec<_>, Vec<_>)>();

        join_handles.extend(si_send.into_iter().enumerate().map(|(mut i, mut ch_send)| {
            let sources = sources.clone();
            let projection = projection.clone();
            spawn(TaskPriority::High, async move {
                let state = ExecutionState::new();
                let mut join_handles = Vec::new();
                while i < sources.len() {
                    join_handles.clear();

                    let source = sources.at(i).into_owned()?;
                    let (mut output_send, output_recv) = connector();
                    let mut source = T::new(source).await?;
                    source.with_projection(projection.as_ref());
                    let mut phase_result_rx = source.spawn_source(
                        num_pipelines,
                        output_recv,
                        &state,
                        &mut join_handles,
                        None,
                    );

                    // Loop until a phase result indicated that the source is empty.
                    loop {
                        let (tx, rx) = if source.is_source_output_parallel() {
                            let (tx, rx) = (0..num_pipelines)
                                .map(|_| connector())
                                .collect::<(Vec<_>, Vec<_>)>();
                            (SourceOutput::Parallel(tx), SourceInput::Parallel(rx))
                        } else {
                            let (tx, rx) = connector();
                            (SourceOutput::Serial(tx), SourceInput::Serial(rx))
                        };

                        // Wait for the orchestrator task to actually be interested in the output
                        // of this file.
                        if ch_send.send(rx).await.is_err() {
                            return Ok(());
                        };
                        // Start draining the source into the created channels.
                        if output_send.send(tx).await.is_err() {
                            return Ok(());
                        };

                        // Wait for the phase to end.
                        let Ok(phase_result) = phase_result_rx.recv().await else {
                            return Ok(());
                        };
                        if phase_result == PhaseResult::Finished {
                            break;
                        }
                    }

                    for handle in join_handles.iter_mut() {
                        handle.await?;
                    }
                    i += num_concurrent_scans;
                }

                PolarsResult::Ok(())
            })
        }));

        let (mut pass_task_send, pass_task_recv) = (0..num_pipelines)
            .map(|_| {
                connector::<(
                    Receiver<Morsel>,
                    linearizer::Inserter<Priority<Reverse<MorselSeq>, Morsel>>,
                )>()
            })
            .collect::<(Vec<_>, Vec<_>)>();
        join_handles.extend(pass_task_recv.into_iter().map(|mut pass_task_recv| {
            spawn(TaskPriority::High, async move {
                while let Ok((mut recv, mut send)) = pass_task_recv.recv().await {
                    while let Ok(v) = recv.recv().await {
                        if send.insert(Priority(Reverse(v.seq()), v)).await.is_err() {
                            break;
                        }
                    }
                }

                Ok(())
            })
        }));

        let sources = sources.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut seq = MorselSeq::default();
            let mut current_scan = 0;

            // Every phase we are given a new send channel.
            'send_port_loop: while let Ok(send) = send_port_recv.recv().await {
                // @TODO: Make this parallel compatible if there is no row count or slice.
                let mut send = send.serial();

                let source_token = SourceToken::new();
                let wait_group = WaitGroup::default();

                while current_scan < sources.len() {
                    let si_recv = &mut si_recv[current_scan % num_concurrent_scans];
                    let Ok(rx) = si_recv.recv().await else {
                        panic!()
                    };

                    match rx {
                        SourceInput::Serial(mut rx) => {
                            while let Ok(rg) = rx.recv().await {
                                let original_source_token = rg.source_token().clone();

                                let df = rg.into_df();
                                let df = process_dataframe(df)?;

                                let mut morsel = Morsel::new(df, seq, source_token.clone());
                                morsel.set_consume_token(wait_group.token());
                                seq = seq.successor();

                                if send.send(morsel).await.is_err() {
                                    break 'send_port_loop;
                                }

                                wait_group.wait().await;
                                if source_token.stop_requested() {
                                    original_source_token.stop();

                                    _ = phase_result_tx.send(PhaseResult::Stopped).await;
                                    continue 'send_port_loop;
                                }
                            }
                        },
                        SourceInput::Parallel(rxs) => {
                            let (mut linearizer, inserters) =
                                Linearizer::new(num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);
                            for ((rx, pass_task_send), inserter) in rxs
                                .into_iter()
                                .zip(pass_task_send.iter_mut())
                                .zip(inserters)
                            {
                                if pass_task_send.send((rx, inserter)).await.is_err() {
                                    return Ok(());
                                };
                            }

                            while let Some(rg) = linearizer.get().await {
                                let rg = rg.1;

                                let original_source_token = rg.source_token().clone();

                                let df = rg.into_df();
                                let df = process_dataframe(df)?;

                                let mut morsel = Morsel::new(df, seq, source_token.clone());
                                morsel.set_consume_token(wait_group.token());
                                seq = seq.successor();

                                if send.send(morsel).await.is_err() {
                                    break 'send_port_loop;
                                }

                                wait_group.wait().await;
                                if source_token.stop_requested() {
                                    original_source_token.stop();

                                    _ = phase_result_tx.send(PhaseResult::Stopped).await;
                                    continue 'send_port_loop;
                                }
                            }
                        },
                    }

                    current_scan += 1;
                }

                _ = phase_result_tx.send(PhaseResult::Finished).await;
                break;
            }

            Ok(())
        }));

        phase_result_rx
    }
}
