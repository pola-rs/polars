use std::cmp::Reverse;
use std::future::Future;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use polars_core::frame::column::ScalarColumn;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_mem_engine::ScanPredicate;
use polars_plan::plans::hive::HivePartitions;
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
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    output_schema: SchemaRef,
    allow_missing_columns: bool,
    _pd: PhantomData<T>,
}

impl<T: MultiScanable> MultiScanNode<T> {
    pub fn new(
        sources: ScanSources,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        output_schema: SchemaRef,
        allow_missing_columns: bool,
    ) -> Self {
        Self {
            name: format!("multi-scan[{}]", T::BASE_NAME),
            sources,
            hive_parts,
            output_schema,
            allow_missing_columns,

            _pd: PhantomData,
        }
    }
}

fn process_dataframe(
    mut df: DataFrame,
    hive_part: Option<&HivePartitions>,
    output_schema: &Schema,
    allow_missing_columns: bool,
) -> PolarsResult<DataFrame> {
    if let Some(hive_part) = hive_part {
        let height = df.height();
        let mut columns = df.take_columns();

        columns.extend(
            hive_part
                .get_statistics()
                .column_stats()
                .iter()
                .map(|column_stat| {
                    ScalarColumn::from_single_value_series(
                        column_stat.get_min_state().unwrap().clone(),
                        height,
                    )
                    .into_column()
                }),
        );

        df = DataFrame::new_with_height(height, columns)?;
    }

    if allow_missing_columns {
        // @TODO: Do this once per file.

        let mut df_extra = Vec::new();
        let mut output_extra = Vec::new();

        df.schema()
            .field_compare(output_schema, &mut df_extra, &mut output_extra);

        if !df_extra.is_empty() {
            // @TODO: Error message.
            panic!();
        }

        for (_, (name, dtype)) in output_extra {
            df.with_column(Column::new_scalar(
                name.clone(),
                Scalar::null(dtype.clone()),
                df.height(),
            ))
            .unwrap();
        }
    }

    df = df.select(output_schema.iter_names_cloned())?;

    Ok(df)
}

pub trait MultiScanable: SourceNode + Sized + Send + Sync {
    const BASE_NAME: &'static str;

    const DOES_PRED_PD: bool;
    const DOES_SLICE_PD: bool;
    const DOES_ROW_INDEX: bool;

    fn new(source: ScanSource) -> impl Future<Output = PolarsResult<Self>> + Send;

    fn with_projection(&mut self, projection: Option<&Bitmap>);
    #[allow(unused)]
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestrication>);
    #[allow(unused)]
    fn with_row_index(&mut self, row_index: Option<PlSmallStr>);

    #[allow(unused)]
    fn row_count(&mut self) -> impl Future<Output = PolarsResult<IdxSize>> + Send;
    fn schema(&mut self) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
}

enum SourceInput {
    Serial(Receiver<Morsel>),
    Parallel(Vec<Receiver<Morsel>>),
}

impl<T: MultiScanable> SourceNode for MultiScanNode<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_source_output_parallel(&self, _is_receiver_serial: bool) -> bool {
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
        let hive_schema = self
            .hive_parts
            .as_ref()
            .and_then(|p| Some(p.first()?.get_statistics().schema().clone()))
            .unwrap_or_else(|| Arc::new(Schema::default()));

        let (si_send, mut si_recv) = (0..num_concurrent_scans)
            .map(|_| connector::<SourceInput>())
            .collect::<(Vec<_>, Vec<_>)>();

        join_handles.extend(si_send.into_iter().enumerate().map(|(mut i, mut ch_send)| {
            let sources = sources.clone();
            let hive_schema = hive_schema.clone();
            spawn(TaskPriority::High, async move {
                let state = ExecutionState::new();
                let mut join_handles = Vec::new();
                while i < sources.len() {
                    join_handles.clear();

                    let source = sources.at(i).into_owned()?;
                    let (mut output_send, output_recv) = connector();
                    let mut source = T::new(source).await?;

                    let source_schema = source.schema().await?;
                    let projection = source_schema
                        .iter_names()
                        .map(|n| !hive_schema.contains(n))
                        .collect::<Bitmap>();
                    source.with_projection(Some(&projection));
                    let mut phase_result_rx = source.spawn_source(
                        num_pipelines,
                        output_recv,
                        &state,
                        &mut join_handles,
                        None,
                    );

                    // Loop until a phase result indicated that the source is empty.
                    loop {
                        let (tx, rx) = if source.is_source_output_parallel(true) {
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

        let hive_parts = self.hive_parts.clone();
        let allow_missing_columns = self.allow_missing_columns;
        let output_schema = self.output_schema.clone();
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
                    let hive_part = hive_parts.as_deref().map(|parts| &parts[current_scan]);
                    let si_recv = &mut si_recv[current_scan % num_concurrent_scans];
                    let Ok(rx) = si_recv.recv().await else {
                        panic!()
                    };

                    match rx {
                        SourceInput::Serial(mut rx) => {
                            while let Ok(rg) = rx.recv().await {
                                let original_source_token = rg.source_token().clone();

                                let df = rg.into_df();
                                let df = process_dataframe(
                                    df,
                                    hive_part,
                                    output_schema.as_ref(),
                                    allow_missing_columns,
                                );
                                let df = match df {
                                    Ok(df) => df,
                                    Err(err) => {
                                        _ = phase_result_tx.send(PhaseResult::Finished).await;
                                        return Err(err);
                                    },
                                };

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
                                let df = process_dataframe(
                                    df,
                                    hive_part,
                                    output_schema.as_ref(),
                                    allow_missing_columns,
                                );
                                let df = match df {
                                    Ok(df) => df,
                                    Err(err) => {
                                        _ = phase_result_tx.send(PhaseResult::Finished).await;
                                        return Err(err);
                                    },
                                };

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
