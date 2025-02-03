use std::collections::VecDeque;
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
use polars_utils::IdxSize;

use super::SourceNode;
use crate::async_executor::{spawn, task_scope};
use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::{
    ComputeNode, JoinHandle, Morsel, MorselSeq, PortState, TaskPriority, TaskScope,
};
use crate::pipe::{RecvPort, SendPort};

pub enum RowRestrication {
    Slice(Range<usize>),
    Predicate(ScanPredicate),
}

pub struct RowGroup<T> {
    pub data: T,
    pub row_count: IdxSize,
}

impl RowGroup<Morsel> {
    fn into_df(self) -> RowGroup<DataFrame> {
        RowGroup {
            data: self.data.into_df(),
            row_count: self.row_count,
        }
    }
}

struct Scan<T: MultiScanable> {
    node: T,
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

fn process_dataframe(df: RowGroup<DataFrame>) -> PolarsResult<DataFrame> {
    Ok(df.data)
}

pub trait MultiScanable: SourceNode + Sized + Send + Sync {
    fn new(
        source: ScanSource,
        projection: Option<&Bitmap>,
        row_restriction: Option<RowRestrication>,
        row_index: Option<PlSmallStr>,
    ) -> impl Future<Output = PolarsResult<Self>> + Send;
    fn row_count(&mut self) -> impl Future<Output = PolarsResult<IdxSize>> + Send;
    fn schema(&mut self) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
}

impl<T: MultiScanable> SourceNode for MultiScanNode<T> {
    const BASE_NAME: &'static str = "multi-scan";

    fn name(&self) -> &str {
        &self.name
    }

    fn source_start(
        &self,
        num_pipelines: usize,
        mut send_port_recv: Receiver<Sender<RowGroup<Morsel>>>,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let num_concurrent_scans = num_pipelines;

        let sources = &self.sources;
        let projection = &self.projection;

        let (mut ch_send, ch_recv) = (0..num_concurrent_scans)
            .map(|_| connector::<Sender<RowGroup<Morsel>>>())
            .collect::<(Vec<_>, Vec<_>)>();

        let source_token = SourceToken::new();
        let wait_group = WaitGroup::default();

        join_handles.extend(ch_recv.into_iter().enumerate().map(|(mut i, mut ch_recv)| {
            let sources = sources.clone();
            let projection = projection.clone();
            spawn(TaskPriority::High, async move {
                let state = ExecutionState::new();
                let mut join_handles = Vec::new();
                while let Ok(rg_send) = ch_recv.recv().await {
                    assert!(i < sources.len());
                    join_handles.clear();

                    let source = sources.at(i).into_owned()?;
                    let source = T::new(source, projection.as_ref(), None, None).await?;

                    let (mut tx, rx) = connector();
                    if tx.send(rg_send).await.is_err() {
                        return Ok(());
                    }
                    source.source_start(num_pipelines, rx, &state, &mut join_handles);

                    for handle in join_handles.iter_mut() {
                        handle.await?;
                    }

                    i += num_concurrent_scans;

                    if i >= sources.len() {
                        break;
                    }
                }

                dbg!(i % num_concurrent_scans);

                PolarsResult::Ok(())
            })
        }));

        let _source_token = source_token.clone();
        let sources = sources.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let source_token = _source_token;
            let wait_group = wait_group;

            let mut seq = MorselSeq::default();
            let mut current_scan = 0;
            let mut buffered = None;
            while let Ok(mut send) = send_port_recv.recv().await {
                if let Some(rg) = buffered.take() {
                    if let Err(rg) = send.send(rg).await {
                        buffered = Some(rg);
                        continue;
                    }
                }

                'source_loop: while current_scan < sources.len() {
                    let ch_send = &mut ch_send[current_scan % num_concurrent_scans];

                    let (tx, mut rx) = connector();

                    if ch_send.send(tx).await.is_err() {
                        panic!();
                    }

                    while let Ok(rg) = rx.recv().await {
                        let row_count = rg.row_count;
                        let df = rg.into_df();
                        let df = process_dataframe(df)?;

                        let mut morsel = Morsel::new(df, seq, source_token.clone());
                        morsel.set_consume_token(wait_group.token());
                        seq = seq.successor();

                        let rg = RowGroup {
                            data: morsel,
                            row_count,
                        };
                        if let Err(rg) = send.send(rg).await {
                            buffered = Some(rg);
                            break 'source_loop;
                        }
                    }

                    current_scan += 1;
                }

                if current_scan >= sources.len() {
                    break;
                }
            }

            dbg!("main");

            Ok(())
        }));

        // Opener
        // Decoders[n]
        // Row Index, Predicate

        // join_handles.push(scope.spawn_task(TaskPriority::High, async move {
        //     let source_token = SourceToken::new();
        //     let wait_group = WaitGroup::default();
        //
        //     while scans.len() < self.num_concurrent_scans && *next_source < sources.len() {
        //         scans.push_back(
        //             <Scan<T>>::open(sources.at(*next_source), projection, None, None).await?,
        //         );
        //         *next_source += 1;
        //     }
        //
        //     for scan in scans.iter_mut() {
        //         scan.spawn(scope, state);
        //     }
        //
        //     while let Some(mut scan) = scans.pop_front() {
        //         while let Ok(rg) = scan.recv().await {
        //             let df = process_dataframe(rg.data)?;
        //             let mut m = Morsel::new(df, *seq, source_token.clone());
        //             m.set_consume_token(wait_group.token());
        //             *seq = seq.successor();
        //
        //             if send.send(m).await.is_err() {
        //                 return Ok(());
        //             };
        //
        //             wait_group.wait().await;
        //             if source_token.stop_requested() {
        //                 scan.request_stop().await;
        //
        //                 while let Ok(rg) = scan.recv().await {
        //                     let df = process_dataframe(rg.data)?;
        //                     let mut m = Morsel::new(df, *seq, source_token.clone());
        //                     m.set_consume_token(wait_group.token());
        //                     *seq = seq.successor();
        //
        //                     if send.send(m).await.is_err() {
        //                         return Ok(());
        //                     };
        //                 }
        //
        //                 // Request stops from all sources and buffer the already creates morsels.
        //                 for scan in scans.iter_mut() {
        //                     if scan.phase.is_none() {
        //                         continue;
        //                     }
        //
        //                     scan.request_stop().await;
        //                     scan.flush().await;
        //                 }
        //
        //                 for scan in scans {
        //                     if let Some(phase) = scan.phase {
        //                         phase.shutdown().await?;
        //                     }
        //                 }
        //
        //                 return Ok(());
        //             }
        //         }
        //
        //         scan.phase.unwrap().shutdown().await?;
        //         if *next_source < sources.len() {
        //             scans.push_back(
        //                 <Scan<T>>::open(sources.at(*next_source), projection, None, None).await?,
        //             );
        //             *next_source += 1;
        //         }
        //     }
        //
        //     Ok(())
        // }));
    }
}
