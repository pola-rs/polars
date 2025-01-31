use std::collections::VecDeque;
use std::future::Future;
use std::ops::Range;

use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_mem_engine::ScanPredicate;
use polars_plan::plans::{ScanSourceRef, ScanSources};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

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

pub trait ScanNode: ComputeNode + Send + Sync + Sized {
    fn name() -> &'static str;

    fn new(
        source: ScanSourceRef<'_>,
        projection: Option<&Bitmap>,
        row_restriction: Option<RowRestrication>,
        row_index: Option<PlSmallStr>,
    ) -> impl Future<Output = PolarsResult<Self>> + Send;
    fn source_spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        send: Sender<RowGroup<Morsel>>,
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );
    fn row_count(&mut self) -> impl Future<Output = PolarsResult<IdxSize>> + Send;
    fn schema(&mut self) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
}

struct ScanPhase {
    receiver: Receiver<RowGroup<Morsel>>,
    source_token: Option<SourceToken>,
    join_handles: Vec<JoinHandle<PolarsResult<()>>>,
}

struct Scan<T: ScanNode> {
    node: T,
    phase: Option<ScanPhase>
    buffered: VecDeque<RowGroup<DataFrame>>,
}

impl ScanPhase {
    async fn shutdown(self) -> PolarsResult<()> {
        for handle in self.join_handles {
            handle.await?;
        }
        Ok(())
    }

    async fn unbuffered_recv(&mut self) -> Result<RowGroup<Morsel>, ()> {
        let m = self.receiver.recv().await?;
        if self.source_token.is_none() {
            self.source_token = Some(m.data.source_token().clone());
        }
        Ok(m)
    }
}

pub struct MultiScanNode<T>
where
    T: ScanNode,
{
    num_pipelines: usize,
    num_concurrent_scans: usize,

    sources: ScanSources,
    scans: VecDeque<Scan<T>>,

    projection: Option<Bitmap>,

    next_source: usize,
    node_index: usize,

    seq: MorselSeq,
}

impl<T> MultiScanNode<T>
where
    T: ScanNode,
{
    pub fn new(sources: ScanSources, projection: Option<Bitmap>) -> Self {
        Self {
            num_pipelines: 1,
            num_concurrent_scans: 1,
            sources,

            scans: VecDeque::new(),

            projection,

            next_source: 0,
            node_index: 0,

            seq: MorselSeq::default(),
        }
    }
}

impl<T: ScanNode> Scan<T> {
    async fn open(
        source: ScanSourceRef<'_>,
        projection: Option<&Bitmap>,
        row_restriction: Option<RowRestrication>,
        row_index: Option<PlSmallStr>,
    ) -> PolarsResult<Self> {
        Ok(Self {
            node: T::new(source, projection, row_restriction, row_index).await?,
            phase: None,
            buffered: VecDeque::new(),
        })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        state: &'s ExecutionState,
    ) {
        let (sender, receiver) = connector::<RowGroup<Morsel>>();
        let mut join_handles = Vec::new();

        self.node.source_spawn(scope, sender, state, &mut join_handles);

        self.phase = Some(ScanPhase {
            receiver,
            source_token: None,
            join_handles,
        });
    }

    async fn recv(&mut self) -> Result<RowGroup<DataFrame>, ()> {
        let Some(phase) = self.phase.as_mut() else {
            unreachable!();
        };

        match self.buffered.pop_front() {
            None => Ok(phase.unbuffered_recv().await?.into_df()),
            Some(df) => Ok(df),
        }
    }

    async fn request_stop(&mut self) {
        let Some(phase) = self.phase.as_mut() else {
            unreachable!();
        };

        match &phase.source_token {
            None => {
                if let Ok(m) = phase.unbuffered_recv().await {
                    m.data.source_token().stop();
                    self.buffered.push_back(m.into_df());
                }
            },
            Some(source_token) => source_token.stop(),
        }
    }
    async fn flush(&mut self) {
        let Some(phase) = self.phase.as_mut() else {
            unreachable!();
        };

        while let Ok(m) = phase.unbuffered_recv().await {
            self.buffered.push_back(m.into_df());
        }
    }
}

fn process_dataframe(df: DataFrame) -> PolarsResult<DataFrame> {
    Ok(df)
}

impl<T: ScanNode> ComputeNode for MultiScanNode<T> {
    fn name(&self) -> &str {
        <T as ScanNode>::name()
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
        self.num_concurrent_scans = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self.next_source >= self.sources.len() {
            send[0] = PortState::Done;
        } else {
            send[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);

        let mut send = send_ports[0].take().unwrap().serial();

        let sources = &self.sources;
        let projection = self.projection.as_ref();
        let scans = &mut self.scans;
        let next_source = &mut self.next_source;
        let seq = &mut self.seq;

        // Opener
        // Decoders[n]
        // Row Index, Predicate

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let source_token = SourceToken::new();
            let wait_group = WaitGroup::default();

            while scans.len() < self.num_concurrent_scans && *next_source < sources.len() {
                scans.push_back(<Scan<T>>::open(sources.at(*next_source), projection, None, None).await?);
                *next_source += 1;
            }

            for scan in scans.iter_mut() {
                scan.spawn(scope, state);
            }

            while let Some(mut scan) = scans.pop_front() {
                while let Ok(rg) = scan.recv().await {
                    let df = process_dataframe(rg.data)?;
                    let mut m = Morsel::new(df, *seq, source_token.clone());
                    m.set_consume_token(wait_group.token());
                    *seq = seq.successor();

                    if send.send(m).await.is_err() {
                        return Ok(());
                    };

                    wait_group.wait().await;
                    if source_token.stop_requested() {
                        scan.request_stop().await;

                        while let Ok(rg) = scan.recv().await {
                            let df = process_dataframe(rg.data)?;
                            let mut m = Morsel::new(df, *seq, source_token.clone());
                            m.set_consume_token(wait_group.token());
                            *seq = seq.successor();

                            if send.send(m).await.is_err() {
                                return Ok(());
                            };
                        }

                        // Request stops from all sources and buffer the already creates morsels.
                        for scan in scans.iter_mut() {
                            if scan.phase.is_none() {
                                continue;
                            }

                            scan.request_stop().await;
                            scan.flush().await;
                        }

                        for scan in scans {
                            if let Some(phase) = scan.phase {
                                phase.shutdown().await?;
                            }
                        }

                        return Ok(());
                    }
                }


                scan.phase.unwrap().shutdown().await?;
                if *next_source < sources.len() {
                    scans.push_back(<Scan<T>>::open(sources.at(*next_source), projection, None, None).await?);
                    *next_source += 1;
                }
            }

            Ok(())
        }));
    }
}
