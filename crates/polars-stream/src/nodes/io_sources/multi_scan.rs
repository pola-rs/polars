use std::future::Future;
use std::ops::Range;

use polars_core::frame::DataFrame;
use polars_core::prelude::PlHashMap;
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_mem_engine::ScanPredicate;
use polars_plan::plans::{ScanSourceRef, ScanSources};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

use crate::async_primitives::connector::connector;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::{self, Linearizer};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::{
    ComputeNode, JoinHandle, Morsel, MorselSeq, PortState, TaskPriority, TaskScope,
};
use crate::pipe::{RecvPort, SendPort};
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;

pub enum RowRestrication {
    Slice(Range<usize>),
    Predicate(ScanPredicate),
}

pub struct RowGroup {
    pub df: DataFrame,
    pub row_count: IdxSize,
}

pub trait ScanNode: ComputeNode + Send + Sync + Sized {
    fn name() -> &'static str;

    fn new(
        source: ScanSourceRef<'_>,
        projection: Option<&Bitmap>,
        row_restriction: Option<RowRestrication>,
        row_index: Option<PlSmallStr>,
    ) -> impl Future<Output = PolarsResult<Self>> + Send;

    fn row_count(&mut self) -> impl Future<Output = PolarsResult<IdxSize>> + Send;
    fn schema(&mut self) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
    fn read_into(
        &mut self,
        channel: Sender<RowGroup>,
    ) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
}

pub struct MultiScanNode<T>
where
    T: ScanNode,
{
    sources: ScanSources,
    scan_nodes: PlHashMap<usize, T>,

    projection: Option<Bitmap>,

    next_source: usize,
    seq: MorselSeq,
}

impl<T> MultiScanNode<T>
where
    T: ScanNode,
{
    pub fn new(sources: ScanSources, projection: Option<Bitmap>) -> Self {
        Self {
            sources,
            scan_nodes: PlHashMap::default(),
            projection,
            next_source: 0,
            seq: MorselSeq::default(),
        }
    }
}

impl<T: ScanNode> ComputeNode for MultiScanNode<T> {
    fn name(&self) -> &str {
        <T as ScanNode>::name()
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
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);

        let source_token = SourceToken::new();
        let mut send = send_ports[0].take().unwrap().serial();

        let sources = &self.sources;
        let projection = self.projection.as_ref();
        let scan_nodes = &mut self.scan_nodes;
        let next_source = &mut self.next_source;
        let seq = &mut self.seq;

        // Opener
        // Decoders[n]
        // Row Index, Predicate
        
        let (mut distributor, scan_recv) = distributor_channel(self.num_pipelines, 1);
        let (linearize, linearized) = Linearizer::new(self.num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);
        let (mut ret_channels, ) = distributor_channel(self.num_pipelines, 1);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut i = *next_source;
            while let Some(source) = sources.get(i) {
                wait_group.wait().await;
                if source_token.stop_requested() {
                    break;
                }

                let node = match scan_nodes.remove(&i) {
                    None => T::new(source, projection, None, None).await?,
                    Some(n) => n,
                };
                if distributor.send(node).await.is_err() {
                    break;
                };
            }

            for ret_channel in ret_channels {
                if let Ok((i, node)) = ret_channel.recv().await {
                    self.scan_nodes.insert(i, node);
                }
            }

            Ok(())
        }));
        
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let source_token = source_token.clone();
            let wait_group = WaitGroup::default();

            let mut i = *next_source;
            while let Some(source) = sources.get(i) {
                wait_group.wait().await;
                if source_token.stop_requested() {
                    break;
                }

                if !scan_nodes.contains_key(&i) {
                    scan_nodes.insert(i, T::new(source, projection, None, None).await?);
                }
                let node = scan_nodes.get_mut(&i).unwrap();

                node.read_into(sender);

                let row_group = node
                    .reader
                    .read_into(node.next_row_group)
                    .await?;
                node.next_row_group += 1;

                let mut morsel = Morsel::new(row_group.df, *seq, source_token.clone());
                morsel.set_consume_token(wait_group.token());
                *seq = seq.successor();
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));

        join_handles.extend(receiver.into_iter().map(|sender
    }
}
