use std::future::Future;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_plan::plans::ScanSource;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

use self::multi_scan::{RowGroup, RowRestrication};
use super::{ComputeNode, JoinHandle, Morsel, PortState, RecvPort, SendPort, TaskPriority};
use crate::async_primitives::connector::{connector, Receiver, Sender};

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
pub mod multi_scan;
#[cfg(feature = "parquet")]
pub mod parquet;

struct StartedSourceComputeNode {
    send_port_send: Sender<Sender<RowGroup<Morsel>>>,
    join_handles: Vec<JoinHandle<PolarsResult<()>>>,
}

pub struct SourceComputeNode<T: SourceNode + Send + Sync> {
    source: T,
    num_pipelines: usize,
    started: Option<StartedSourceComputeNode>,
}

impl<T: SourceNode + Send + Sync> SourceComputeNode<T> {
    pub fn new(source: T) -> Self {
        Self {
            source,
            num_pipelines: 0,
            started: None,
        }
    }
}

impl<T: SourceNode> ComputeNode for SourceComputeNode<T> {
    fn name(&self) -> &str {
        self.source.name()
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
    ) -> polars_error::PolarsResult<()> {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self
            .started
            .as_ref()
            .is_some_and(|s| s.join_handles.is_empty())
        {
            send[0] = PortState::Done;
        }

        if send[0] != PortState::Done {
            send[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s super::TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);

        let started = self.started.get_or_insert_with(|| {
            let (tx, rx) = connector();
            let mut join_handles = Vec::new();
            self.source
                .source_start(self.num_pipelines, rx, state, &mut join_handles);

            StartedSourceComputeNode {
                send_port_send: tx,
                join_handles,
            }
        });

        let mut send = send_ports[0].take().unwrap().serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let (tx, mut rx) = connector();
            if started.send_port_send.send(tx).await.is_err() {
                return Ok(());
            };

            while let Ok(rg) = rx.recv().await {
                if send.send(rg.data).await.is_err() {
                    return Ok(());
                };
            }

            dbg!("waiting for handles");

            for handle in std::mem::take(&mut started.join_handles) {
                handle.await?;
            }

            Ok(())
        }));
    }
}

pub trait SourceNode: Sized + Send + Sync {
    const BASE_NAME: &'static str;

    fn name(&self) -> &str {
        Self::BASE_NAME
    }

    fn source_start(
        &self,
        num_pipelines: usize,
        send_port_recv: Receiver<Sender<RowGroup<Morsel>>>,
        state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );
}
