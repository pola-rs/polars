use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, IdxCa, IntoColumn, PlSeedableRandomStateQuality};
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::aliases::PlHashMap;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct UniqueIdNode {
    // Used only when dense=true
    map: PlHashMap<AnyValue<'static>, IdxSize>,
    counter: IdxSize,
    maintain_order: bool,
    dense: bool,
}

impl UniqueIdNode {
    pub fn new(maintain_order: bool, dense: bool) -> Self {
        Self {
            map: Default::default(),
            counter: 0,
            maintain_order,
            dense,
        }
    }
}

impl ComputeNode for UniqueIdNode {
    fn name(&self) -> &str {
        "unique_id"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        if self.dense || self.maintain_order {
            // Serial processing: needed for dense IDs (counter) or maintain_order (first-occurrence tracking)
            let mut recv = recv_ports[0].take().unwrap().serial();
            let mut send = send_ports[0].take().unwrap().serial();

            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(mut m) = recv.recv().await {
                    let df = m.df_mut();
                    if df.height() == 0 {
                        if send.send(m).await.is_err() {
                            break;
                        }
                        continue;
                    }

                    assert_eq!(df.width(), 1);
                    let column = &df[0];
                    let name = column.name().clone();

                    let mut ids = Vec::with_capacity(column.len());
                    for val in column.as_materialized_series().iter() {
                        let id = *self.map.entry(val.into_static()).or_insert_with(|| {
                            let id = self.counter;
                            self.counter += 1;
                            id
                        });
                        ids.push(id);
                    }

                    let id_column = IdxCa::new_vec(name, ids).into_column();
                    *df = unsafe { DataFrame::new_unchecked(id_column.len(), vec![id_column]) };

                    if send.send(m).await.is_err() {
                        break;
                    }
                }

                Ok(())
            }));
        } else {
            // Parallel mode: dense=False AND maintain_order=False
            // Use hash values as IDs - same value → same hash → same ID
            let receivers = recv_ports[0].take().unwrap().parallel();
            let senders = send_ports[0].take().unwrap().parallel();

            // Use a fixed random state so same value always produces same hash
            let random_state = PlSeedableRandomStateQuality::fixed();

            for (mut recv, mut send) in receivers.into_iter().zip(senders) {
                let rs = random_state.clone();
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(mut m) = recv.recv().await {
                        let df = m.df_mut();
                        if df.height() == 0 {
                            if send.send(m).await.is_err() {
                                break;
                            }
                            continue;
                        }

                        assert_eq!(df.width(), 1);
                        let column = &df[0];
                        let name = column.name().clone();
                        let series = column.as_materialized_series();

                        // Compute hash for each value using VecHash trait
                        let mut hashes = Vec::with_capacity(series.len());
                        series.vec_hash(rs.clone(), &mut hashes)?;

                        // Convert u64 hashes to IdxSize
                        let ids: Vec<IdxSize> = hashes.into_iter().map(|h| h as IdxSize).collect();

                        let id_column = IdxCa::new_vec(name, ids).into_column();
                        *df = unsafe { DataFrame::new_unchecked(id_column.len(), vec![id_column]) };

                        if send.send(m).await.is_err() {
                            break;
                        }
                    }

                    Ok(())
                }));
            }
        }
    }
}
