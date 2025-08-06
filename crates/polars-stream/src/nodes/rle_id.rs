use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, Column, DataType};
use polars_core::scalar::Scalar;
use polars_error::PolarsResult;
use polars_utils::IdxSize;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct RleIdNode {
    index: IdxSize,

    dtype: DataType,
    last: Option<AnyValue<'static>>,
}

impl RleIdNode {
    pub fn new(dtype: DataType) -> Self {
        Self {
            index: 0,
            dtype,
            last: None,
        }
    }
}

impl ComputeNode for RleIdNode {
    fn name(&self) -> &str {
        "rle_id"
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

        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut lengths = Vec::new();
            while let Ok(mut m) = recv.recv().await {
                let df = m.df_mut();
                if df.height() == 0 {
                    continue;
                }

                assert_eq!(df.width(), 1);
                let column = &df[0];

                let name = column.name().clone();

                lengths.clear();
                polars_ops::series::rle_lengths(column, &mut lengths)?;

                // If the last value seen is different from this first value here, bump the index
                // by 1.
                if let Some(last) = self.last.take() {
                    let fst = Scalar::new(self.dtype.clone(), column.get(0).unwrap().into_static());
                    let last = Scalar::new(self.dtype.clone(), last);
                    self.index += IdxSize::from(fst != last);
                }
                self.last = Some(column.get(column.len() - 1).unwrap().into_static());

                let column = if lengths.len() == 1 {
                    // If we only have one unique value, just give a Scalar column.
                    Column::new_scalar(name, Scalar::from(self.index), lengths[0] as usize)
                } else {
                    let mut values = Vec::with_capacity(column.len());
                    values.extend(std::iter::repeat_n(self.index, lengths[0] as usize));
                    for length in lengths.iter().skip(1) {
                        self.index += 1;
                        values.extend(std::iter::repeat_n(self.index, *length as usize));
                    }
                    let mut column = Column::new(name, values);
                    column.set_sorted_flag(polars_core::series::IsSorted::Ascending);
                    column
                };

                *df = unsafe { DataFrame::new_no_checks(column.len(), vec![column]) };

                if send.send(m).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
