use arrow::bitmap::BitmapBuilder;
use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::encode_rows_unordered;
use polars_core::prelude::{AnyValue, BooleanChunked, Column, IntoColumn};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::ComputeNode;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct SortedUnique {
    keys: Vec<usize>,
    row_encode: bool,
    last: Vec<Option<AnyValue<'static>>>,
}

impl SortedUnique {
    pub fn new(keys: &[PlSmallStr], schema: &Schema) -> Self {
        assert!(!keys.is_empty());
        let mut row_encode = keys.len() > 1;
        let last = vec![None; keys.len()];
        let keys = keys
            .iter()
            .map(|key| {
                let (idx, _, dtype) = schema.get_full(key).unwrap();
                row_encode |= dtype.is_nested();
                idx
            })
            .collect();
        Self {
            keys,
            row_encode,
            last,
        }
    }
}

impl ComputeNode for SortedUnique {
    fn name(&self) -> &str {
        "sorted_unique"
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

        let mut receiver = recv_ports[0].take().unwrap().serial();
        let senders = send_ports[0].take().unwrap().parallel();

        let (mut distributor, distr_receivers) =
            distributor_channel(senders.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let last = &mut self.last;
        let keys = &self.keys;
        let row_encode = self.row_encode;

        // Serial receiver.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = receiver.recv().await {
                let df = morsel.df();
                let height = df.height();
                if height == 0 {
                    continue;
                }

                let mut is_first_new_run = false;
                for (key, last) in keys.iter().zip(last.iter_mut()) {
                    let column = &df[*key];
                    is_first_new_run |= last
                        .take()
                        .is_none_or(|last| column.get(0).unwrap().into_static() != last);
                    *last = Some(column.get(height - 1).unwrap().into_static());
                }

                if distributor.send((morsel, is_first_new_run)).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));

        // Parallel worker threads.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();
                let mut lengths: Vec<IdxSize> = Vec::new();
                let mut columns: Vec<Column> = Vec::new();

                while let Ok((morsel, is_first_new_run)) = recv.recv().await {
                    let mut morsel = morsel.try_map(|df| {
                        let column = if row_encode {
                            columns.clear();
                            columns.extend(keys.iter().map(|i| df[*i].clone()));
                            encode_rows_unordered(&columns)?.into_column()
                        } else {
                            df[keys[0]].clone()
                        };

                        lengths.clear();
                        polars_ops::series::rle_lengths(&column, &mut lengths)?;

                        if !is_first_new_run && lengths.len() == 1 {
                            return Ok(DataFrame::empty());
                        }

                        // Build a boolean buffer: true only at the start of each new run.
                        let mut values = BitmapBuilder::with_capacity(column.len());
                        values.push(is_first_new_run);
                        values.extend_constant(lengths[0] as usize - 1, false);
                        for &length in &lengths[1..] {
                            values.push(true);
                            values.extend_constant(length as usize - 1, false);
                        }
                        let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, values.freeze());

                        // We already parallelize, call the sequential filter.
                        df.filter_seq(mask.as_ref())
                    })?;

                    if morsel.df().height() == 0 {
                        continue;
                    }

                    morsel.set_consume_token(wait_group.token());
                    if send.send(morsel).await.is_err() {
                        break;
                    }
                    wait_group.wait().await;
                }

                Ok(())
            }));
        }
    }
}
