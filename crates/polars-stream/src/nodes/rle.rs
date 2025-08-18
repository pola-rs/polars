use arrow::array::builder::ShareStrategy;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, DataType, Field, IDX_DTYPE, IntoColumn, NamedFrom, StructChunked,
};
use polars_core::scalar::Scalar;
use polars_core::series::Series;
use polars_core::series::builder::SeriesBuilder;
use polars_error::PolarsResult;
use polars_ops::series::{RLE_LENGTH_COLUMN_NAME, RLE_VALUE_COLUMN_NAME};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::{RecvPort, SendPort};

pub struct RleNode {
    name: PlSmallStr,
    dtype: DataType,

    seq: MorselSeq,

    // Invariant: last == None <=> last_length == 0
    last_length: IdxSize,
    last: Option<AnyValue<'static>>,
}

impl RleNode {
    pub fn new(name: PlSmallStr, dtype: DataType) -> Self {
        Self {
            name,
            dtype,
            seq: MorselSeq::default(),
            last_length: 0,
            last: None,
        }
    }
}

impl ComputeNode for RleNode {
    fn name(&self) -> &str {
        "rle"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            self.last_length = 0;
            self.last.take();
        } else if recv[0] == PortState::Done {
            if self.last.is_some() {
                send[0] = PortState::Ready;
            } else {
                send[0] = PortState::Done;
            }
        } else {
            recv.swap_with_slice(send);
        }

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

        let recv = recv_ports[0].take();
        let mut send = send_ports[0].take().unwrap().serial();

        let fields = vec![
            Field::new(PlSmallStr::from_static(RLE_LENGTH_COLUMN_NAME), IDX_DTYPE),
            Field::new(
                PlSmallStr::from_static(RLE_VALUE_COLUMN_NAME),
                self.dtype.clone(),
            ),
        ];
        let output_dtype = DataType::Struct(fields.clone());

        match recv {
            None => {
                // This happens when we have received out last morsel and we need to return one
                // more value.
                let last = self.last.take().unwrap();
                if self.last_length > 0 {
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        let column = Scalar::new(
                            output_dtype,
                            AnyValue::StructOwned(Box::new((
                                vec![AnyValue::from(self.last_length), last],
                                fields,
                            ))),
                        )
                        .into_column(self.name.clone());

                        let df = DataFrame::new(vec![column]).unwrap();
                        _ = send
                            .send(Morsel::new(df, self.seq.successor(), SourceToken::new()))
                            .await;

                        self.last_length = 0;
                        Ok(())
                    }));
                }
            },

            Some(recv) => {
                let mut recv = recv.serial();
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    let mut idxs = Vec::new();
                    let mut lengths = Vec::new();
                    while let Ok(mut m) = recv.recv().await {
                        self.seq = m.seq();
                        if m.df().height() == 0 {
                            continue;
                        }

                        assert_eq!(m.df().width(), 1);
                        let column = &m.df()[0];

                        lengths.clear();
                        polars_ops::series::rle_lengths(column, &mut lengths)?;

                        let mut new_first_is_last = false;
                        if let Some(last) = &self.last {
                            let fst = Scalar::new(
                                self.dtype.clone(),
                                column.get(0).unwrap().into_static(),
                            );
                            let last = Scalar::new(self.dtype.clone(), last.clone());
                            new_first_is_last = fst == last;
                        }

                        // If we have a morsel that is all the same value and we already know that
                        // value. Just add it to the length and continue.
                        if lengths.len() == 1 && new_first_is_last {
                            self.last_length += lengths[0];
                            continue;
                        }

                        let mut values = SeriesBuilder::new(self.dtype.clone());
                        values.reserve(lengths.len());

                        // Create the gather indices.
                        idxs.clear();
                        idxs.reserve(lengths.len() - 1);
                        let mut idx = 0;
                        for l in &lengths[0..lengths.len() - 1] {
                            idxs.push(idx);
                            idx += *l;
                        }

                        // Update the lengths to match what is being gathered and with the last
                        // element.
                        if new_first_is_last || self.last.is_none() {
                            lengths[0] += self.last_length;
                            self.last_length = lengths.pop().unwrap();
                        } else {
                            let mut prev = self.last_length;
                            for l in lengths.iter_mut() {
                                std::mem::swap(l, &mut prev);
                            }
                            self.last_length = prev;
                        }
                        let old_last = self
                            .last
                            .replace(column.get(column.len() - 1).unwrap().into_static());

                        // If we have nothing to return, just continue.
                        if lengths.is_empty() {
                            continue;
                        }

                        // If the morsel starts with a new value. We need to make sure to push it
                        // into the output values.
                        if !new_first_is_last && let Some(last) = old_last {
                            values.push_any_value(last);
                        }

                        // Actually gather the remaining values.
                        unsafe {
                            values.gather_extend(
                                column.as_materialized_series(),
                                &idxs,
                                ShareStrategy::Always,
                            )
                        };

                        let lengths = Series::new(
                            PlSmallStr::from_static(RLE_LENGTH_COLUMN_NAME),
                            std::mem::take(&mut lengths),
                        );
                        let series = values.freeze(PlSmallStr::from_static(RLE_VALUE_COLUMN_NAME));

                        let rle_struct = StructChunked::from_series(
                            self.name.clone(),
                            lengths.len(),
                            [&lengths, &series].into_iter(),
                        )
                        .unwrap();
                        *m.df_mut() = DataFrame::new(vec![rle_struct.into_column()]).unwrap();

                        if send.send(m).await.is_err() {
                            break;
                        }
                    }
                    Ok(())
                }));
            },
        }
    }
}
