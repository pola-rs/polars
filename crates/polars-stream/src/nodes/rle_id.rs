use std::borrow::Cow;

use polars_core::prelude::{AnyValue, Column, DataType, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaExt};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct RleIdNode {
    name: PlSmallStr,
    index: IdxSize,

    dtype: DataType,
    last: Option<AnyValue<'static>>,
}

impl RleIdNode {
    pub fn new(name: PlSmallStr, schema: &Schema) -> Self {
        let dtype = if schema.len() == 1 {
            schema.get_at_index(0).unwrap().1.clone()
        } else {
            DataType::Struct(schema.iter_fields().collect())
        };
        Self {
            name,
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
        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut lengths = Vec::new();
            while let Ok(mut m) = recv.recv().await {
                let df = m.df_mut();
                if df.height() == 0 {
                    continue;
                }

                let column = if df.width() == 1 {
                    Cow::Borrowed(&df[0])
                } else {
                    Cow::Owned(
                        std::mem::take(df)
                            .into_struct(PlSmallStr::EMPTY)
                            .into_column(),
                    )
                };
                let column = column.as_ref();

                lengths.clear();
                polars_ops::series::rle_lengths(column, &mut lengths)?;

                // If the last value seen is different from this first value here, bump the index
                // by 1.
                self.index += IdxSize::from(self.last.take().is_some_and(|last| {
                    let fst = Scalar::new(self.dtype.clone(), column.get(0).unwrap().into_static());
                    let last = Scalar::new(self.dtype.clone(), last);
                    fst != last
                }));
                self.last = Some(column.get(column.len() - 1).unwrap().into_static());

                let column = if lengths.len() == 1 {
                    // If we only have one unique value, just give a Scalar column.
                    Column::new_scalar(
                        self.name.clone(),
                        Scalar::from(self.index),
                        lengths[0] as usize,
                    )
                } else {
                    let mut values = Vec::with_capacity(column.len());
                    values.extend(std::iter::repeat_n(self.index, lengths[0] as usize));
                    for length in lengths.iter().skip(1) {
                        self.index += 1;
                        values.extend(std::iter::repeat_n(self.index, *length as usize));
                    }
                    let mut column = Column::new(self.name.clone(), values);
                    column.set_sorted_flag(polars_core::series::IsSorted::Ascending);
                    column
                };

                // SAFETY:
                // - No name collision, because we insert only one column.
                // - Height updated to that columns length.
                // - Schema cache cleared after.
                unsafe {
                    df.set_height(column.len());
                    let columns = df.get_columns_mut();
                    if columns.len() != 1 {
                        *columns = vec![column];
                    } else {
                        columns[0] = column;
                    }
                }
                df.clear_schema();

                if send.send(m).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
