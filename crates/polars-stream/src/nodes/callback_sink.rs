use std::num::NonZeroUsize;

use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_core::frame::DataFrame;
use polars_core::prelude::{DataType, PlSmallStr};
use polars_core::runtime::ASYNC;
use polars_error::{PolarsResult, polars_ensure};
use polars_plan::prelude::PlanCallback;

use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

pub struct CallbackSinkNode {
    function: PlanCallback<DataFrame, bool>,
    maintain_order: bool,

    buffer: DataFrame,
    chunk_size: Option<NonZeroUsize>,
    is_done: bool,
    /// The names and dtypes of the first morsel, which every later one must match.
    schema: Option<Vec<(PlSmallStr, DataType)>>,
}

/// Checks the names and types of `df` against `schema` and casts the columns that match it only
/// through a Null type.
fn conform_to_schema(schema: &[(PlSmallStr, DataType)], df: DataFrame) -> PolarsResult<DataFrame> {
    polars_ensure!(
        schema.len() == df.width()
            && schema.iter().zip(df.columns()).all(|((n, _), c)| n == c.name()),
        SchemaMismatch: "schema of a morsel differs from that of the first"
    );
    let mut must_cast = false;
    for ((_, expected), c) in schema.iter().zip(df.columns()) {
        must_cast |= c.dtype().matches_schema_type(expected)?;
    }
    if !must_cast {
        return Ok(df);
    }
    let height = df.height();
    let columns = schema
        .iter()
        .zip(df.into_columns())
        .map(|((_, expected), c)| {
            if c.dtype() == expected {
                Ok(c)
            } else {
                c.cast(expected)
            }
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: casts keep the height.
    Ok(unsafe { DataFrame::new_unchecked(height, columns) })
}

impl CallbackSinkNode {
    pub fn new(
        function: PlanCallback<DataFrame, bool>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            function,
            maintain_order,

            buffer: DataFrame::empty(),
            chunk_size,
            is_done: false,
            schema: None,
        }
    }
}

impl ComputeNode for CallbackSinkNode {
    fn name(&self) -> &str {
        "sink_batches"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.is_empty());

        if self.is_done || recv[0] == PortState::Done {
            recv[0] = PortState::Done;

            // Flush the last buffer
            if self.buffer.height() > 0 && !self.is_done {
                let function = self.function.clone();
                let df = std::mem::take(&mut self.buffer);

                assert!(
                    self.chunk_size
                        .is_some_and(|chunk_size| self.buffer.height() <= chunk_size.get())
                );
                state.spawn_subphase_task(async move {
                    ASYNC
                        .spawn_blocking(move || function.call(df))
                        .await
                        .unwrap()?;
                    Ok(())
                });
                return Ok(());
            }
        } else {
            recv[0] = PortState::Ready;
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
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let mut recv = recv_ports[0]
            .take()
            .unwrap()
            .serial_with_maintain_order(self.maintain_order);

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while !self.is_done
                && let Ok(m) = recv.recv().await
            {
                let (sf, _, _, consume_token) = m.into_inner();

                let df = sf.into_df().await;

                // Morsels are not always appended to the buffer, so check them here, and cast
                // a column whose type is (nested) Null to the type of the first morsel's, as the
                // vstack did: all batches have the same types.
                let df = match &self.schema {
                    None => {
                        self.schema = Some(
                            df.columns()
                                .iter()
                                .map(|c| (c.name().clone(), c.dtype().clone()))
                                .collect(),
                        );
                        df
                    },
                    Some(schema) => conform_to_schema(schema, df)?,
                };

                // Without a chunk size every morsel is passed on as it is: buffering it would
                // only append it to an empty frame.
                if self.chunk_size.is_none() {
                    debug_assert_eq!(self.buffer.height(), 0);
                    if df.height() > 0 {
                        let function = self.function.clone();
                        let should_stop = ASYNC
                            .spawn_blocking(move || function.call(df))
                            .await
                            .unwrap()?;
                        if should_stop {
                            self.is_done = true;
                        }
                    }
                    drop(consume_token);
                    continue;
                }

                // @NOTE: This also performs schema validation.
                self.buffer.vstack_mut_owned(df)?;

                while self.buffer.height() > 0
                    && self
                        .chunk_size
                        .is_none_or(|chunk_size| self.buffer.height() >= chunk_size.get())
                {
                    let chunk_size = self.chunk_size.map_or(usize::MAX, Into::into);

                    let df;
                    (df, self.buffer) = self
                        .buffer
                        .split_at(self.buffer.height().min(chunk_size) as i64);
                    if self.buffer.height() == 0 {
                        // Start afresh: an empty remainder would keep the flushed dictionaries
                        // (or a materialized column) and have the next morsels merged into them.
                        self.buffer = DataFrame::empty();
                    }

                    let function = self.function.clone();
                    let should_stop = ASYNC
                        .spawn_blocking(move || function.call(df))
                        .await
                        .unwrap()?;

                    if should_stop {
                        self.is_done = true;
                        break;
                    }
                }
                drop(consume_token);
                // Increase the backpressure. Only free up a pipeline when the morsel has been
                // processed in its entirety.
            }

            Ok(())
        }));
    }
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn null_morsels_are_cast_to_the_first_schema() {
        let schema = vec![("a".into(), DataType::Int64)];
        let null =
            DataFrame::new_infer_height(vec![Column::full_null("a".into(), 3, &DataType::Null)])
                .unwrap();
        let out = conform_to_schema(&schema, null).unwrap();
        assert_eq!(out.height(), 3);
        assert_eq!(out.column("a").unwrap().dtype(), &DataType::Int64);
        assert_eq!(out.column("a").unwrap().null_count(), 3);

        let strings = df!("a" => ["x"]).unwrap();
        assert!(conform_to_schema(&schema, strings).is_err());
        let renamed = df!("b" => [1i64]).unwrap();
        assert!(conform_to_schema(&schema, renamed).is_err());
    }
}
