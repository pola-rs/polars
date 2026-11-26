use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use num_traits::AsPrimitive;
use polars_async::executor::TaskMetricAggregator;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::*;
use polars_core::series::ops::int_range::new_int_range;
use polars_core::with_match_physical_integer_polars_type;
use polars_expr::dispatch::int_range_bounds;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use crate::morsel::{SourceToken, get_ideal_morsel_size};

/// Generates `int_range(start, end, step)` from single-value `start` and `end` inputs.
pub struct IntRangeNode {
    name: PlSmallStr,
    dtype: DataType,
    step: i64,
    state: IntRangeState,
}

enum IntRangeState {
    GatheringBounds {
        start: InMemorySinkNode,
        end: InMemorySinkNode,
    },
    Emitting {
        /// The `start` and `end` columns.
        bounds: Box<[Column; 2]>,
        len: usize,
        /// Offset of the next morsel to claim.
        next_offset: AtomicU64,
    },
    Done,
}

impl IntRangeNode {
    pub fn new(
        name: PlSmallStr,
        dtype: DataType,
        step: i64,
        start_schema: Arc<Schema>,
        end_schema: Arc<Schema>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        Self {
            name,
            dtype,
            step,
            state: IntRangeState::GatheringBounds {
                start: InMemorySinkNode::new(start_schema, task_metrics.clone()),
                end: InMemorySinkNode::new(end_schema, task_metrics),
            },
        }
    }
}

impl ComputeNode for IntRangeNode {
    fn name(&self) -> &str {
        "int-range"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = IntRangeState::Done;
        }

        if let IntRangeState::GatheringBounds { start, end } = &mut self.state
            && recv.iter().all(|p| *p == PortState::Done)
        {
            let start = start.get_output()?.unwrap().columns()[0].clone();
            let end = end.get_output()?.unwrap().columns()[0].clone();
            let len = with_match_physical_integer_polars_type!(&self.dtype, |$T| {
                let (start, end) = int_range_bounds::<$T>(&start, &end)?;
                int_range_len::<$T>(start, end, self.step)?
            });
            self.state = IntRangeState::Emitting {
                bounds: Box::new([start, end]),
                len,
                next_offset: AtomicU64::new(0),
            };
        }

        match &mut self.state {
            IntRangeState::GatheringBounds { start, end } => {
                start.update_state(&mut recv[0..1], &mut [], state)?;
                end.update_state(&mut recv[1..2], &mut [], state)?;
                send[0] = PortState::Blocked;
            },
            IntRangeState::Emitting {
                len, next_offset, ..
            } => {
                recv.fill(PortState::Done);
                send[0] = if (*next_offset.get_mut() as usize) < *len {
                    PortState::Ready
                } else {
                    PortState::Done
                };
            },
            IntRangeState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);
        match &mut self.state {
            IntRangeState::GatheringBounds { start, end } => {
                assert!(send_ports[0].is_none());
                if recv_ports[0].is_some() {
                    start.spawn(scope, &mut recv_ports[0..1], &mut [], state, join_handles);
                }
                if recv_ports[1].is_some() {
                    end.spawn(scope, &mut recv_ports[1..2], &mut [], state, join_handles);
                }
            },
            IntRangeState::Emitting {
                bounds,
                len,
                next_offset,
            } => {
                assert!(recv_ports.iter().all(|p| p.is_none()));
                let senders = send_ports[0].take().unwrap().parallel();
                let [start, end] = &**bounds;
                let (len, next_offset) = (*len, &*next_offset);

                let remaining = len - next_offset.load(Ordering::Relaxed) as usize;
                let ideal_morsel_count = (remaining / get_ideal_morsel_size()).max(1);
                let morsel_count = ideal_morsel_count.next_multiple_of(state.num_pipelines);
                let morsel_size = remaining.div_ceil(morsel_count).max(1);
                let (name, dtype, step) = (&self.name, &self.dtype, self.step);

                let source_token = SourceToken::new();
                for mut send in senders {
                    let source_token = source_token.clone();
                    join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                        let wait_group = WaitGroup::default();
                        loop {
                            let offset =
                                next_offset.fetch_add(morsel_size as u64, Ordering::Relaxed) as usize;
                            if offset >= len {
                                break;
                            }
                            let height = morsel_size.min(len - offset);
                            let range = with_match_physical_integer_polars_type!(dtype, |$T| {
                                type N = <$T as PolarsNumericType>::Native;
                                let (start, end) = int_range_bounds::<$T>(start, end)?;
                                // Wrapping arithmetic is exact for representable elements.
                                let step_n: N = step.as_();
                                let at = |i: usize| -> N {
                                    start.wrapping_add(AsPrimitive::<N>::as_(i).wrapping_mul(step_n))
                                };
                                // Element `len` may not be representable, so the last morsel ends at `end`.
                                let morsel_end = if offset + height == len {
                                    end
                                } else {
                                    at(offset + height)
                                };
                                new_int_range::<$T>(at(offset), morsel_end, step, name.clone())?
                            });
                            let mut morsel = Morsel::new_unregistered(
                                range.into_frame(),
                                MorselSeq::new(offset as u64),
                                source_token.clone(),
                            );
                            morsel.set_consume_token(wait_group.token());
                            if send.send(morsel).await.is_err() {
                                break;
                            }

                            wait_group.wait().await;
                            if source_token.stop_requested() {
                                break;
                            }
                        }

                        Ok(())
                    }));
                }
            },
            IntRangeState::Done => unreachable!(),
        }
    }
}

/// Return the number of elements of `int_range(start, end, step)`.
fn int_range_len<T>(start: T::Native, end: T::Native, step: i64) -> PolarsResult<usize>
where
    T: PolarsIntegerType,
    T::Native: AsPrimitive<u128>,
{
    polars_ensure!(step != 0, InvalidOperation: "step must not be zero");
    let (lo, hi) = if step > 0 { (start, end) } else { (end, start) };
    if hi <= lo {
        return Ok(0);
    }
    // Sign extension makes the wrapping difference exact.
    let diff = u128::wrapping_sub(hi.as_(), lo.as_());
    let len = diff.div_ceil(step.unsigned_abs() as u128);
    usize::try_from(len)
        .map_err(|_| polars_err!(ComputeError: "`int_range` length {len} overflows usize"))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_int_range_len() {
        assert_eq!(int_range_len::<Int64Type>(0, 10, 3).unwrap(), 4);
        assert_eq!(int_range_len::<Int64Type>(10, 0, -3).unwrap(), 4);
        assert_eq!(int_range_len::<Int64Type>(0, 10, -1).unwrap(), 0);
        assert_eq!(int_range_len::<Int64Type>(5, 5, 1).unwrap(), 0);
        assert_eq!(int_range_len::<Int8Type>(-128, 127, 1).unwrap(), 255);
        assert_eq!(int_range_len::<Int8Type>(127, -128, -2).unwrap(), 128);
        assert_eq!(int_range_len::<UInt8Type>(255, 0, -1).unwrap(), 255);
        assert_eq!(
            int_range_len::<Int64Type>(i64::MIN, i64::MAX, i64::MAX).unwrap(),
            3
        );
        assert_eq!(
            int_range_len::<UInt64Type>(u64::MAX, 0, i64::MIN).unwrap(),
            2
        );
        assert!(int_range_len::<Int64Type>(0, 1, 0).is_err());
        assert!(int_range_len::<Int128Type>(i128::MIN, i128::MAX, 1).is_err());
    }
}
