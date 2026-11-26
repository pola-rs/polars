use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use num_traits::{AsPrimitive, WrappingAdd, WrappingMul};
use polars_async::executor::TaskMetricAggregator;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_integer_polars_type;
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
        start: Scalar,
        len: usize,
        morsel_size: usize,
        next_morsel: AtomicU64,
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
            let start = extract_bound(start, "start")?;
            let end = extract_bound(end, "end")?;
            let len = with_match_physical_integer_polars_type!(&self.dtype, |$T| {
                type N = <$T as PolarsNumericType>::Native;
                int_range_len::<$T>(
                    start.value().extract::<N>().unwrap(),
                    end.value().extract::<N>().unwrap(),
                    self.step,
                )?
            });
            let ideal_morsel_count = (len / get_ideal_morsel_size()).max(1);
            let morsel_count = ideal_morsel_count.next_multiple_of(state.num_pipelines);
            self.state = IntRangeState::Emitting {
                start,
                len,
                morsel_size: len.div_ceil(morsel_count).max(1),
                next_morsel: AtomicU64::new(0),
            };
        }

        match &mut self.state {
            IntRangeState::GatheringBounds { start, end } => {
                start.update_state(&mut recv[0..1], &mut [], state)?;
                end.update_state(&mut recv[1..2], &mut [], state)?;
                send[0] = PortState::Blocked;
            },
            IntRangeState::Emitting {
                len,
                morsel_size,
                next_morsel,
                ..
            } => {
                recv.fill(PortState::Done);
                let emitted = *next_morsel.get_mut() as usize * *morsel_size;
                send[0] = if emitted < *len {
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
                start,
                len,
                morsel_size,
                next_morsel,
            } => {
                assert!(recv_ports.iter().all(|p| p.is_none()));
                let senders = send_ports[0].take().unwrap().parallel();
                let (start, len, morsel_size, next_morsel) =
                    (&*start, *len, *morsel_size, &*next_morsel);
                let (name, dtype, step) = (&self.name, &self.dtype, self.step);

                let source_token = SourceToken::new();
                for mut send in senders {
                    let source_token = source_token.clone();
                    join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                        let wait_group = WaitGroup::default();
                        loop {
                            let seq = next_morsel.fetch_add(1, Ordering::Relaxed);
                            let offset = seq as usize * morsel_size;
                            if offset >= len {
                                break;
                            }
                            let height = morsel_size.min(len - offset);
                            let range = with_match_physical_integer_polars_type!(dtype, |$T| {
                                let start = start.value().extract::<<$T as PolarsNumericType>::Native>().unwrap();
                                int_range_slice::<$T>(start, step, offset, height, name.clone())
                            });
                            let mut morsel = Morsel::new_unregistered(
                                range.into_frame(),
                                MorselSeq::new(seq),
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

/// Return the single value of the bound `name`, failing on a null or a count other than one.
fn extract_bound(sink: &mut InMemorySinkNode, name: &str) -> PolarsResult<Scalar> {
    let df = sink.get_output()?.unwrap();
    let col = &df.columns()[0];
    polars_ensure!(
        col.len() == 1,
        ComputeError: "`{name}` must contain exactly one value, got {} values", col.len()
    );
    let value = col.get(0)?.into_static();
    polars_ensure!(!value.is_null(), ComputeError: "invalid null input for `int_range`");
    Ok(Scalar::new(col.dtype().clone(), value))
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

/// Return elements `offset..offset + len` of `int_range(start, _, step)`.
///
/// All returned elements must be representable in `T`.
fn int_range_slice<T>(
    start: T::Native,
    step: i64,
    offset: usize,
    len: usize,
    name: PlSmallStr,
) -> Series
where
    T: PolarsIntegerType,
    T::Native: WrappingAdd + WrappingMul,
    i64: AsPrimitive<T::Native>,
    usize: AsPrimitive<T::Native>,
{
    // Wrapping arithmetic is exact modulo 2^bits, so it is exact for representable results.
    let step_t: T::Native = step.as_();
    let first = start.wrapping_add(&offset.as_().wrapping_mul(&step_t));
    let mut ca = if step == 1 {
        ChunkedArray::<T>::from_iter_values(name, (0..len).map(|k| first.wrapping_add(&k.as_())))
    } else {
        ChunkedArray::<T>::from_iter_values(
            name,
            (0..len).map(|k| first.wrapping_add(&k.as_().wrapping_mul(&step_t))),
        )
    };
    ca.set_sorted_flag(if step > 0 {
        IsSorted::Ascending
    } else {
        IsSorted::Descending
    });
    ca.into_series()
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

    #[test]
    fn test_int_range_slice() {
        let s = int_range_slice::<UInt8Type>(255, -100, 1, 2, PlSmallStr::EMPTY);
        assert_eq!(s.u8().unwrap().to_vec(), [Some(155), Some(55)]);
        let s = int_range_slice::<Int8Type>(-128, 100, 2, 1, PlSmallStr::EMPTY);
        assert_eq!(s.i8().unwrap().to_vec(), [Some(72)]);
    }
}
