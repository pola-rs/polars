use std::cmp::Reverse;
use std::sync::Arc;

use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
use polars_async::primitives::linearizer::Linearizer;
use polars_async::primitives::wait_group::WaitGroup;
use polars_core::prelude::{AnyValue, DataType, IntoColumn, Scalar, Series};
use polars_error::PolarsResult;
use polars_ops::series::{
    cum_count, cum_max, cum_max_with_init, cum_min, cum_min_with_init, cum_prod, cum_sum,
};
use polars_utils::kahan_sum::KahanSum;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::relaxed_cell::RelaxedCell;

use super::ComputeNode;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};
use crate::utils::morsel_distributor::morsel_distributor;
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

/// Computes the cumulative aggregate of each morsel in parallel, then combines
/// each of them with the aggregate of all preceding morsels, again in parallel.
pub struct CumAggNode {
    kind: CumAggKind,
    dtype: DataType,
    running_agg: Option<RunningAgg>,
    seq_offset: Arc<RelaxedCell<u64>>,
}

#[derive(Debug, Clone, Copy, strum_macros::IntoStaticStr)]
pub enum CumAggKind {
    Min,
    Max,
    Sum,
    Count,
    Prod,
}

impl CumAggKind {
    /// The cumulative aggregate of a single morsel. Floats are summed in f64.
    fn local_cum_agg(self, s: &Series) -> PolarsResult<Series> {
        match self {
            Self::Min => cum_min(s, false),
            Self::Max => cum_max(s, false),
            Self::Sum if s.dtype().is_float() => cum_sum(&s.cast(&DataType::Float64)?, false),
            Self::Sum => cum_sum(s, false),
            Self::Count => cum_count(s, false),
            Self::Prod => cum_prod(s, false),
        }
    }

    /// Combines every value of a local cumulative aggregate with the aggregate
    /// of all preceding morsels.
    fn apply_preceding(self, local: &Series, preceding: &Scalar) -> PolarsResult<Series> {
        let preceding_series = || preceding.clone().into_series(PlSmallStr::EMPTY);
        match self {
            // The local aggregate is monotonic, so its cumulative min/max
            // starting from `preceding` is the element-wise min/max.
            Self::Min => cum_min_with_init(local, false, preceding.value()),
            Self::Max => cum_max_with_init(local, false, preceding.value()),
            Self::Sum | Self::Count => local + &preceding_series(),
            Self::Prod => local * &preceding_series(),
        }
    }
}

/// The aggregate of the non-null values of all morsels seen so far.
enum RunningAgg {
    FloatSum(KahanSum<f64>),
    Exact(Scalar),
}

impl RunningAgg {
    fn new(kind: CumAggKind, value: Scalar) -> Self {
        if matches!(kind, CumAggKind::Sum) && value.dtype().is_float() {
            Self::FloatSum(KahanSum::new(value.value().extract().unwrap()))
        } else {
            Self::Exact(value)
        }
    }

    fn combine(&mut self, kind: CumAggKind, value: Scalar) -> PolarsResult<()> {
        match self {
            Self::FloatSum(sum) => *sum += value.value().extract::<f64>().unwrap(),
            Self::Exact(agg) => {
                *agg = kind
                    .apply_preceding(&value.into_series(PlSmallStr::EMPTY), agg)?
                    .first()
            },
        }
        Ok(())
    }

    fn to_scalar(&self) -> Scalar {
        match self {
            Self::FloatSum(sum) => Scalar::new(DataType::Float64, AnyValue::Float64(sum.sum())),
            Self::Exact(agg) => agg.clone(),
        }
    }
}

impl CumAggNode {
    pub fn new(kind: CumAggKind, dtype: DataType) -> Self {
        Self {
            kind,
            dtype,
            running_agg: None,
            seq_offset: Arc::default(),
        }
    }
}

impl ComputeNode for CumAggNode {
    fn name(&self) -> &str {
        match self.kind {
            CumAggKind::Min => "cum_min",
            CumAggKind::Max => "cum_max",
            CumAggKind::Sum => "cum_sum",
            CumAggKind::Count => "cum_count",
            CumAggKind::Prod => "cum_prod",
        }
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
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        let receivers = recv_ports[0].take().unwrap().parallel();
        let senders = send_ports[0].take().unwrap().parallel();

        let (mut linearizer, inserters) =
            Linearizer::new(receivers.len(), *DEFAULT_LINEARIZER_BUFFER_SIZE);
        let (mut distributor, distr_receivers) = morsel_distributor(
            senders.len(),
            *DEFAULT_DISTRIBUTOR_BUFFER_SIZE,
            self.seq_offset.clone(),
        );

        let kind = self.kind;
        let dtype = &self.dtype;
        let running_agg = &mut self.running_agg;

        // Compute the cumulative aggregate within each morsel, and its last non-null value.
        for (mut recv, mut inserter) in receivers.into_iter().zip(inserters) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                while let Ok(mut morsel) = recv.recv().await {
                    if morsel.height() == 0 {
                        continue;
                    }

                    let consume_token = morsel.take_consume_token();
                    let mut last = Scalar::null(DataType::Null);
                    let morsel = morsel
                        .try_map(|df| {
                            assert_eq!(df.width(), 1);
                            let local = kind.local_cum_agg(df[0].as_materialized_series())?;
                            last = local.last_non_null();
                            PolarsResult::Ok(local.into_column().into_frame())
                        })
                        .await?;

                    if inserter
                        .insert(Priority(Reverse(morsel.seq()), (morsel, last)))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    drop(consume_token);
                }

                Ok(())
            }));
        }

        // Determine the aggregate of all preceding morsels, in order.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Some(Priority(_, (morsel, last))) = linearizer.get().await {
                let preceding = running_agg.as_ref().map(RunningAgg::to_scalar);
                if !last.is_null() {
                    match running_agg {
                        Some(agg) => agg.combine(kind, last)?,
                        None => *running_agg = Some(RunningAgg::new(kind, last)),
                    }
                }
                if distributor.send((morsel, preceding)).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));

        // Combine with the aggregate of all preceding morsels.
        for (mut send, mut recv) in senders.into_iter().zip(distr_receivers) {
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let wait_group = WaitGroup::default();
                while let Ok((morsel, preceding)) = recv.recv().await {
                    let mut morsel = morsel
                        .try_map(|df| {
                            let mut s = df[0].as_materialized_series().clone();
                            if let Some(preceding) = &preceding {
                                s = kind.apply_preceding(&s, preceding)?;
                            }
                            PolarsResult::Ok(s.cast(dtype)?.into_column().into_frame())
                        })
                        .await?;
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
