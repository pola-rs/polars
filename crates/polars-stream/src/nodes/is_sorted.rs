use std::sync::Arc;

use polars_core::datatypes::AnyValue;
use polars_core::prelude::{Column, PlSmallStr, Series, SortOptions};
use polars_ops::prelude::SeriesMethods;
use polars_ops::series::resolve_sort_options;
use polars_utils::sort::reorder_cmp;

use super::compute_node_prelude::*;
use super::in_memory_source::InMemorySourceNode;
use crate::nodes::ComputeNode;

enum IsSortedState {
    Sink {
        is_sorted: bool,
        /// Sort direction, once known: seeded from the hint, then fixed by the first distinct pair.
        committed_descending: Option<bool>,
        /// Null placement, once known: seeded from the hint, then fixed by the first null/non-null
        /// transition.
        committed_nulls_last: Option<bool>,
        /// Last value of the previous morsel, for the cross-morsel boundary check.
        last_value: Option<AnyValue<'static>>,
    },
    Source(InMemorySourceNode),
    Done,
}

pub struct IsSortedNode {
    state: IsSortedState,
    output_name: PlSmallStr,
}

impl IsSortedNode {
    pub fn new(
        descending: Option<bool>,
        nulls_last: Option<bool>,
        output_name: PlSmallStr,
    ) -> Self {
        Self {
            state: IsSortedState::Sink {
                is_sorted: true,
                committed_descending: descending,
                committed_nulls_last: nulls_last,
                last_value: None,
            },
            output_name,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_sink<'env, 's>(
        is_sorted: &'env mut bool,
        committed_descending: &'env mut Option<bool>,
        committed_nulls_last: &'env mut Option<bool>,
        last_value: &'env mut Option<AnyValue<'static>>,
        scope: &'s TaskScope<'s, 'env>,
        recv: RecvPort<'_>,
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let mut recv = recv.serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await {
                if !*is_sorted {
                    return Ok(());
                }

                let df = morsel.into_df();
                if df.height() == 0 {
                    continue;
                }
                assert_eq!(df.width(), 1);
                let series = df[0].as_materialized_series();

                if !process_morsel(
                    series,
                    committed_descending,
                    committed_nulls_last,
                    last_value,
                )? {
                    *is_sorted = false;
                    return Ok(());
                }
            }
            Ok(())
        }))
    }
}

/// Commits a newly observed value for an option axis. If the axis is not yet known it is fixed to
/// `value`; if it is already known, a mismatch means the input is not sorted (returns `false`).
fn observe(committed: &mut Option<bool>, value: bool) -> bool {
    match committed {
        None => {
            *committed = Some(value);
            true
        },
        Some(c) => *c == value,
    }
}

/// Folds one morsel into the running sortedness check. Returns `false` as soon as the input is known
/// to be unsorted. Inference is done with vectorized series operations; only the two morsel
/// endpoints are inspected scalar-wise, so this stays `O(1)` per morsel in element lookups.
fn process_morsel(
    series: &Series,
    committed_descending: &mut Option<bool>,
    committed_nulls_last: &mut Option<bool>,
    last_value: &mut Option<AnyValue<'static>>,
) -> PolarsResult<bool> {
    let first = series.get(0).unwrap();

    // Reconcile the boundary with the previous morsel before inferring from this one.
    if let Some(prev) = last_value.as_ref() {
        match (prev.is_null(), first.is_null()) {
            // A non-null followed by a null fixes nulls-last; a null followed by a non-null fixes
            // nulls-first. A contradiction means the nulls are not all on one side.
            (false, true) if !observe(committed_nulls_last, true) => return Ok(false),
            (true, false) if !observe(committed_nulls_last, false) => return Ok(false),
            _ => {},
        }
        // The first distinct non-null pair fixes the direction, even across a boundary.
        if committed_descending.is_none() && !prev.is_null() && !first.is_null() && &first != prev {
            *committed_descending = Some(&first < prev);
        }
    }

    // Infer whatever this morsel reveals about the still-unknown axes (vectorized).
    let (morsel_descending, morsel_nulls_last) =
        resolve_sort_options(series, *committed_descending, *committed_nulls_last)?;
    if committed_descending.is_none() {
        *committed_descending = morsel_descending;
    }
    if committed_nulls_last.is_none() {
        *committed_nulls_last = morsel_nulls_last;
    }

    // Unknown axes are trivially sorted along that axis, so default them to `false`. Any genuine
    // violation (wrong direction, misplaced nulls) is caught by `is_sorted` / `boundary_ok`.
    let opts = SortOptions {
        descending: committed_descending.unwrap_or(false),
        nulls_last: committed_nulls_last.unwrap_or(false),
        ..Default::default()
    };

    if !series.is_sorted(opts)? {
        return Ok(false);
    }
    if !boundary_ok(last_value.as_ref(), &first, opts) {
        return Ok(false);
    }

    *last_value = Some(series.get(series.len() - 1).unwrap().into_static());
    Ok(true)
}

fn boundary_ok(prev: Option<&AnyValue<'static>>, first: &AnyValue<'_>, opts: SortOptions) -> bool {
    match prev {
        None => true,
        // The boundary is sorted iff `prev` orders before-or-equal to `first` under the resolved
        // options. `reorder_cmp` handles both null placement and sort direction.
        Some(prev) => reorder_cmp(prev, first, opts.descending, opts.nulls_last).is_le(),
    }
}

impl ComputeNode for IsSortedNode {
    fn name(&self) -> &str {
        "is_sorted"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self.state, IsSortedState::Done) {
            self.state = IsSortedState::Done;
        }

        // Once we know the answer (early `false`, or the input is exhausted),
        // transition to emitting the single-row result.
        if let IsSortedState::Sink { is_sorted, .. } = &self.state {
            if !*is_sorted || recv[0] == PortState::Done {
                let column = Column::new(self.output_name.clone(), &[*is_sorted]);
                let df = unsafe { DataFrame::new_unchecked(1, vec![column]) };
                let source = InMemorySourceNode::new(Arc::new(df), MorselSeq::default());
                self.state = IsSortedState::Source(source);
            }
        }

        match &mut self.state {
            IsSortedState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            IsSortedState::Source(source) => {
                recv[0] = PortState::Done;
                source.update_state(&mut [], send, state)?;
            },
            IsSortedState::Done => {
                recv[0] = PortState::Done;
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
        assert!(send_ports.len() == 1 && recv_ports.len() == 1);

        match &mut self.state {
            IsSortedState::Sink {
                is_sorted,
                committed_descending,
                committed_nulls_last,
                last_value,
            } => {
                assert!(send_ports[0].is_none());
                let recv = recv_ports[0].take().unwrap();
                Self::spawn_sink(
                    is_sorted,
                    committed_descending,
                    committed_nulls_last,
                    last_value,
                    scope,
                    recv,
                    state,
                    join_handles,
                );
            },
            IsSortedState::Source(source) => {
                assert!(recv_ports[0].is_none());
                source.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            IsSortedState::Done => unreachable!(),
        }
    }
}
