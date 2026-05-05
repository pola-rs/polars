use polars_core::datatypes::AnyValue;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, PlSmallStr, Series, SortOptions};
use polars_ops::prelude::SeriesMethods;
use polars_ops::series::resolve_sort_options;

use super::compute_node_prelude::*;
use crate::morsel::SourceToken;
use crate::nodes::ComputeNode;

enum IsSortedState {
    Sink {
        is_sorted: bool,
        opts: Option<SortOptions>,
        last_value: Option<AnyValue<'static>>,
        last_was_null: bool,
        first_non_null: Option<AnyValue<'static>>,
        seen_null: bool,
        seen_value_after_null: bool,
        committed_descending: Option<bool>,
    },
    Source(Option<DataFrame>),
    Done,
}

pub struct IsSortedNode {
    state: IsSortedState,
    descending_hint: Option<bool>,
    nulls_last_hint: Option<bool>,
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
                opts: None,
                last_value: None,
                last_was_null: false,
                first_non_null: None,
                seen_null: false,
                seen_value_after_null: false,
                committed_descending: None,
            },
            descending_hint: descending,
            nulls_last_hint: nulls_last,
            output_name,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_sink<'env, 's>(
        is_sorted: &'env mut bool,
        opts: &'env mut Option<SortOptions>,
        last_value: &'env mut Option<AnyValue<'static>>,
        last_was_null: &'env mut bool,
        first_non_null: &'env mut Option<AnyValue<'static>>,
        seen_null: &'env mut bool,
        seen_value_after_null: &'env mut bool,
        committed_descending: &'env mut Option<bool>,
        descending_hint: Option<bool>,
        nulls_last_hint: Option<bool>,
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

                if opts.is_none() {
                    if let Some(o) =
                        resolve_sort_options(series, descending_hint, nulls_last_hint)?
                    {
                        *opts = Some(o);
                    }
                }

                if opts.is_none()
                    && !scan_unresolved(
                        series,
                        last_value,
                        *last_was_null,
                        first_non_null,
                        seen_null,
                        seen_value_after_null,
                        committed_descending,
                        descending_hint,
                        nulls_last_hint,
                        opts,
                    )?
                {
                    *is_sorted = false;
                    return Ok(());
                }

                if let Some(resolved) = *opts {
                    if !series.is_sorted(resolved)? {
                        *is_sorted = false;
                        return Ok(());
                    }

                    let first = series.get(0).unwrap();
                    if !boundary_ok(last_value.as_ref(), *last_was_null, &first, resolved) {
                        *is_sorted = false;
                        return Ok(());
                    }
                }

                update_last(series, last_value, last_was_null);
            }
            Ok(())
        }))
    }

    fn spawn_source<'env, 's>(
        df: &'env mut Option<DataFrame>,
        scope: &'s TaskScope<'s, 'env>,
        send: SendPort<'_>,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        let mut send = send.serial();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let morsel = Morsel::new(df.take().unwrap(), MorselSeq::new(0), SourceToken::new());
            let _ = send.send(morsel).await;
            Ok(())
        }));
    }
}

fn update_last(
    series: &Series,
    last_value: &mut Option<AnyValue<'static>>,
    last_was_null: &mut bool,
) {
    let last = series.get(series.len() - 1).unwrap();
    *last_was_null = last.is_null();
    *last_value = Some(last.into_static());
}

fn boundary_ok(
    prev: Option<&AnyValue<'static>>,
    prev_was_null: bool,
    first: &AnyValue<'_>,
    opts: SortOptions,
) -> bool {
    let Some(prev) = prev else {
        return true;
    };

    match (prev_was_null, first.is_null()) {
        (true, true) => true,
        (true, false) => !opts.nulls_last,
        (false, true) => opts.nulls_last,
        (false, false) => {
            if opts.descending {
                prev >= first
            } else {
                prev <= first
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_unresolved(
    series: &Series,
    last_value: &Option<AnyValue<'static>>,
    prev_last_was_null: bool,
    first_non_null: &mut Option<AnyValue<'static>>,
    seen_null: &mut bool,
    seen_value_after_null: &mut bool,
    committed_descending: &mut Option<bool>,
    descending_hint: Option<bool>,
    nulls_last_hint: Option<bool>,
    opts: &mut Option<SortOptions>,
) -> PolarsResult<bool> {
    let len = series.len();
    if len == 0 {
        return Ok(true);
    }

    let first = series.get(0).unwrap();
    if prev_last_was_null && !first.is_null() {
        *seen_value_after_null = true;
    }

    let mut prev_non_null: Option<AnyValue<'static>> = if prev_last_was_null {
        None
    } else {
        last_value.clone()
    };

    for i in 0..len {
        let v = series.get(i).unwrap();

        if v.is_null() {
            if *seen_value_after_null {
                return Ok(false);
            }
            *seen_null = true;
            continue;
        }

        if *seen_null {
            *seen_value_after_null = true;
        }

        if first_non_null.is_none() {
            *first_non_null = Some(v.clone().into_static());
        }

        if let Some(prev) = prev_non_null.as_ref() {
            if &v != prev {
                let step_descending = &v < prev;

                match *committed_descending {
                    None => {
                        if let Some(d) = descending_hint {
                            if d != step_descending {
                                return Ok(false);
                            }
                        }
                        *committed_descending = Some(step_descending);
                    },
                    Some(d) => {
                        if d != step_descending {
                            return Ok(false);
                        }
                    },
                }
            }
        }

        prev_non_null = Some(v.into_static());
    }

    if let Some(d) = *committed_descending {
        if *seen_null || nulls_last_hint.is_some() {
            let nulls_last = match nulls_last_hint {
                Some(n) => n,
                None => *seen_value_after_null,
            };
            *opts = Some(SortOptions {
                descending: d,
                nulls_last,
                ..Default::default()
            });
        }
    }

    Ok(true)
}

impl ComputeNode for IsSortedNode {
    fn name(&self) -> &str {
        "is_sorted"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        match &mut self.state {
            _ if send[0] == PortState::Done => {
                self.state = IsSortedState::Done;
            },
            IsSortedState::Sink { is_sorted, .. } if !*is_sorted => {
                let column = Column::new(self.output_name.clone(), &[false]);
                let out = unsafe { DataFrame::new_unchecked(1, vec![column]) };
                self.state = IsSortedState::Source(Some(out));
            },
            IsSortedState::Sink { is_sorted, .. } if matches!(recv[0], PortState::Done) => {
                let column = Column::new(self.output_name.clone(), &[*is_sorted]);
                let out = unsafe { DataFrame::new_unchecked(1, vec![column]) };
                self.state = IsSortedState::Source(Some(out));
            },
            IsSortedState::Source(df) if df.is_none() => {
                self.state = IsSortedState::Done;
            },
            IsSortedState::Done | IsSortedState::Sink { .. } | IsSortedState::Source(_) => {},
        }

        match &self.state {
            IsSortedState::Sink { .. } => {
                send[0] = PortState::Blocked;
                recv[0] = PortState::Ready;
            },
            IsSortedState::Source(..) => {
                recv[0] = PortState::Done;
                send[0] = PortState::Ready;
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
                opts,
                last_value,
                last_was_null,
                first_non_null,
                seen_null,
                seen_value_after_null,
                committed_descending,
            } => {
                assert!(send_ports[0].is_none());
                let recv = recv_ports[0].take().unwrap();
                Self::spawn_sink(
                    is_sorted,
                    opts,
                    last_value,
                    last_was_null,
                    first_non_null,
                    seen_null,
                    seen_value_after_null,
                    committed_descending,
                    self.descending_hint,
                    self.nulls_last_hint,
                    scope,
                    recv,
                    state,
                    join_handles,
                );
            },
            IsSortedState::Source(df) => {
                assert!(recv_ports[0].is_none());
                let send = send_ports[0].take().unwrap();
                Self::spawn_source(df, scope, send, join_handles);
            },
            IsSortedState::Done => unreachable!(),
        }
    }
}