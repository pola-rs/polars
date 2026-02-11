use std::collections::VecDeque;

use polars_core::prelude::*;
use polars_core::utils::Container;
use polars_ops::frame::{AsOfOptions, AsofStrategy, JoinArgs, JoinType};
use polars_utils::format_pl_smallstr;

use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel as dc;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::DataFrameSearchBuffer;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

#[derive(Debug)]
pub struct AsOfJoinSideParams {
    pub on: PlSmallStr,
    pub tmp_key_col: Option<PlSmallStr>,
}

impl AsOfJoinSideParams {
    fn key_col(&self) -> &PlSmallStr {
        self.tmp_key_col.as_ref().unwrap_or(&self.on)
    }
}

#[derive(Debug)]
struct AsOfJoinParams {
    left: AsOfJoinSideParams,
    right: AsOfJoinSideParams,
    args: JoinArgs,
}

impl AsOfJoinParams {
    fn as_of_options(&self) -> &AsOfOptions {
        let JoinType::AsOf(ref options) = self.args.how else {
            unreachable!("incorrect join type");
        };
        options
    }
}

#[derive(Debug, Default, PartialEq)]
enum AsOfJoinState {
    #[default]
    Running,
    FlushInputBuffer,
    Done,
}

#[derive(Debug)]
pub struct AsOfJoinNode {
    params: AsOfJoinParams,
    state: AsOfJoinState,
    /// We may need to stash a morsel on the left side whenever we do not
    /// have enough data on the right side, but the right side is empty.
    /// In these cases, we stash that morsel here.
    left_buffer: VecDeque<(DataFrame, MorselSeq)>,
    /// Buffer of the live range of right AsOf join rows.
    right_buffer: DataFrameSearchBuffer,
}

impl AsOfJoinNode {
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        args: JoinArgs,
    ) -> Self {
        let left_key_col = tmp_left_key_col.as_ref().unwrap_or(&left_on);
        let right_key_col = tmp_right_key_col.as_ref().unwrap_or(&right_on);
        let left_key_dtype = left_input_schema.get(left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(right_key_col).unwrap();
        assert_eq!(left_key_dtype, right_key_dtype);
        let left = AsOfJoinSideParams {
            on: left_on,
            tmp_key_col: tmp_left_key_col,
        };
        let right = AsOfJoinSideParams {
            on: right_on,
            tmp_key_col: tmp_right_key_col,
        };

        let params = AsOfJoinParams { left, right, args };
        AsOfJoinNode {
            params,
            state: AsOfJoinState::default(),
            left_buffer: Default::default(),
            right_buffer: DataFrameSearchBuffer::empty_with_schema(right_input_schema),
        }
    }
}

impl ComputeNode for AsOfJoinNode {
    fn name(&self) -> &str {
        "asof-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        if send[0] == PortState::Done {
            self.state = AsOfJoinState::Done;
        }

        if self.state == AsOfJoinState::Running && recv[0] == PortState::Done {
            self.state = AsOfJoinState::FlushInputBuffer;
        }

        if self.state == AsOfJoinState::FlushInputBuffer && self.left_buffer.is_empty() {
            self.state = AsOfJoinState::Done;
        }

        let recv0_blocked = recv[0] == PortState::Blocked;
        let recv1_blocked = recv[1] == PortState::Blocked;
        let send_blocked = send[0] == PortState::Blocked;
        match self.state {
            AsOfJoinState::Running => {
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv0_blocked {
                    recv[1] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if recv1_blocked {
                    recv[0] = PortState::Blocked;
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[0] = PortState::Blocked;
                    recv[1] = PortState::Blocked;
                }
            },
            AsOfJoinState::FlushInputBuffer => {
                recv[0] = PortState::Done;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv1_blocked {
                    send[0] = PortState::Blocked;
                }
                if send_blocked {
                    recv[1] = PortState::Blocked;
                }
            },
            AsOfJoinState::Done => {
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        match &self.state {
            AsOfJoinState::Running | AsOfJoinState::FlushInputBuffer => {
                let params = &self.params;
                let recv_left = match self.state {
                    AsOfJoinState::Running => Some(recv_ports[0].take().unwrap().serial()),
                    _ => None,
                };
                let recv_right = recv_ports[1].take().map(RecvPort::serial);
                let send = send_ports[0].take().unwrap().parallel();
                let (distributor, dist_recv) =
                    dc::distributor_channel(send.len(), *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
                let left_buffer = &mut self.left_buffer;
                let right_buffer = &mut self.right_buffer;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    distribute_work_task(
                        recv_left,
                        recv_right,
                        distributor,
                        left_buffer,
                        right_buffer,
                        params,
                    )
                    .await
                }));

                join_handles.extend(dist_recv.into_iter().zip(send).map(|(recv, send)| {
                    scope.spawn_task(TaskPriority::High, async move {
                        compute_and_emit_task(recv, send, params).await
                    })
                }));
            },
            AsOfJoinState::Done => {
                unreachable!();
            },
        }
    }
}

/// Tell the sender to this port to stop, and buffer everything that is still in the pipe.
async fn stop_and_buffer_pipe_contents<F>(port: Option<&mut PortReceiver>, buffer_morsel: &mut F)
where
    F: FnMut(DataFrame, MorselSeq),
{
    let Some(port) = port else {
        return;
    };

    while let Ok(morsel) = port.recv().await {
        morsel.source_token().stop();
        let (df, seq, _, _) = morsel.into_inner();
        buffer_morsel(df, seq);
    }
}

async fn distribute_work_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut distributor: dc::Sender<(DataFrame, DataFrameSearchBuffer, MorselSeq, SourceToken)>,
    left_buffer: &mut VecDeque<(DataFrame, MorselSeq)>,
    right_buffer: &mut DataFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let source_token = SourceToken::new();
    let right_done = recv_right.is_none();

    loop {
        if source_token.stop_requested() {
            stop_and_buffer_pipe_contents(recv_left.as_mut(), &mut |df, seq| {
                left_buffer.push_back((df, seq))
            })
            .await;
            stop_and_buffer_pipe_contents(recv_right.as_mut(), &mut |df, _| {
                right_buffer.push_df(df)
            })
            .await;
            return Ok(());
        }

        let (left_df, seq, st) = if let Some((df, seq)) = left_buffer.pop_front() {
            (df, seq, source_token.clone())
        } else if let Some(ref mut recv) = recv_left
            && let Ok(m) = recv.recv().await
        {
            let (df, seq, st, _) = m.into_inner();
            (df, seq, st)
        } else {
            stop_and_buffer_pipe_contents(recv_right.as_mut(), &mut |df, _| {
                right_buffer.push_df(df)
            })
            .await;
            return Ok(());
        };

        while need_more_right_side(&left_df, right_buffer, params)? && !right_done {
            if let Some(ref mut recv) = recv_right
                && let Ok(morsel_right) = recv.recv().await
            {
                right_buffer.push_df(morsel_right.into_df());
            } else {
                // The right pipe is empty at this stage, we will need to wait for
                // a new stage and try again.
                left_buffer.push_back((left_df, seq));
                stop_and_buffer_pipe_contents(recv_left.as_mut(), &mut |df, seq| {
                    left_buffer.push_back((df, seq))
                })
                .await;
                return Ok(());
            }
        }

        distributor
            .send((left_df.clone(), right_buffer.clone(), seq, st))
            .await
            .unwrap();
        prune_right_side(&left_df, right_buffer, params)?;
    }
}

/// Do we need more values on the right side before we can compute the AsOf join
/// between the right side and the complete left side?
fn need_more_right_side(
    left: &DataFrame,
    right: &DataFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<bool> {
    let options = params.as_of_options();
    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    if left_key.is_empty() {
        return Ok(false);
    }
    // SAFETY: We just checked that left_key is not empty
    let left_last_val = unsafe { left_key.get_unchecked(left_key.len() - 1) };
    let right_range_end = match (options.strategy, options.allow_eq) {
        (AsofStrategy::Forward, true) => {
            right.binary_search(|x| *x >= left_last_val, params.right.key_col(), false)
        },
        (AsofStrategy::Forward, false) | (AsofStrategy::Backward, true) => {
            right.binary_search(|x| *x > left_last_val, params.right.key_col(), false)
        },
        (AsofStrategy::Backward, false) | (AsofStrategy::Nearest, _) => {
            let first_greater =
                right.binary_search(|x| *x > left_last_val, params.right.key_col(), false);
            if first_greater >= right.height() {
                return Ok(true);
            }
            // In the Backward/Nearest cases, there may be a chunk of consecutive equal
            // values following the match value on the left side.  In this case, the AsOf
            // join is greedy and should until the *end* of that chunk.

            // SAFETY: We just checked that right_range_end is in bounds
            let fst_greater_val =
                unsafe { right.get_bypass_validity(params.right.key_col(), first_greater, false) };
            right.binary_search(|x| *x > fst_greater_val, params.right.key_col(), false)
        },
    };
    Ok(right_range_end >= right.height())
}

fn prune_right_side(
    left: &DataFrame,
    right: &mut DataFrameSearchBuffer,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let left_key = left.column(params.left.key_col())?.as_materialized_series();
    if left.len() == 0 {
        return Ok(());
    }
    // SAFETY: We just checked that left_key is not empty
    let left_first_val = unsafe { left_key.get_unchecked(0) };
    let right_range_start = right
        .binary_search(|x| *x >= left_first_val, params.right.key_col(), false)
        .saturating_sub(1);
    right.split_at(right_range_start);
    Ok(())
}

async fn compute_and_emit_task(
    mut dist_recv: dc::Receiver<(DataFrame, DataFrameSearchBuffer, MorselSeq, SourceToken)>,
    mut send: PortSender,
    params: &AsOfJoinParams,
) -> PolarsResult<()> {
    let options = params.as_of_options();
    while let Ok((left_df, right_buffer, seq, st)) = dist_recv.recv().await {
        let right_df = right_buffer.into_df();

        let left_key = left_df.column(params.left.key_col())?;
        let right_key = right_df.column(params.right.key_col())?;
        let any_key_is_temporary_col =
            params.left.tmp_key_col.is_some() || params.right.tmp_key_col.is_some();
        let mut out = polars_ops::frame::AsofJoin::_join_asof(
            &left_df,
            &right_df,
            left_key.as_materialized_series(),
            right_key.as_materialized_series(),
            options.strategy,
            options.tolerance.clone().map(Scalar::into_value),
            params.args.suffix.clone(),
            None,
            any_key_is_temporary_col || params.args.should_coalesce(),
            options.allow_eq,
            options.check_sortedness,
        )?;

        // Drop any temporary key columns that were added
        for tmp_key_col in [&params.left.tmp_key_col, &params.right.tmp_key_col] {
            if let Some(tmp_col) = tmp_key_col
                && out.schema().contains(tmp_col)
            {
                out.drop_in_place(tmp_col)?;
            }
        }

        // If the join key passed to _join_asof() was a temporary key column,
        // we still need to coalesce the real 'on' columns ourselves.
        if any_key_is_temporary_col
            && params.args.should_coalesce()
            && params.left.on == params.right.on
        {
            let right_on_name = format_pl_smallstr!("{}{}", params.right.on, params.args.suffix());
            out.drop_in_place(&right_on_name)?;
        }

        let morsel = Morsel::new(out, seq, st);
        if send.send(morsel).await.is_err() {
            return Ok(());
        }
    }
    Ok(())
}
