use std::collections::VecDeque;

use polars_core::prelude::*;
use polars_ops::frame::{AsOfOptions, JoinArgs, JoinType};

use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::DataFrameBuffer;
use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};

// TODO [amber]: Distribute the AsOf work across a pool of workers

pub const KEY_COL_NAME: &'static str = "__POLARS_JOIN_KEY";

#[derive(Debug)]
pub struct AsOfJoinSideParams {
    pub input_schema: SchemaRef,
    pub on: PlSmallStr,
    pub key_col: PlSmallStr,
}

#[derive(Debug)]
struct AsOfJoinParams {
    left: AsOfJoinSideParams,
    right: AsOfJoinSideParams,
    key_dtype: DataType,
    args: JoinArgs,
}

impl AsOfJoinParams {
    fn as_of_options(&self) -> &AsOfOptions {
        let JoinType::AsOf(ref options) = self.args.how else {
            panic!("incorrect join type");
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
    left_buffer: VecDeque<Morsel>,
    /// Buffer of the live range of right AsOf join rows.
    right_buffer: DataFrameBuffer,
}

impl AsOfJoinNode {
    pub fn new(
        left_input_schema: SchemaRef,
        right_input_schema: SchemaRef,
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        args: JoinArgs,
    ) -> Self {
        let left_key_col = match left_input_schema.contains(KEY_COL_NAME) {
            true => KEY_COL_NAME.into(),
            false => left_on.clone(),
        };
        let right_key_col = match right_input_schema.contains(KEY_COL_NAME) {
            true => KEY_COL_NAME.into(),
            false => right_on.clone(),
        };
        let left_key_dtype = left_input_schema.get(&left_key_col).unwrap();
        let right_key_dtype = right_input_schema.get(&right_key_col).unwrap();
        assert_eq!(left_key_dtype, right_key_dtype);
        let key_dtype = left_key_dtype.clone();
        let left = AsOfJoinSideParams {
            input_schema: left_input_schema.clone(),
            on: left_on,
            key_col: left_key_col,
        };
        let right = AsOfJoinSideParams {
            input_schema: right_input_schema.clone(),
            on: right_on,
            key_col: right_key_col,
        };

        let params = AsOfJoinParams {
            left,
            right,
            key_dtype,
            args,
        };
        AsOfJoinNode {
            params,
            state: AsOfJoinState::default(),
            left_buffer: Default::default(),
            right_buffer: DataFrameBuffer::empty_with_schema(right_input_schema),
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
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 2 && send.len() == 1);

        let left_input_channel_done = recv[0] == PortState::Done;
        let right_input_channel_done = recv[1] == PortState::Done;

        if send[0] == PortState::Done {
            self.state = AsOfJoinState::Done;
        }

        if self.state == AsOfJoinState::Running && left_input_channel_done {
            self.state = AsOfJoinState::FlushInputBuffer;
        }

        if self.state == AsOfJoinState::FlushInputBuffer && self.left_buffer.is_empty() {
            self.state = AsOfJoinState::Done;
        }

        let recv0_blocked = recv[0] == PortState::Blocked;
        let recv1_blocked = recv[1] == PortState::Blocked;
        let send_blocked = send[0] == PortState::Blocked;
        match self.state {
            AsOfJoinState::Running { .. } => {
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
            AsOfJoinState::FlushInputBuffer { .. } => {
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
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 2 && send_ports.len() == 1);

        match &mut self.state {
            AsOfJoinState::Running | AsOfJoinState::FlushInputBuffer => {
                let params = &self.params;
                let mut recv_left = match self.state {
                    AsOfJoinState::Running => Some(recv_ports[0].take().unwrap().serial()),
                    _ => None,
                };
                let recv_right = recv_ports[1].take().map(RecvPort::serial);
                dbg!(&self.state);
                dbg!(&self.left_buffer);
                dbg!(recv_left.is_some());
                let send = send_ports[0].take().unwrap().serial();
                let left_buffer = &mut self.left_buffer;
                let right_buffer = &mut self.right_buffer;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    process_asof_join_task(
                        recv_left,
                        recv_right,
                        send,
                        left_buffer,
                        right_buffer,
                        params,
                    )
                    .await;
                    Ok(())
                }));
            },
            AsOfJoinState::Done => {
                unreachable!();
            },
        }
    }
}

/// Tell the sender to this port to stop, and buffer everything that is still in the pipe.
async fn stop_and_buffer_pipe_contents<F>(port: Option<&mut PortReceiver>, mut buffer_morsel: F)
where
    F: FnMut(Morsel),
{
    let Some(port) = port else {
        return;
    };

    while let Ok(morsel) = port.recv().await {
        morsel.source_token().stop();
        buffer_morsel(morsel);
    }
}

async fn process_asof_join_task(
    mut recv_left: Option<PortReceiver>,
    mut recv_right: Option<PortReceiver>,
    mut send: PortSender,
    left_buffer: &mut VecDeque<Morsel>,
    right_buffer: &mut DataFrameBuffer,
    params: &AsOfJoinParams,
) {
    let right_done = recv_right.is_none();

    dbg!(right_done);
    let options = params.as_of_options();
    let mut recv_left_morsel = async |recv: Option<&mut PortReceiver>| {
        if let Some(m) = left_buffer.pop_front() {
            Ok(m)
        } else if let Some(recv) = recv {
            recv.recv().await
        } else {
            Err(())
        }
    };

    while let Ok(m) = recv_left_morsel(recv_left.as_mut()).await {
        // TODO [amber]: Check if the right side is sufficientl complete
        let have_enough_right_side = false;

        while !right_done && !have_enough_right_side {
            if let Some(ref mut recv) = recv_right
                && let Ok(m_right) = recv.recv().await
            {
                right_buffer.push_df(m_right.into_df());
            } else {
                // The right pipe is empty at this stage, we will need to wait for
                // a new stage and try again.
                left_buffer.push_back(m);
                stop_and_buffer_pipe_contents(recv_left.as_mut(), |m| left_buffer.push_back(m))
                    .await;
                return;
            }
        }

        // Compute AsOf join
        let (left_df, seq, st, wt) = m.into_inner();
        let right_df = right_buffer.clone().into_df();
        let left_key = left_df.column(&params.left.key_col).unwrap();
        let right_key = right_df.column(&params.right.key_col).unwrap();
        let mut out = polars_ops::frame::AsofJoin::_join_asof(
            &left_df,
            &right_df,
            left_key.as_materialized_series(),
            right_key.as_materialized_series(),
            options.strategy,
            options.tolerance.clone().map(Scalar::into_value),
            params.args.suffix.clone(),
            None, // slice: Option<(i64, usize)>
            params.args.coalesce.coalesce(&params.args.how),
            options.allow_eq,
            options.check_sortedness,
        )
        .unwrap();
        let _ = out.drop_in_place(KEY_COL_NAME);

        let mut m = Morsel::new(out, seq, st);
        if let Some(wt) = wt {
            m.set_consume_token(wt);
        }
        if send.send(m).await.is_err() {
            dbg!("cannot send");
            return;
        }

        // TODO [amber]: Prune the right buffer
    }
    stop_and_buffer_pipe_contents(recv_right.as_mut(), |m| right_buffer.push_df(m.into_df())).await;
}
