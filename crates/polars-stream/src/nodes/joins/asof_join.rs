use polars_core::prelude::*;
use polars_ops::frame::JoinArgs;

use crate::async_executor::{JoinHandle, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::nodes::joins::utils::DataFrameBuffer;
use crate::pipe::{RecvPort, SendPort};

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

#[derive(Debug, Default)]
enum AsOfJoinState {
    #[default]
    Running,
    Done,
}

#[derive(Debug)]
pub struct AsOfJoinNode {
    params: AsOfJoinParams,
    state: AsOfJoinState,
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
            input_schema: right_input_schema,
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
            right_buffer: DataFrameBuffer::empty_with_schema(left_input_schema),
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
        use AsOfJoinState::*;

        assert!(recv.len() == 2 && send.len() == 1);

        let input_channels_done = recv.iter().all(|r| *r == PortState::Done);

        if matches!(self.state, Running) && input_channels_done {
            self.state = Done;
        }

        match self.state {
            Running => {
                let recv0_blocked = recv[0] == PortState::Blocked;
                let recv1_blocked = recv[1] == PortState::Blocked;
                let send_blocked = send[0] == PortState::Blocked;
                recv[0] = PortState::Ready;
                recv[1] = PortState::Ready;
                send[0] = PortState::Ready;
                if recv0_blocked || recv1_blocked {
                    send[0] = PortState::Blocked;
                }
                if recv1_blocked || send_blocked {
                    recv[0] = PortState::Blocked;
                }
                if recv0_blocked || send_blocked {
                    recv[1] = PortState::Blocked;
                }
            },
            Done => {
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
        use AsOfJoinState::*;

        todo!()
    }
}
