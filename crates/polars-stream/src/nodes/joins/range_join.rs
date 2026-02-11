use polars_core::prelude::*;
use polars_ops::frame::{IEJoinOptions, JoinArgs};

use crate::async_executor::{JoinHandle, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::nodes::ComputeNode;
use crate::pipe::{RecvPort, SendPort};

#[derive(Debug)]
enum RangeJoinState {
    Running,
}

#[derive(Debug)]
pub struct RangeJoinNode {
    state: RangeJoinState,
    params: RangeJoinParams,
}

#[derive(Debug)]
struct RangeJoinParams {
    left: RangeJoinSideParams,
    right: RangeJoinSideParams,
    args: JoinArgs,
    options: IEJoinOptions,
}

#[derive(Debug)]
struct RangeJoinSideParams {
    schema: SchemaRef,
    on: (PlSmallStr, Option<PlSmallStr>),
    tmp_key_cols: (Option<PlSmallStr>, Option<PlSmallStr>),
}

impl RangeJoinSideParams {
    fn fst_key_col(&self) -> &PlSmallStr {
        self.tmp_key_cols.0.as_ref().unwrap_or(&self.on.0)
    }
    fn snd_key_col(&self) -> Option<&PlSmallStr> {
        self.tmp_key_cols.1.as_ref().or(self.on.1.as_ref())
    }
}

impl RangeJoinNode {
    pub fn new(
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        left_on: (PlSmallStr, Option<PlSmallStr>),
        right_on: (PlSmallStr, Option<PlSmallStr>),
        tmp_left_key_cols: (Option<PlSmallStr>, Option<PlSmallStr>),
        tmp_right_key_cols: (Option<PlSmallStr>, Option<PlSmallStr>),
        args: JoinArgs,
        options: IEJoinOptions,
    ) -> Self {
        let have_second_predicate = options.operator2.is_some();
        assert!(have_second_predicate == left_on.1.is_some());
        assert!(have_second_predicate == right_on.1.is_some());
        let left = RangeJoinSideParams {
            schema: left_schema,
            on: left_on,
            tmp_key_cols: tmp_left_key_cols,
        };
        let right = RangeJoinSideParams {
            schema: right_schema,
            on: right_on,
            tmp_key_cols: tmp_right_key_cols,
        };
        let params = RangeJoinParams {
            left,
            right,
            args,
            options,
        };
        RangeJoinNode {
            state: RangeJoinState::Running,
            params,
        }
    }
}

impl ComputeNode for RangeJoinNode {
    fn name(&self) -> &str {
        "range-join"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        // TODO: [amber] LEFT HERE
        //
        // I think it is time to start implementing the range-join node here.

        todo!()
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        todo!()
    }
}
