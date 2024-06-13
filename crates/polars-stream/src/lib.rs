#![allow(unused)] // TODO: remove.
#![allow(clippy::all)] // TODO: remove.

#[allow(unused)] // TODO: remove.
mod async_executor;
#[allow(unused)] // TODO: remove.
mod async_primitives;
mod skeleton;

use polars_expr::state::ExecutionState;
pub use skeleton::run_query;

use crate::nodes::ComputeNode;

mod execute;
mod graph;
mod morsel;
mod nodes;
mod physical_plan;

pub async fn dummy() {
    let num_threads = 8;
    async_executor::set_num_threads(num_threads);

    let node: nodes::filter::FilterNode = todo!();

    let state = ExecutionState::new();
    async_executor::task_scope(|s| {
        node.spawn(s, 0, Vec::new(), Vec::new(), &state);
    });

    let (mut send, mut recv) = async_primitives::pipe::pipe::<u32>();
    send.send(42).await.ok();
    recv.recv().await.ok();
    let (mut send, mut recvs) =
        async_primitives::distributor_channel::distributor_channel::<u32>(num_threads, 8);
    send.send(42).await.ok();
    recvs[0].recv().await.ok();
}
