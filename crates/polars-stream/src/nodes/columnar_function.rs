use std::sync::Arc;

use polars_core::schema::Schema;
use polars_plan::dsl::ColumnsUdf;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use super::in_memory_sink::InMemorySinkNode;
use super::in_memory_source::InMemorySourceNode;

pub enum ColumnarFunctionNode {
    Sink {
        sink_nodes: Vec<InMemorySinkNode>,
        func: Arc<dyn ColumnsUdf>,
        output_name: PlSmallStr,
    },
    Source(InMemorySourceNode),
    Done,
}

impl ColumnarFunctionNode {
    pub fn new(
        input_schemas: Vec<Arc<Schema>>,
        func: Arc<dyn ColumnsUdf>,
        output_name: PlSmallStr,
    ) -> Self {
        Self::Sink {
            sink_nodes: input_schemas
                .into_iter()
                .map(InMemorySinkNode::new)
                .collect(),
            func,
            output_name,
        }
    }
}

impl ComputeNode for ColumnarFunctionNode {
    fn name(&self) -> &str {
        "columnar-function"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(send.len() == 1);

        // If the output doesn't want any more data, transition to being done.
        if send[0] == PortState::Done && !matches!(self, Self::Done) {
            *self = Self::Done;
        }

        // If all inputs are done, transition to being a source.
        if let Self::Sink {
            sink_nodes,
            func,
            output_name,
        } = self
        {
            assert!(recv.len() == sink_nodes.len());
            if recv.iter().all(|p| *p == PortState::Done) {
                let mut cols = Vec::new();
                for sink_node in sink_nodes {
                    let df = sink_node.get_output()?.unwrap();
                    cols.extend(df.into_columns());
                }
                let out_col = func.call_udf(&mut cols)?.with_name(output_name.clone());
                let source_node = InMemorySourceNode::new(
                    Arc::new(DataFrame::new(out_col.len(), vec![out_col])?),
                    MorselSeq::default(),
                );
                *self = Self::Source(source_node);
            }
        }

        match self {
            Self::Sink { sink_nodes, .. } => {
                for (sink_node, r) in sink_nodes.iter_mut().zip_eq(recv) {
                    sink_node.update_state(core::slice::from_mut(r), &mut [], state)?;
                }
                send[0] = PortState::Blocked;
            },
            Self::Source(source_node) => {
                recv[0] = PortState::Done;
                source_node.update_state(&mut [], send, state)?;
            },
            Self::Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self, Self::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        match self {
            Self::Sink { sink_nodes, .. } => {
                for (sink_node, recv) in sink_nodes.iter_mut().zip_eq(recv_ports) {
                    if recv.is_some() {
                        sink_node.spawn(
                            scope,
                            core::slice::from_mut(recv),
                            &mut [],
                            state,
                            join_handles,
                        )
                    }
                }
            },
            Self::Source(source) => source.spawn(scope, &mut [], send_ports, state, join_handles),
            Self::Done => unreachable!(),
        }
    }
}
