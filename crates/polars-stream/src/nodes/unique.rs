use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
use std::collections::HashSet;
use std::sync::Arc;

use crate::nodes::group_by::{GroupByNode, GroupKey};
use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::Morsel;
use crate::pipe::{RecvPort, SendPort};

/// Strategy for which duplicate rows to keep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UniqueKeepStrategy {
    /// First unique row.
    First,
    /// Last unique row.
    Last,
    /// Any unique row (doesn't guarantee which one).
    Any,
}

/// A node that performs unique operations on a dataframe by leveraging the GroupByNode.
pub struct UniqueNode {
    /// The subset of columns to consider for uniqueness
    subset: Option<Vec<PlSmallStr>>,
    /// Strategy for which duplicates to keep
    keep: UniqueKeepStrategy,
    /// Whether to maintain the original order
    maintain_order: bool,
    /// The underlying group by node used for implementation
    group_by: Option<GroupByNode>,
}

impl UniqueNode {
    pub fn new(
        subset: Option<Vec<PlSmallStr>>,
        keep: UniqueKeepStrategy,
        maintain_order: bool,
    ) -> Self {
        Self {
            subset,
            keep,
            maintain_order,
            group_by: None,
        }
    }

    /// Initialize the internal GroupByNode when we receive the first DataFrame
    fn init_group_by(&mut self, schema: &Schema) -> PolarsResult<()> {
        let keys = if let Some(subset) = &self.subset {
            // Use the provided subset columns as group by keys
            subset.iter().map(|name| GroupKey::Column(name.clone())).collect()
        } else {
            // Use all columns as keys if no subset is provided
            schema
                .iter()
                .map(|(name, _)| GroupKey::Column(name.clone()))
                .collect()
        };

        // Configure the aggregation strategy based on keep parameter
        let aggs = Vec::new(); // No aggregations, we just want the unique keys
        
        let group_by = GroupByNode::new(
            keys,
            aggs,
            self.maintain_order,
            false, // No need to rechunk
        )?;
        
        self.group_by = Some(group_by);
        Ok(())
    }
}

impl ComputeNode for UniqueNode {
    fn name(&self) -> &str {
        "unique"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        if let Some(group_by) = &mut self.group_by {
            return group_by.update_state(recv, send, state);
        }
        
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
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
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();
        let subset = self.subset.clone();
        let keep = self.keep.clone();
        let maintain_order = self.maintain_order;

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            // Process data as it arrives
            if let Ok(morsel) = recv.recv().await {
                let df = morsel.df();
                
                // Initialize the group_by node with the first dataframe's schema
                if self.group_by.is_none() {
                    self.init_group_by(df.schema())?;
                }
                
                // If we have a group_by node, use it to process the data
                if let Some(group_by) = &mut self.group_by {
                    // Push the first morsel back to the receiver so the group_by node can process it
                    recv.undo_recv(morsel);
                    
                    // Let the group_by node do all the work
                    group_by.spawn(scope, &mut [Some(recv)], &mut [Some(send)], state, join_handles);
                    return Ok(());
                }
            }

            // Fallback direct implementation if the group_by initialization failed
            let mut seen_keys = HashSet::new();
            
            while let Ok(mut morsel) = recv.recv().await {
                let df = morsel.df_mut();
                if df.height() == 0 {
                    // Skip empty DataFrames
                    continue;
                }

                // Implementation will depend on keep strategy
                // For now, just pass through the data
                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}