use polars_core::prelude::*;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use slotmap::SlotMap;
use std::sync::Arc;

use crate::nodes::unique::UniqueKeepStrategy;
use crate::physical_plan::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream};

/// Create a streaming unique node.
///
/// This function creates a streaming unique node that filters out duplicate rows based on the
/// given subset of columns.
///
/// # Arguments
///
/// * `graph` - The graph to add the node to
/// * `input` - The input node
/// * `subset` - The subset of columns to consider for uniqueness (if None, uses all columns)
/// * `keep` - Strategy for which duplicates to keep
/// * `maintain_order` - Whether to maintain the original order
pub fn create_streaming_unique_node(
    graph: &mut SlotMap<PhysNodeKey, PhysNode>,
    input: PhysNodeKey,
    subset: Option<Vec<PlSmallStr>>,
    keep: UniqueKeepStrategy,
    maintain_order: bool,
) -> PhysNodeKey {
    let input_stream = PhysStream::first(input);
    let input_schema = graph[input].output_schema.clone();

    let unique_node_kind = PhysNodeKind::Unique {
        input: input_stream,
        subset,
        keep,
        maintain_order,
    };

    // Output schema is the same as the input schema
    graph.insert(PhysNode::new(input_schema, unique_node_kind))
}