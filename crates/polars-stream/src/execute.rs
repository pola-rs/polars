use polars_core::frame::DataFrame;
use slotmap::SparseSecondaryMap;

use crate::graph::{Graph, GraphNodeKey};

pub fn execute_graph(graph: &mut Graph) -> SparseSecondaryMap<GraphNodeKey, DataFrame> {
    let mut out = SparseSecondaryMap::new();
    out
}
