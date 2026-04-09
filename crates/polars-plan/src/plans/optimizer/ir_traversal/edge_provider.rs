use polars_core::prelude::PlHashMap;
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use polars_utils::collection::{Collection, CollectionWrap};
use slotmap::SlotMap;

use crate::plans::ir_traversal::{IRNodeEdgeKeys, IRNodeKey, unpack_edges_mut};

pub trait NodeEdgesProvider<Edge> {
    type InputsProvider<'a>: Collection<Edge>
    where
        Self: 'a,
        Edge: 'a;
    type OutputsProvider<'a>: Collection<Edge>
    where
        Self: 'a,
        Edge: 'a;

    fn unpack_edges_mut<
        'a,
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])>
    where
        Edge: 'a;

    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, Self::InputsProvider<'a>>
    where
        Edge: 'a;

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, Self::OutputsProvider<'a>>
    where
        Edge: 'a;
}

pub struct IRTraversalGraphEdgeProvider<'a, EdgeKey: slotmap::Key, Edge> {
    pub ir_node_edge_keys: &'a IRNodeEdgeKeys<EdgeKey>,
    pub edges_map: &'a mut SlotMap<EdgeKey, Edge>,
}

impl<'provider, EdgeKey: slotmap::Key, Edge> NodeEdgesProvider<Edge>
    for IRTraversalGraphEdgeProvider<'provider, EdgeKey, Edge>
{
    type InputsProvider<'a>
        = Inputs<'a, 'provider, EdgeKey, Edge>
    where
        Self: 'a,
        Edge: 'a;
    type OutputsProvider<'a>
        = Outputs<'a, 'provider, EdgeKey, Edge>
    where
        Self: 'a,
        Edge: 'a;

    fn unpack_edges_mut<
        'a,
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
        // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
        const TOTAL_EDGES: usize,
    >(
        &'a mut self,
    ) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])>
    where
        Edge: 'a,
    {
        unpack_edges_mut::<EdgeKey, Edge, NUM_INPUTS, NUM_OUTPUTS, TOTAL_EDGES>(
            self.ir_node_edge_keys,
            self.edges_map,
        )
    }

    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, Self::InputsProvider<'a>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(Inputs(self))
    }

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, Self::OutputsProvider<'a>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(Outputs(self))
    }
}

pub struct Inputs<'outer, 'inner, EdgeKey: slotmap::Key, Edge>(
    &'outer mut IRTraversalGraphEdgeProvider<'inner, EdgeKey, Edge>,
);

impl<'outer, 'inner, EdgeKey: slotmap::Key, Edge> Collection<Edge>
    for Inputs<'outer, 'inner, EdgeKey, Edge>
{
    fn len(&self) -> usize {
        self.0.ir_node_edge_keys.in_edges.len()
    }

    fn get(&self, idx: usize) -> Option<&Edge> {
        self.0
            .ir_node_edge_keys
            .in_edges
            .get(idx)
            .map(|k| self.0.edges_map.get(*k).unwrap())
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Edge> {
        self.0
            .ir_node_edge_keys
            .in_edges
            .get(idx)
            .map(|k| self.0.edges_map.get_mut(*k).unwrap())
    }
}

pub struct Outputs<'outer, 'inner, EdgeKey: slotmap::Key, Edge>(
    &'outer mut IRTraversalGraphEdgeProvider<'inner, EdgeKey, Edge>,
);

impl<'outer, 'inner, EdgeKey: slotmap::Key, Edge> Collection<Edge>
    for Outputs<'outer, 'inner, EdgeKey, Edge>
{
    fn len(&self) -> usize {
        self.0.ir_node_edge_keys.out_edges.len()
    }

    fn get(&self, idx: usize) -> Option<&Edge> {
        self.0
            .ir_node_edge_keys
            .out_edges
            .get(idx)
            .map(|k| self.0.edges_map.get(*k).unwrap())
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Edge> {
        self.0
            .ir_node_edge_keys
            .out_edges
            .get(idx)
            .map(|k| self.0.edges_map.get_mut(*k).unwrap())
    }
}
