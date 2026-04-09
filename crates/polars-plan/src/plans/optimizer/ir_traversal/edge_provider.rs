use polars_utils::collection::{Collection, CollectionWrap};
use slotmap::SlotMap;

use crate::apply_dyn_len_array;
use crate::plans::ir_traversal::IRNodeEdgeKeys;
use crate::traversal::edge_provider::{DynLenArray, EdgesUnpacker, NodeEdgesProvider};

pub struct IRTraversalGraphEdgeProvider<'a, EdgeKey: slotmap::Key, Edge> {
    pub ir_node_edge_keys: &'a IRNodeEdgeKeys<EdgeKey>,
    pub edges_map: &'a mut SlotMap<EdgeKey, Edge>,
}

impl<'provider, EdgeKey: slotmap::Key, Edge> NodeEdgesProvider<Edge>
    for IRTraversalGraphEdgeProvider<'provider, EdgeKey, Edge>
{
    fn unpacker<'a>(&'a mut self) -> EdgesUnpacker<'a, Edge>
    where
        Edge: 'a,
    {
        apply_dyn_len_array!(
            DynLenArray::from_iter(
                self.ir_node_edge_keys
                    .in_edges
                    .iter()
                    .chain(self.ir_node_edge_keys.out_edges.iter())
                    .copied()
            )
            .unwrap(),
            |a| { self.edges_map.get_disjoint_mut(a).unwrap() }
        )
        .into()
    }

    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(unsafe {
            std::mem::transmute::<&'a mut Self, &'a mut Inputs<Self>>(self)
        })
    }

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(unsafe {
            std::mem::transmute::<&'a mut Self, &'a mut Outputs<Self>>(self)
        })
    }
}

#[repr(transparent)]
struct Inputs<T>(T);

impl<'a, EdgeKey: slotmap::Key, Edge> Collection<Edge>
    for Inputs<IRTraversalGraphEdgeProvider<'a, EdgeKey, Edge>>
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

#[repr(transparent)]
struct Outputs<T>(T);

impl<'a, EdgeKey: slotmap::Key, Edge> Collection<Edge>
    for Outputs<IRTraversalGraphEdgeProvider<'a, EdgeKey, Edge>>
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
