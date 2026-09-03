use std::ops::{ControlFlow, Range};

use polars_utils::collection::{Collection, CollectionWrap};
use polars_utils::{UnitVec, unitvec};

use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

pub trait GetNodeInputs<Key> {
    fn get_node_inputs(&self, key: Key, push_fn: &mut dyn FnMut(Key));

    fn num_inputs(&self, key: Key) -> usize {
        let mut n: usize = 0;

        self.get_node_inputs(key, &mut |_| n += 1);

        n
    }
}

pub fn tree_traversal<Key, Storage, Edge, BreakValue>(
    root_key: Key,
    storage: &mut Storage,
    visit_stack: &mut Vec<Key>,
    edges: &mut Vec<Edge>,
    visitor: &mut dyn NodeVisitor<Key = Key, Storage = Storage, Edge = Edge, BreakValue = BreakValue>,
) -> ControlFlow<BreakValue, Edge>
where
    Key: Clone,
    Storage: GetNodeInputs<Key>,
{
    let root_edge_idx = edges.len();
    edges.push(visitor.default_edge(root_key.clone(), None));
    let root_edge_deleted = visitor.is_deleted_edge(&edges[root_edge_idx]) == Some(true);

    TreeTraversalImpl {
        storage,
        visit_stack,
        edges,
        persist_input_edge_idxs: None,
        graph_visit_order_fn: None,
        visitor,
    }
    .traverse_rec(root_key, root_edge_idx, root_edge_deleted)?;

    assert_eq!(edges.len(), root_edge_idx + 1);

    ControlFlow::Continue(edges.pop().unwrap())
}

pub enum PersistInputEdgeIdxs<'a> {
    Build(&'a mut Vec<usize>),
    Use(&'a [usize]),
}

#[allow(clippy::type_complexity)]
pub struct TreeTraversalImpl<'a, Key, Storage, Edge, BreakValue> {
    pub storage: &'a mut Storage,
    pub visit_stack: &'a mut Vec<Key>,
    pub edges: &'a mut Vec<Edge>,
    pub persist_input_edge_idxs: Option<&'a mut PersistInputEdgeIdxs<'a>>,
    /// Turn this into a graph traversal (multi-output nodes).
    pub graph_visit_order_fn: Option<
        &'a mut dyn for<'storage> FnMut(
            Key,
            &'storage mut Storage,
            // Edge key (usize). `None` if the edge was skipped.
            Option<usize>,
        ) -> GraphVisitOrder<usize>,
    >,
    pub visitor: &'a mut dyn NodeVisitor<Key = Key, Storage = Storage, Edge = Edge, BreakValue = BreakValue>,
}

impl<'a, Key, Storage, Edge, BreakValue> TreeTraversalImpl<'a, Key, Storage, Edge, BreakValue>
where
    Key: Clone,
    Storage: GetNodeInputs<Key>,
{
    #[recursive::recursive]
    pub fn traverse_rec(
        &mut self,
        current_key: Key,
        current_key_out_edge_idx: usize,
        // The visitor's pre/post_visit will not be called for this node, but we are still traversing
        // for the `graph_visit_order_fn`.
        in_skipped_subtree: bool,
    ) -> ControlFlow<BreakValue> {
        let Self {
            storage,
            visit_stack,
            edges,
            persist_input_edge_idxs,
            graph_visit_order_fn,
            visitor,
        } = self;

        let base_visit_stack_len = visit_stack.len();

        let current_key_out_edge_idxs = if let Some(graph_visit_order_fn) = graph_visit_order_fn {
            match (graph_visit_order_fn)(
                current_key.clone(),
                storage,
                (!in_skipped_subtree).then_some(current_key_out_edge_idx),
            ) {
                GraphVisitOrder::HasUnvisitedOutputs => return ControlFlow::Continue(()),
                GraphVisitOrder::Visit { output_keys } => output_keys,
            }
        } else if !in_skipped_subtree {
            unitvec![current_key_out_edge_idx]
        } else {
            unitvec![]
        };

        storage.get_node_inputs(current_key.clone(), &mut |key| visit_stack.push(key));

        let num_inputs = visit_stack.len() - base_visit_stack_len;

        let input_edges_start_idx: usize = match persist_input_edge_idxs {
            Some(PersistInputEdgeIdxs::Use(idxs)) => idxs[current_key_out_edge_idx],
            _ => {
                let base_edges_len = edges.len();

                edges.extend((0..num_inputs).map(|i| {
                    visitor.default_edge(
                        visit_stack[base_visit_stack_len + i].clone(),
                        Some((current_key.clone(), i)),
                    )
                }));

                match persist_input_edge_idxs {
                    Some(PersistInputEdgeIdxs::Build(idxs)) => {
                        idxs[current_key_out_edge_idx] = base_edges_len;
                        idxs.extend((0..num_inputs).map(|_| usize::MAX));
                        assert_eq!(edges.len(), idxs.len());
                    },
                    None => {},
                    Some(PersistInputEdgeIdxs::Use(_)) => unreachable!(),
                };

                base_edges_len
            },
        };

        assert!(input_edges_start_idx != usize::MAX); // Visiting a node that was not visited during `PersistInputEdgeIdxs::Build()`

        let subtree_visit = if current_key_out_edge_idxs.is_empty() {
            SubtreeVisit::Skip
        } else {
            visitor.pre_visit(
                current_key.clone(),
                storage,
                &mut SliceEdgeProvider {
                    edges,
                    input_range: input_edges_start_idx..input_edges_start_idx + num_inputs,
                    output_idxs: &current_key_out_edge_idxs,
                },
            )?
        };

        let mut deleted_inputs: usize = 0;
        if match subtree_visit {
            SubtreeVisit::Visit => true,
            SubtreeVisit::Skip => {
                self.graph_visit_order_fn.is_some()
                    || self
                        .visitor
                        .is_deleted_edge(&self.edges[current_key_out_edge_idx])
                        .is_some()
            },
        } {
            for i in 0..num_inputs {
                let key = self.visit_stack[base_visit_stack_len + i].clone();
                let deleted = matches!(subtree_visit, SubtreeVisit::Skip)
                    || self
                        .visitor
                        .is_deleted_edge(&self.edges[input_edges_start_idx + i])
                        == Some(true);

                self.traverse_rec(key, input_edges_start_idx + i, deleted)?;

                if deleted {
                    deleted_inputs += 1;
                } else if deleted_inputs != 0 {
                    self.edges.swap(i, i - deleted_inputs)
                }
            }
        }

        let Self {
            storage,
            visit_stack,
            edges,
            persist_input_edge_idxs,
            graph_visit_order_fn,
            visitor,
        } = self;

        assert_eq!(visit_stack.len(), base_visit_stack_len + num_inputs);
        visit_stack.truncate(base_visit_stack_len);

        let persist_edges = persist_input_edge_idxs.is_some() || graph_visit_order_fn.is_some();

        if !persist_edges {
            assert_eq!(edges.len(), input_edges_start_idx + num_inputs);
        }

        if !current_key_out_edge_idxs.is_empty() {
            visitor.post_visit(
                current_key,
                storage,
                &mut SliceEdgeProvider {
                    edges,
                    input_range: input_edges_start_idx
                        ..input_edges_start_idx + num_inputs - deleted_inputs,
                    output_idxs: &current_key_out_edge_idxs,
                },
            )?;
        }

        if !persist_edges {
            edges.truncate(input_edges_start_idx);
        }

        ControlFlow::Continue(())
    }
}

pub enum GraphVisitOrder<EdgeKey> {
    HasUnvisitedOutputs,
    Visit { output_keys: UnitVec<EdgeKey> },
}

struct SliceEdgeProvider<'a, Edge> {
    edges: &'a mut [Edge],
    input_range: Range<usize>,
    output_idxs: &'a [usize],
}

impl<'provider, Edge> NodeEdgesProvider<Edge> for SliceEdgeProvider<'provider, Edge> {
    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(unsafe {
            std::mem::transmute::<
                &'a mut SliceEdgeProvider<'provider, Edge>,
                &'a mut Inputs<SliceEdgeProvider<'provider, Edge>>,
            >(self)
        })
    }

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(unsafe {
            std::mem::transmute::<
                &'a mut SliceEdgeProvider<'provider, Edge>,
                &'a mut Outputs<SliceEdgeProvider<'provider, Edge>>,
            >(self)
        })
    }

    fn swap_input_output(&mut self, input_idx: usize, output_idx: usize) {
        assert!(input_idx < self.input_range.len());
        assert!(output_idx < self.output_idxs.len());

        let input_idx = self.input_range.start + input_idx;
        let output_idx = self.output_idxs[output_idx];

        self.edges.swap(input_idx, output_idx);
    }

    fn get_input_output_mut(&mut self, input_idx: usize, output_idx: usize) -> [&mut Edge; 2] {
        assert!(input_idx < self.input_range.len());
        assert!(output_idx < self.output_idxs.len());

        let input_idx = self.input_range.start + input_idx;
        let output_idx = self.output_idxs[output_idx];

        self.edges
            .get_disjoint_mut([input_idx, output_idx])
            .unwrap()
    }
}

#[repr(transparent)]
struct Inputs<T>(T);

impl<'a, Edge> Collection<Edge> for Inputs<SliceEdgeProvider<'a, Edge>> {
    fn len(&self) -> usize {
        self.0.input_range.len()
    }

    fn get(&self, idx: usize) -> Option<&Edge> {
        (idx < self.0.input_range.len()).then(|| &self.0.edges[self.0.input_range.start + idx])
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Edge> {
        (idx < self.0.input_range.len()).then(|| &mut self.0.edges[self.0.input_range.start + idx])
    }
}

#[repr(transparent)]
struct Outputs<T>(T);

impl<'a, Edge> Collection<Edge> for Outputs<SliceEdgeProvider<'a, Edge>> {
    fn len(&self) -> usize {
        self.0.output_idxs.len()
    }

    fn get(&self, idx: usize) -> Option<&Edge> {
        Some(&self.0.edges[*self.0.output_idxs.get(idx)?])
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Edge> {
        Some(&mut self.0.edges[*self.0.output_idxs.get(idx)?])
    }
}
