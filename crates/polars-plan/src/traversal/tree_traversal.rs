use std::ops::{ControlFlow, Range};

use polars_utils::collection::{Collection, CollectionWrap};

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
    edges.push(visitor.default_edge());

    TreeTraversalImpl {
        storage,
        visit_stack,
        edges,
        persist_input_edge_idxs: None,
        visitor,
    }
    .traverse_rec(root_key, root_edge_idx)?;

    assert_eq!(edges.len(), root_edge_idx + 1);

    ControlFlow::Continue(edges.pop().unwrap())
}

pub enum PersistInputEdgeIdxs<'a> {
    Build(&'a mut Vec<usize>),
    Use(&'a [usize]),
}

pub struct TreeTraversalImpl<'a, Key, Storage, Edge, BreakValue> {
    pub storage: &'a mut Storage,
    pub visit_stack: &'a mut Vec<Key>,
    pub edges: &'a mut Vec<Edge>,
    pub persist_input_edge_idxs: Option<&'a mut PersistInputEdgeIdxs<'a>>,
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
    ) -> ControlFlow<BreakValue> {
        let Self {
            storage,
            visit_stack,
            edges,
            persist_input_edge_idxs,
            visitor,
        } = self;

        let base_visit_stack_len = visit_stack.len();

        storage.get_node_inputs(current_key.clone(), &mut |key| visit_stack.push(key));

        let num_inputs = visit_stack.len() - base_visit_stack_len;

        let input_edges_start_idx: usize = match persist_input_edge_idxs {
            Some(PersistInputEdgeIdxs::Use(idxs)) => idxs[current_key_out_edge_idx],
            _ => {
                let base_edges_len = edges.len();

                edges.extend((0..num_inputs).map(|_| visitor.default_edge()));

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

        match visitor.pre_visit(
            current_key.clone(),
            storage,
            &mut SliceEdgeProvider {
                edges,
                input_range: input_edges_start_idx..input_edges_start_idx + num_inputs,
                output_idx: current_key_out_edge_idx,
            },
        )? {
            SubtreeVisit::Visit => {
                for i in 0..num_inputs {
                    let key = self.visit_stack[base_visit_stack_len + i].clone();
                    self.traverse_rec(key, input_edges_start_idx + i)?;
                }
            },
            SubtreeVisit::Skip => {},
        }

        let Self {
            storage,
            visit_stack,
            edges,
            persist_input_edge_idxs,
            visitor,
        } = self;

        assert_eq!(visit_stack.len(), base_visit_stack_len + num_inputs);
        visit_stack.truncate(base_visit_stack_len);

        if persist_input_edge_idxs.is_none() {
            assert_eq!(edges.len(), input_edges_start_idx + num_inputs);
        }

        visitor.post_visit(
            current_key,
            storage,
            &mut SliceEdgeProvider {
                edges,
                input_range: input_edges_start_idx..input_edges_start_idx + num_inputs,
                output_idx: current_key_out_edge_idx,
            },
        )?;

        if persist_input_edge_idxs.is_none() {
            edges.truncate(input_edges_start_idx);
        }

        ControlFlow::Continue(())
    }
}

struct SliceEdgeProvider<'a, Edge> {
    edges: &'a mut [Edge],
    input_range: Range<usize>,
    output_idx: usize,
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
        1
    }

    fn get(&self, idx: usize) -> Option<&Edge> {
        (idx == 0).then(|| &self.0.edges[self.0.output_idx])
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Edge> {
        (idx == 0).then(|| &mut self.0.edges[self.0.output_idx])
    }
}
