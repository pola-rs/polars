use std::marker::PhantomData;
use std::ops::{ControlFlow, Range};

use polars_utils::collection::{Collection, CollectionWrap};

use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

pub trait GetNodeInputs<Key> {
    fn push_inputs_for_key<C>(&self, key: Key, container: &mut C)
    where
        C: Extend<Key>;

    fn num_inputs(&self, key: Key) -> usize {
        struct Counter<T>(usize, PhantomData<T>);

        impl<T> Extend<T> for Counter<T> {
            fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
                iter.into_iter().for_each(|_| self.0 += 1);
            }
        }

        let mut c = Counter::<Key>(0, PhantomData);
        self.push_inputs_for_key(key, &mut c);

        c.0
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

    tree_traversal_impl::<Key, Storage, Edge, BreakValue>(
        root_key,
        root_edge_idx,
        storage,
        visit_stack,
        edges,
        visitor,
    )?;

    assert_eq!(edges.len(), root_edge_idx + 1);
    ControlFlow::Continue(edges.pop().unwrap())
}

#[recursive::recursive]
pub fn tree_traversal_impl<Key, Storage, Edge, BreakValue>(
    current_key: Key,
    current_key_out_edge_idx: usize,
    storage: &mut Storage,
    visit_stack: &mut Vec<Key>,
    edges: &mut Vec<Edge>,
    visitor: &mut dyn NodeVisitor<Key = Key, Storage = Storage, Edge = Edge, BreakValue = BreakValue>,
) -> ControlFlow<BreakValue>
where
    Key: Clone,
    Storage: GetNodeInputs<Key>,
{
    let base_visit_stack_len = visit_stack.len();
    let base_edges_len = edges.len();

    storage.push_inputs_for_key(current_key.clone(), visit_stack);

    let num_inputs = visit_stack.len() - base_visit_stack_len;

    edges.extend((0..num_inputs).map(|_| visitor.default_edge()));

    match visitor.pre_visit(
        current_key.clone(),
        storage,
        &mut SliceEdgeProvider {
            edges,
            input_range: base_edges_len..base_edges_len + num_inputs,
            output_idx: current_key_out_edge_idx,
        },
    )? {
        SubtreeVisit::Visit => {
            for i in 0..num_inputs {
                tree_traversal_impl(
                    visit_stack[base_visit_stack_len + i].clone(),
                    base_edges_len + i,
                    storage,
                    visit_stack,
                    edges,
                    visitor,
                )?;
            }
        },
        SubtreeVisit::Skip => {},
    }

    assert_eq!(visit_stack.len(), base_visit_stack_len + num_inputs);
    visit_stack.truncate(base_visit_stack_len);

    assert_eq!(edges.len(), base_edges_len + num_inputs);

    let control_flow = visitor.post_visit(
        current_key,
        storage,
        &mut SliceEdgeProvider {
            edges,
            input_range: base_edges_len..base_edges_len + num_inputs,
            output_idx: current_key_out_edge_idx,
        },
    );

    edges.truncate(base_edges_len);

    control_flow
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
