use std::ops::Range;

use polars_error::PolarsResult;
use polars_utils::collection::{Collection, CollectionWrap};

use crate::apply_dyn_len_array;
use crate::traversal::edge_provider::{DynLenArray, EdgesUnpacker, NodeEdgesProvider};
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};

pub trait GetNodeInputs<Key> {
    fn push_inputs_for_key<C>(&self, key: Key, container: &mut C)
    where
        C: Extend<Key>;
}

pub fn tree_traversal<'visitor, Visitor, Edge>(
    root_key: Visitor::Key,
    storage: &mut Visitor::Storage,
    visit_stack: &mut Vec<Visitor::Key>,
    edges: &mut Vec<Edge>,
    visitor: &'visitor mut Visitor,
) -> PolarsResult<Edge>
where
    Visitor::Storage: GetNodeInputs<Visitor::Key>,
    Edge: Default,
    Visitor::Key: Clone,
    Visitor: NodeVisitor<Edge = Edge> + ?Sized,
{
    let root_edge_idx = edges.len();
    edges.push(Edge::default());

    tree_traversal_impl::<Visitor, Edge>(
        root_key,
        root_edge_idx,
        storage,
        visit_stack,
        edges,
        visitor,
    )?;

    assert_eq!(edges.len(), root_edge_idx + 1);
    Ok(edges.pop().unwrap())
}

#[recursive::recursive]
pub fn tree_traversal_impl<Visitor, Edge>(
    current_key: Visitor::Key,
    current_key_out_edge_idx: usize,
    storage: &mut Visitor::Storage,
    visit_stack: &mut Vec<Visitor::Key>,
    edges: &mut Vec<Edge>,
    visitor: &mut Visitor,
) -> PolarsResult<()>
where
    Visitor::Storage: GetNodeInputs<Visitor::Key>,
    Visitor::Key: Clone,
    Edge: Default,
    Visitor: NodeVisitor<Edge = Edge> + ?Sized,
{
    let base_visit_stack_len = visit_stack.len();
    let base_edges_len = edges.len();

    storage.push_inputs_for_key(current_key.clone(), visit_stack);

    let num_inputs = visit_stack.len() - base_visit_stack_len;

    edges.extend((0..num_inputs).map(|_| Edge::default()));

    match visitor.pre_visit(
        current_key.clone(),
        storage,
        &mut SliceEdgesProvider {
            edges,
            input_range: base_edges_len..base_edges_len + num_inputs,
            output_idx: current_key_out_edge_idx,
        },
    )? {
        SubtreeVisit::Visit => {
            for i in 0..num_inputs {
                tree_traversal_impl::<Visitor, Edge>(
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

    visitor.post_visit(
        current_key,
        storage,
        &mut SliceEdgesProvider {
            edges,
            input_range: base_edges_len..base_edges_len + num_inputs,
            output_idx: current_key_out_edge_idx,
        },
    )?;

    edges.truncate(base_edges_len);

    Ok(())
}

struct SliceEdgesProvider<'a, Edge> {
    edges: &'a mut [Edge],
    input_range: Range<usize>,
    output_idx: usize,
}

impl<'provider, Edge> NodeEdgesProvider<Edge> for SliceEdgesProvider<'provider, Edge> {
    fn unpacker<'a>(&'a mut self) -> EdgesUnpacker<'a, Edge>
    where
        Edge: 'a,
    {
        apply_dyn_len_array!(
            DynLenArray::from_iter(
                self.input_range
                    .clone()
                    .into_iter()
                    .chain([self.output_idx])
            )
            .unwrap(),
            |a| { self.edges.get_disjoint_mut(a).unwrap() }
        )
        .into()
    }

    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(unsafe {
            std::mem::transmute::<
                &'a mut SliceEdgesProvider<'provider, Edge>,
                &'a mut Inputs<SliceEdgesProvider<'provider, Edge>>,
            >(self)
        })
    }

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a,
    {
        CollectionWrap::new(unsafe {
            std::mem::transmute::<
                &'a mut SliceEdgesProvider<'provider, Edge>,
                &'a mut Outputs<SliceEdgesProvider<'provider, Edge>>,
            >(self)
        })
    }
}

#[repr(transparent)]
struct Inputs<T>(T);

impl<'a, Edge> Collection<Edge> for Inputs<SliceEdgesProvider<'a, Edge>> {
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

impl<'a, Edge> Collection<Edge> for Outputs<SliceEdgesProvider<'a, Edge>> {
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
