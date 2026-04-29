use std::marker::PhantomData;
use std::ops::ControlFlow;

use crate::traversal::edge_provider::NodeEdgesProvider;

pub enum SubtreeVisit {
    Visit,
    Skip,
}

pub trait NodeVisitor {
    type Key;
    type Storage;
    type Edge;
    type BreakValue;

    fn default_edge(&mut self) -> Self::Edge;

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit>;

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue>;
}

pub struct FnVisitors<Key, Storage, Edge, BreakValue, DefaultEdgeFn, PreVisitFn, PostVisitFn>
where
    PreVisitFn: FnMut(
        Key,
        &mut Storage,
        &mut dyn NodeEdgesProvider<Edge>,
    ) -> ControlFlow<BreakValue, SubtreeVisit>,
    PostVisitFn:
        FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> ControlFlow<BreakValue>,
{
    default_edge_fn: DefaultEdgeFn,
    pre_visit_fn: PreVisitFn,
    post_visit_fn: PostVisitFn,
    phantom: PhantomData<(Key, Storage, Edge, BreakValue)>,
}

impl<Key, Storage, Edge, BreakValue, DefaultEdgeFn, PreVisitFn, PostVisitFn>
    FnVisitors<Key, Storage, Edge, BreakValue, DefaultEdgeFn, PreVisitFn, PostVisitFn>
where
    DefaultEdgeFn: FnMut() -> Edge,
    PreVisitFn: FnMut(
        Key,
        &mut Storage,
        &mut dyn NodeEdgesProvider<Edge>,
    ) -> ControlFlow<BreakValue, SubtreeVisit>,
    PostVisitFn:
        FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> ControlFlow<BreakValue>,
{
    pub fn new(
        default_edge_fn: DefaultEdgeFn,
        pre_visit_fn: PreVisitFn,
        post_visit_fn: PostVisitFn,
    ) -> Self {
        Self {
            default_edge_fn,
            pre_visit_fn,
            post_visit_fn,
            phantom: PhantomData,
        }
    }
}

impl<Key, Storage, Edge, BreakValue, DefaultEdgeFn, PreVisitFn, PostVisitFn> NodeVisitor
    for FnVisitors<Key, Storage, Edge, BreakValue, DefaultEdgeFn, PreVisitFn, PostVisitFn>
where
    DefaultEdgeFn: FnMut() -> Edge,
    PreVisitFn: FnMut(
        Key,
        &mut Storage,
        &mut dyn NodeEdgesProvider<Edge>,
    ) -> ControlFlow<BreakValue, SubtreeVisit>,
    PostVisitFn:
        FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> ControlFlow<BreakValue>,
{
    type Key = Key;
    type Storage = Storage;
    type Edge = Edge;
    type BreakValue = BreakValue;

    fn default_edge(&mut self) -> Self::Edge {
        (self.default_edge_fn)()
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue, SubtreeVisit> {
        (self.pre_visit_fn)(key, storage, edges)
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> ControlFlow<Self::BreakValue> {
        (self.post_visit_fn)(key, storage, edges)
    }
}
