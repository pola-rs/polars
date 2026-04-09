use std::marker::PhantomData;

use polars_error::PolarsResult;

use crate::traversal::edge_provider::NodeEdgesProvider;

pub enum SubtreeVisit {
    Visit,
    Skip,
}

pub trait NodeVisitor {
    type Key;
    type Storage;
    type Edge;

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> PolarsResult<SubtreeVisit>;

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> PolarsResult<()>;
}

pub struct FnVisitors<Key, Storage, Edge, PreVisitFn, PostVisitFn>
where
    PreVisitFn:
        FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> PolarsResult<SubtreeVisit>,
    PostVisitFn: FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> PolarsResult<()>,
{
    pre_visit: PreVisitFn,
    post_visit: PostVisitFn,
    phantom: PhantomData<(Key, Storage, Edge)>,
}

impl<Key, Storage, Edge, PreVisitFn, PostVisitFn>
    FnVisitors<Key, Storage, Edge, PreVisitFn, PostVisitFn>
where
    PreVisitFn:
        FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> PolarsResult<SubtreeVisit>,
    PostVisitFn: FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> PolarsResult<()>,
{
    pub fn new(pre_visit: PreVisitFn, post_visit: PostVisitFn) -> Self {
        Self {
            pre_visit,
            post_visit,
            phantom: PhantomData,
        }
    }
}

impl<Key, Storage, Edge, PreVisitFn, PostVisitFn> NodeVisitor
    for FnVisitors<Key, Storage, Edge, PreVisitFn, PostVisitFn>
where
    PreVisitFn:
        FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> PolarsResult<SubtreeVisit>,
    PostVisitFn: FnMut(Key, &mut Storage, &mut dyn NodeEdgesProvider<Edge>) -> PolarsResult<()>,
{
    type Key = Key;
    type Storage = Storage;
    type Edge = Edge;

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> PolarsResult<SubtreeVisit> {
        (self.pre_visit)(key, storage, edges)
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> PolarsResult<()> {
        (self.post_visit)(key, storage, edges)
    }
}
