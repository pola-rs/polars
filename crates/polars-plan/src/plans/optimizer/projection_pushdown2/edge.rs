use std::sync::{Arc, LazyLock};

use polars_core::prelude::PlIndexSet;
use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use crate::plans::IR;
use crate::plans::optimizer::projection_pushdown2::{iters_eq, min_dtype_size_col};

#[derive(Debug, Clone)]
pub struct Edge(ProjectionState, ParentKeyAndPort);

impl Edge {
    pub fn new(
        projection: Projection,
        names: Option<Box<PlIndexSet<PlSmallStr>>>,
        parent_key_and_port: ParentKeyAndPort,
    ) -> Self {
        Self(ProjectionState { projection, names }, parent_key_and_port)
    }
}

const _: () = {
    assert!(std::mem::size_of::<Edge>() <= 32);
};

#[derive(Debug, Clone, Default)]
pub struct ProjectionState {
    pub projection: Projection,
    /// If `projection` is `Projection::All` and this is non-empty, it is assumed to match the
    /// full set of output columns (in order) for the current node.
    pub names: Option<Box<PlIndexSet<PlSmallStr>>>,
}

static EMPTY_NAMES: LazyLock<PlIndexSet<PlSmallStr>> = LazyLock::new(Default::default);

pub trait GetProjectionState {
    fn projection_state(&self) -> &ProjectionState;
    fn projection_state_mut(&mut self) -> &mut ProjectionState;

    fn projection(&self) -> Projection {
        self.projection_state().projection.clone()
    }

    fn projection_mut(&mut self) -> &mut Projection {
        &mut self.projection_state_mut().projection
    }

    fn names(&self) -> &PlIndexSet<PlSmallStr> {
        self.projection_state()
            .names
            .as_deref()
            .unwrap_or(&*EMPTY_NAMES)
    }

    fn names_mut(&mut self) -> &mut PlIndexSet<PlSmallStr> {
        self.projection_state_mut()
            .names
            .get_or_insert_default()
            .as_mut()
    }

    fn take_names(&mut self) -> Option<Box<PlIndexSet<PlSmallStr>>> {
        self.projection_state_mut().names.take()
    }

    /// Returns `None` if the projected names already match the input schema.
    fn compute_projected_names(
        &mut self,
        input_schema: &Schema,
    ) -> Option<&PlIndexSet<PlSmallStr>> {
        match self.projection() {
            Projection::All => return None,
            Projection::Len => {
                if input_schema.len() <= 1 {
                    return None;
                }

                if self.names().is_empty() {
                    self.names_mut()
                        .insert(min_dtype_size_col(input_schema.iter()).unwrap().clone());
                } else {
                    assert_eq!(self.names().len(), 1);
                    assert!(input_schema.contains(self.names().first().unwrap()));
                }
            },
            Projection::Names => {
                if iters_eq(self.names().iter(), input_schema.iter_names()) {
                    return None;
                }
            },
        };

        Some(self.names())
    }

    /// Returns `None` if the projected names are not a subset of the input schema.
    fn compute_projected_names_subset(
        &mut self,
        input_schema: &Schema,
    ) -> Option<&PlIndexSet<PlSmallStr>> {
        self.compute_projected_names(input_schema)
            .filter(|x| x.len() != input_schema.len())
    }
}

impl GetProjectionState for ProjectionState {
    fn projection_state(&self) -> &ProjectionState {
        self
    }

    fn projection_state_mut(&mut self) -> &mut ProjectionState {
        self
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub enum Projection {
    /// Project all columns
    #[default]
    All,
    /// Project the names in the names set.
    Names,
    /// Consumer is `select(len())`.
    Len,
}

impl GetProjectionState for Edge {
    fn projection_state(&self) -> &ProjectionState {
        &self.0
    }

    fn projection_state_mut(&mut self) -> &mut ProjectionState {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
pub struct ParentKeyAndPort {
    /// Node that consumes the output of the current node.
    pub node: Node,
    /// Index in the inputs of that node at which the current node sends its
    /// output.
    pub idx: usize,
}

const DELETED_BIT: usize = 1 << (usize::BITS - 1);

pub trait GetParentKeyAndPort {
    fn parent_key_and_port(&self) -> &ParentKeyAndPort;
    fn parent_key_and_port_mut(&mut self) -> &mut ParentKeyAndPort;
}

impl GetParentKeyAndPort for Edge {
    fn parent_key_and_port(&self) -> &ParentKeyAndPort {
        &self.1
    }

    fn parent_key_and_port_mut(&mut self) -> &mut ParentKeyAndPort {
        &mut self.1
    }
}

impl ParentKeyAndPort {
    pub fn is_deleted(&self) -> bool {
        self.idx & DELETED_BIT != 0
    }

    pub fn set_deleted(&mut self, deleted: bool) {
        if deleted {
            self.idx |= DELETED_BIT
        }
    }

    pub fn attach_simple_projection(
        &mut self,
        schema: Arc<Schema>,
        ir_arena: &mut Arena<IR>,
    ) -> bool {
        let ir = ir_arena.get(self.node);

        debug_assert!(self.idx & DELETED_BIT == 0);
        debug_assert!(self.idx != usize::MAX);

        let new_consumer_ir = IR::SimpleProjection {
            input: ir.inputs().nth(self.idx).unwrap(),
            columns: schema,
        };

        let new_consumer_node = ir_arena.add(new_consumer_ir);

        *ir_arena
            .get_mut(self.node)
            .inputs_mut()
            .nth(self.idx)
            .unwrap() = new_consumer_node;

        *self = Self {
            node: new_consumer_node,
            idx: 0,
        };

        true
    }
}
