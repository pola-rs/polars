use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_utils::arena::Node;

mod lower_ir;
mod to_graph;

pub use lower_ir::lower_ir;
pub use to_graph::physical_plan_to_graph;

slotmap::new_key_type! {
    /// Key used for PNodes.
    pub struct PhysNodeKey;
}

/// A node in the physical plan.
///
/// A physical plan is created when the `IR` is translated to a directed
/// acyclic graph of operations that can run on the streaming engine.
#[derive(Clone, Debug)]
pub enum PhysNode {
    InMemorySource {
        df: Arc<DataFrame>,
    },
    Filter {
        input: PhysNodeKey,
        predicate: ExprIR,
    },
    SimpleProjection {
        input: PhysNodeKey,
        schema: SchemaRef,
    },
    InMemorySink {
        input: PhysNodeKey,
    },
    // Fallback to the in-memory engine.
    Fallback(Node),
}
