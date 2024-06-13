use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_plan::prelude::expr_ir::ExprIR;
use polars_utils::arena::Node;

mod lower_ir;
mod to_graph;

pub use lower_ir::lower_ir;

slotmap::new_key_type! {
    /// Key used for PNodes.
    pub struct PhysNodeKey;
}

/// A node in the physical plan.
#[derive(Clone, Debug)]
pub enum PhysNode {
    DataFrameScan {
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
    // Fallback to the in-memory engine.
    Fallback(Node),
}
