use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::SortMultipleOptions;
use polars_core::schema::Schema;
use polars_plan::plans::DataFrameUdf;
use polars_plan::prelude::expr_ir::ExprIR;

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

    Select {
        input: PhysNodeKey,
        selectors: Vec<ExprIR>,
        extend_original: bool,
        output_schema: Arc<Schema>,
    },

    Reduce {
        input: PhysNodeKey,
        exprs: Vec<ExprIR>,
        input_schema: Arc<Schema>,
        output_schema: Arc<Schema>,
    },

    StreamingSlice {
        input: PhysNodeKey,
        offset: usize,
        length: usize,
    },

    Filter {
        input: PhysNodeKey,
        predicate: ExprIR,
    },

    SimpleProjection {
        input: PhysNodeKey,
        input_schema: Arc<Schema>,
        columns: Vec<String>,
    },

    InMemorySink {
        input: PhysNodeKey,
        schema: Arc<Schema>,
    },

    InMemoryMap {
        input: PhysNodeKey,
        input_schema: Arc<Schema>,
        map: Arc<dyn DataFrameUdf>,
    },

    Map {
        input: PhysNodeKey,
        map: Arc<dyn DataFrameUdf>,
    },

    Sort {
        input: PhysNodeKey,
        input_schema: Arc<Schema>, // TODO: remove when not using fallback impl.
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },

    OrderedUnion {
        inputs: Vec<PhysNodeKey>,
    },

    Zip {
        inputs: Vec<PhysNodeKey>,
    },
}
