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
pub struct PhysNode {
    output_schema: Arc<Schema>,
    kind: PhysNodeKind,
}

impl PhysNode {
    pub fn new(output_schema: Arc<Schema>, kind: PhysNodeKind) -> Self {
        Self {
            output_schema,
            kind,
        }
    }
}

#[derive(Clone, Debug)]
pub enum PhysNodeKind {
    InMemorySource {
        df: Arc<DataFrame>,
    },

    Select {
        input: PhysNodeKey,
        selectors: Vec<ExprIR>,
        extend_original: bool,
    },

    Reduce {
        input: PhysNodeKey,
        exprs: Vec<ExprIR>,
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
        columns: Vec<String>,
    },

    InMemorySink {
        input: PhysNodeKey,
    },

    InMemoryMap {
        input: PhysNodeKey,
        map: Arc<dyn DataFrameUdf>,
    },

    Map {
        input: PhysNodeKey,
        map: Arc<dyn DataFrameUdf>,
    },

    Sort {
        input: PhysNodeKey,
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },

    OrderedUnion {
        inputs: Vec<PhysNodeKey>,
    },

    Zip {
        inputs: Vec<PhysNodeKey>,
        /// If true shorter inputs are extended with nulls to the longest input,
        /// if false all inputs must be the same length, or have length 1 in
        /// which case they are broadcast.
        null_extend: bool,
    },

    #[allow(unused)]
    Multiplexer {
        input: PhysNodeKey,
    },
}
