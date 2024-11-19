use std::path::PathBuf;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{IdxSize, InitHashMaps, PlHashMap, SortMultipleOptions};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_ops::frame::JoinArgs;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::{AExpr, DataFrameUdf, FileInfo, FileScan, ScanSources, IR};
use polars_plan::prelude::expr_ir::ExprIR;

mod fmt;
mod lower_expr;
mod lower_ir;
mod to_graph;

pub use fmt::visualize_plan;
use polars_plan::prelude::{FileScanOptions, FileType};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use slotmap::{Key, SecondaryMap, SlotMap};
pub use to_graph::physical_plan_to_graph;

use crate::physical_plan::lower_expr::ExprCache;

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

    WithRowIndex {
        input: PhysNodeKey,
        name: PlSmallStr,
        offset: Option<IdxSize>,
    },

    InputIndependentSelect {
        selectors: Vec<ExprIR>,
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
        columns: Vec<PlSmallStr>,
    },

    InMemorySink {
        input: PhysNodeKey,
    },

    FileSink {
        path: Arc<PathBuf>,
        file_type: FileType,
        input: PhysNodeKey,
    },

    /// Generic fallback for (as-of-yet) unsupported streaming mappings.
    /// Fully sinks all data to an in-memory data frame and uses the in-memory
    /// engine to perform the map.
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

    FileScan {
        scan_sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<ExprIR>,
        output_schema: Option<SchemaRef>,
        scan_type: FileScan,
        file_options: FileScanOptions,
    },

    GroupBy {
        input: PhysNodeKey,
        key: Vec<ExprIR>,
        aggs: Vec<ExprIR>,
    },
    
    EquiJoin {
        input_left: PhysNodeKey,
        input_right: PhysNodeKey,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
    },

    /// Generic fallback for (as-of-yet) unsupported streaming joins.
    /// Fully sinks all data to in-memory data frames and uses the in-memory
    /// engine to perform the join.
    InMemoryJoin {
        input_left: PhysNodeKey,
        input_right: PhysNodeKey,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
    },
}

#[recursive::recursive]
fn insert_multiplexers(
    node: PhysNodeKey,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    referenced: &mut SecondaryMap<PhysNodeKey, ()>,
) {
    let seen_before = referenced.insert(node, ()).is_some();
    if seen_before && !matches!(phys_sm[node].kind, PhysNodeKind::Multiplexer { .. }) {
        // This node is referenced at least twice. We first set the input key to
        // null and then update it to avoid a double-mutable-borrow issue.
        let input_schema = phys_sm[node].output_schema.clone();
        let orig_input_node = core::mem::replace(
            &mut phys_sm[node],
            PhysNode::new(
                input_schema,
                PhysNodeKind::Multiplexer {
                    input: PhysNodeKey::null(),
                },
            ),
        );
        let orig_input_key = phys_sm.insert(orig_input_node);
        phys_sm[node].kind = PhysNodeKind::Multiplexer {
            input: orig_input_key,
        };
    }

    if !seen_before {
        match &phys_sm[node].kind {
            PhysNodeKind::InMemorySource { .. }
            | PhysNodeKind::FileScan { .. }
            | PhysNodeKind::InputIndependentSelect { .. } => {},
            PhysNodeKind::Select { input, .. }
            | PhysNodeKind::WithRowIndex { input, .. }
            | PhysNodeKind::Reduce { input, .. }
            | PhysNodeKind::StreamingSlice { input, .. }
            | PhysNodeKind::Filter { input, .. }
            | PhysNodeKind::SimpleProjection { input, .. }
            | PhysNodeKind::InMemorySink { input }
            | PhysNodeKind::FileSink { input, .. }
            | PhysNodeKind::InMemoryMap { input, .. }
            | PhysNodeKind::Map { input, .. }
            | PhysNodeKind::Sort { input, .. }
            | PhysNodeKind::Multiplexer { input }
            | PhysNodeKind::GroupBy { input, .. } => {
                insert_multiplexers(*input, phys_sm, referenced);
            },

            PhysNodeKind::InMemoryJoin { input_left, input_right, .. } | PhysNodeKind::EquiJoin {
                input_left,
                input_right,
                ..
            } => {
                let input_right = *input_right;
                insert_multiplexers(*input_left, phys_sm, referenced);
                insert_multiplexers(input_right, phys_sm, referenced);
            },

            PhysNodeKind::OrderedUnion { inputs } | PhysNodeKind::Zip { inputs, .. } => {
                for input in inputs.clone() {
                    insert_multiplexers(input, phys_sm, referenced);
                }
            },
        }
    }
}

pub fn build_physical_plan(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
) -> PolarsResult<PhysNodeKey> {
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let mut expr_cache = ExprCache::with_capacity(expr_arena.len());
    let phys_root = lower_ir::lower_ir(
        root,
        ir_arena,
        expr_arena,
        phys_sm,
        &mut schema_cache,
        &mut expr_cache,
    )?;
    let mut referenced = SecondaryMap::with_capacity(phys_sm.capacity());
    insert_multiplexers(phys_root, phys_sm, &mut referenced);
    Ok(phys_root)
}
