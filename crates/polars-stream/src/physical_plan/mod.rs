use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{IdxSize, InitHashMaps, PlHashMap, SortMultipleOptions};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
use polars_ops::frame::JoinArgs;
use polars_plan::dsl::deletion::DeletionFilesList;
use polars_plan::dsl::{
    CastColumnsPolicy, FileSinkOptions, JoinTypeOptionsIR, MissingColumnsPolicy,
    PartitionedSinkOptionsIR, PredicateFileSkip, ScanSources, TableStatistics,
};
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_plan::plans::{AExpr, DataFrameUdf, DynamicPred, IR};

mod fmt;
mod io;
mod lower_expr;
mod lower_group_by;
mod lower_ir;
mod to_graph;

pub use fmt::{NodeStyle, visualize_plan};
use polars_plan::prelude::PlanCallback;
#[cfg(feature = "dynamic_group_by")]
use polars_time::DynamicGroupOptions;
use polars_time::{ClosedWindow, Duration};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;
use slotmap::{SecondaryMap, SlotMap};
pub use to_graph::physical_plan_to_graph;

pub use self::lower_ir::StreamingLowerIRContext;
use crate::nodes::io_sources::multi_scan::components::forbid_extra_columns::ForbidExtraColumns;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::physical_plan::lower_expr::ExprCache;

slotmap::new_key_type! {
    /// Key used for physical nodes.
    pub struct PhysNodeKey;
}

impl PhysNodeKey {
    pub fn as_ffi(&self) -> u64 {
        self.0.as_ffi()
    }
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

    pub fn kind(&self) -> &PhysNodeKind {
        &self.kind
    }
}

/// A handle representing a physical stream of data with a fixed schema in the
/// physical plan. It consists of a reference to a physical node as well as the
/// output port on that node to connect to receive the stream.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct PhysStream {
    pub node: PhysNodeKey,
    pub port: usize,
}

impl PhysStream {
    #[allow(unused)]
    pub fn new(node: PhysNodeKey, port: usize) -> Self {
        Self { node, port }
    }

    // Convenience method to refer to the first output port of a physical node.
    pub fn first(node: PhysNodeKey) -> Self {
        Self { node, port: 0 }
    }
}

/// Behaviour when handling multiple DataFrames with different heights.

#[derive(Clone, Debug, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "physical_plan_visualization_schema",
    derive(schemars::JsonSchema)
)]
pub enum ZipBehavior {
    /// Fill the shorter DataFrames with nulls to the height of the longest DataFrame.
    NullExtend,
    /// All inputs must be the same height, or have length 1 in which case they are broadcast.
    Broadcast,
    /// Raise an error if the DataFrames have different heights.
    Strict,
}

#[derive(Clone, Debug)]
pub enum PhysNodeKind {
    InMemorySource {
        df: Arc<DataFrame>,
        disable_morsel_split: bool,
    },

    Select {
        input: PhysStream,
        selectors: Vec<ExprIR>,
        extend_original: bool,
    },

    InputIndependentSelect {
        selectors: Vec<ExprIR>,
    },

    WithRowIndex {
        input: PhysStream,
        name: PlSmallStr,
        offset: Option<IdxSize>,
    },

    Reduce {
        input: PhysStream,
        exprs: Vec<ExprIR>,
    },

    StreamingSlice {
        input: PhysStream,
        offset: usize,
        length: usize,
    },

    NegativeSlice {
        input: PhysStream,
        offset: i64,
        length: usize,
    },

    DynamicSlice {
        input: PhysStream,
        offset: PhysStream,
        length: PhysStream,
    },

    Shift {
        input: PhysStream,
        offset: PhysStream,
        fill: Option<PhysStream>,
    },

    Filter {
        input: PhysStream,
        predicate: ExprIR,
    },

    SimpleProjection {
        input: PhysStream,
        columns: Vec<PlSmallStr>,
    },

    InMemorySink {
        input: PhysStream,
    },

    CallbackSink {
        input: PhysStream,
        function: PlanCallback<DataFrame, bool>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
    },

    FileSink {
        input: PhysStream,
        options: FileSinkOptions,
    },

    PartitionedSink {
        input: PhysStream,
        options: PartitionedSinkOptionsIR,
    },

    SinkMultiple {
        sinks: Vec<PhysNodeKey>,
    },

    /// Generic fallback for (as-of-yet) unsupported streaming mappings.
    /// Fully sinks all data to an in-memory data frame and uses the in-memory
    /// engine to perform the map.
    InMemoryMap {
        input: PhysStream,
        map: Arc<dyn DataFrameUdf>,

        /// A formatted explain of what the in-memory map. This usually calls format on the IR.
        format_str: Option<String>,
    },

    Map {
        input: PhysStream,
        map: Arc<dyn DataFrameUdf>,

        /// A formatted explain of what the in-memory map. This usually calls format on the IR.
        format_str: Option<String>,
    },

    SortedGroupBy {
        input: PhysStream,
        key: PlSmallStr,
        aggs: Vec<ExprIR>,
        slice: Option<(IdxSize, IdxSize)>,
    },

    Sort {
        input: PhysStream,
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },

    TopK {
        input: PhysStream,
        k: PhysStream,
        by_column: Vec<ExprIR>,
        reverse: Vec<bool>,
        nulls_last: Vec<bool>,
        dyn_pred: Option<DynamicPred>,
    },

    Repeat {
        value: PhysStream,
        repeats: PhysStream,
    },

    #[cfg(feature = "cum_agg")]
    CumAgg {
        input: PhysStream,
        kind: crate::nodes::cum_agg::CumAggKind,
    },

    // Parameter is the input stream
    GatherEvery {
        input: PhysStream,
        n: usize,
        offset: usize,
    },
    Rle(PhysStream),
    RleId(PhysStream),
    PeakMinMax {
        input: PhysStream,
        is_peak_max: bool,
    },

    OrderedUnion {
        inputs: Vec<PhysStream>,
    },

    UnorderedUnion {
        inputs: Vec<PhysStream>,
    },

    Zip {
        inputs: Vec<PhysStream>,
        zip_behavior: ZipBehavior,
    },

    #[allow(unused)]
    Multiplexer {
        input: PhysStream,
    },

    MultiScan {
        scan_sources: ScanSources,

        file_reader_builder: Arc<dyn FileReaderBuilder>,
        cloud_options: Option<Arc<CloudOptions>>,

        /// Columns to project from the file.
        file_projection_builder: ProjectionBuilder,
        /// Final output schema of morsels being sent out of MultiScan.
        output_schema: SchemaRef,

        row_index: Option<RowIndex>,
        pre_slice: Option<Slice>,
        predicate: Option<ExprIR>,
        predicate_file_skip_applied: Option<PredicateFileSkip>,

        hive_parts: Option<HivePartitionsDf>,
        include_file_paths: Option<PlSmallStr>,
        cast_columns_policy: CastColumnsPolicy,
        missing_columns_policy: MissingColumnsPolicy,
        forbid_extra_columns: Option<ForbidExtraColumns>,

        deletion_files: Option<DeletionFilesList>,
        table_statistics: Option<TableStatistics>,

        /// Schema of columns contained in the file. Does not contain external columns (e.g. hive / row_index).
        file_schema: SchemaRef,
        disable_morsel_split: bool,
    },

    #[cfg(feature = "python")]
    PythonScan {
        options: polars_plan::plans::python::PythonOptions,
    },

    GroupBy {
        inputs: Vec<PhysStream>,
        // Must have the same schema when applied for each input.
        key_per_input: Vec<Vec<ExprIR>>,
        // Must be a 'simple' expression, a singular column feeding into a single aggregate, or Len.
        aggs_per_input: Vec<Vec<ExprIR>>,
    },

    #[cfg(feature = "dynamic_group_by")]
    DynamicGroupBy {
        input: PhysStream,
        options: DynamicGroupOptions,
        aggs: Vec<ExprIR>,
        slice: Option<(IdxSize, IdxSize)>,
    },

    #[cfg(feature = "dynamic_group_by")]
    RollingGroupBy {
        input: PhysStream,
        index_column: PlSmallStr,
        period: Duration,
        offset: Duration,
        closed: ClosedWindow,
        slice: Option<(IdxSize, IdxSize)>,
        aggs: Vec<ExprIR>,
    },

    EquiJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
    },

    MergeJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        descending: bool,
        nulls_last: bool,
        keys_row_encoded: bool,
        args: JoinArgs,
    },

    SemiAntiJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
        output_bool: bool,
    },

    CrossJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        args: JoinArgs,
    },

    AsOfJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        tmp_left_key_col: Option<PlSmallStr>,
        tmp_right_key_col: Option<PlSmallStr>,
        args: JoinArgs,
    },

    /// Generic fallback for (as-of-yet) unsupported streaming joins.
    /// Fully sinks all data to in-memory data frames and uses the in-memory
    /// engine to perform the join.
    InMemoryJoin {
        input_left: PhysStream,
        input_right: PhysStream,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        args: JoinArgs,
        options: Option<JoinTypeOptionsIR>,
    },

    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        input_left: PhysStream,
        input_right: PhysStream,
    },

    #[cfg(feature = "ewma")]
    EwmMean {
        input: PhysStream,
        options: polars_ops::series::EWMOptions,
    },

    #[cfg(feature = "ewma")]
    EwmVar {
        input: PhysStream,
        options: polars_ops::series::EWMOptions,
    },

    #[cfg(feature = "ewma")]
    EwmStd {
        input: PhysStream,
        options: polars_ops::series::EWMOptions,
    },
}

fn visit_node_inputs_mut(
    roots: Vec<PhysNodeKey>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    mut visit: impl FnMut(&mut PhysStream),
) {
    let mut to_visit = roots;
    let mut seen: SecondaryMap<PhysNodeKey, ()> =
        to_visit.iter().copied().map(|n| (n, ())).collect();
    macro_rules! rec {
        ($n:expr) => {
            let n = $n;
            if seen.insert(n, ()).is_none() {
                to_visit.push(n)
            }
        };
    }
    while let Some(node) = to_visit.pop() {
        match &mut phys_sm[node].kind {
            PhysNodeKind::InMemorySource { .. }
            | PhysNodeKind::MultiScan { .. }
            | PhysNodeKind::InputIndependentSelect { .. } => {},
            #[cfg(feature = "python")]
            PhysNodeKind::PythonScan { .. } => {},
            PhysNodeKind::Select { input, .. }
            | PhysNodeKind::WithRowIndex { input, .. }
            | PhysNodeKind::Reduce { input, .. }
            | PhysNodeKind::StreamingSlice { input, .. }
            | PhysNodeKind::NegativeSlice { input, .. }
            | PhysNodeKind::Filter { input, .. }
            | PhysNodeKind::SimpleProjection { input, .. }
            | PhysNodeKind::InMemorySink { input }
            | PhysNodeKind::CallbackSink { input, .. }
            | PhysNodeKind::FileSink { input, .. }
            | PhysNodeKind::PartitionedSink { input, .. }
            | PhysNodeKind::InMemoryMap { input, .. }
            | PhysNodeKind::SortedGroupBy { input, .. }
            | PhysNodeKind::Map { input, .. }
            | PhysNodeKind::Sort { input, .. }
            | PhysNodeKind::Multiplexer { input }
            | PhysNodeKind::GatherEvery { input, .. }
            | PhysNodeKind::Rle(input)
            | PhysNodeKind::RleId(input)
            | PhysNodeKind::PeakMinMax { input, .. } => {
                rec!(input.node);
                visit(input);
            },

            #[cfg(feature = "dynamic_group_by")]
            PhysNodeKind::DynamicGroupBy { input, .. } => {
                rec!(input.node);
                visit(input);
            },
            #[cfg(feature = "dynamic_group_by")]
            PhysNodeKind::RollingGroupBy { input, .. } => {
                rec!(input.node);
                visit(input);
            },

            #[cfg(feature = "cum_agg")]
            PhysNodeKind::CumAgg { input, .. } => {
                rec!(input.node);
                visit(input);
            },

            PhysNodeKind::InMemoryJoin {
                input_left,
                input_right,
                ..
            }
            | PhysNodeKind::EquiJoin {
                input_left,
                input_right,
                ..
            }
            | PhysNodeKind::MergeJoin {
                input_left,
                input_right,
                ..
            }
            | PhysNodeKind::SemiAntiJoin {
                input_left,
                input_right,
                ..
            }
            | PhysNodeKind::CrossJoin {
                input_left,
                input_right,
                ..
            }
            | PhysNodeKind::AsOfJoin {
                input_left,
                input_right,
                ..
            } => {
                rec!(input_left.node);
                rec!(input_right.node);
                visit(input_left);
                visit(input_right);
            },

            #[cfg(feature = "merge_sorted")]
            PhysNodeKind::MergeSorted {
                input_left,
                input_right,
                ..
            } => {
                rec!(input_left.node);
                rec!(input_right.node);
                visit(input_left);
                visit(input_right);
            },

            PhysNodeKind::TopK { input, k, .. } => {
                rec!(input.node);
                rec!(k.node);
                visit(input);
                visit(k);
            },

            PhysNodeKind::DynamicSlice {
                input,
                offset,
                length,
            } => {
                rec!(input.node);
                rec!(offset.node);
                rec!(length.node);
                visit(input);
                visit(offset);
                visit(length);
            },

            PhysNodeKind::Shift {
                input,
                offset,
                fill,
            } => {
                rec!(input.node);
                rec!(offset.node);
                if let Some(fill) = fill {
                    rec!(fill.node);
                }
                visit(input);
                visit(offset);
                if let Some(fill) = fill {
                    visit(fill);
                }
            },

            PhysNodeKind::Repeat { value, repeats } => {
                rec!(value.node);
                rec!(repeats.node);
                visit(value);
                visit(repeats);
            },

            PhysNodeKind::GroupBy { inputs, .. }
            | PhysNodeKind::OrderedUnion { inputs }
            | PhysNodeKind::UnorderedUnion { inputs }
            | PhysNodeKind::Zip { inputs, .. } => {
                for input in inputs {
                    rec!(input.node);
                    visit(input);
                }
            },

            PhysNodeKind::SinkMultiple { sinks } => {
                for sink in sinks {
                    rec!(*sink);
                    visit(&mut PhysStream::first(*sink));
                }
            },

            #[cfg(feature = "ewma")]
            PhysNodeKind::EwmMean { input, options: _ }
            | PhysNodeKind::EwmVar { input, options: _ }
            | PhysNodeKind::EwmStd { input, options: _ } => {
                rec!(input.node);
                visit(input)
            },
        }
    }
}

fn insert_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>) {
    let mut refcount = PlHashMap::new();
    visit_node_inputs_mut(roots.clone(), phys_sm, |i| {
        *refcount.entry(*i).or_insert(0) += 1;
    });

    let mut multiplexer_map: PlHashMap<PhysStream, PhysStream> = refcount
        .into_iter()
        .filter(|(_stream, refcount)| *refcount > 1)
        .map(|(stream, _refcount)| {
            let input_schema = phys_sm[stream.node].output_schema.clone();
            let multiplexer_node = phys_sm.insert(PhysNode::new(
                input_schema,
                PhysNodeKind::Multiplexer { input: stream },
            ));
            (stream, PhysStream::first(multiplexer_node))
        })
        .collect();

    visit_node_inputs_mut(roots, phys_sm, |i| {
        if let Some(m) = multiplexer_map.get_mut(i) {
            *i = *m;
            m.port += 1;
        }
    });
}

pub fn build_physical_plan(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    ctx: StreamingLowerIRContext,
) -> PolarsResult<PhysNodeKey> {
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let mut expr_cache = ExprCache::with_capacity(expr_arena.len());
    let mut cache_nodes = PlHashMap::new();
    let phys_root = lower_ir::lower_ir(
        root,
        ir_arena,
        expr_arena,
        phys_sm,
        &mut schema_cache,
        &mut expr_cache,
        &mut cache_nodes,
        ctx,
        None,
    )?;
    insert_multiplexers(vec![phys_root.node], phys_sm);
    Ok(phys_root.node)
}
