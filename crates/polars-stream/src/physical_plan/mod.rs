use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::SortMultipleOptions;
use polars_core::schema::{Schema, SchemaRef};
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::{AExpr, DataFrameUdf, EscapeLabel, FileInfo, FileScan, PathsDisplay};
use polars_plan::prelude::expr_ir::ExprIR;

mod lower_expr;
mod lower_ir;
mod to_graph;

pub use lower_ir::lower_ir;
use polars_plan::prelude::FileScanOptions;
use polars_utils::arena::Arena;
use polars_utils::itertools::Itertools;
use slotmap::{Key, SecondaryMap, SlotMap};
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

    FileScan {
        paths: Arc<Vec<PathBuf>>,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<ExprIR>,
        output_schema: Option<SchemaRef>,
        scan_type: FileScan,
        file_options: FileScanOptions,
    },
}

fn escape_graphviz(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

fn fmt_exprs(exprs: &[ExprIR], expr_arena: &Arena<AExpr>) -> String {
    exprs
        .iter()
        .map(|e| escape_graphviz(&e.display(expr_arena).to_string()))
        .collect_vec()
        .join("\\n")
}

#[recursive::recursive]
fn visualize_plan_rec(
    node_key: PhysNodeKey,
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
    visited: &mut SecondaryMap<PhysNodeKey, ()>,
    out: &mut Vec<String>,
) {
    if visited.contains_key(node_key) {
        return;
    }
    visited.insert(node_key, ());

    use std::slice::from_ref;
    let (label, inputs) = match &phys_sm[node_key].kind {
        PhysNodeKind::InMemorySource { df } => (
            format!(
                "in-memory-source\\ncols: {}",
                df.get_column_names().join(", ")
            ),
            &[][..],
        ),
        PhysNodeKind::Select {
            input,
            selectors,
            extend_original,
        } => {
            let label = if *extend_original {
                "with-columns"
            } else {
                "select"
            };
            (
                format!("{label}\\n{}", fmt_exprs(selectors, expr_arena)),
                from_ref(input),
            )
        },
        PhysNodeKind::Reduce { input, exprs } => (
            format!("reduce\\n{}", fmt_exprs(exprs, expr_arena)),
            from_ref(input),
        ),
        PhysNodeKind::StreamingSlice {
            input,
            offset,
            length,
        } => (
            format!("slice\\noffset: {offset}, length: {length}"),
            from_ref(input),
        ),
        PhysNodeKind::Filter { input, predicate } => (
            format!("filter\\n{}", fmt_exprs(from_ref(predicate), expr_arena)),
            from_ref(input),
        ),
        PhysNodeKind::SimpleProjection { input, columns } => (
            format!("select\\ncols: {}", columns.join(", ")),
            from_ref(input),
        ),
        PhysNodeKind::InMemorySink { input } => ("in-memory-sink".to_string(), from_ref(input)),
        PhysNodeKind::InMemoryMap { input, map: _ } => {
            ("in-memory-map".to_string(), from_ref(input))
        },
        PhysNodeKind::Map { input, map: _ } => ("map".to_string(), from_ref(input)),
        PhysNodeKind::Sort {
            input,
            by_column,
            slice: _,
            sort_options: _,
        } => (
            format!("sort\\n{}", fmt_exprs(by_column, expr_arena)),
            from_ref(input),
        ),
        PhysNodeKind::OrderedUnion { inputs } => ("ordered-union".to_string(), inputs.as_slice()),
        PhysNodeKind::Zip {
            inputs,
            null_extend,
        } => {
            let label = if *null_extend {
                "zip-null-extend"
            } else {
                "zip"
            };
            (label.to_string(), inputs.as_slice())
        },
        PhysNodeKind::Multiplexer { input } => ("multiplexer".to_string(), from_ref(input)),
        PhysNodeKind::FileScan {
            paths,
            file_info,
            hive_parts,
            output_schema: _,
            scan_type,
            predicate,
            file_options,
        } => {
            let name = match scan_type {
                FileScan::Parquet { .. } => "parquet-source",
                FileScan::Csv { .. } => "csv-source",
                FileScan::Ipc { .. } => "ipc-source",
                FileScan::NDJson { .. } => "ndjson-source",
                FileScan::Anonymous { .. } => "anonymous-source",
            };

            let mut out = name.to_string();
            let mut f = EscapeLabel(&mut out);

            {
                let paths_display = PathsDisplay(paths.as_ref());

                write!(f, "\npaths: {}", paths_display).unwrap();
            }

            {
                let total_columns =
                    file_info.schema.len() - usize::from(file_options.row_index.is_some());
                let n_columns = file_options
                    .with_columns
                    .as_ref()
                    .map(|columns| columns.len());

                if let Some(n) = n_columns {
                    write!(f, "\nprojection: {}/{total_columns}", n).unwrap();
                } else {
                    write!(f, "\nprojection: */{total_columns}").unwrap();
                }
            }

            if let Some(polars_io::RowIndex { name, offset }) = &file_options.row_index {
                write!(f, r#"\nrow index: name: "{}", offset: {}"#, name, offset).unwrap();
            }

            if let Some((offset, len)) = file_options.slice {
                write!(f, "\nslice: offset: {}, len: {}", offset, len).unwrap();
            }

            if let Some(predicate) = predicate.as_ref() {
                write!(f, "\nfilter: {}", predicate.display(expr_arena)).unwrap();
            }

            if let Some(v) = hive_parts
                .as_deref()
                .map(|x| x[0].get_statistics().column_stats().len())
            {
                write!(f, "\nhive: {} columns", v).unwrap();
            }

            (out, &[][..])
        },
    };

    out.push(format!(
        "{} [label=\"{}\"];",
        node_key.data().as_ffi(),
        label
    ));
    for input in inputs {
        visualize_plan_rec(*input, phys_sm, expr_arena, visited, out);
        out.push(format!(
            "{} -> {};",
            input.data().as_ffi(),
            node_key.data().as_ffi()
        ));
    }
}

pub fn visualize_plan(
    root: PhysNodeKey,
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> String {
    let mut visited: SecondaryMap<PhysNodeKey, ()> = SecondaryMap::new();
    let mut out = Vec::with_capacity(phys_sm.len() + 2);
    out.push("digraph polars {\nrankdir=\"BT\"".to_string());
    visualize_plan_rec(root, phys_sm, expr_arena, &mut visited, &mut out);
    out.push("}".to_string());
    out.join("\n")
}
