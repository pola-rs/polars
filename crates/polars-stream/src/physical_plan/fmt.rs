use std::fmt::Write;

use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, EscapeLabel, FileScan, ScanSourcesDisplay};
use polars_plan::prelude::FileType;
use polars_utils::arena::Arena;
use polars_utils::itertools::Itertools;
use slotmap::{Key, SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind};

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
                df.get_column_names_owned().join(", ")
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
        PhysNodeKind::WithRowIndex {
            input,
            name,
            offset,
        } => (
            format!("with-row-index\\nname: {name}\\noffset: {offset:?}"),
            from_ref(input),
        ),
        PhysNodeKind::InputIndependentSelect { selectors } => (
            format!(
                "input-independent-select\\n{}",
                fmt_exprs(selectors, expr_arena)
            ),
            &[][..],
        ),
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
        PhysNodeKind::NegativeSlice {
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
        PhysNodeKind::FileSink {
            input, file_type, ..
        } => match file_type {
            #[cfg(feature = "parquet")]
            FileType::Parquet(_) => ("parquet-sink".to_string(), from_ref(input)),
            #[cfg(feature = "ipc")]
            FileType::Ipc(_) => ("ipc-sink".to_string(), from_ref(input)),
            #[cfg(feature = "csv")]
            FileType::Csv(_) => ("csv-sink".to_string(), from_ref(input)),
            #[cfg(feature = "json")]
            FileType::Json(_) => ("json-sink".to_string(), from_ref(input)),
            #[allow(unreachable_patterns)]
            _ => todo!(),
        },
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
            scan_sources,
            file_info,
            hive_parts,
            output_schema: _,
            scan_type,
            predicate,
            file_options,
        } => {
            let name = match scan_type {
                #[cfg(feature = "parquet")]
                FileScan::Parquet { .. } => "parquet-source",
                #[cfg(feature = "csv")]
                FileScan::Csv { .. } => "csv-source",
                #[cfg(feature = "ipc")]
                FileScan::Ipc { .. } => "ipc-source",
                #[cfg(feature = "json")]
                FileScan::NDJson { .. } => "ndjson-source",
                FileScan::Anonymous { .. } => "anonymous-source",
            };

            let mut out = name.to_string();
            let mut f = EscapeLabel(&mut out);

            {
                let disp = ScanSourcesDisplay(scan_sources);

                write!(f, "\npaths: {}", disp).unwrap();
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
        PhysNodeKind::GroupBy { input, key, aggs } => (
            format!(
                "group-by\\nkey:\\n{}\\naggs:\\n{}",
                fmt_exprs(key, expr_arena),
                fmt_exprs(aggs, expr_arena)
            ),
            from_ref(input),
        ),
        PhysNodeKind::InMemoryJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            ..
        }
        | PhysNodeKind::EquiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
        } => {
            let mut label = if matches!(phys_sm[node_key].kind, PhysNodeKind::EquiJoin { .. }) {
                "equi-join".to_string()
            } else {
                "in-memory-join".to_string()
            };
            write!(label, r"\nleft_on:\n{}", fmt_exprs(left_on, expr_arena)).unwrap();
            write!(label, r"\nright_on:\n{}", fmt_exprs(right_on, expr_arena)).unwrap();
            write!(
                label,
                r"\nhow: {}",
                escape_graphviz(&format!("{:?}", args.how))
            )
            .unwrap();
            if args.join_nulls {
                write!(label, r"\njoin-nulls").unwrap();
            }
            (label, &[*input_left, *input_right][..])
        },
        #[cfg(feature = "merge_sorted")]
        PhysNodeKind::MergeSorted {
            input_left,
            input_right,
            key,
        } => (
            format!("merge sorted on '{key}'"),
            &[*input_left, *input_right][..],
        ),
    };

    out.push(format!(
        "{} [label=\"{}\"];",
        node_key.data().as_ffi(),
        label
    ));
    for input in inputs {
        visualize_plan_rec(input.node, phys_sm, expr_arena, visited, out);
        out.push(format!(
            "{} -> {};",
            input.node.data().as_ffi(),
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
