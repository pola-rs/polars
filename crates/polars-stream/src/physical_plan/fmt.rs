use std::fmt::Write;

use polars_plan::dsl::PartitionVariantIR;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, EscapeLabel};
use polars_plan::prelude::FileType;
use polars_utils::arena::Arena;
use polars_utils::slice_enum::Slice;
use slotmap::{Key, SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind};

/// A style of a graph node.
enum NodeStyle {
    InMemoryFallback,
    MemoryIntensive,
    Generic,
}

impl NodeStyle {
    const COLOR_IN_MEM_FALLBACK: &str = "0.0 0.3 1.0"; // Pastel red
    const COLOR_MEM_INTENSIVE: &str = "0.16 0.3 1.0"; // Pastel yellow

    /// Returns a style for a node kind.
    pub fn for_node_kind(kind: &PhysNodeKind) -> Self {
        use PhysNodeKind as K;
        match kind {
            K::InMemoryMap { .. } => Self::InMemoryFallback,
            K::InMemorySource { .. }
            | K::InputIndependentSelect { .. }
            | K::NegativeSlice { .. }
            | K::InMemorySink { .. }
            | K::Sort { .. }
            | K::GroupBy { .. }
            | K::EquiJoin { .. }
            | K::SemiAntiJoin { .. }
            | K::InMemoryJoin { .. }
            | K::Multiplexer { .. } => Self::MemoryIntensive,
            #[cfg(feature = "merge_sorted")]
            K::MergeSorted { .. } => Self::MemoryIntensive,
            _ => Self::Generic,
        }
    }

    /// Returns extra styling attributes (if any) for the graph node.
    pub fn node_attrs(&self) -> Option<String> {
        match self {
            Self::InMemoryFallback => Some(format!(
                "style=filled,fillcolor=\"{}\"",
                Self::COLOR_IN_MEM_FALLBACK
            )),
            Self::MemoryIntensive => Some(format!(
                "style=filled,fillcolor=\"{}\"",
                Self::COLOR_MEM_INTENSIVE
            )),
            Self::Generic => None,
        }
    }

    /// Returns a legend explaining the node style meaning.
    pub fn legend() -> String {
        format!(
            "fontname=\"Helvetica\"\nfontsize=\"10\"\nlabelloc=\"b\"\nlabel=<<BR/><BR/><B>Legend</B><BR/><BR/>◯ streaming engine node <FONT COLOR=\"{}\">⬤</FONT> potentially memory-intensive node <FONT COLOR=\"{}\">⬤</FONT> in-memory engine fallback>",
            Self::COLOR_MEM_INTENSIVE,
            Self::COLOR_IN_MEM_FALLBACK,
        )
    }
}

fn escape_graphviz(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

fn fmt_expr(f: &mut dyn Write, expr: &ExprIR, expr_arena: &Arena<AExpr>) -> std::fmt::Result {
    // Remove the alias to make the display better
    let without_alias = ExprIR::from_node(expr.node(), expr_arena);
    write!(
        f,
        "{} = {}",
        expr.output_name(),
        without_alias.display(expr_arena)
    )
}

pub enum FormatExprStyle {
    Select,
    NoAliases,
}

pub fn fmt_exprs_to_label(
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    style: FormatExprStyle,
) -> String {
    let mut buffer = String::new();
    let mut f = EscapeLabel(&mut buffer);
    fmt_exprs(&mut f, exprs, expr_arena, style);
    buffer
}

pub fn fmt_exprs(
    f: &mut dyn Write,
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    style: FormatExprStyle,
) {
    if matches!(style, FormatExprStyle::Select) {
        let mut formatted = Vec::new();

        let mut max_name_width = 0;
        let mut max_expr_width = 0;

        for e in exprs {
            let mut name = String::new();
            let mut expr = String::new();

            // Remove the alias to make the display better
            let without_alias = ExprIR::from_node(e.node(), expr_arena);

            write!(name, "{}", e.output_name()).unwrap();
            write!(expr, "{}", without_alias.display(expr_arena)).unwrap();

            max_name_width = max_name_width.max(name.chars().count());
            max_expr_width = max_expr_width.max(expr.chars().count());

            formatted.push((name, expr));
        }

        for (name, expr) in formatted {
            write!(f, "{name:>max_name_width$} = {expr:<max_expr_width$}\\n").unwrap();
        }
    } else {
        let Some(e) = exprs.first() else {
            return;
        };

        fmt_expr(f, e, expr_arena).unwrap();

        for e in &exprs[1..] {
            f.write_str("\\n").unwrap();
            fmt_expr(f, e, expr_arena).unwrap();
        }
    }
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

    let kind = &phys_sm[node_key].kind;

    use std::slice::from_ref;
    let (label, inputs) = match kind {
        PhysNodeKind::InMemorySource { df } => (
            format!(
                "in-memory-source\\ncols: {}",
                df.get_column_names_owned().join(", ")
            ),
            &[][..],
        ),
        #[cfg(feature = "python")]
        PhysNodeKind::PythonScan { .. } => ("python-scan".to_string(), &[][..]),
        PhysNodeKind::SinkMultiple { sinks } => {
            for sink in sinks {
                visualize_plan_rec(*sink, phys_sm, expr_arena, visited, out);
            }
            return;
        },
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
                format!(
                    "{label}\\n{}",
                    fmt_exprs_to_label(selectors, expr_arena, FormatExprStyle::Select)
                ),
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
                fmt_exprs_to_label(selectors, expr_arena, FormatExprStyle::Select)
            ),
            &[][..],
        ),
        PhysNodeKind::Reduce { input, exprs } => (
            format!(
                "reduce\\n{}",
                fmt_exprs_to_label(exprs, expr_arena, FormatExprStyle::Select)
            ),
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
            format!(
                "filter\\n{}",
                fmt_exprs_to_label(from_ref(predicate), expr_arena, FormatExprStyle::Select)
            ),
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
            FileType::Json(_) => ("ndjson-sink".to_string(), from_ref(input)),
            #[allow(unreachable_patterns)]
            _ => todo!(),
        },
        PhysNodeKind::PartitionSink {
            input,
            file_type,
            variant,
            ..
        } => {
            let variant = match variant {
                PartitionVariantIR::ByKey { .. } => "partition-by-key-sink",
                PartitionVariantIR::MaxSize { .. } => "partition-max-size-sink",
                PartitionVariantIR::Parted { .. } => "partition-parted-sink",
            };

            match file_type {
                #[cfg(feature = "parquet")]
                FileType::Parquet(_) => (format!("{}[parquet]", variant), from_ref(input)),
                #[cfg(feature = "ipc")]
                FileType::Ipc(_) => (format!("{}[ipc]", variant), from_ref(input)),
                #[cfg(feature = "csv")]
                FileType::Csv(_) => (format!("{}[csv]", variant), from_ref(input)),
                #[cfg(feature = "json")]
                FileType::Json(_) => (format!("{}[ndjson]", variant), from_ref(input)),
                #[allow(unreachable_patterns)]
                _ => todo!(),
            }
        },
        PhysNodeKind::InMemoryMap {
            input,
            map: _,
            format_str,
        } => {
            let mut label = String::new();
            label.push_str("in-memory-map");
            if let Some(format_str) = format_str {
                label.push('\n');

                let mut f = EscapeLabel(&mut label);
                write!(f, "{format_str}").unwrap();
            }
            (label, from_ref(input))
        },
        PhysNodeKind::Map { input, map: _ } => ("map".to_string(), from_ref(input)),
        PhysNodeKind::Sort {
            input,
            by_column,
            slice: _,
            sort_options: _,
        } => (
            format!(
                "sort\\n{}",
                fmt_exprs_to_label(by_column, expr_arena, FormatExprStyle::NoAliases)
            ),
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
        PhysNodeKind::MultiScan {
            scan_sources,
            file_reader_builder,
            cloud_options: _,
            projected_file_schema,
            output_schema,
            row_index,
            pre_slice,
            predicate,
            hive_parts,
            include_file_paths,
            cast_columns_policy: _,
            missing_columns_policy: _,
            extra_columns_policy: _,
            file_schema: _,
        } => {
            let mut out = format!("multi-scan[{}]", file_reader_builder.reader_name());
            let mut f = EscapeLabel(&mut out);

            write!(f, "\n{} source", scan_sources.len()).unwrap();

            if scan_sources.len() != 1 {
                write!(f, "s").unwrap();
            }

            write!(
                f,
                "\nproject: {} total, {} from file",
                output_schema.len(),
                projected_file_schema.len()
            )
            .unwrap();

            if let Some(ri) = row_index {
                write!(f, "\nrow index: name: {}, offset: {:?}", ri.name, ri.offset).unwrap();
            }

            if let Some(col_name) = include_file_paths {
                write!(f, "\nfile path column: {}", col_name).unwrap();
            }

            if let Some(pre_slice) = pre_slice {
                write!(f, "\nslice: offset: ").unwrap();

                match pre_slice {
                    Slice::Positive { offset, len: _ } => write!(f, "{}", *offset),
                    Slice::Negative {
                        offset_from_end,
                        len: _,
                    } => write!(f, "-{}", *offset_from_end),
                }
                .unwrap();

                write!(f, ", len: {}", pre_slice.len()).unwrap()
            }

            if let Some(predicate) = predicate {
                write!(f, "\nfilter: {}", predicate.display(expr_arena)).unwrap();
            }

            if let Some(v) = hive_parts.as_ref().map(|h| h.df().width()) {
                write!(f, "\nhive: {} column", v).unwrap();

                if v != 1 {
                    write!(f, "s").unwrap();
                }
            }

            (out, &[][..])
        },
        PhysNodeKind::GroupBy { input, key, aggs } => (
            format!(
                "group-by\\nkey:\\n{}\\naggs:\\n{}",
                fmt_exprs_to_label(key, expr_arena, FormatExprStyle::Select),
                fmt_exprs_to_label(aggs, expr_arena, FormatExprStyle::Select)
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
        }
        | PhysNodeKind::SemiAntiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            output_bool: _,
        } => {
            let label = match phys_sm[node_key].kind {
                PhysNodeKind::EquiJoin { .. } => "equi-join",
                PhysNodeKind::InMemoryJoin { .. } => "in-memory-join",
                PhysNodeKind::CrossJoin { .. } => "cross-join",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: false, ..
                } if args.how.is_semi() => "semi-join",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: false, ..
                } if args.how.is_anti() => "anti-join",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: true, ..
                } if args.how.is_semi() => "is-in",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: true, ..
                } if args.how.is_anti() => "is-not-in",
                _ => unreachable!(),
            };
            let mut label = label.to_string();
            write!(
                label,
                r"\nleft_on:\n{}",
                fmt_exprs_to_label(left_on, expr_arena, FormatExprStyle::NoAliases)
            )
            .unwrap();
            write!(
                label,
                r"\nright_on:\n{}",
                fmt_exprs_to_label(right_on, expr_arena, FormatExprStyle::NoAliases)
            )
            .unwrap();
            if args.how.is_equi() {
                write!(
                    label,
                    r"\nhow: {}",
                    escape_graphviz(&format!("{:?}", args.how))
                )
                .unwrap();
            }
            if args.nulls_equal {
                write!(label, r"\njoin-nulls").unwrap();
            }
            (label, &[*input_left, *input_right][..])
        },
        PhysNodeKind::CrossJoin {
            input_left,
            input_right,
            args: _,
        } => ("cross-join".to_string(), &[*input_left, *input_right][..]),
        #[cfg(feature = "merge_sorted")]
        PhysNodeKind::MergeSorted {
            input_left,
            input_right,
            key,
        } => {
            let mut out = "merge-sorted".to_string();
            let mut f = EscapeLabel(&mut out);

            write!(f, "\nkey: {}", key).unwrap();

            (out, &[*input_left, *input_right][..])
        },
    };

    let node_id = node_key.data().as_ffi();
    let style = NodeStyle::for_node_kind(kind);

    if let Some(attrs) = style.node_attrs() {
        out.push(format!("{node_id} [label=\"{label}\",{attrs}];"));
    } else {
        out.push(format!("{node_id} [label=\"{label}\"];"));
    }
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
    let mut out = Vec::with_capacity(phys_sm.len() + 3);
    out.push("digraph polars {\nrankdir=\"BT\"\nnode [fontname=\"Monospace\"]".to_string());
    out.push(NodeStyle::legend());
    visualize_plan_rec(root, phys_sm, expr_arena, &mut visited, &mut out);
    out.push("}".to_string());
    out.join("\n")
}
